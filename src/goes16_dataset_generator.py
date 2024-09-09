import os
import datetime
import logging
import fire
import numpy as np
import cv2
from tqdm import tqdm
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd

# Generate goes-16 lat-lon conversion files with script provided in data_handlers/goes_16_metadata_generator.py
# NOAA goes-16 Amazon S3 Bucket: https://noaa-goes16.s3.amazonaws.com/index.html
import satellite.constants as sat_cts
import satellite.functions as sat_functions
from utils import timeit
from PIL import Image
from typing import Dict, List


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GOES16 Dataset Generator")

# Define environment variable to speed up gdal
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = "nc"
os.environ["GDAL_HTTP_VERSION"] = "2"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_INGESTED_BYTES_AT_OPEN"] = "2000000"  # 2MB
os.environ["CPL_VSIL_CURL_CHUNK_SIZE"] = "5000000"  # 5MB
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
bucket = s3.Bucket(sat_cts.BUCKET)

if sat_cts.REGION == "C":
    METADATA_FOLDER = "datasets/ABI_L2_CMIP_M6C02_G16/CONUS"
elif sat_cts.REGION == "F":
    METADATA_FOLDER = "datasets/ABI_L2_CMIP_M6C02_G16/FULL_DISK"


@timeit
def convert_coordinates_to_pixel(
    lat: float,
    lon: float,
    REF_LAT: np.ndarray,
    REF_LON: np.ndarray,
    size: int,
    verbose: bool = True,
):
    if not (np.min(REF_LAT) < lat < np.max(REF_LAT)):
        raise ValueError(
            f"Latitude {lat} out of range. Latitude degrees covered: "
            f"{np.min(REF_LAT)} - {np.max(REF_LAT)}"
        )

    if not (np.min(REF_LON) < lon < np.max(REF_LON)):
        raise ValueError(
            f"Longitude {lon} out of range. Longitude degrees covered: "
            f"{np.min(REF_LON)} - {np.max(REF_LON)}"
        )

    # find closest coordinates to the given lat, lon and transform to pixel coordinates
    distances = abs(REF_LAT - lat) + abs(REF_LON - lon)
    y, x = np.unravel_index(np.nanargmin(distances), distances.shape)
    # avoid negative pixel coordinates

    if x - size // 2 < 0 or y - size // 2 == 0:
        raise ValueError(
            "Crop around lat-lon coordinates is outside the area covered. "
            "Reduce size or change coordinates."
        )

    if x + size // 2 >= REF_LAT.shape[1] or y + size // 2 >= REF_LAT.shape[0]:
        raise ValueError(
            "Crop around lat-lon coordinates is outside the area covered. "
            "Reduce size or change coordinates."
        )

    crop_lats = REF_LAT[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2]
    crop_lons = REF_LON[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2]

    if verbose:
        sat_functions.print_coordinates_square(x, y, lat, lon, size, REF_LAT, REF_LON)

    return crop_lats, crop_lons, x, y


@timeit
def get_S3_files_in_range(
    start_date: str, end_date: str, output_folder: str
) -> Dict[datetime.datetime, List[str]]:
    date_range = pd.date_range(start=start_date, end=end_date).tolist()

    for date in date_range:
        out_path = os.path.join(
            output_folder, f"{date.year}_{str(date.timetuple().tm_yday).zfill(3)}"
        )
        os.makedirs(out_path, exist_ok=True)
    files_in_s3_per_date = {}
    date_range = pd.date_range(start=start_date, end=end_date).tolist()
    for date in date_range:
        all_files_in_day = sat_functions.get_day_filenames(
            bucket, date.timetuple().tm_yday, date.year
        )
        files_in_s3_per_date[date] = all_files_in_day
    return files_in_s3_per_date


@timeit
def crop_processing(CMI_DQF_crop: np.ndarray, cosangs: np.ndarray) -> np.ndarray:
    """
    Process a crop of satellite imagery data.

    This function performs the following steps:
    1. Inpaints the image based on a mask derived from the DQF (Data Quality Flag).
    2. Normalizes the image to calculate planetary reflectance.
    3. Clips the reflectance values to the range [0, 1].
    4. Reduces the precision of the output to float16.

    Args:
        CMI_DQF_crop (np.ndarray): A 3D array containing the CMI (Cloud and Moisture Imagery)
                                   data in the first channel and DQF in the second channel.
        cosangs (np.ndarray): An array of cosine angles used in the normalization step.

    Returns:
        np.ndarray: The processed planetary reflectance as a 2D float16 array.

    Notes:
        - The function logs the percentage of pixels to be inpainted and the min-max values
          of the final planetary reflectance.
        - NaN values in the planetary reflectance are replaced with 0.
    """
    inpaint_mask = np.uint8(CMI_DQF_crop[1] != 0)
    logger.info(f"Percentage of pixels to inpaint: {np.mean(inpaint_mask) * 100:.2f}%")
    CMI_DQF_crop[0] = cv2.inpaint(CMI_DQF_crop[0], inpaint_mask, 3, cv2.INPAINT_NS)
    # normalizar
    planetary_reflectance = sat_functions.normalize(CMI_DQF_crop[0], cosangs, 0.15)
    planetary_reflectance[np.isnan(planetary_reflectance)] = 0
    planetary_reflectance = np.clip(planetary_reflectance, 0, 1.0)
    # reduce precision
    planetary_reflectance = planetary_reflectance.astype(np.float16)
    logger.info(
        f"    - min-max values for PR: {np.min(planetary_reflectance)} "
        f"- {np.max(planetary_reflectance)}"
    )
    return planetary_reflectance


@timeit
def main(
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-01",
    lat: float = -31.390502,
    lon: float = -57.954138,  # Salto, Uruguay
    size: int = 1024,
    output_folder: str = "datasets/goes16/",
    skip_night: bool = True,
    save_only_first: bool = False,
    save_as_npy: bool = True,
    verbose: bool = True,
):
    """Download/load images and perform a background substraction.

    Args:
        date: Reference date, in iso format YYYY-MM-DD.
        lat: latitude.
        lon: longitude.
        size: Size of the crop.
        ndates: Number of dates used for the processing (1 for the reference
            and ndates for the background).
        out: Where results should be saved.
        save_only_first_img: Save only the first image of the day.
        verbose: print extra info
    """

    # Load lat-lon conversion files created with data_handlers/goes_16_metadata_generator.py
    REF_LAT = np.load(os.path.join(METADATA_FOLDER, "lat.npy"))
    REF_LAT = np.ma.masked_array(REF_LAT[0], mask=REF_LAT[1])
    REF_LON = np.load(os.path.join(METADATA_FOLDER, "lon.npy"))
    REF_LON = np.ma.masked_array(REF_LON[0], mask=REF_LON[1])

    crop_lats, crop_lons, x, y = convert_coordinates_to_pixel(
        lat, lon, REF_LAT, REF_LON, size, verbose
    )
    files_in_s3_per_date = get_S3_files_in_range(start_date, end_date, output_folder)

    for date, all_files_in_s3 in files_in_s3_per_date.items():
        logger.info(
            f"Processing date: {date}. Num of available files: {len(all_files_in_s3)}"
        )
        for filename in tqdm(all_files_in_s3):
            t_coverage = filename.split("/")[-1].split("_")[3][1:]

            yl = t_coverage[0:4]
            day_of_year = t_coverage[4:7]
            hh = t_coverage[7:9]
            mm = t_coverage[9:11]
            ss = t_coverage[11:13]
            dd = str(date.day).zfill(2)
            mt = str(date.month).zfill(2)

            str_date = (
                str(dd) + "/" + str(mt) + "/" + str(yl) + " " + str(hh) + ":" + str(mm)
            )

            filename_date = datetime.datetime.strptime(str_date, "%d/%m/%Y %H:%M")
            if skip_night:
                cosangs, _ = sat_functions.get_cosangs(
                    filename_date, crop_lats, crop_lons
                )
                download_img = sat_functions.is_a_full_day_crop(cosangs)
            else:
                download_img = True

            if download_img:
                CMI_DQF_crop = sat_functions.read_crop(
                    filename, x, y, size, verbose
                )  # shape: (2, size, size)

                planetary_reflectance = crop_processing(CMI_DQF_crop, cosangs)

                out_path = os.path.join(output_folder, f"{yl}_{day_of_year}")
                crop_filename = f"{yl}_{day_of_year}_UTC_{hh}{mm}{ss}"

                if save_as_npy:
                    # Save image as npy array with float16 precision
                    np.save(out_path + f"/{crop_filename}.npy", planetary_reflectance)
                else:
                    planetary_reflectance = planetary_reflectance.astype(np.float32)
                    planetary_reflectance = (planetary_reflectance * 2 ** 16 - 1).astype(
                        np.uint16
                    )
                    Image.fromarray(planetary_reflectance, mode="I;16").save(
                        out_path + f"/{crop_filename}_L.png", fromat="PNG"
                    )

                if save_only_first:
                    break


if __name__ == "__main__":
    fire.Fire(main)
