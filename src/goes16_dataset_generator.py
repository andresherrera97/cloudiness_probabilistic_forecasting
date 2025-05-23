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

import satellite.constants as sat_cts
import satellite.functions as sat_functions
from utils import timeit
from PIL import Image
from typing import Dict, List, Optional


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GOES16 Dataset Generator")

# Define environment variable to speed up gdal
# os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = "nc"
os.environ["GDAL_HTTP_VERSION"] = "2"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
# os.environ["GDAL_INGESTED_BYTES_AT_OPEN"] = "2000000"  # 2MB
os.environ["GDAL_INGESTED_BYTES_AT_OPEN"] = "500000"  # 0.5MB
# os.environ["CPL_VSIL_CURL_CHUNK_SIZE"] = "5000000"  # 5MB
os.environ["CPL_VSIL_CURL_CHUNK_SIZE"] = "104857"
os.environ["CPL_VSIL_CURL_CACHE_SIZE"] = "200000000"  # 200MB cache to reduce re-reads
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
s3_client = boto3.client(
    "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=32)
)


def get_bucket_name(use_goes_16: bool = True):
    if use_goes_16:
        return "noaa-goes16"
    else:
        return "noaa-goes17"


def get_metadata_folder(use_goes_16: bool = True):
    if use_goes_16:
        if sat_cts.REGION == "C":
            return "ABI_L2_CMIP_M6C02_G16/CONUS"  # need to be relative to provided root
        elif sat_cts.REGION == "F":
            return "ABI_L2_CMIP_M6C02_G16/FULL_DISK"
    else:
        if sat_cts.REGION == "C":
            return "ABI_L2_CMIP_M6C02_G17/CONUS"  # need to be relative to provided root
        elif sat_cts.REGION == "F":
            return "ABI_L2_CMIP_M6C02_G17/FULL_DISK"


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
    bucket_name, start_date: str, end_date: Optional[str], output_folder: str
) -> Dict[datetime.datetime, List[str]]:
    if end_date is None:
        end_date = start_date
    if datetime.datetime.strptime(end_date, "%Y-%m-%d") < datetime.datetime.strptime(
        start_date, "%Y-%m-%d"
    ):
        raise ValueError(
            f"End date ({end_date}) must be greater than start date ({start_date})"
        )
    date_range = pd.date_range(start=start_date, end=end_date).tolist()

    files_in_s3_per_date = {}
    for date in date_range:
        all_files_in_day = sat_functions.get_day_filenames(
            # bucket, date.timetuple().tm_yday, date.year
            bucket_name, s3_client, date.timetuple().tm_yday, date.year
        )
        if len(all_files_in_day) > 0:
            out_path = os.path.join(
                output_folder, f"{date.year}_{str(date.timetuple().tm_yday).zfill(3)}"
            )
            os.makedirs(out_path, exist_ok=True)
            files_in_s3_per_date[date] = all_files_in_day
    return files_in_s3_per_date


@timeit
def crop_processing(CMI_DQF_crop: np.ndarray, cosangs: np.ndarray, calculate_pr: bool = True) -> np.ndarray:
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
    pixel_pct_to_inpaint = np.mean(inpaint_mask) * 100
    logger.info(f"Percentage of pixels to inpaint: {pixel_pct_to_inpaint:.6f}%")
    CMI_DQF_crop[0] = cv2.inpaint(CMI_DQF_crop[0], inpaint_mask, 3, cv2.INPAINT_NS)
    if calculate_pr:
        # return Planetary Reflectance (PR) crop
        processed_crop = sat_functions.normalize(CMI_DQF_crop[0], cosangs, 0.15)
        processed_crop[np.isnan(processed_crop)] = 0
        processed_crop = np.clip(processed_crop, 0, 1.0)
        # reduce precision
        processed_crop = processed_crop.astype(np.float16)
    else:
        # return Cloud and Moisture Imagery (CMI) crop
        processed_crop = CMI_DQF_crop[0].astype(np.float16)
    logger.info(
        f"    - min-max values for processed crop: {np.min(processed_crop)} "
        f"- {np.max(processed_crop)}"
    )
    return processed_crop, pixel_pct_to_inpaint


@timeit
def main(
    metadata_root="datasets",
    start_date: str = "2024-01-05",
    end_date: Optional[str] = None,
    lat: float = -31.390502,
    lon: float = -57.954138,  # Salto, Uruguay
    size: int = 1024,
    output_folder: str = "datasets/goes16/",
    skip_night: bool = True,
    save_only_first: bool = False,
    save_as_npy: bool = True,
    calculate_pr: bool = True,
    use_goes_16: bool = True,
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
    metadata_folder = get_metadata_folder(use_goes_16)

    # Load lat-lon conversion files created with data_handlers/goes_16_metadata_generator.py
    REF_LAT = np.load(os.path.join(metadata_root, metadata_folder, "lat.npy"))
    REF_LAT = np.ma.masked_array(REF_LAT[0], mask=REF_LAT[1])
    REF_LON = np.load(os.path.join(metadata_root, metadata_folder, "lon.npy"))
    REF_LON = np.ma.masked_array(REF_LON[0], mask=REF_LON[1])

    crop_lats, crop_lons, x, y = convert_coordinates_to_pixel(
        lat, lon, REF_LAT, REF_LON, size, verbose
    )

    del REF_LAT
    del REF_LON

    bucket_name = get_bucket_name(use_goes_16)
    bucket = s3.Bucket(bucket_name)
    files_in_s3_per_date = get_S3_files_in_range(bucket_name, start_date, end_date, output_folder)

    for date, all_files_in_s3 in files_in_s3_per_date.items():
        logger.info(
            f"Processing date: {date}. Num of available files: {len(all_files_in_s3)}"
        )
        is_day = []
        inpaint_pct = []
        filenames_processed = []

        for filename in tqdm(all_files_in_s3):
            filenames_processed.append(filename)
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
            out_path = os.path.join(output_folder, f"{yl}_{day_of_year}")
            cosangs, _ = sat_functions.get_cosangs(
                filename_date, crop_lats, crop_lons
            )
            if skip_night:
                download_img = sat_functions.is_a_full_day_crop(cosangs)
            else:
                download_img = True

            is_day.append(download_img)
            if download_img:
                # check if there is an improvement in the download time
                CMI_DQF_crop = sat_functions.read_crop_concurrent(
                    bucket_name, filename, x, y, size, verbose
                )  # shape: (2, size, size)

                processed_crop, pixel_pct_to_inpaint = crop_processing(
                    CMI_DQF_crop, cosangs, calculate_pr
                )
                inpaint_pct.append(pixel_pct_to_inpaint)
                crop_filename = f"{yl}_{day_of_year}_UTC_{hh}{mm}{ss}"

                if save_as_npy:
                    # Save image as npy array with float16 precision
                    np.save(out_path + f"/{crop_filename}.npy", processed_crop)
                else:
                    processed_crop = processed_crop.astype(np.float32)
                    processed_crop = (
                        processed_crop * 2**16 - 1
                    ).astype(np.uint16)
                    Image.fromarray(processed_crop, mode="I;16").save(
                        out_path + f"/{crop_filename}_L.png", fromat="PNG"
                    )

                if save_only_first:
                    break
            else:
                inpaint_pct.append(None)

        df = pd.DataFrame(
            {
                "filenames": filenames_processed,
                "is_day": is_day,
                "inpaint_pct": inpaint_pct,
            }
        )
        df.to_csv(
            os.path.join(
                out_path,
                f"data_{date.year}_{str(date.timetuple().tm_yday).zfill(3)}.csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    fire.Fire(main)
