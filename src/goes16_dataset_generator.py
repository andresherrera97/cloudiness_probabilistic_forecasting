import os
import time
import datetime
import numpy as np
import cv2
from tqdm import tqdm
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Generate goes-16 lat-lon conversion files with script provided in data_handlers/goes_16_metadata_generator.py
# NOAA goes-16 Amazon S3 Bucket: https://noaa-goes16.s3.amazonaws.com/index.html
import satellite.constants as sat_cts
import satellite.functions as sat_functions


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


def main(
    date: str,
    lat: float,
    lon: float,
    size: int,
    out: str,
    save_only_first_img: bool = False,
    verbose: bool = False,
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

    # Transform string date to year + ordinal day and hour
    date = datetime.datetime.fromisoformat(date)

    out_path = os.path.join(
        out, f"{date.year}_{str(date.timetuple().tm_yday).zfill(3)}"
    )
    os.makedirs(out_path, exist_ok=True)

    # Download ref date
    time_download_start = time.time()

    all_files_in_day = sat_functions.get_day_filenames(
        bucket, date.timetuple().tm_yday, date.year
    )

    # for filename in tqdm(all_files_in_day):
    for filename in all_files_in_day:

        t_coverage = filename.split("/")[-1].split("_")[3][1:]

        # exmaple time_coverage_start -> start of scan time
        # dataset_name: s20201971301225
        # time_coverage_start: 2020-07-15T13:01:22.5Z

        yl = t_coverage[0:4]
        day_of_year = t_coverage[4:7]
        hh = t_coverage[7:9]
        mm = t_coverage[9:11]
        ss = t_coverage[11:13]

        res = datetime.datetime.strptime(yl + "-" + day_of_year, "%Y-%j").strftime(
            "%m-%d-%Y"
        )

        dd = res.split("-")[1]
        mt = res.split("-")[0]

        str_date = (
            str(dd) + "/" + str(mt) + "/" + str(yl) + " " + str(hh) + ":" + str(mm)
        )

        filename_date = datetime.datetime.strptime(str_date, "%d/%m/%Y %H:%M")
        print(filename_date)
        cosangs, _ = sat_functions.get_cosangs(filename_date, crop_lats, crop_lons)
        is_full_day = sat_functions.is_a_full_day_crop(cosangs)
        print(f"    - is_full_day: {is_full_day}")

        if is_full_day:
            CMI_DQF_crop = sat_functions.read_crop(
                filename, x, y, size, verbose
            )  # shape: (2, size, size)
            inpaint_mask = np.uint8(CMI_DQF_crop[1] != 0)
            CMI_DQF_crop[0] = cv2.inpaint(
                CMI_DQF_crop[0], inpaint_mask, 3, cv2.INPAINT_NS
            )
            # normalizar
            planetary_reflectance = sat_functions.normalize(
                CMI_DQF_crop[0], cosangs, 0.15
            )
            planetary_reflectance[np.isnan(planetary_reflectance)] = 0
            planetary_reflectance = np.clip(planetary_reflectance, 0, 1.0)
            planetary_reflectance *= 100
            print(
                f"    - min-max values for PR: {np.nanmin(planetary_reflectance)} "
                f"- {np.nanmax(planetary_reflectance)}"
            )
            # Save image as npy array
            crop_filename = f"{yl}_{day_of_year}_UTC_{hh}{mm}{ss}"

            np.save(out_path + f"/{crop_filename}.npy", planetary_reflectance)

            if save_only_first_img:
                break

    time_download_end = time.time()
    print(
        f"Downloading and processing time {(time_download_end - time_download_start):.2f}"
    )


if __name__ == "__main__":
    main(
        date="2024-01-21",
        lat=-34,
        lon=-55,
        size=512,
        out="datasets/goes16/",
        save_only_first_img=True,
        verbose=True,
    )
