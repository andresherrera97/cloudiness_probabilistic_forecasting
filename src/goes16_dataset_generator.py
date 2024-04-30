# Download goes-16 lat-lon conversion files: https://www.star.nesdis.noaa.gov/atmospheric-composition-training/satellite_data_goes_imager_projection.php
# NOAA goes-16 Amazon S3 Bucket: https://noaa-goes16.s3.amazonaws.com/index.html

import rasterio
from tqdm import tqdm
from natsort import natsorted
import numpy as np
import datetime
from typing import List
import os
import time
from PIL import Image
import netCDF4

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Define environment variable to speed up gdal
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = "nc"
os.environ["GDAL_HTTP_VERSION"] = "2"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_INGESTED_BYTES_AT_OPEN"] = "2000000"  # 2MB
os.environ["CPL_VSIL_CURL_CHUNK_SIZE"] = "5000000"  # 5MB
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

BUCKET = "noaa-goes16"

SENSOR = "ABI"  # Advanced Baseline Imager
PROCESSING_LEVEL = "L2"  # [L1b, L2]
PRODUCT = "CMIP"  # [CMIP, Rad]
HDF5_PRODUCT = "CMI"  # [CMI, Rad]

REGION = "C"  # [C, F]
# For CONUS there is an image every 5 minutes (60/5 = 12 images per hour)
# Full-Disk has an image every 10 minutes (60/10 = 6 images per hour)


PREFIX = f"{SENSOR}-{PROCESSING_LEVEL}-{PRODUCT}{REGION}"

CHANNEL = "C02"  # Visible Red Band

if REGION == "C":
    LAT_LON_FILE = "datasets/goes16_abi_conus_lat_lon.nc"
elif REGION == "F":
    LAT_LON_FILE = "datasets/goes16_abi_full_disk_lat_lon.nc"


def read_crop(f, x, y, size, verbose=False):
    timing_start = time.time()
    with rasterio.open(f"HDF5:/vsis3/{BUCKET}/{f}://{HDF5_PRODUCT}") as ds:
        # Read only a window from the entire file
        crop = ds.read(
            window=((y - size // 2, y + size // 2), (x - size // 2, x + size // 2))
        )[0, ...]
    if verbose:
        print(
            f"Downloading crop: HDF5:/vsis3/{BUCKET}/{f}://{PRODUCT} in {(time.time() - timing_start):.2f} sec"
        )
    return crop.astype(np.float32)


def download_and_process(
    x: int,
    y: int,
    size: int,
    year: int,
    date: int,
    hour: int,
    save_only_first_of_hour: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Download and process GOES 16 images.

    Args:
        x: Position of the crop (width).
        y: Position of the crop (height).
        year: Year of the day considered.
        date: Ordinal day.
        hour: Hour of the day.
        verbose: print extra info
    """

    timing_start = time.time()
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(BUCKET)
    filter_prefix = PREFIX + f"/{year}/{date:03}/{hour}/"
    objects = [o.key for o in bucket.objects.filter(Prefix=filter_prefix)]
    # objects: List[str]

    if verbose:
        print(f"Filtering: {filter_prefix} in {time.time() - timing_start}")

    # Filter C02 files from the objects list
    channel_files = natsorted([f for f in objects if CHANNEL in f])

    if save_only_first_of_hour:
        channel_files = channel_files[:1]

    crops = np.asarray(
        [read_crop(f, x, y, size, verbose) for f in channel_files]
    )  # Array: [12, size, size]

    return crops, channel_files


def main(
    date: str,
    lat: float,
    lon: float,
    size: int,
    out: str,
    hours: List[int] = [16, 17, 18, 19],
    lat_lon_file: str = "goes16_abi_conus_lat_lon.nc",
    save_only_first_of_hour: bool = False,
    verbose: bool = False,
):
    """Download/load images and perform a background substraction.

    Args:
        date: Reference date, in iso format YYYY-MM-DD.
        x: Position of the crop (width).
        y: Position of the crop (height).
        size: Size of the crop.
        ndates: Number of dates used for the processing (1 for the reference and ndates for the background).
        out: Where results should be saved.
        hours: List of hours (UTM) to use.
        cache: Path to cache (if used).
        lat_lon_file: Path to the ABI Conus lat lon file
        verbose: print extra info
    """

    ds = netCDF4.Dataset(lat_lon_file)

    REF_LAT, REF_LON = ds["latitude"][:], ds["longitude"][:]

    if not (np.min(REF_LAT) < lat < np.max(REF_LAT)):
        raise ValueError(
            f"Latitude {lat} out of range. Latitude degrees covered: {np.min(REF_LAT)} - {np.max(REF_LAT)}"
        )

    if not (np.min(REF_LON) < lon < np.max(REF_LON)):
        raise ValueError(
            f"Longitude {lon} out of range. Longitude degrees covered: {np.min(REF_LON)} - {np.max(REF_LON)}"
        )

    # find closest coordinates to the given lat, lon and transform to pixel coordinates
    distances = abs(REF_LAT - lat) + abs(REF_LON - lon)
    # TODO: check if this is the correct way to get the pixel coordinates
    y, x = np.unravel_index(np.nanargmin(distances), distances.shape)
    # avoid negative pixel coordinates

    if x - size // 2 < 0 or y - size // 2 == 0:
        raise ValueError(
            "Crop around lat-lon coordinates is outside the area covered. Reduce size or change coordinates."
        )

    if x + size // 2 >= REF_LAT.shape[1] or y + size // 2 >= REF_LAT.shape[0]:
        raise ValueError(
            "Crop around lat-lon coordinates is outside the area covered. Reduce size or change coordinates."
        )

    top_left_pixel_coord = (y - size // 2, x - size // 2)
    top_right_pixel_coord = (y - size // 2, x + size // 2)
    bottom_left_pixel_coord = (y + size // 2, x - size // 2)
    bottom_right_pixel_coord = (y + size // 2, x + size // 2)

    print(f"lat: {lat}, lon: {lon} -> pixel coords x={x}, y={y}")
    print("geographic coords covered:")
    print(
        f"({REF_LAT[top_left_pixel_coord]:.1f}, {REF_LON[top_left_pixel_coord]:.1f}) -- ({REF_LAT[top_right_pixel_coord]:.1f}, {REF_LON[top_right_pixel_coord]:.1f})"
    )
    print("      |                  |")
    print(
        f"({REF_LAT[bottom_left_pixel_coord]:.1f}, {REF_LON[bottom_left_pixel_coord]:.1f}) -- ({REF_LAT[bottom_right_pixel_coord]:.1f}, {REF_LON[bottom_right_pixel_coord]:.1f})"
    )

    # Transform string date to year + ordinal day and hour
    date = datetime.datetime.fromisoformat(date)

    out_path = os.path.join(out, f"{date.year}", f"{date.timetuple().tm_yday}")
    os.makedirs(out_path, exist_ok=True)

    # Download ref date
    time_download_start = time.time()

    day_crops_per_hour = []
    day_filenames_per_hour = []

    for h in tqdm(sorted(hours)):
        hour_crops, hour_filenames = download_and_process(
            x,
            y,
            size,
            date.year,
            date.timetuple().tm_yday,
            h,
            save_only_first_of_hour,
            verbose=verbose,
        )
        day_crops_per_hour.append(hour_crops)
        day_filenames_per_hour += hour_filenames

    day_crops_array = np.concatenate(day_crops_per_hour, axis=0)

    time_download_end = time.time()
    print(
        f"Downloading and processing time for {day_crops_array.shape[0]} images: {(time_download_end - time_download_start):.2f}"
    )
    print(
        f"    - Mean time per image: {(time_download_end - time_download_start) / day_crops_array.shape[0]:.2f}"
    )

    print("Saving images...")

    # TODO: Check if its better to save images per hour or per day

    for i, im in enumerate(day_crops_array):
        crop_filename = day_filenames_per_hour[i].split("/")[-1].split(".")[0]
        im = Image.fromarray(
            ((im - np.min(im)) / (np.max(im) - np.min(im))) * 255
        ).convert("L")
        # im = Image.fromarray(im, mode="F")  # float32
        # im.save(out + f"/{i}.tiff", "TIFF")
        im.save(out_path + f"/{crop_filename}.jpeg")


if __name__ == "__main__":
    main(
        date="2023-05-31",
        lat=36.5,
        lon=-80,
        size=128,
        out="datasets/goes16/",
        hours=[16],
        lat_lon_file=LAT_LON_FILE,
        save_only_first_of_hour=True,
        verbose=True,
    )
