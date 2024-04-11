# noaa goes 16 files https://noaa-goes16.s3.amazonaws.com/index.html

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
# PREFIX = "ABI-L1b-RadF"  # Advanced Baseline Imager Level 1b Full Disk
PREFIX = "ABI-L1b-RadC"  # Advanced Baseline Imager Level 1b CONUS
CHANNEL = "C02"  # Visible Red Band


def read_crop(f, x, y, size, verbose=False):
    timing_start = time.time()
    with rasterio.open("HDF5:/vsis3/" + BUCKET + "/" + f + "://Rad") as ds:
        # Read only a window from the entire file
        crop = ds.read(window=((y, y + size), (x, x + size)))[0, ...]
    if verbose:
        print(
            f"Downloading crop: {'HDF5:/vsis3/' + BUCKET + '/' + f + '://Rad'} in {time.time() - timing_start}"
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
    if save_only_first_of_hour:
        objects = objects[:1]
    # objects[0] type -> str

    if verbose:
        print(f"Filtering: {filter_prefix} in {time.time() - timing_start}")

    # Filter C02 files from the objects list
    files_c02 = natsorted([f for f in objects if CHANNEL in f])

    crops_c02 = np.asarray(
        [read_crop(f, 2 * x, 2 * y, 2 * size, verbose) for f in files_c02]
    )  # Array: [12, 2*size, 2*size]

    return crops_c02


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

    print(f"REF_LAT: {REF_LAT.shape}, REF_LON: {REF_LON.shape}")

    distances = np.sqrt((REF_LAT - lat) ** 2 + (REF_LON - lon) ** 2)
    print(f"dists: {distances.shape}")

    y, x = np.unravel_index(np.nanargmin(distances), distances.shape)
    x = max(0, x - size // 2)
    y = max(0, y - size // 2)
    print(f"x: {x}, y: {y}")
    # Transform string date to year + ordinal day and hour
    date = datetime.datetime.fromisoformat(date)

    # Download ref date
    ref = np.concatenate(
        [
            download_and_process(
                x,
                y,
                size,
                date.year,
                date.timetuple().tm_yday,
                h,
                save_only_first_of_hour,
                verbose=verbose,
            )
            for h in tqdm(hours)
        ],
        axis=0,
    )  # Array: [12*n_hours, 2*size, 2*size]
    # For CONUS there is an image every 5 minutes (60/5 = 12 images per hour)
    print(f"ref min-max values: {np.min(ref)}, {np.max(ref)}")

    # TODO better naming of the result
    os.makedirs(out, exist_ok=True)
    print("Saving images...")

    for i, im in enumerate(ref):
        im = Image.fromarray(
            ((im - np.min(im)) / (np.max(im) - np.min(im))) * 255
        ).convert("L")
        # im = Image.fromarray(im, mode="F")  # float32
        # im.save(out + f"/{i}.tiff", "TIFF")
        im.save(out + f"/{i}.png")


if __name__ == "__main__":
    main(
        date="2021-05-28",
        lat=500,
        lon=500,
        size=64,
        out="datasets/goes16_conus_64x64/",
        hours=[16],
        lat_lon_file="datasets/goes16_abi_conus_lat_lon.nc",
        save_only_first_of_hour=False,
        verbose=True,
    )
