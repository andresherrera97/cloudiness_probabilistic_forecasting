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

SENSOR = "ABI"  # Advanced Baseline Imager
PROCESSING_LEVEL = "L2"  # [L1b, L2]
PRODUCT = "CMIP"  # [CMIP, Rad]
HDF5_PRODUCT = "CMI"  # [CMI, Rad]
REGION = "C"  # [C, F]

PREFIX = f"{SENSOR}-{PROCESSING_LEVEL}-{PRODUCT}{REGION}"

CHANNEL = "C02"  # Visible Red Band


def read_crop(f, x, y, size, verbose=False):
    timing_start = time.time()
    with rasterio.open(f"HDF5:/vsis3/{BUCKET}/{f}://{HDF5_PRODUCT}") as ds:
        # Read only a window from the entire file
        crop = ds.read(window=((y, y + size), (x, x + size)))[0, ...]
    if verbose:
        print(
            f"Downloading crop: HDF5:/vsis3/{BUCKET}/{f}://{PRODUCT} in {(time.time() - timing_start):.2f}"
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
    # objects[0] type -> str

    if verbose:
        print(f"Filtering: {filter_prefix} in {time.time() - timing_start}")

    # Filter C02 files from the objects list
    files_c02 = natsorted([f for f in objects if CHANNEL in f])

    if save_only_first_of_hour:
        files_c02 = files_c02[:1]

    print(f"files_c02: {files_c02}")

    crops_c02 = np.asarray(
        [read_crop(f, x, y, size, verbose) for f in files_c02]
    )  # Array: [12, size, size]

    print(crops_c02.shape)

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

    distances = np.sqrt((REF_LAT - lat) ** 2 + (REF_LON - lon) ** 2)

    y, x = np.unravel_index(np.nanargmin(distances), distances.shape)
    x = max(0, x - size // 2)
    y = max(0, y - size // 2)
    print(f"x: {x}, y: {y}")
    # Transform string date to year + ordinal day and hour
    date = datetime.datetime.fromisoformat(date)

    # Download ref date
    time_download_start = time.time()
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
    )  # Array: [12*n_hours, size, size]
    print(
        f"Downloading and procedding time for {ref.shape[0]} images: {time.time() - time_download_start:.2f}"
    )

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
        im.save(out + f"/{i}.jpeg")


if __name__ == "__main__":
    main(
        date="2023-05-28",
        lat=0,
        lon=0,
        size=1024,
        out="datasets/goes16/",
        hours=[16],
        lat_lon_file="datasets/goes16_abi_conus_lat_lon.nc",
        save_only_first_of_hour=True,
        verbose=True,
    )
