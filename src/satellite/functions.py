import calendar
import concurrent.futures
import datetime
import numpy as np
import rasterio
import time
from natsort import natsorted
import satellite.constants as sat_cts
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Satellite Functions")


def read_crop(f: str, x: int, y: int, size: int, verbose: bool = True):
    '''
    Download a crop from the NetCDF file in the S3 bucket and return the CMI and DQF crops.
    The CMI crop is the Cloud and Moisture Imagery, and the DQF crop is the Data Quality Flag.
    The CMI crop is the actual image, and the DQF crop is a flag that indicates the quality of the data.
    Args:
        f: The filename of the NetCDF file in the S3 bucket.
        x: The x coordinate of the center of the crop.
        y: The y coordinate of the center of the crop.
        size: The size of the crop.
        verbose: If True, print the time taken to download the crop.
    Returns:
        CMI_DQF_crop: A numpy array with shape (2, size, size) where the first
        element is the CMI crop and the second element is the DQF crop.
    '''
    timing_start = time.time()
    # Read only a window from the entire file
    # for CONUS the whole image is 6000x10000

    with rasterio.open(f"HDF5:/vsis3/{sat_cts.BUCKET}/{f}://CMI") as ds:
        CMI_crop = ds.read(
            window=((y - size // 2, y + size // 2), (x - size // 2, x + size // 2))
        )[0, ...]
        # CMI_crop dtype: int16
        CMI_crop = CMI_crop.astype(np.float32)
        # Process CMI crop to match original data
        CMI_crop[CMI_crop == -1] = np.nan

        CMI_crop /= sat_cts.CORRECTION_FACTOR

        ds.close()

    with rasterio.open(f"HDF5:/vsis3/{sat_cts.BUCKET}/{f}://DQF") as ds:
        # Download data quality flag
        DQF_crop = ds.read(
            window=((y - size // 2, y + size // 2), (x - size // 2, x + size // 2))
        )[0, ...]
        DQF_crop = DQF_crop.astype(np.float32)
        ds.close()

        # 0 Good pixels
        # 1 Conditionally usable pixels
        # 2 Out of range pixels
        # 3 No value pixels
        # 4 Focal plane temperature threshold exceeded

    if verbose:
        logging.info(
            f"Downloading crops: HDF5:/vsis3/{sat_cts.BUCKET}/{f}://{sat_cts.PRODUCT} "
            f"in {(time.time() - timing_start):.2f} sec"
        )

    CMI_DQF_crop = np.stack([CMI_crop, DQF_crop], axis=0)

    return CMI_DQF_crop


def read_crop_concurrent(bucket_name, f: str, x: int, y: int, size: int, verbose: bool = True):
    timing_start = time.time()

    def read_data(product):
        with rasterio.open(f"HDF5:/vsis3/{bucket_name}/{f}://{product}") as ds:
            crop = ds.read(
                window=((y - size // 2, y + size // 2), (x - size // 2, x + size // 2))
            )[0, ...]
            crop = crop.astype(np.float32)
            if product == "CMI":
                crop[crop == -1] = np.nan
                crop /= sat_cts.CORRECTION_FACTOR
            return product, crop

    # Use ThreadPoolExecutor to run the reads concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_product = {
            executor.submit(read_data, product): product
            for product in ["CMI", "DQF"]
        }

        results = {}
        for future in as_completed(future_to_product):
            product, crop = future.result()
            results[product] = crop

    if verbose:
        logging.info(
            f"Downloaded crop {f}://{sat_cts.PRODUCT} "
            f"in {(time.time() - timing_start):.2f} sec"
        )

    return np.stack([results["CMI"], results["DQF"]], axis=0)


def print_coordinates_square(x, y, lat, lon, size, REF_LAT, REF_LON):
    top_left_pixel_coord = (y - size // 2, x - size // 2)
    top_right_pixel_coord = (y - size // 2, x + size // 2)
    bottom_left_pixel_coord = (y + size // 2, x - size // 2)
    bottom_right_pixel_coord = (y + size // 2, x + size // 2)

    print(f"lat: {lat}, lon: {lon} -> pixel coords x={x}, y={y}")
    print("geographic coords covered:")
    print(
        f"({REF_LAT[top_left_pixel_coord]:.1f}, {REF_LON[top_left_pixel_coord]:.1f}) "
        f"-- ({REF_LAT[top_right_pixel_coord]:.1f}, {REF_LON[top_right_pixel_coord]:.1f})"
    )
    print("      |                  |")
    print(
        f"({REF_LAT[bottom_left_pixel_coord]:.1f}, {REF_LON[bottom_left_pixel_coord]:.1f}) "
        f"-- ({REF_LAT[bottom_right_pixel_coord]:.1f}, {REF_LON[bottom_right_pixel_coord]:.1f})"
    )


# def get_day_filenames(bucket, date, year):
#     logging.info(f"Extracting all available images for {date} in NOAA S3 ...")
#     all_files_in_day = []
#     for hour in range(0, 24):
#         filter_prefix = sat_cts.PREFIX + f"/{year}/{date:03}/{str(hour).zfill(2)}/"
#         objects = [o.key for o in bucket.objects.filter(Prefix=filter_prefix)]
#         # objects: List[str]

#         # Filter C02 files from the objects list
#         hour_channel_files = [f for f in objects if sat_cts.CHANNEL in f]
#         all_files_in_day += hour_channel_files

#     all_files_in_day = natsorted(all_files_in_day)
#     logging.info("Done.")
#     return all_files_in_day


def get_day_filenames(bucket, s3_client, doy, year):
    """
    Retrieves S3 file list for a specific day using list_objects_v2.
    """
    prefix = sat_cts.PREFIX + f"/{year}/{doy:03}/"
    files = []

    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket, "Prefix": prefix}

    for page in paginator.paginate(**operation_parameters):
        if "Contents" in page:
            files.extend(
                [obj["Key"] for obj in page["Contents"] if sat_cts.CHANNEL in obj["Key"]]
            )

    return natsorted(files)


def daily_solar_angles(year: int, doy: int):
    days_year = calendar.isleap(year) * 1 + 365
    gamma_rad = 2 * np.pi * (doy - 1) / days_year
    delta_rad = (
        0.006918
        - 0.399912 * np.cos(gamma_rad)
        + 0.070257 * np.sin(gamma_rad)
        - 0.006758 * np.cos(2 * gamma_rad)
        + 0.000907 * np.sin(2 * gamma_rad)
        - 0.002697 * np.cos(3 * gamma_rad)
        + 0.00148 * np.sin(3 * gamma_rad)
    )
    eot_min = (
        60
        * 3.8196667
        * (
            0.000075
            + 0.001868 * np.cos(gamma_rad)
            - 0.032077 * np.sin(gamma_rad)
            - 0.014615 * np.cos(2 * gamma_rad)
            - 0.040849 * np.sin(2 * gamma_rad)
        )
    )
    return delta_rad, eot_min


def solar_angles(
    lat, lon, delta_rad, eot_min, hour, minute, second, ret_mask: bool = True
):
    lat_rad = lat * np.pi / 180
    h_sol = hour + minute / 60 + lon / 15 + eot_min / 60 + second / 3600
    w_rad = np.pi * (h_sol / 12 - 1)
    cos_zenith = np.sin(lat_rad) * np.sin(delta_rad) + np.cos(lat_rad) * np.cos(
        delta_rad
    ) * np.cos(w_rad)
    mask = cos_zenith > 0
    if cos_zenith.size > 1:
        cos_zenith[~mask] = 0
    elif cos_zenith.size == 1:
        cos_zenith = cos_zenith * mask
    if ret_mask:
        return cos_zenith, mask
    else:
        return cos_zenith


def get_cosangs(dtime: datetime.date, lats: np.ndarray, lons: np.ndarray):
    """Combine the two solar angle functions"""

    delta_rad, eot_min = daily_solar_angles(dtime.year, dtime.timetuple().tm_yday)
    cosangs, cos_mask = solar_angles(
        lats,
        lons,
        delta_rad,
        eot_min,
        dtime.hour,
        dtime.minute,
        dtime.second,
    )
    return cosangs, cos_mask


def is_a_full_day_crop(cosangs: np.ndarray, threshold: float = 0.15):
    return np.min(cosangs) > threshold


def normalize(img: np.ndarray, cosangs: np.ndarray, thresh: float = 0.15):
    """Normalization involves dividing by cosangs if greater
    than 0, and clipping if cosangs is below threshold"""
    n1 = np.divide(img, cosangs, out=np.zeros_like(img), where=thresh < cosangs)
    n2 = np.divide(
        img,
        cosangs,
        out=np.zeros_like(img),
        where=(0 < cosangs) * (cosangs <= thresh),
    )
    return n1 + np.clip(n2, 0, np.nanmean(n1))  # np.amax(n1))
