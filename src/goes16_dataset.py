import os

# Configure environment variables to optimize GDAL performance
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = "nc"
os.environ["GDAL_HTTP_VERSION"] = "2"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_INGESTED_BYTES_AT_OPEN"] = "20000000"  # 2MB
os.environ["CPL_VSIL_CURL_CHUNK_SIZE"] = "50000000"  # 5MB
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

import datetime
import logging
import time
import json
import tempfile
import calendar
import numpy as np
import fire
import boto3
import rasterio
import cv2
import tqdm
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional
from natsort import natsorted
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GOES16 Unified Script")

########################################
# Constants (from satellite/constants.py)
########################################
BUCKET = "noaa-goes16"
SENSOR = "ABI"
PROCESSING_LEVEL = "L2"
PRODUCT = "CMIP"
REGION = "F"  # [C, F]
PREFIX = f"{SENSOR}-{PROCESSING_LEVEL}-{PRODUCT}{REGION}"
CHANNEL = "C02"
CORRECTION_FACTOR = (2**12 - 1) / 1.3


########################################
# Decorator
########################################
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__} in {time.time() - start:.2f} sec")
        return result

    return wrapper


########################################
# Metadata Calculation
########################################
def calculate_degrees(file_id):
    """Convert GOES ABI fixed grid coords to lat/lon."""
    logger.info("Calculating lat/lon from the ABI projection...")

    x_coordinate_1d = file_id.variables["x"][:]
    y_coordinate_1d = file_id.variables["y"][:]
    proj_info = file_id.variables["goes_imager_projection"]
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height + proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis
    x2d, y2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    lambda_0 = (lon_origin * np.pi) / 180.0
    a = (np.sin(x2d) ** 2) + (
        (np.cos(x2d) ** 2)
        * ((np.cos(y2d) ** 2) + (((r_eq**2) / (r_pol**2)) * (np.sin(y2d) ** 2)))
    )
    b = -2.0 * H * np.cos(x2d) * np.cos(y2d)
    c = (H**2) - (r_eq**2)
    r_s = (-b - np.sqrt((b**2) - (4.0 * a * c))) / (2.0 * a)

    s_x = r_s * np.cos(x2d) * np.cos(y2d)
    s_y = -r_s * np.sin(x2d)
    s_z = r_s * np.cos(x2d) * np.sin(y2d)

    # Avoid warnings for domain issues on partial coverage
    np.seterr(all="ignore")

    abi_lat = (180.0 / np.pi) * np.arctan(
        ((r_eq**2) / (r_pol**2)) * (s_z / np.sqrt(((H - s_x) ** 2) + (s_y**2)))
    )
    abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)
    logger.info("Lat/lon calculated successfully!")
    return abi_lat, abi_lon


@timeit
def create_metadata(region: str, output_folder: str, download_ref: bool):
    """
    If needed, download a reference .nc file from NOAA S3 or load locally,
    then extract lat/lon and store them in .npy, plus metadata.json.
    """
    from netCDF4 import Dataset  # local import to avoid overhead

    logger.info("Preparing metadata generation...")
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # Some random reference files
    PATH_TO_LOCAL_CONUS = "datasets/CONUS_nc_files/OR_ABI-L2-CMIPC-M6C02_G16_s20241110601171_e20241110603544_c20241110604037.nc"
    PATH_TO_LOCAL_FULL_DISK = "datasets/full_disk_nc_files/OR_ABI-L2-CMIPF-M6C02_G16_s20220120310204_e20220120319512_c20220120319587.nc"
    S3_PATH_TO_CONUS = "ABI-L2-CMIPC/2024/111/10/OR_ABI-L2-CMIPC-M6C02_G16_s20241111011171_e20241111013544_c20241111014046.nc"
    S3_PATH_TO_FULL_DISK = "ABI-L2-CMIPF/2024/111/10/OR_ABI-L2-CMIPF-M6C02_G16_s20241111050205_e20241111059513_c20241111059581.nc"

    if region.lower() == "conus":
        file_key = S3_PATH_TO_CONUS
        local_file = PATH_TO_LOCAL_CONUS
        data_folder = "CONUS"
    else:
        file_key = S3_PATH_TO_FULL_DISK
        local_file = PATH_TO_LOCAL_FULL_DISK
        data_folder = "FULL_DISK"

    # Download or load
    if download_ref:
        logger.info("Downloading reference nc from S3...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
            s3.download_fileobj(BUCKET, file_key, tmp)
            tmp_name = tmp.name
        nc_dataset = Dataset(tmp_name)
    else:
        logger.info("Using local reference nc...")
        nc_dataset = Dataset(local_file)

    out_folder = os.path.join(output_folder, data_folder)
    os.makedirs(out_folder, exist_ok=True)

    metadata = {}
    # Minimal set of keys
    for key in nc_dataset.variables:
        if key in ["geospatial_lat_lon_extent", "goes_imager_projection"]:
            metadata[key] = nc_dataset.variables[key].__dict__
            for sub_key in metadata[key]:
                if not isinstance(metadata[key][sub_key], str):
                    metadata[key][sub_key] = metadata[key][sub_key].tolist()
        if key in ["x", "y", "x_image_bounds", "y_image_bounds"]:
            metadata[key] = list(nc_dataset.variables[key][:].astype(float))

    with open(os.path.join(out_folder, "metadata.json"), "w") as outfile:
        json.dump(metadata, outfile, indent=4)

    abi_lat, abi_lon = calculate_degrees(nc_dataset)
    lat_data_mask = np.concatenate((abi_lat.data[None], abi_lat.mask[None]), axis=0)
    lon_data_mask = np.concatenate((abi_lon.data[None], abi_lon.mask[None]), axis=0)
    np.save(os.path.join(out_folder, "lat.npy"), lat_data_mask)
    np.save(os.path.join(out_folder, "lon.npy"), lon_data_mask)
    logger.info(f"Metadata stored in {out_folder}")


########################################
# AWS Data Reading and Utilities
########################################
def read_crop(f: str, x: int, y: int, size: int, verbose: bool):
    """
    Download a windowed crop from S3. Returns stack of (CMI, DQF).
    """
    with rasterio.open(f"HDF5:/vsis3/{BUCKET}/{f}://CMI") as ds:
        CMI_crop = ds.read(
            window=((y - size // 2, y + size // 2), (x - size // 2, x + size // 2))
        )[0].astype(np.float32)
        CMI_crop[CMI_crop == -1] = np.nan
        CMI_crop /= CORRECTION_FACTOR
    with rasterio.open(f"HDF5:/vsis3/{BUCKET}/{f}://DQF") as ds:
        DQF_crop = ds.read(
            window=((y - size // 2, y + size // 2), (x - size // 2, x + size // 2))
        )[0].astype(np.float32)
    if verbose:
        logger.info(f"Downloaded crop from {f}.")
    return np.stack([CMI_crop, DQF_crop], axis=0)


def read_crop_concurrent(f: str, x: int, y: int, size: int, verbose: bool):
    def read_data(product, max_tries=3):
        num_tries = 0
        while num_tries < max_tries:
            try:
                with rasterio.open(f"HDF5:/vsis3/{BUCKET}/{f}://{product}") as ds:
                    c = ds.read(
                        window=((y - size // 2, y + size // 2), (x - size // 2, x + size // 2))
                    )[0].astype(np.float32)
                    if product == "CMI":
                        c[c == -1] = np.nan
                        c /= CORRECTION_FACTOR
            except error as e:
                num_tries += 1
                logger.error(f"Error reading {product} from {f}. Retrying...")
                time.sleep(1)
                continue
        return c

    with ThreadPoolExecutor(max_workers=2) as exe:
        futs = {exe.submit(read_data, p): p for p in ["CMI", "DQF"]}
        results = {}
        for fu in as_completed(futs):
            p = futs[fu]
            results[p] = fu.result()
    if verbose:
        logger.info(f"Downloaded concurrent crop from {f}.")
    return np.stack([results["CMI"], results["DQF"]], axis=0)


def print_coordinates_square(x, y, lat, lon, size, REF_LAT, REF_LON):
    """Diagnostic console print for bounding box coverage."""
    tl = (y - size // 2, x - size // 2)
    tr = (y - size // 2, x + size // 2)
    bl = (y + size // 2, x - size // 2)
    br = (y + size // 2, x + size // 2)
    print(f"(lat={lat}, lon={lon}) -> pixel (x={x}, y={y}). Region corners:")
    print(
        f"TL=({REF_LAT[tl]:.1f}, {REF_LON[tl]:.1f}) - TR=({REF_LAT[tr]:.1f}, {REF_LON[tr]:.1f})"
    )
    print("           |                   |")
    print(
        f"BL=({REF_LAT[bl]:.1f}, {REF_LON[bl]:.1f}) - BR=({REF_LAT[br]:.1f}, {REF_LON[br]:.1f})"
    )


def get_day_filenames(bucket, doy, year):
    """Collect all C02 files for that day in NOAA S3."""
    # logger.info(f"Extracting NOAA S3 images for DOY={doy} {year} ...")
    s3_objs = []
    for hour in range(24):
        prefix = PREFIX + f"/{year}/{doy:03}/{str(hour).zfill(2)}/"
        s3_objs += [o.key for o in bucket.objects.filter(Prefix=prefix)]
    return natsorted([f for f in s3_objs if CHANNEL in f])


def daily_solar_angles(year: int, doy: int):
    """For solar angle computations."""
    days_year = (1 if calendar.isleap(year) else 0) + 365
    gamma = 2 * np.pi * (doy - 1) / days_year
    delta = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    eot = (
        60
        * 3.8196667
        * (
            0.000075
            + 0.001868 * np.cos(gamma)
            - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma)
            - 0.040849 * np.sin(2 * gamma)
        )
    )
    return delta, eot


def solar_angles(lat, lon, delta, eot, h, m, s, ret_mask=True):
    lat_r = lat * np.pi / 180
    h_sol = h + m / 60 + lon / 15 + eot / 60 + s / 3600
    w_rad = np.pi * (h_sol / 12 - 1)
    cos_zen = np.sin(lat_r) * np.sin(delta) + np.cos(lat_r) * np.cos(delta) * np.cos(
        w_rad
    )
    mask = cos_zen > 0
    if cos_zen.size > 1:
        cos_zen[~mask] = 0
    elif cos_zen.size == 1:
        cos_zen = cos_zen * mask
    return (cos_zen, mask) if ret_mask else cos_zen


def get_cosangs(dtime: datetime.datetime, lats: np.ndarray, lons: np.ndarray):
    """Main wrapper for solar angles."""
    delta, eot = daily_solar_angles(dtime.year, dtime.timetuple().tm_yday)
    cosz, mask = solar_angles(
        lats, lons, delta, eot, dtime.hour, dtime.minute, dtime.second
    )
    return cosz, mask


def is_a_full_day_crop(cosangs: np.ndarray, threshold: float = 0.15):
    return np.min(cosangs) > threshold


def normalize(img: np.ndarray, cosangs: np.ndarray, thresh: float = 0.15):
    """Divide by cosangs > threshold, partially by cosangs <= threshold."""
    n1 = np.divide(img, cosangs, out=np.zeros_like(img), where=cosangs > thresh)
    n2 = np.divide(
        img, cosangs, out=np.zeros_like(img), where=(0 < cosangs) & (cosangs <= thresh)
    )
    return n1 + np.clip(n2, 0, np.nanmean(n1))


########################################
# Coordinate Crop
########################################
@timeit
def convert_coordinates_to_pixel(lat, lon, REF_LAT, REF_LON, size, verbose=True):
    if not (np.min(REF_LAT) < lat < np.max(REF_LAT)):
        raise ValueError(
            f"Lat {lat} out of range: {np.min(REF_LAT)} - {np.max(REF_LAT)}"
        )
    if not (np.min(REF_LON) < lon < np.max(REF_LON)):
        raise ValueError(
            f"Lon {lon} out of range: {np.min(REF_LON)} - {np.max(REF_LON)}"
        )
    distances = abs(REF_LAT - lat) + abs(REF_LON - lon)
    y, x = np.unravel_index(np.nanargmin(distances), distances.shape)

    if x - size // 2 < 0 or y - size // 2 == 0:
        raise ValueError("Crop too large or outside coverage. Adjust.")
    if x + size // 2 >= REF_LAT.shape[1] or y + size // 2 >= REF_LAT.shape[0]:
        raise ValueError("Crop too large or outside coverage. Adjust.")

    c_lats = REF_LAT[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2]
    c_lons = REF_LON[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2]
    if verbose:
        print_coordinates_square(x, y, lat, lon, size, REF_LAT, REF_LON)
    return c_lats, c_lons, x, y


########################################
# Get S3 files in date range
########################################
@timeit
def get_S3_files_in_range(start_date, end_date, output_folder):
    if end_date is None:
        end_date = start_date
    if datetime.datetime.strptime(end_date, "%Y-%m-%d") < datetime.datetime.strptime(
        start_date, "%Y-%m-%d"
    ):
        raise ValueError("End date must be >= start date")
    date_range = pd.date_range(start=start_date, end=end_date).tolist()

    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(BUCKET)
    files_in_s3_per_date = {}
    for dt in tqdm.tqdm(date_range):
        # Year check: skip < 2017 or > current year
        if dt.year < 2017 or dt.year > datetime.datetime.now().year:
            logger.info(f"Skipping {dt.year}. Outside supported range.")
            continue
        day_files = get_day_filenames(bucket, dt.timetuple().tm_yday, dt.year)
        if day_files:
            out_path = os.path.join(
                output_folder, f"{dt.year}_{str(dt.timetuple().tm_yday).zfill(3)}"
            )
            os.makedirs(out_path, exist_ok=True)
            files_in_s3_per_date[dt] = day_files
    return files_in_s3_per_date


########################################
# Crop processing
########################################
@timeit
def crop_processing(CMI_DQF_crop: np.ndarray, cosangs: np.ndarray) -> np.ndarray:
    """
    1. Inpaint using DQF mask, 2. normalize, 3. clip to [0,1], 4. float16.
    """
    inpaint_mask = np.uint8(CMI_DQF_crop[1] != 0)
    pct_inpaint = np.mean(inpaint_mask) * 100
    logger.info(f"Pixels to inpaint: {pct_inpaint:.4f}%")
    CMI_DQF_crop[0] = cv2.inpaint(CMI_DQF_crop[0], inpaint_mask, 3, cv2.INPAINT_NS)
    pr = normalize(CMI_DQF_crop[0], cosangs, 0.15)
    pr[np.isnan(pr)] = 0
    pr = np.clip(pr, 0, 1.0).astype(np.float16)
    logger.info(f"PR range: {pr.min()} - {pr.max()}")
    return pr, pct_inpaint


########################################
# Main
########################################
@timeit
def main(
    rootdir="datasets",
    outdir="salto1024",
    region: str = "full_disk",
    # metadata_root="datasets",
    start_date: str = "2024-01-05",
    end_date: Optional[str] = None,
    lat: float = -31.390502,
    lon: float = -57.954138,
    size: int = 1024,
    # output_folder: str = "datasets/goes16/",
    skip_night: bool = True,
    save_only_first: bool = False,
    save_as_npy: bool = True,
    verbose: bool = True,
    # download_metadata: bool = False,
):
    """
    Unifies metadata creation (once) + dataset download. Skips year < 2017 or year > current.
    """
    # 1) Create or ensure metadata is present
    metadata_dir = os.path.join(rootdir, "ABI_L2_CMIP_M6C02_G16")
    if region == "C":
        region_folder = "CONUS"
    else:
        region_folder = "FULL_DISK"
    if not os.path.exists(os.path.join(metadata_dir, region_folder)):
        create_metadata(
            region, metadata_dir, True
        )

    # 2) Load lat/lon from stored .npy

    ref_lat_path = os.path.join(
        metadata_dir, region_folder, "lat.npy"
    )
    ref_lon_path = os.path.join(
        metadata_dir, region_folder, "lon.npy"
    )

    REF_LAT = np.load(ref_lat_path)
    lat_data = np.ma.masked_array(REF_LAT[0], mask=REF_LAT[1])
    REF_LON = np.load(ref_lon_path)
    lon_data = np.ma.masked_array(REF_LON[0], mask=REF_LON[1])
    del REF_LAT, REF_LON

    c_lats, c_lons, x, y = convert_coordinates_to_pixel(
        lat, lon, lat_data, lon_data, size, verbose
    )
    del lat_data, lon_data

    output_folder = os.path.join(rootdir, outdir)
    files_per_date = get_S3_files_in_range(start_date, end_date, output_folder)
    # s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    # bucket = s3.Bucket(BUCKET)

    for dt, day_files in tqdm.tqdm(files_per_date.items()):
        logger.info(f"Date={dt}, total={len(day_files)}")
        is_day_list, inpaint_pct_list, filenames_processed = [], [], []
        out_day_path = os.path.join(
            output_folder, f"{dt.year}_{str(dt.timetuple().tm_yday).zfill(3)}"
        )

        for f in day_files:
            filenames_processed.append(f)
            t_str = f.split("/")[-1].split("_")[3][1:]  # sYYYYDDDHHMMSSS
            y4, dayy, hh, mm, ss = (
                t_str[0:4],
                t_str[4:7],
                t_str[7:9],
                t_str[9:11],
                t_str[11:13],
            )
            file_dt = datetime.datetime.strptime(
                f"{dt.day:02}/{dt.month:02}/{y4} {hh}:{mm}:{ss}", "%d/%m/%Y %H:%M:%S"
            )
            cosangs, _ = get_cosangs(file_dt, c_lats, c_lons)
            if skip_night:
                download_img = is_a_full_day_crop(cosangs)
            else:
                download_img = True
            is_day_list.append(download_img)

            if download_img:
                cmi_dqf_crop = read_crop_concurrent(f, x, y, size, verbose)
                pr, pct_inp = crop_processing(cmi_dqf_crop, cosangs)
                inpaint_pct_list.append(pct_inp)
                out_fname = f"{y4}_{dayy}_UTC_{hh}{mm}{ss}"
                if save_as_npy:
                    np.save(os.path.join(out_day_path, out_fname + ".npy"), pr)
                else:
                    pr_16u = ((pr.astype(np.float32) * (2**16 - 1))).astype(np.uint16)
                    Image.fromarray(pr_16u, mode="I;16").save(
                        os.path.join(out_day_path, out_fname + "_L.png"), format="PNG"
                    )
                if save_only_first:
                    break
            else:
                inpaint_pct_list.append(None)

        df = pd.DataFrame(
            {
                "filenames": filenames_processed,
                "is_day": is_day_list,
                "inpaint_pct": inpaint_pct_list,
            }
        )
        df.to_csv(
            os.path.join(
                out_day_path,
                f"data_{dt.year}_{str(dt.timetuple().tm_yday).zfill(3)}.csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    fire.Fire(main)
