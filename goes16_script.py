import os
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import json
import tempfile
import calendar
import datetime
import logging
import numpy as np
import fire
import boto3
import rasterio
import cv2
import tqdm
import pandas as pd

from PIL import Image
from typing import Optional
from natsort import natsorted
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore import UNSIGNED
from botocore.config import Config

########################################
# 1) SHARED CONSTANTS & FUNCTIONS
########################################

# Configure environment variables to optimize GDAL performance
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = "nc"
os.environ["GDAL_HTTP_VERSION"] = "2"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_INGESTED_BYTES_AT_OPEN"] = "20000000"
os.environ["CPL_VSIL_CURL_CHUNK_SIZE"] = "10485760"
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GOES16 Modular Script")

# Some constants
BUCKET = "noaa-goes16"
SENSOR = "ABI"
PROCESSING_LEVEL = "L2"
PRODUCT = "CMIP"
REGION = "F"  # e.g., Full Disk
PREFIX = f"{SENSOR}-{PROCESSING_LEVEL}-{PRODUCT}{REGION}"
CHANNEL = "C02"
CORRECTION_FACTOR = (2**12 - 1) / 1.3


def timeit(func):
    """Decorator to time function execution."""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__} in {time.time() - start:.2f} sec")
        return result

    return wrapper


def calculate_degrees(file_id):
    """
    Convert GOES ABI fixed-grid coordinates to latitude/longitude arrays.
    """
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
    r_s = (-b - np.sqrt(np.maximum(0, (b**2) - (4.0 * a * c)))) / (2.0 * a)

    s_x = r_s * np.cos(x2d) * np.cos(y2d)
    s_y = -r_s * np.sin(x2d)
    s_z = r_s * np.cos(x2d) * np.sin(y2d)

    # Avoid warnings for domain issues on partial coverage
    np.seterr(all="ignore")

    abi_lat = (180.0 / np.pi) * np.arctan(
        ((r_eq**2) / (r_pol**2)) * (s_z / np.sqrt(((H - s_x) ** 2) + (s_y**2)))
    )
    abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

    # Ensure masks are always arrays
    if np.isscalar(abi_lat.mask):
        abi_lat = np.ma.masked_array(abi_lat, mask=np.zeros_like(abi_lat, dtype=bool))
    if np.isscalar(abi_lon.mask):
        abi_lon = np.ma.masked_array(abi_lon, mask=np.zeros_like(abi_lon, dtype=bool))

    logger.info("Lat/lon calculated successfully!")
    return abi_lat, abi_lon


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
    """
    Compute cos(zenith) given a lat/lon and a time of day (UTC).
    """
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


@timeit
def get_cosangs(dtime: datetime.datetime, lats: np.ndarray, lons: np.ndarray):
    """Main wrapper for solar angles for a given date/time."""
    delta, eot = daily_solar_angles(dtime.year, dtime.timetuple().tm_yday)
    cosz, mask = solar_angles(
        lats, lons, delta, eot, dtime.hour, dtime.minute, dtime.second
    )
    return cosz, mask


def is_a_full_day_crop(cosangs: np.ndarray, threshold: float = 0.15):
    """
    Decide if the entire crop is in daytime (cos(zenith) > threshold).
    Return True if minimum cos(zenith) across the crop > threshold.
    """
    return np.min(cosangs) > threshold


def normalize(img: np.ndarray, cosangs: np.ndarray, thresh: float = 0.15):
    """
    Normalize reflectances by dividing by cos(zenith) in well-lit areas,
    limit overly bright areas, clip to [0,1].
    """
    n1 = np.divide(img, cosangs, out=np.zeros_like(img), where=cosangs > thresh)
    n2 = np.divide(
        img, cosangs, out=np.zeros_like(img), where=(0 < cosangs) & (cosangs <= thresh)
    )
    return n1 + np.clip(n2, 0, np.nanmean(n1))


def print_coordinates_square(x, y, lat, lon, size, REF_LAT, REF_LON):
    """Diagnostic console print for bounding box coverage."""
    tl = (y - size // 2, x - size // 2)
    tr = (y - size // 2, x + size // 2)
    bl = (y + size // 2, x - size // 2)
    br = (y + size // 2, x + size // 2)
    print(f"(lat={lat}, lon={lon}) -> pixel (x={x}, y={y}). Region corners:")
    print(
        f"TL=({REF_LAT[tl]:.1f}, {REF_LON[tl]:.1f}) "
        f"- TR=({REF_LAT[tr]:.1f}, {REF_LON[tr]:.1f})"
    )
    print("           |                   |")
    print(
        f"BL=({REF_LAT[bl]:.1f}, {REF_LON[bl]:.1f}) "
        f"- BR=({REF_LAT[br]:.1f}, {REF_LON[br]:.1f})"
    )


def convert_coordinates_to_pixel(lat, lon, REF_LAT, REF_LON, size, verbose=True):
    """
    Return the sub-array of lat/lon plus the x,y pixel coordinates
    that correspond to the requested lat/lon center.
    """
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

    # check coverage
    if x - size // 2 < 0 or y - size // 2 < 0:
        raise ValueError("Crop too large or outside coverage. Adjust.")
    if x + size // 2 >= REF_LAT.shape[1] or y + size // 2 >= REF_LAT.shape[0]:
        raise ValueError("Crop too large or outside coverage. Adjust.")

    c_lats = REF_LAT[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2]
    c_lons = REF_LON[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2]

    if verbose:
        print_coordinates_square(x, y, lat, lon, size, REF_LAT, REF_LON)
    return c_lats, c_lons, x, y


########################################
# 1) DOWNLOADING & PROCESSING MODULE (used at last)
########################################
class Downloader:
    """
    Handles the actual cropping, reading, solar angle checks, inpainting, saving, etc.
    Now includes batch-based parallel downloading/processing.
    """

    def __init__(self):
        # If needed, you can initialize more things here
        pass

    @timeit
    def _read_crop(self, f: str, x: int, y: int, size: int, verbose: bool):
        """
        Non-concurrent version of read_crop for clarity when using batch parallelization.
        Fetches CMI & DQF sequentially.
        """

        def read_data(product, max_tries=3):
            num_tries = 0
            while num_tries < max_tries:
                try:
                    with rasterio.open(f"HDF5:/vsis3/{BUCKET}/{f}://{product}") as ds:
                        c = ds.read(
                            window=(
                                (y - size // 2, y + size // 2),
                                (x - size // 2, x + size // 2),
                            )
                        )[0].astype(np.float32)
                        if product == "CMI":
                            c[c == -1] = np.nan
                            c /= CORRECTION_FACTOR
                    return c
                except Exception as e:
                    num_tries += 1
                    logger.error(f"Error {e} reading {product} from {f}. Retrying...")
                    time.sleep(1)
                    continue
            raise ValueError(
                f"Failed to read {product} from {f} after {max_tries} tries."
            )

        CMI = read_data("CMI")
        DQF = read_data("DQF")
        if verbose:
            logger.info(f"Downloaded crop from {f}")
        return np.stack([CMI, DQF], axis=0)

    @timeit
    def crop_processing(
        self, CMI_DQF_crop: np.ndarray, cosangs: np.ndarray
    ) -> np.ndarray:
        """
        1) Inpaint using DQF mask, 2) normalize, 3) clip to [0,1], 4) float16.
        Returns (processed_image, inpaint_percent).
        """
        inpaint_mask = np.uint8(CMI_DQF_crop[1] != 0)
        pct_inpaint = np.mean(inpaint_mask) * 100
        logger.info(f"Pixels to inpaint: {pct_inpaint:.4f}%")

        # Step 1: inpaint
        CMI_DQF_crop[0] = cv2.inpaint(CMI_DQF_crop[0], inpaint_mask, 3, cv2.INPAINT_NS)

        # Step 2: normalize
        pr = normalize(CMI_DQF_crop[0], cosangs, 0.15)

        # Step 3: remove nans and clip
        pr[np.isnan(pr)] = 0
        pr = np.clip(pr, 0, 1.0).astype(np.float16)

        logger.info(f"PR range: {pr.min()} - {pr.max()}")
        return pr, pct_inpaint

    def _download_and_process_file(
        self,
        f: str,
        x: int,
        y: int,
        size: int,
        c_lats: np.ndarray,
        c_lons: np.ndarray,
        skip_night: bool,
        verbose: bool,
        out_day_path: str,
        save_as: str,
        save_only_first: bool,
    ):
        """
        Helper function to download a single file, process it,
        and return logging info for CSV later.
        """
        # Parse time from filename
        t_str = f.split("/")[-1].split("_")[3][1:]  # sYYYYDDDHHMMSSS
        y4, dayy, hh, mm, ss = (
            t_str[0:4],
            t_str[4:7],
            t_str[7:9],
            t_str[9:11],
            t_str[11:13],
        )
        file_dt = datetime.datetime.strptime(
            f"{dayy}/{y4} {hh}:{mm}:{ss}", "%j/%Y %H:%M:%S"
        )
        # 1) Compute solar angles
        cosangs, _ = get_cosangs(file_dt, c_lats, c_lons)
        # 2) Skip nighttime if requested
        if skip_night:
            download_img = is_a_full_day_crop(cosangs)
        else:
            download_img = True

        if not download_img:
            return f, False, None  # nighttime -> skip

        # 3) Download the CMI+DQF crop
        cmi_dqf_crop = self._read_crop(f, x, y, size, verbose)
        # 4) Process (inpaint, normalize, clip)
        pr, pct_inp = self.crop_processing(cmi_dqf_crop, cosangs)

        # 5) Save results
        out_fname = f"{y4}_{dayy}_UTC_{hh}{mm}{ss}"
        os.makedirs(
            os.path.dirname(os.path.join(out_day_path, out_fname)), exist_ok=True
        )

        if save_as == "npy":
            np.save(
                os.path.join(out_day_path, out_fname + ".npy"), pr.astype(np.float16)
            )
        elif save_as == "png16":
            pr_16u = (np.round(pr.astype(np.float32) * (2**16 - 1))).astype(np.uint16)
            Image.fromarray(pr_16u, mode="I;16").save(
                os.path.join(out_day_path, out_fname + "_L.png"), format="PNG"
            )
        elif save_as == "png8":
            pr_8u = np.round(pr * 255).astype(np.uint8)
            Image.fromarray(pr_8u, mode="L").save(
                os.path.join(out_day_path, out_fname + "_L.png"), format="PNG"
            )

        # 6) If save_only_first is True, we can short-circuit here
        if save_only_first:
            # Not returning a "special" condition — just note that the caller can handle breaking
            pass

        return f, True, pct_inp

    @timeit
    def download_files(
        self,
        metadata_path: str,
        files_per_date_path: str,
        outdir: str = "salto1024",
        lat: float = -31.390502,
        lon: float = -57.954138,
        size: int = 1024,
        skip_night: bool = True,
        save_only_first: bool = False,
        save_as: str = "npy",
        verbose: bool = True,
        batch_size: int = 4,
    ):
        """
        Download and process GOES16 data for the given lat/lon region in parallel batches.

        - `metadata_path`: path containing lat.npy and lon.npy
        - `files_per_date_path`: JSON file containing {datetime_string: [list_of_files]}
        - `outdir`: base folder to write outputs
        - `batch_size`: number of files to process per batch
        - `save_as`: 'npy', 'png16', or 'png8'
        - `skip_night`: if True, skip nighttime
        - `save_only_first`: if True, save only the first file per day (if daytime)
        - etc.
        """
        t0 = time.time()
        assert save_as in ["npy", "png16", "png8"], "Invalid save_as option."

        # 1) Load metadata
        lat_path = os.path.join(metadata_path, "lat.npy")
        lon_path = os.path.join(metadata_path, "lon.npy")
        if not (os.path.exists(lat_path) and os.path.exists(lon_path)):
            raise FileNotFoundError("lat.npy or lon.npy not found in metadata_path!")

        REF_LAT = np.load(lat_path)
        lat_data = np.ma.masked_array(REF_LAT[0], mask=REF_LAT[1])
        REF_LON = np.load(lon_path)
        lon_data = np.ma.masked_array(REF_LON[0], mask=REF_LON[1])
        del REF_LAT, REF_LON

        # 2) Convert user lat/lon to pixel coords
        c_lats, c_lons, x, y = convert_coordinates_to_pixel(
            lat, lon, lat_data, lon_data, size, verbose
        )
        del lat_data, lon_data

        # 3) Load the JSON containing {datetime-string: [list_of_files]}
        if not os.path.exists(files_per_date_path):
            raise FileNotFoundError(f"{files_per_date_path} not found!")
        with open(files_per_date_path, "r") as f:
            raw_dict = json.load(f)

        # Convert string keys to datetime
        files_per_date = {}
        for date_str, file_list in raw_dict.items():
            dt = datetime.datetime.fromisoformat(date_str)
            files_per_date[dt] = file_list

        # 4) Process each day
        for dt, day_files in tqdm.tqdm(files_per_date.items()):
            logger.info(f"Date={dt}, total files={len(day_files)}")

            if not day_files:
                continue

            # Prepare output folder for this day
            out_day_path = os.path.join(
                outdir, f"{dt.year}_{str(dt.timetuple().tm_yday).zfill(3)}"
            )
            os.makedirs(out_day_path, exist_ok=True)

            # If user only wants the first (daytime) file,
            # we can short-circuit if we find one quickly, but let's keep it consistent:
            if save_only_first:
                # We'll do the same approach with batches, but we'll stop on the first day file we get.
                pass

            # Split day_files into sub-batches
            day_batches = [
                day_files[i : i + batch_size]
                for i in range(0, len(day_files), batch_size)
            ]

            # We'll accumulate results from all batches here
            filenames_processed = []
            is_day_list = []
            inpaint_pct_list = []

            # 4a) Process each batch in parallel
            #    Each worker processes the entire batch sequentially
            #    so we only spawn as many workers as sub-batches (not files).
            with ProcessPoolExecutor(max_workers=len(day_batches)) as executor:
                future_to_batch_idx = {}
                for b_idx, file_batch in enumerate(day_batches):
                    future = executor.submit(
                        _process_batch,
                        file_batch,
                        x,
                        y,
                        size,
                        c_lats,
                        c_lons,
                        skip_night,
                        verbose,
                        out_day_path,
                        save_as,
                        save_only_first,
                        self._download_and_process_file,  # Pass your method
                    )
                    future_to_batch_idx[future] = b_idx

                # 4b) Collect results
                #     Remember that if save_only_first=True, a batch may stop early,
                #     but other batches may still be running (unless you add extra logic to cancel them).
                for future in as_completed(future_to_batch_idx):
                    batch_results = (
                        future.result()
                    )  # list of (filename, is_day, inpaint_pct)
                    for filename, is_day, pct_inp in batch_results:
                        filenames_processed.append(filename)
                        is_day_list.append(is_day)
                        inpaint_pct_list.append(pct_inp)

                        # If user wants only the first daytime file, and we found it,
                        # you *could* break and try to cancel other futures, but that
                        # requires more advanced logic. We'll keep it simple here.
                        if save_only_first and is_day:
                            # We found the first day file, so let's ignore the rest.
                            # But note: the other futures are still running in the background.
                            break

            # 5) Summarize each day
            if filenames_processed:
                df = pd.DataFrame(
                    {
                        "filenames": filenames_processed,
                        "is_day": is_day_list,
                        "inpaint_pct": inpaint_pct_list,
                    }
                )
                df_path = os.path.join(
                    out_day_path,
                    f"data_{dt.year}_{str(dt.timetuple().tm_yday).zfill(3)}.csv",
                )
                df.to_csv(df_path, index=False)

        logger.info("All downloads completed.")
        logger.info(f"Total time: {time.time() - t0:.2f} sec")


def _process_batch(
    file_batch,
    x,
    y,
    size,
    c_lats,
    c_lons,
    skip_night,
    verbose,
    out_day_path,
    save_as,
    save_only_first,
    download_and_process_fn,
):
    """
    Helper function that processes a *batch* of files sequentially (or in any custom manner).
    Returns a list of (filename, is_day, inpaint_pct).
    """
    results = []
    for f in file_batch:
        result = download_and_process_fn(
            f,
            x,
            y,
            size,
            c_lats,
            c_lons,
            skip_night,
            verbose,
            out_day_path,
            save_as,
            save_only_first,
        )
        if result is not None:
            # result is a tuple (filename, is_day, pct_inp)
            filename, is_day, pct_inp = result
            results.append((filename, is_day, pct_inp))

            # If user wants only the first daytime file, stop processing further in this batch
            if save_only_first and is_day:
                break

    return results


########################################
# 2) METADATA MODULE
########################################
class Metadata:
    """
    Handles the creation of metadata files (lat.npy, lon.npy, minimal metadata.json)
    if they do not exist or if the user desires a fresh download.
    """

    @timeit
    def create_metadata(
        self,
        region: str,
        output_folder: str,
        download_ref: bool = True,
    ):
        """
        1) Download a reference .nc file from NOAA S3 or use a local one
        2) Compute lat/lon arrays
        3) Save lat/lon as .npy along with a minimal metadata.json
        """
        from netCDF4 import Dataset  # local import for convenience

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

        # 1) Download or load reference .nc
        if download_ref:
            logger.info("Downloading reference nc from S3...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
                s3.download_fileobj(BUCKET, file_key, tmp)
                tmp_name = tmp.name
            nc_dataset = Dataset(tmp_name)
        else:
            logger.info("Using local reference nc...")
            nc_dataset = Dataset(local_file)

        # Create folder
        out_folder = os.path.join(output_folder, data_folder)
        os.makedirs(out_folder, exist_ok=True)

        # 2) Minimal metadata
        metadata = {}
        for key in nc_dataset.variables:
            if key in ["geospatial_lat_lon_extent", "goes_imager_projection"]:
                metadata[key] = nc_dataset.variables[key].__dict__
                # Convert non-JSON-serializable data to list
                for sub_key in metadata[key]:
                    if not isinstance(metadata[key][sub_key], str):
                        metadata[key][sub_key] = metadata[key][sub_key].tolist()
            if key in ["x", "y", "x_image_bounds", "y_image_bounds"]:
                metadata[key] = list(nc_dataset.variables[key][:].astype(float))

        with open(os.path.join(out_folder, "metadata.json"), "w") as outfile:
            json.dump(metadata, outfile, indent=4)

        # 3) Lat/lon .npy
        abi_lat, abi_lon = calculate_degrees(nc_dataset)
        lat_data_mask = np.concatenate((abi_lat.data[None], abi_lat.mask[None]), axis=0)
        lon_data_mask = np.concatenate((abi_lon.data[None], abi_lon.mask[None]), axis=0)
        np.save(os.path.join(out_folder, "lat.npy"), lat_data_mask)
        np.save(os.path.join(out_folder, "lon.npy"), lon_data_mask)

        logger.info(f"Metadata stored in {out_folder}")
        logger.info("Done creating metadata.")


########################################
# 3) LISTING MODULE
########################################
class Listing:
    """
    Responsible for listing all relevant GOES16 files in a date range.
    """

    @timeit
    def get_S3_files_in_range(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        output_path: str = "dataset/to_download.json",
    ):
        """
        Return a dictionary mapping datetime -> list of S3 keys for that day.
        """
        if end_date is None:
            end_date = start_date
        if datetime.datetime.strptime(
            end_date, "%Y-%m-%d"
        ) < datetime.datetime.strptime(start_date, "%Y-%m-%d"):
            raise ValueError("End date must be >= start date")

        date_range = pd.date_range(start=start_date, end=end_date).tolist()
        s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
        bucket = s3.Bucket(BUCKET)

        files_in_s3_per_date = {}
        for dt in tqdm.tqdm(date_range):
            # year 2017 has partial data; skip <2018 or beyond current
            if dt.year < 2018 or dt.year > datetime.datetime.now().year:
                logger.info(f"Skipping {dt.year}. Outside supported range.")
                continue

            # Gather all C02 files
            day_files = self._get_day_filenames(bucket, dt.timetuple().tm_yday, dt.year)
            if day_files:
                # out_path = os.path.join(
                #     output_folder, f"{dt.year}_{str(dt.timetuple().tm_yday).zfill(3)}"
                # )
                # os.makedirs(out_path, exist_ok=True)
                files_in_s3_per_date[dt] = day_files

        # save as json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(os.path.join(output_path), "w") as outfile:
            json.dump(
                {str(k): v for k, v in files_in_s3_per_date.items()},
                outfile,
                indent=4,
            )

        return files_in_s3_per_date

    def _get_day_filenames(self, bucket, doy, year):
        prefix = PREFIX + f"/{year}/{doy:03}/"
        return natsorted(
            [f.key for f in bucket.objects.filter(Prefix=prefix) if CHANNEL in f.key]
        )


########################################
# 5) FIRE-BASED COMMAND LINE
########################################
class GOES16CLI:
    """
    Main entry point for Fire CLI.
    Provides three sub-commands:
    1) metadata -> create_metadata(...)
    2) listing -> get_S3_files_in_range(...)
    3) downloader -> download_files(...)
    """

    def __init__(self):
        self.metadata = Metadata()
        self.listing = Listing()
        self.downloader = Downloader()


if __name__ == "__main__":
    fire.Fire(GOES16CLI)

# example, run commands as follows:

# python goes16_script.py metadata create_metadata --region="full_disk" --output_folder="/export/home/projects/franchesoni/goes16/metadata"
# python goes16_script.py listing get_S3_files_in_range --start_date="2018-05-01" --end_date=None --output_path="/export/home/projects/franchesoni/goes16/tmp/to_download.json"
# python goes16_script.py downloader download_files --metadata_path="/export/home/projects/franchesoni/goes16/metadata/FULL_DISK" --files_per_date_path="/export/home/projects/franchesoni/goes16/tmp/to_download.json" --outdir="/export/home/projects/franchesoni/goes16/tmp/salto1024_try1" --size=1024 --skip_night=True --save_only_first=False --save_as='png8' --verbose=True --batch_size=1
# INFO:GOES16 Modular Script:Total time: 2236.89 sec

# python goes16_script.py listing get_S3_files_in_range --start_date="2018-05-02" --end_date=None --output_path="/export/home/projects/franchesoni/goes16/tmp/to_download2.json"
# python goes16_script.py downloader download_files --metadata_path="/export/home/projects/franchesoni/goes16/metadata/FULL_DISK" --files_per_date_path="/export/home/projects/franchesoni/goes16/tmp/to_download2.json" --outdir="/export/home/projects/franchesoni/goes16/tmp/salto1024_try2" --size=1024 --skip_night=True --save_only_first=False --save_as='png8' --verbose=True --batch_size=4
# INFO:GOES16 Modular Script:Finished download_files in 340.52 sec

#  python goes16_script.py listing get_S3_files_in_range --start_date="2019-04-01" --end_date=None --output_path="/export/home/projects/franchesoni/goes16/tmp/to_download3.json"
# python goes16_script.py downloader download_files --metadata_path="/export/home/projects/franchesoni/goes16/metadata/FULL_DISK" --files_per_date_path="/export/home/projects/franchesoni/goes16/tmp/to_download3.json" --outdir="/export/home/projects/franchesoni/goes16/tmp/salto1024_try3" --size=1024 --skip_night=True --save_only_first=False --save_as='png8' --verbose=True --batch_size=24
# INFO:GOES16 Modular Script:Finished download_files in 1625.03 sec

# python goes16_script.py listing get_S3_files_in_range --start_date="2019-03-01" --end_date=None --output_path="/export/home/projects/franchesoni/goes16/tmp/to_download4.json"
# python goes16_script.py downloader download_files --metadata_path="/export/home/projects/franchesoni/goes16/metadata/FULL_DISK" --files_per_date_path="/export/home/projects/franchesoni/goes16/tmp/to_download4.json" --outdir="/export/home/projects/franchesoni/goes16/tmp/salto1024_try4" --size=1024 --skip_night=True --save_only_first=False --save_as='png8' --verbose=True --batch_size=12

# final (we'll use 4)
# python goes16_script.py listing get_S3_files_in_range --start_date="2019-04-02" --end_date="2025-01-01" --output_path="/export/home/projects/franchesoni/goes16/all.json"
# python goes16_script.py downloader download_files --metadata_path="/export/home/projects/franchesoni/goes16/metadata/FULL_DISK" --files_per_date_path="/export/home/projects/franchesoni/goes16/all.json" --outdir="/export/home/projects/franchesoni/goes16/tmp/salto1024_all" --size=1024 --skip_night=True --save_only_first=False --save_as='png8' --verbose=True --batch_size=4
