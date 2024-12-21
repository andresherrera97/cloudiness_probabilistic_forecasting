import os
import json
import datetime
import tempfile
import logging
import numpy as np
import pandas as pd
import cv2
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from netCDF4 import Dataset
from PIL import Image
from tqdm import tqdm
import fire

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GOES16")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
BUCKET = "noaa-goes16"
REGION = "F"  # C (CONUS) or F (FULL_DISK) 
CHANNEL = "C02"
S3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
S3_CLIENT = boto3.client("s3", config=Config(signature_version=UNSIGNED))
# data_root = "datasets/ABI_L2_CMIP_M6C02_G16"

# ---------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------
def calculate_degrees(nc_file):
    """Convert GOES ABI projection to lat/lon."""
    x = nc_file.variables["x"][:]
    y = nc_file.variables["y"][:]
    proj = nc_file.variables["goes_imager_projection"]
    lon0 = proj.longitude_of_projection_origin * np.pi / 180.0
    H = proj.perspective_point_height + proj.semi_major_axis
    r_eq, r_pol = proj.semi_major_axis, proj.semi_minor_axis

    x2d, y2d = np.meshgrid(x, y)
    a = (np.sin(x2d)**2) + np.cos(x2d)**2 * (
        np.cos(y2d)**2 + (r_eq**2 / r_pol**2) * np.sin(y2d)**2
    )
    b = -2 * H * np.cos(x2d) * np.cos(y2d)
    c = H**2 - r_eq**2
    r_s = (-b - np.sqrt(np.maximum(0, b**2 - 4*a*c))) / (2*a)
    s_x = r_s * np.cos(x2d) * np.cos(y2d)
    s_y = -r_s * np.sin(x2d)
    s_z = r_s * np.cos(x2d) * np.sin(y2d)

    lat = np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((H - s_x)**2 + s_y**2)))
    lon = lon0 - np.arctan(s_y / (H - s_x))
    return np.degrees(lat), np.degrees(lon)

def generate_metadata(data_root, region="full_disk", out_folder="ABI_L2_CMIP_M6C02_G16", download_ref=True):
    """Pull one .nc file (local or from S3) and save lat/lon + small metadata."""
    logger.info("Generating metadata...")
    conus_s3 = "ABI-L2-CMIPC/2024/111/10/OR_ABI-L2-CMIPC-M6C02_G16_s20241111011171_e20241111013544_c20241111014046.nc"
    full_s3  = "ABI-L2-CMIPF/2024/111/10/OR_ABI-L2-CMIPF-M6C02_G16_s20241111050205_e20241111059513_c20241111059581.nc"

    file_key = conus_s3 if region.lower().startswith("c") else full_s3
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as temp_file:
        S3_CLIENT.download_fileobj(BUCKET, file_key, temp_file)
        filename = temp_file.name

    ds = Dataset(filename)
    folder = "CONUS" if region.lower().startswith("c") else "FULL_DISK"
    out_folder = os.path.join(data_root, folder)
    os.makedirs(out_folder, exist_ok=True)

    meta = {}
    for k in ds.variables:
        if k in ["geospatial_lat_lon_extent", "goes_imager_projection"]:
            meta[k] = {subk: ds.variables[k].__dict__[subk] for subk in ds.variables[k].__dict__}
        if k in ["x", "y", "x_image_bounds", "y_image_bounds"]:
            meta[k] = ds.variables[k][:].astype(float).tolist()

    with open(os.path.join(out_folder, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    lat, lon = calculate_degrees(ds)
    lat_data_mask = np.stack([lat.data, lat.mask]) if hasattr(lat, 'mask') else np.stack([lat, np.zeros_like(lat, dtype=bool)])
    lon_data_mask = np.stack([lon.data, lon.mask]) if hasattr(lon, 'mask') else np.stack([lon, np.zeros_like(lon, dtype=bool)])
    np.save(os.path.join(out_folder, "lat.npy"), lat_data_mask)
    np.save(os.path.join(out_folder, "lon.npy"), lon_data_mask)
    logger.info("Metadata ready!")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def is_valid_date(d):
    # NOAA GOES-16 data starts ~2017. If user picks e.g. year=2000, skip quickly
    earliest = datetime.datetime(2017, 1, 1)
    latest = datetime.datetime.now()
    if d < earliest or d > latest: 
        return False
    return True

def get_s3_filenames(bucket, day_of_year, year):
    """List S3 objects for a specific day. Quick fail if out of range."""
    # NOAA GOES-16 started in 2017, so skip if year < 2017 or year too big
    if year < 2017 or year > 2050: 
        return []
    prefix = f"ABI-L2-CMIP{REGION}/{year}/{str(day_of_year).zfill(3)}/"
    objs = bucket.objects.filter(Prefix=prefix)
    return [o.key for o in objs if o.key.endswith(".nc")]

def read_crop_concurrent(s3key, x, y, size):
    # Just a placeholder read; you'd do real concurrency if you want
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    S3_CLIENT.download_fileobj(BUCKET, s3key, tmp)
    ds = Dataset(tmp.name)
    cmi = ds.variables["CMI"][:]
    dqf = ds.variables["DQF"][:]
    ds.close()
    # Crop
    cmi_crop = cmi[y : y+size, x : x+size]
    dqf_crop = dqf[y : y+size, x : x+size]
    return np.stack([cmi_crop, dqf_crop], axis=0)

def get_cosangs(dateobj, lat_crop, lon_crop):
    """Compute approximate cos(sun zenith), skip details."""
    # This is just a simple check to see if it's daytime
    # (placeholder formula, not super accurate)
    # Return all 1's if it's day, all 0's if night, for the lat/lon box
    hour = dateobj.hour
    # Very rough approximation: daytime if 6 <= hour <= 18
    if 6 <= hour < 18:
        return np.ones_like(lat_crop), True
    return np.zeros_like(lat_crop), False

def normalize(cmi, cosangs, factor=0.15):
    """Convert reflectance, trivial approach."""
    with np.errstate(invalid='ignore'):
        return (cmi / (cosangs + 1e-5)) * factor

# ---------------------------------------------------------------------
# Main data download
# ---------------------------------------------------------------------
def check_and_generate_metadata(data_root, region):
    folder = "CONUS" if region.upper() == "C" else "FULL_DISK"
    meta_path = os.path.join(data_root, folder)
    if not os.path.exists(meta_path):
        logger.info("Metadata missing. Generating...")
        generate_metadata(data_root, region)

def download_goes16(
    data_root="datasets",
    dataset_name="salto1024",
    region="C", 
    start_date="2024-01-05", 
    end_date=None,
    lat=-31.3905,
    lon=-57.9541,
    size=1024,
    skip_night=True,
    save_only_first=False,
    save_as_npy=True
):
    """Download GOES16 data in a lat/lon crop for the specified date range."""
    out_folder = os.path.join(data_root, dataset_name)
    check_and_generate_metadata(data_root, region)
    logger.info("Loading precomputed lat/lon grids...")
    folder = "CONUS" if region.upper() == "C" else "FULL_DISK"
    lat_data = np.load(os.path.join(data_root, folder, "lat.npy"))
    lon_data = np.load(os.path.join(data_root, folder, "lon.npy"))

    ref_lat = np.ma.masked_array(lat_data[0], mask=lat_data[1])
    ref_lon = np.ma.masked_array(lon_data[0], mask=lon_data[1])

    # Convert lat/lon to pixel coords
    # Simple approach: find nearest index
    distances = np.abs(ref_lat - lat) + np.abs(ref_lon - lon)
    y, x = np.unravel_index(np.nanargmin(distances), distances.shape)
    # quick boundary check
    half = size // 2
    if (x - half < 0) or (y - half < 0) or (x + half >= ref_lat.shape[1]) or (y + half >= ref_lat.shape[0]):
        raise ValueError("Requested crop is partially outside the domain.")

    # Build date range
    if not end_date: end_date = start_date
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.datetime.strptime(end_date,   "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError("End date must be >= start date.")
    dates = pd.date_range(start_dt, end_dt).to_pydatetime().tolist()

    # For each date, find S3 keys
    bucket = S3.Bucket(BUCKET)
    for day in dates:
        if not is_valid_date(day):
            logger.info(f"Skipping {day.date()}, date out of range (2017-now).")
            continue
        dofy = day.timetuple().tm_yday
        year = day.year
        keys = [o.key for o in bucket.objects.filter(Prefix=f"ABI-L2-CMIP{region}/{year}/{str(dofy).zfill(3)}/") if o.key.endswith(".nc")]
        if not keys:
            logger.info(f"No files for {day.date()}.")
            continue
        # Output folder
        odir = os.path.join(out_folder, f"{year}_{str(dofy).zfill(3)}")
        os.makedirs(odir, exist_ok=True)
        logger.info(f"Date: {day.date()}, #files={len(keys)}")

        is_day, inpaint_pct, processed = [], [], []
        for k in tqdm(keys):
            # Parse time
            t_str = k.split("/")[-1].split("_")[3][1:]  # s20241111011171...
            hh, mm, ss = t_str[7:9], t_str[9:11], t_str[11:13]
            dt_str = f"{day.year}-{day.month:02}-{day.day:02} {hh}:{mm}:{ss}"
            dt_obj = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

            # Check day/night
            cosangs, sunlit = get_cosangs(dt_obj, ref_lat[y-half:y+half, x-half:x+half],
                                          ref_lon[y-half:y+half, x-half:x+half])
            if skip_night and not sunlit:
                is_day.append(False)
                inpaint_pct.append(None)
                processed.append(k)
                continue

            # Download + crop
            cmi_dqf = read_crop_concurrent(k, x-half, y-half, size)
            mask = (cmi_dqf[1] != 0).astype(np.uint8)
            to_inpaint_pct = mask.mean() * 100
            # Inpaint
            cmi_dqf[0] = cv2.inpaint(cmi_dqf[0], mask, 3, cv2.INPAINT_NS)
            # Reflectance
            refl = normalize(cmi_dqf[0], cosangs, 0.15)
            refl[np.isnan(refl)] = 0
            refl = np.clip(refl, 0, 1).astype(np.float16)

            # Save
            fname = f"{year}_{str(dofy).zfill(3)}_UTC_{hh}{mm}{ss}"
            if save_as_npy:
                np.save(os.path.join(odir, fname + ".npy"), refl)
            else:
                out_img = (refl.astype(np.float32)*65535).astype(np.uint16)
                Image.fromarray(out_img, mode="I;16").save(os.path.join(odir, fname + ".png"))
            inpaint_pct.append(to_inpaint_pct)
            is_day.append(sunlit)
            processed.append(k)
            if save_only_first: break

        # Save CSV
        pd.DataFrame({
            "filenames": processed,
            "is_day": is_day,
            "inpaint_pct": inpaint_pct
        }).to_csv(os.path.join(odir, f"data_{year}_{str(dofy).zfill(3)}.csv"), index=False)

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    fire.Fire(download_goes16)
