import os
import calendar
import datetime
import tqdm
import numpy as np


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


def get_cosangs(dtime: datetime.datetime, lats: np.ndarray, lons: np.ndarray):
    """Main wrapper for solar angles for a given date/time."""
    delta, eot = daily_solar_angles(dtime.year, dtime.timetuple().tm_yday)
    cosz, mask = solar_angles(
        lats, lons, delta, eot, dtime.hour, dtime.minute, dtime.second
    )
    return cosz, mask


def find_nearest_pixel(REF_LAT, REF_LON, target_lat, target_lon):
    rows, cols = REF_LAT.shape
    y, x = rows // 2, cols // 2
    step = max(rows, cols) // 2

    def manhattan(i, j):
        return abs(REF_LAT[i, j] - target_lat) + abs(REF_LON[i, j] - target_lon)

    best = manhattan(y, x)
    while step:
        improved = False
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    d = manhattan(ny, nx)
                    if d < best:
                        best, y, x = d, ny, nx
                        improved = True
        if not improved:
            step //= 2
    return y, x


def convert_coordinates_to_pixel(lat, lon, REF_LAT, REF_LON, size, verbose=True):
    """
    Return the sub-array of lat/lon plus the x,y pixel coordinates
    that correspond to the requested lat/lon center.
    """
    print("Preparing to crop...")
    y, x = find_nearest_pixel(REF_LAT, REF_LON, lat, lon)

    # check coverage
    if x - size // 2 < 0 or y - size // 2 < 0:
        raise ValueError("Crop too large or outside coverage. Adjust.")
    if x + size // 2 >= REF_LAT.shape[1] or y + size // 2 >= REF_LAT.shape[0]:
        raise ValueError("Crop too large or outside coverage. Adjust.")

    print("Cropping...")
    c_lats = REF_LAT[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2]
    c_lons = REF_LON[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2]

    print("Crop ready!")
    return c_lats, c_lons, x, y


if __name__ == "__main__":
    DATASET_PATH = "../data/salto1024_all/"
    OUTPUT_PATH = "../data/salto_crop/"

    # load lats and lons info
    metadata_path = "../data/goes16/metadata/FULL_DISK/"
    lat = -31.390502
    lon = -57.954138
    size = 32

    lat_path = os.path.join(metadata_path, "lat.npy")
    lon_path = os.path.join(metadata_path, "lon.npy")

    print("Loading metadata...")
    REF_LAT = np.load(lat_path)
    lat_data = np.ma.masked_array(REF_LAT[0], mask=REF_LAT[1])
    REF_LON = np.load(lon_path)
    lon_data = np.ma.masked_array(REF_LON[0], mask=REF_LON[1])
    del REF_LAT, REF_LON

    # 2) Convert user lat/lon to pixel coords
    print("Converting coordinates to pixel...")
    c_lats, c_lons, x, y = convert_coordinates_to_pixel(
        lat, lon, lat_data, lon_data, size, True
    )
    del lat_data, lon_data

    try:
        os.mkdir(os.path.join(OUTPUT_PATH, "metadata"))
    except:
        pass

    print(c_lats.shape)
    print(type(c_lats))

    np.save(os.path.join(OUTPUT_PATH, "metadata", "lats"), c_lats.data)
    np.save(os.path.join(OUTPUT_PATH, "metadata", "lons"), c_lons.data)

    for day in tqdm.tqdm(sorted(os.listdir(DATASET_PATH))):
        try:
            os.mkdir(os.path.join(OUTPUT_PATH, "CMI", day))
            os.mkdir(os.path.join(OUTPUT_PATH, "cosangs", day))
        except:
            pass
        images = os.listdir(os.path.join(DATASET_PATH, day))
        for image in images:
            if not image.endswith(".npy"):
                continue
            # image = YYYY_DOY_UTC_HHMMSS.npy
            planetary_reflectance = np.load(
                os.path.join(DATASET_PATH, day, image)
            )  # uint8
            # crop 32 x 32 center
            border_size = (1024 - 32) // 2
            planetary_reflectance_crop = planetary_reflectance[
                border_size:-border_size, border_size:-border_size
            ]
            # Transform from uint8 to float32
            planetary_reflectance_crop = (
                planetary_reflectance_crop.astype(np.float32) / 255.0
            )

            # Parse time from filename
            y4 = image.split("_")[0]
            dayy = image.split("_")[1]
            hh = image.split("_")[-1][:2]
            mm = image.split("_")[-1][2:4]
            ss = image.split("_")[-1][4:6]
            file_dt = datetime.datetime.strptime(
                f"{dayy}/{y4} {hh}:{mm}:{ss}", "%j/%Y %H:%M:%S"
            )
            # 1) Compute solar angles
            cosangs, _ = get_cosangs(file_dt, c_lats, c_lons)
            # Calculate Cloud and Moisture Imagery values
            cmi = planetary_reflectance_crop * cosangs.data

            # save values in respective folder
            np.save(os.path.join(OUTPUT_PATH, "CMI", day, image), cmi)
            np.save(os.path.join(OUTPUT_PATH, "cosangs", day, image), cosangs.data)
