import os
import numpy as np
import fire


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


def main(satellite: str = "goes16", lat: float = 29.0253, lon: float = -111.1434, size: int = 512):
    # load lats and lons info
    output_path = f"../data/sonora/metadata/{satellite}/"
    metadata_path = f"../data/{satellite}/metadata/FULL_DISK/"

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

    print(c_lats.shape)
    print(type(c_lats))

    np.save(os.path.join(output_path, "lats"), c_lats.data)
    np.save(os.path.join(output_path, "lons"), c_lons.data)


if __name__ == "__main__":
    fire.Fire(main)
