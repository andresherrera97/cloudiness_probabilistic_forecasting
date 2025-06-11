import os
import cv2
import fire
import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


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

    if verbose:
        print_coordinates_square(x, y, lat, lon, size, REF_LAT, REF_LON)
    print("Crop ready!")
    return c_lats, c_lons, x, y


def plot_centered_square(
    lat: float, lon: float, size: int, c_lats: np.ndarray, c_lons: np.ndarray
) -> None:
    row = size // 2 - 1  # Center of the crop
    column = size // 2 - 1  # Center of the crop
    box_size = 2  # 2x2 box

    lat_box_0 = c_lats[row : row + box_size, column : column + box_size]
    lon_box_0 = c_lons[row : row + box_size, column : column + box_size]

    corner_lons = [
        lon_box_0[0, 0],
        lon_box_0[0, -1],
        lon_box_0[-1, -1],
        lon_box_0[-1, 0],
        lon_box_0[0, 0],
    ]
    corner_lats = [
        lat_box_0[0, 0],
        lat_box_0[0, -1],
        lat_box_0[-1, -1],
        lat_box_0[-1, 0],
        lat_box_0[0, 0],
    ]

    plt.title("Center of the crop")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.plot(lon, lat, "ro", markersize=8, markeredgecolor="black", markeredgewidth=2)
    plt.plot(corner_lons, corner_lats, "r-", linewidth=2)  # Red square line
    plt.show()


def main(
    dataset_path: str = "datasets/LE_Crop_128x128/",
    output_path: str = "datasets/les/",
    metadata_path: str = "datasets/LE_Crop_128x128/metadata/",
    lat: float = -31.2827,
    lon: float = -57.9181,
):
    lat_path = os.path.join(metadata_path, "lats.npy")
    lon_path = os.path.join(metadata_path, "lons.npy")

    print("Loading metadata...")
    lats = np.load(lat_path)
    lons = np.load(lon_path)

    size = lats.shape[0]

    # Path(os.path.join(output_path, "crop_64x64_HR", "metadata")).mkdir(
    #     parents=True, exist_ok=True
    # )
    # Path(os.path.join(output_path, "crop_64x64_MR", "metadata")).mkdir(
    #     parents=True, exist_ok=True
    # )
    # Path(os.path.join(output_path, "crop_64x64_HR", "CMI")).mkdir(
    #     parents=True, exist_ok=True
    # )
    # Path(os.path.join(output_path, "crop_64x64_MR", "CMI")).mkdir(
    #     parents=True, exist_ok=True
    # )
    Path(os.path.join(output_path, "crop_64x64_MR_bilinear", "metadata")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(output_path, "crop_64x64_MR_bilinear", "CMI")).mkdir(
        parents=True, exist_ok=True
    )

    lats_HR = lats[size // 4 : -size // 4, size // 4 : -size // 4]
    lons_HR = lons[size // 4 : -size // 4, size // 4 : -size // 4]
    lats_MR = lats[::2, ::2]
    lons_MR = lons[::2, ::2]
    lats_MR_bilinear = cv2.resize(
        lats,
        (lats.shape[1] // 2, lats.shape[0] // 2),
        interpolation=cv2.INTER_LINEAR,
    )
    lons_MR_bilinear = cv2.resize(
        lons,
        (lons.shape[1] // 2, lons.shape[0] // 2),
        interpolation=cv2.INTER_LINEAR,
    )

    plot_centered_square(lat=lat, lon=lon, size=size, c_lats=lats, c_lons=lons)
    plot_centered_square(
        lat=lat, lon=lon, size=64, c_lats=lats_MR, c_lons=lons_MR
    )
    plot_centered_square(
        lat=lat,
        lon=lon,
        size=64,
        c_lats=lats_HR,
        c_lons=lons_HR,
    )
    plot_centered_square(
        lat=lat,
        lon=lon,
        size=64,
        c_lats=lats_MR_bilinear,
        c_lons=lons_MR_bilinear,
    )

    # np.save(os.path.join(output_path, "crop_64x64_HR", "metadata", "lats"), lats_HR)
    # np.save(os.path.join(output_path, "crop_64x64_HR", "metadata", "lons"), lons_HR)
    # np.save(os.path.join(output_path, "crop_64x64_MR", "metadata", "lats"), lats_MR)
    # np.save(os.path.join(output_path, "crop_64x64_MR", "metadata", "lons"), lons_MR)
    np.save(os.path.join(output_path, "crop_64x64_MR_bilinear", "metadata", "lons"), lons_MR_bilinear)
    np.save(os.path.join(output_path, "crop_64x64_MR_bilinear", "metadata", "lats"), lats_MR_bilinear)

    for day in tqdm.tqdm(sorted(os.listdir(os.path.join(dataset_path, "CMI")))):
        files_in_day = sorted(
            os.listdir(os.path.join(dataset_path, "CMI", day))
        )
        # Path(os.path.join(output_path, "crop_64x64_HR", "CMI", day)).mkdir(
        #     parents=True, exist_ok=True
        # )
        # Path(os.path.join(output_path, "crop_64x64_MR", "CMI", day)).mkdir(
        #     parents=True, exist_ok=True
        # )
        Path(os.path.join(output_path, "crop_64x64_MR_bilinear", "CMI", day)).mkdir(
            parents=True, exist_ok=True
        )
        for day_img in files_in_day:
            if not day_img.endswith(".npy"):
                continue

            day_img_path = os.path.join(dataset_path, "CMI", day, day_img)

            # Load the image data
            img_data = np.load(day_img_path)

            # Save cropped images
            # np.save(
            #     os.path.join(output_path, "crop_64x64_HR", "CMI", day, day_img),
            #     img_data[size // 4 : -size // 4, size // 4 : -size // 4],
            # )
            # np.save(
            #     os.path.join(output_path, "crop_64x64_MR", "CMI", day, day_img),
            #     img_data[::2, ::2],
            # )
            img_data = img_data.astype(np.float32)  # Ensure float32 for cv2.resize
            img_bilinear = cv2.resize(
                img_data,
                (img_data.shape[1] // 2, img_data.shape[0] // 2),
                interpolation=cv2.INTER_LINEAR,
            )
            img_bilinear = img_bilinear.astype(np.float16)  # Ensure float32 for saving
            np.save(
                os.path.join(output_path, "crop_64x64_MR_bilinear", "CMI", day, day_img),
                img_bilinear,
            )


if __name__ == "__main__":
    fire.Fire(main)
