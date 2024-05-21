"""
This file downloads a nc file from the goes-16 AWS S3 bucket, and extracts the
metadata from it. The metadata is constant for all the other files, saving this
data saves processing when using other files.
NOAA goes-16 Amazon S3 Bucket: https://noaa-goes16.s3.amazonaws.com/index.html
"""

from netCDF4 import Dataset
import numpy as np
import json
import fire
import os


PATH_TO_CONUS_FILE = "datasets/CONUS_nc_files/OR_ABI-L2-CMIPC-M6C02_G16_s20241110601171_e20241110603544_c20241110604037.nc"
PATH_TO_FULL_DISK_FILE = "datasets/full_disk_nc_files/OR_ABI-L2-CMIPF-M6C02_G16_s20220120310204_e20220120319512_c20220120319587.nc"


def calculate_degrees(file_id):

    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables["x"][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables["y"][:]  # N/S elevation angle in radians
    projection_info = file_id.variables["goes_imager_projection"]
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height + projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis

    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin * np.pi) / 180.0
    a_var = np.power(np.sin(x_coordinate_2d), 2.0) + (
        np.power(np.cos(x_coordinate_2d), 2.0)
        * (
            np.power(np.cos(y_coordinate_2d), 2.0)
            + (
                ((r_eq * r_eq) / (r_pol * r_pol))
                * np.power(np.sin(y_coordinate_2d), 2.0)
            )
        )
    )
    b_var = -2.0 * H * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    c_var = (H**2.0) - (r_eq**2.0)
    r_s = (-1.0 * b_var - np.sqrt((b_var**2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
    s_x = r_s * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    s_y = -r_s * np.sin(x_coordinate_2d)
    s_z = r_s * np.cos(x_coordinate_2d) * np.sin(y_coordinate_2d)

    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all="ignore")

    abi_lat = (180.0 / np.pi) * (
        np.arctan(
            ((r_eq * r_eq) / (r_pol * r_pol))
            * ((s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y))))
        )
    )
    abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

    return abi_lat, abi_lon


def main(region: str = "conus"):
    # TODO: Download the files from the AWS S3 bucket
    if region.lower() == "conus":
        nc_dataset = Dataset(PATH_TO_CONUS_FILE)
        data_folder = "CONUS"
    if region.lower() == "full_disk":
        nc_dataset = Dataset(PATH_TO_FULL_DISK_FILE)
        data_folder = "FULL_DISK"

    OUTPUT_FOLDER = "datasets/ABI_L2_CMIP_M6C02_G16/" + data_folder

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    metadata = {}

    for key in nc_dataset.variables:
        if key in [
            "geospatial_lat_lon_extent",
            "goes_imager_projection",
        ]:
            metadata[key] = nc_dataset.variables[key].__dict__
            for sub_key in metadata[key]:
                if not isinstance(metadata[key][sub_key], str):
                    metadata[key][sub_key] = metadata[key][sub_key].tolist()
        if key in ["x", "y", "x_image_bounds", "y_image_bounds"]:
            metadata[key] = list(nc_dataset.variables[key][:].astype(float))

    # Convert and write JSON object to file
    with open(os.path.join(OUTPUT_FOLDER, "metadata.json"), "w") as outfile:
        json.dump(metadata, outfile, indent=4)

    abi_lat, abi_lon = calculate_degrees(nc_dataset)

    abi_lat_data_mask = np.concatenate(
        (abi_lat.data[None, ...], abi_lat.mask[None, ...]), axis=0
    )
    abi_lon_data_mask = np.concatenate(
        (abi_lon.data[None, ...], abi_lon.mask[None, ...]), axis=0
    )

    np.save(os.path.join(OUTPUT_FOLDER, "lat.npy"), abi_lat_data_mask)
    np.save(os.path.join(OUTPUT_FOLDER, "lon.npy"), abi_lon_data_mask)


if __name__ == "__main__":
    fire.Fire(main)
