from netCDF4 import Dataset
import numpy as np
import os


# Calculate latitude and longitude from GOES ABI fixed grid projection data
# GOES ABI fixed grid projection is a map projection relative to the GOES satellite
# Units: latitude in °N (°S < 0), longitude in °E (°W < 0)
# See GOES-R Product User Guide (PUG) Volume 5 (L2 products) Section 4.2.8 for details & example of calculations
# "file_id" is an ABI L1b or L2 .nc file opened using the netCDF4 library


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


if __name__ == "__main__":
    folder_path = "datasets/full_disk_nc_files"
    filenames = os.listdir(folder_path)
    filenames = [filename for filename in filenames if filename[0] != "."]

    nc_files = {}
    for filename in filenames:
        date_time = filename.split("_s")[1].split("_")[0]
        nc_files[date_time] = Dataset(f"{folder_path}/{filename}")

    print("=== Dataset name ===")
    for key in nc_files:
        print(f"{key}: {nc_files[key].dataset_name}")
        
    print("=== Scan Mode ===")
    for key in nc_files:
        print(f"{key}: {nc_files[key].dataset_name.split('-')[3][:2]}")

    print("=== X and Y Image Bounds ===")
    for key in nc_files:
        print(nc_files[key].variables["x_image_bounds"][:])
        print(nc_files[key].variables["y_image_bounds"][:])
    
    print("=== Semi Major Axis ===")
    for key in nc_files:
        print(nc_files[key].variables["goes_imager_projection"].semi_major_axis)
        
    print("=== Semi Minor Axis ===")
    for key in nc_files:
        print(nc_files[key].variables["goes_imager_projection"].semi_minor_axis)
    
    print("=== Point Height ===")
    for key in nc_files:
        print(nc_files[key].variables["goes_imager_projection"].perspective_point_height)
    
    print("=== longitude_of_projection_origin ===")
    for key in nc_files:
        scan_operation = key
        print(f'{key}, {nc_files[key].dataset_name.split("-")[3][:2]}, {nc_files[key].variables["goes_imager_projection"].longitude_of_projection_origin}')
        
    print("=== x ===")
    for n, key in enumerate(nc_files):
        if n == 0:
            aux = nc_files[key].variables["x"][:]
        print((nc_files[key].variables["x"][:] == nc_files[key].variables["x"][:]).all())
        
    print("=== y ===")
    for key in nc_files:
        if n == 0:
            aux = nc_files[key].variables["y"][:]
        print((nc_files[key].variables["y"][:] == nc_files[key].variables["y"][:]).all())

    # nc_2024_011_12 = Dataset(
    #     "datasets/CONUS_nc_files/OR_ABI-L2-CMIPC-M6C02_G16_s20241110601171_e20241110603544_c20241110604037.nc"
    #     # "datasets/full_disk_nc_files/OR_ABI-L2-CMIPF-M6C02_G16_s20220120310204_e20220120319512_c20220120319587.nc"
    # )
    # nc_2024_011_13 = Dataset(
    #     "datasets/CONUS_nc_files/OR_ABI-L2-CMIPC-M6C02_G16_s20240112006171_e20240112008544_c20240112009047.nc"
    #     # "datasets/full_disk_nc_files/OR_ABI-L2-CMIPF-M6C02_G16_s20220120330204_e20220120339512_c20220120339575.nc"
    # )
    # nc_2020_197_13 = Dataset(
    #     "datasets/CONUS_nc_files/OR_ABI-L2-CMIPC-M6C02_G16_s20201971301225_e20201971303598_c20201971304130.nc"
    #     # "datasets/full_disk_nc_files/OR_ABI-L2-CMIPF-M6C02_G16_s20240461620209_e20240461629517_c20240461629576.nc"
    # )

    # nc_prueba = Dataset(
    #     "datasets/CONUS_nc_files/OR_ABI-L2-CMIPC-M6C02_G16_s20231511601173_e20231511603546_c20231511604055.nc"
    # )

    # print(nc_prueba["x_offset"])

    # sat_h_nc_2024_011_12 = nc_2024_011_12.variables[
    #     "goes_imager_projection"
    # ].perspective_point_height
    # sat_h_nc_2024_011_13 = nc_2024_011_13.variables[
    #     "goes_imager_projection"
    # ].perspective_point_height
    # sat_h_nc_2020_197_13 = nc_2020_197_13.variables[
    #     "goes_imager_projection"
    # ].perspective_point_height
    # print("=== Perspective Point Height ===")
    # print(f"sat_h_nc_2024_011_12: {sat_h_nc_2024_011_12}")
    # print(f"sat_h_nc_2024_011_13: {sat_h_nc_2024_011_13}")
    # print(f"sat_h_nc_2020_197_13: {sat_h_nc_2020_197_13}")

    # print("=== Longitude of Projection Origin ===")
    # sat_lon_nc_2024_011_12 = nc_2024_011_12.variables[
    #     "goes_imager_projection"
    # ].longitude_of_projection_origin
    # sat_lon_nc_2024_011_13 = nc_2024_011_13.variables[
    #     "goes_imager_projection"
    # ].longitude_of_projection_origin
    # sat_lon_nc_2020_197_13 = nc_2020_197_13.variables[
    #     "goes_imager_projection"
    # ].longitude_of_projection_origin
    # print(f"sat_lon_nc_2024_011_12: {sat_lon_nc_2024_011_12}")
    # print(f"sat_lon_nc_2024_011_13: {sat_lon_nc_2024_011_13}")
    # print(f"sat_lon_nc_2020_197_13: {sat_lon_nc_2020_197_13}")

    # print("=== Sweep Angle Axis ===")
    # sat_sweep_nc_2024_011_12 = nc_2024_011_12.variables[
    #     "goes_imager_projection"
    # ].sweep_angle_axis
    # sat_sweep_nc_2024_011_13 = nc_2024_011_13.variables[
    #     "goes_imager_projection"
    # ].sweep_angle_axis
    # sat_sweep_nc_2020_197_13 = nc_2020_197_13.variables[
    #     "goes_imager_projection"
    # ].sweep_angle_axis
    # print(f"sat_sweep_nc_2024_011_12: {sat_sweep_nc_2024_011_12}")
    # print(f"sat_sweep_nc_2024_011_13: {sat_sweep_nc_2024_011_13}")
    # print(f"sat_sweep_nc_2020_197_13: {sat_sweep_nc_2020_197_13}")

    # print("=== Satellite Longitude ===")
    # print(f"shape: {nc_2024_011_13.variables['y'][:].shape}")
    # print(nc_2024_011_12.variables["y"][:])
    # print(nc_2024_011_13.variables["y"][:])
    # print(nc_2020_197_13.variables["y"][:])

    # print("=== Satellite Latitude ===")
    # print(f"shape: {nc_2024_011_13.variables['x'][:].shape}")
    # print(nc_2024_011_12.variables["x"][:])
    # print(nc_2024_011_13.variables["x"][:])
    # print(nc_2020_197_13.variables["x"][:])

    # print("=== X Image Bounds ===")
    # print(nc_2024_011_12.variables["x_image_bounds"][:])
    # print(nc_2024_011_13.variables["x_image_bounds"][:])
    # print(nc_2020_197_13.variables["x_image_bounds"][:])

    # print("=== Y Image Bounds ===")
    # print(nc_2024_011_12.variables["y_image_bounds"][:])
    # print(nc_2024_011_13.variables["y_image_bounds"][:])
    # print(nc_2020_197_13.variables["y_image_bounds"][:])

    # print("=== Min-Max-Mean-StdDev Reflectance Factor ===")
    # print(
    #     f"{nc_2024_011_12.variables['min_reflectance_factor'][:]}, "
    #     f"{nc_2024_011_12.variables['max_reflectance_factor'][:]}, "
    #     f"{nc_2024_011_12.variables['mean_reflectance_factor'][:]}, "
    #     f"{nc_2024_011_12.variables['std_dev_reflectance_factor'][:]}"
    # )
    # print(
    #     f"{nc_2024_011_13.variables['min_reflectance_factor'][:]}, "
    #     f"{nc_2024_011_13.variables['max_reflectance_factor'][:]}, "
    #     f"{nc_2024_011_13.variables['mean_reflectance_factor'][:]}, "
    #     f"{nc_2024_011_13.variables['std_dev_reflectance_factor'][:]}"
    # )
    # print(
    #     f"{nc_2020_197_13.variables['min_reflectance_factor'][:]}, "
    #     f"{nc_2020_197_13.variables['max_reflectance_factor'][:]}, "
    #     f"{nc_2020_197_13.variables['mean_reflectance_factor'][:]}, "
    #     f"{nc_2020_197_13.variables['std_dev_reflectance_factor'][:]}"
    # )
    # print(
    #     f"{nc_prueba.variables['min_reflectance_factor'][:]}, "
    #     f"{nc_prueba.variables['max_reflectance_factor'][:]}, "
    #     f"{nc_prueba.variables['mean_reflectance_factor'][:]}, "
    #     f"{nc_prueba.variables['std_dev_reflectance_factor'][:]}"
    # )

    # print("=== Type CMI ===")
    # print(
    #     type(nc_2020_197_13.variables["CMI"]),
    #     type(nc_2020_197_13.variables["CMI"][:]),
    #     nc_2020_197_13.variables["CMI"][:].dtype,
    # )
    # print(
    #     type(nc_2024_011_12.variables["CMI"]),
    #     type(nc_2024_011_12.variables["CMI"][:]),
    #     nc_2024_011_12.variables["CMI"][:].dtype,
    # )
    # print(
    #     type(nc_2024_011_13.variables["CMI"]),
    #     type(nc_2024_011_13.variables["CMI"][:]),
    #     nc_2024_011_13.variables["CMI"][:].dtype,
    # )

    # print("=== Min-Max CMI ===")
    # print(
    #     np.min(nc_2024_011_12.variables["CMI"][:]),
    #     np.max(nc_2024_011_12.variables["CMI"][:]),
    # )
    # print(
    #     np.min(nc_2024_011_13.variables["CMI"][:]),
    #     np.max(nc_2024_011_13.variables["CMI"][:]),
    # )
    # print(
    #     np.min(nc_2020_197_13.variables["CMI"][:]),
    #     np.max(nc_2020_197_13.variables["CMI"][:]),
    # )
    # print(
    #     np.min(nc_prueba.variables["CMI"][:]),
    #     np.max(nc_prueba.variables["CMI"][:]),
    # )
    # print(
    #     np.min(nc_prueba.variables["CMI"][:][1847:1975, 6317:6445]),
    #     np.max(nc_prueba.variables["CMI"][:][1847:1975, 6317:6445]),
    # )

    # # print(nc_2024_011_13.variables["y_image_bounds"][:])
    # # print(nc_2020_197_13.variables["y_image_bounds"][:])

    # # calculate lat lons
    # # Print arrays of calculated latitude and longitude
    # # Call function to calculate latitude and longitude from GOES ABI fixed grid projection data
    # # print("=== Latitude and Longitude Calculation ===")
    # # abi_lat, abi_lon = calculate_degrees(nc_2024_011_12)
    # # abi_lat_2, abi_lon_2 = calculate_degrees(nc_2024_011_13)
    # # abi_lat_3, abi_lon_3 = calculate_degrees(nc_2020_197_13)

    # # print("    - Latitude:")
    # # print(abi_lat.shape)
    # # print((abi_lat_2 == abi_lat).all())
    # # print((abi_lat_2 == abi_lat_3).all())

    # # print("    - Longitude:")
    # # print(abi_lon.shape)
    # # print((abi_lon_2 == abi_lon).all())
    # # print((abi_lon_2 == abi_lon_3).all())
