# Cloudiness Probabilistic Forecasting

## Introduction [WIP]

...

## Setup [WIP]

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/andresherrera97/cloudiness_probabilistic_forecasting.git
cd cloudiness_probabilistic_forecasting
```

### 2. Create a Virtual Environment

### 3. Activate the Virtual Environment

### 4. Install Dependencies
With the virtual environment activated, install the project dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Dataset Generation

In this repository we provide a script which is capable of downloading specific crops from the satellita images captured by the GOES-16 satellite and saved in the NOAA S3 [bucket](https://noaa-goes16.s3.amazonaws.com/index.html). The product used is the band 02 (Visible red channel) from the ABI-L2-CMIPF, which provides information about the cloudiness of the region. These Cloud and Moisture Imagery Full Disk products are updated every 10 minutes, and they represent the Reflectance Factor. The Reflectance Factor images are normalized by the cosine zenital angle to get the **Planetary Reflectance** images.

Before running the script to download the images for the dataset, it is necessary to generate the reference coordinates for each pixel in the images taken by the satellite. To do so run the script `src/satellite/goes16_metadata_generator.py` with the following arguments:


| Parameter  | Description         | Type  | Default              |
|------------|---------------------|-------|------------------------|
| region | Selected region to generate the metada. Options: "full_disk" or "conus" | str  | "full_disk"           |
| output_folder   | Output directory to save the metadata generated. | str   | "datasets/ABI_L2_CMIP_M6C02_G16/"  |
| download_reference | If True, the script will download the necessary nc files from the NOAA S3 bucket. | bool  | True   |


The script will generate the metadata for the selected region and save it in the output folder. The metadata is saved as two npy files with the name `lat.npy` and `lon.npy` and a json with general metadata from the satellite. The npy files contain the reference coordinates for each pixel in the images. The coordinates are used to crop the images in a specifc geographic location. It is important to note that the satellite imagery product with ABI Level 2+ are fixed to a specific grid, so it is expected that each pixel in CMI products is associated with a specific and constante latitude and longitude in the Earth's surface. This is referenced in the following document [GOES-R SERIES PRODUCT DEFINITION AND USERS’ GUIDE ](https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf) in the section 4.2.

After generating the metadata for the desired region, it is possible to download the images from the desired time range.
To download the images from a desired time range run the script `src/goes16_dataset_generator.py` with the following arguments:


| Parameter  | Description         | Type  | Default              |
|------------|---------------------|-------|------------------------|
| start_date | First date (included) from which to download images. | str   | "2023-01-01"           |
| end_date   | Last date (included) from which to download images. | str   | "2023-12-31"           |
| skip_night | If True, the script will not download the images with night pixels. | bool  | True                   |
| lat        | Latitude for the center of the crop.     | float | -34                    |
| lon        | Longitude for the center of the crop.    | float | -55                    |
| size       | Size of the sides of the crop.  | int   | 512                    |
| output_folder        | Output directory to save images divided by day.  | str   | "datasets/goes16/"     |


The images are saved in the output folder divided in folders for each day. The folders are named with the format `YYYY_DOY` and inside each folder the images are saved as npy file in the format `YYYY_DOY_UTC_HHMMSS.npy`. The CMIP files are downloaded with a 12-bit precion from the NOAA S3 bucket, after processing them the images are saved as npy files with a 16-bit precision as float16 arrays.


## Model Training [WIP]

To train the model run the script `train.py` ... COMPLETE

# Comments (franchesoni)
- readme talks about dataset but not about the `download_dataset.sh` file
- I'm trying to download whole history but it's hard, my command is:
```
python src/goes16_dataset_generator.py \
    --start_date=2000-01-01 \
    --end_date=2001-01-01 \
    --lat=-31.2827 \
    --lon=-57.9181 \
    --size=32 \
    --output_folder=/export/home/projects/goes16/salto32/ \
    --skip_night=True \
    --save_only_first=True
```

