# Cloudiness Probabilistic Forecasting

### Introduction

Work in Progress

### Dataset Generation

In this repository we provide a script which is capable of downloading specific crops from the satellita images captured by the GOES-16 satellite. The product used is the band 02 (Visible red channel) from the ABI-L2-CMIPF, which provides information about the cloudiness of the region. The script is capable of downloading only the desired crop from the NOAA S3 [bucket](https://noaa-goes16.s3.amazonaws.com/index.html). These Cloud and Moisture Imagery Full Disk products are updated every 10 minutes, and they represent the Reflectance Factor. When the Reflectance Factor images are normalized by the cosine zenital angle, the Planetary Reflectance images are obtained. The images used as input and the output of the models are the Planetary Reflectance images.

To download the images from a day run the script `goes16_dataset_generator.py` ... COMPLETE
