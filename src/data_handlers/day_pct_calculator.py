import numpy as np
import os
import csv
import fire
import logging

from utils.utils import get_cosangs_mask


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Day Cosangs CSV Generator")


META_PATH = "/clusteruy/home03/DeepCloud/deepCloud/data/raw/meta"


def day_pct_per_img_calculator(
    csv_path: str = "/clusteruy/home03/DeepCloud/deepCloud/data/mvd/val_cosangs_mvd.csv",
    img_folder_path: str = "/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation",
    region: str = "mvd",
):
    filenames_days = sorted(os.listdir(img_folder_path))
    with open(csv_path, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        for filename_day in filenames_days:
            logger.info(f"filename_day: {filename_day}")
            day_folder_path = os.path.join(img_folder_path, filename_day)
            image_names = sorted(os.listdir(day_folder_path))
            for image_name in image_names:
                image_data = []
                image_data.append(image_name)
                _, cosangs_thresh = get_cosangs_mask(
                    meta_path=META_PATH, img_name=image_name
                )
                if region == "mvd":
                    cosangs_img = cosangs_thresh[1550: 1550 + 256, 1600: 1600 + 256]
                elif region == "uru":
                    cosangs_img = cosangs_thresh[1205: 1205 + 512, 1450: 1450 + 512]
                elif region == "region3":
                    cosangs_img = cosangs_thresh[800: 800 + 1024, 1250: 1250 + 1024]
                else:
                    raise ValueError("Invalid region")
                image_data.append(np.count_nonzero(cosangs_img == 1) / cosangs_img.size)
                writer.writerow(image_data)


if __name__ == "__main__":
    fire.Fire(day_pct_per_img_calculator)
