import os
import pandas as pd
import numpy as np
import fire
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DATASET ANALYZER")


def main(path):
    folders = os.listdir(path)
    logger.info(f"Number of folders: {len(folders)}")
    folders_2022 = [folder for folder in folders if folder.startswith("2022")]
    logger.info(f"Number of folders starting with '2022': {len(folders_2022)}")
    folders_2023 = [folder for folder in folders if folder.startswith("2023")]
    logger.info(f"Number of folders starting with '2023': {len(folders_2023)}")

    num_files_in_folder = []
    inpaint_pcts = []
    highest_inpaint = 0
    highest_mean_inpaint = 0

    for folder in sorted(folders):
        files = os.listdir(os.path.join(path, folder))
        npy_files = [file for file in files if file.endswith(".npy")]
        num_files_in_folder.append(len(npy_files))
        logger.info(f"Folder: {folder}, num_files: {len(npy_files)}")
        try:
            data_df = pd.read_csv(os.path.join(path, folder, f"data_{folder}.csv"))
            # print(data_df.loc[(data_df["inpaint_pct"] > 1.9) & (data_df["inpaint_pct"] < 5)]["filenames"])
            inpaint_pcts += data_df["inpaint_pct"].dropna().to_list()
            if data_df['inpaint_pct'].max() > highest_inpaint:
                highest_inpaint =  data_df['inpaint_pct'].max()
                highest_inpaint_img = data_df.iloc[data_df['inpaint_pct'].idxmax()]['filenames']
            if data_df['inpaint_pct'].mean() > highest_mean_inpaint:
                highest_mean_inpaint = data_df['inpaint_pct'].mean()
                highest_mean_inpaint_day = folder
            if len(npy_files) != len(data_df.loc[data_df["is_day"] == True]):
                logger.error("Length of Dataframe does not match num ofimages in day")
        except Exception as error:
            logger.error(error)

    logger.info(f"{'=' * 5} SUMMARY {'=' * 5}")
    logger.info(f"Total images: {np.sum(num_files_in_folder)}")
    logger.info(f"Highest inpaint pct is in image {highest_inpaint_img} with: {highest_inpaint} %")
    logger.info(f"Highest mean day is {highest_mean_inpaint_day} with inpaint pct: {highest_mean_inpaint} %")

    # Calculate quantiles
    quantiles = [0.05, 0.5, 0.75, 0.90, 0.95, 0.99]
    results = np.quantile(inpaint_pcts, quantiles)
    
    # Display results
    for q, value in zip(quantiles, results):
        logger.info(f"{q*100}th percentile: {value:.2f}")
    
    # Additional statistics
    logger.info(f"\nMean: {np.mean(inpaint_pcts):.2f}")
    logger.info(f"Median: {np.median(inpaint_pcts):.2f}")
    logger.info(f"Standard deviation: {np.std(inpaint_pcts):.2f}")


if __name__ == '__main__':
    fire.Fire(main)
