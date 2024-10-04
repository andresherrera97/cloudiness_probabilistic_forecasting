import os
import pandas as pd
import numpy as np
import fire


def main(path):
    folders = os.listdir(path)
    print(f"Number of folders: {len(folders)}")
    folders_2022 = [folder for folder in folders if folder.startswith("2022")]
    print(f"Number of folders starting with '2022': {len(folders_2022)}")
    folders_2023 = [folder for folder in folders if folder.startswith("2023")]
    print(f"Number of folders starting with '2023': {len(folders_2023)}")

    num_files_in_folder = []

    for folder in folders:
        print(f"Folder: {folder}")
        files = os.listdir(os.path.join(path, folder))
        npy_files = [file for file in files if file.endswith(".npy")]
        num_files_in_folder.append(len(npy_files))
        data_df = pd.load_csv(os.path.join(path, folder, f"data_{folder}.csv"))


if __name__ == '__main__':
    fire.Fire(main)
