import os
import re
import json
import pandas as pd
import cv2 as cv
import numpy as np
import utils.preprocessing_functions as pf


def load_img(
    meta_path="data/meta",
    img_name="ART_2020020_111017.FR",
    mk_folder_path="data/C02-MK/2020",
    img_folder_path="data/C02-FR/2020",
):
    """Loads image from .FR .MK and metadata files into Numpy array

    Args:
        meta_path (str, optional): Defaults to 'data/meta'.
        img_name (str, optional): Defaults to 'ART_2020020_111017.FR'.
        mk_folder_path (str, optional): Defaults to 'data/C02-MK/2020'.
        img_folder_path (str, optional): Defaults to 'data/C02-FR/2020'.
    """

    lats, lons = pf.read_meta(meta_path)

    dtime = pf.get_dtime(img_name)

    cosangs, cos_mask = pf.get_cosangs(dtime, lats, lons)
    img_mask = pf.load_mask(img_name, mk_folder_path, lats.size, lons.size)
    img = pf.load_img(img_name, img_folder_path, lats.size, lons.size)
    rimg = cv.inpaint(img, img_mask, 3, cv.INPAINT_NS)
    rp_image = pf.normalize(rimg, cosangs, 0.15)

    return rp_image


def save_imgs_2npy(
    meta_path="data/meta",
    mk_folder_path="data/C02-MK/2020",
    img_folder_path="data/C02-FR/2020",
    destintation_path="data/images",
    split_days_into_folders=True,
):
    """Saves images from "img_folder_path" to "destintation_path" as Numpy arrays
       (Uses load_img() function)

    Args:
        meta_path (str, optional): Defaults to 'data/meta'.
        mk_folder_path (str, optional): Defaults to 'data/C02-MK/2020'.
        img_folder_path (str, optional): Defaults to 'data/C02-FR/2020'.
        destintation_path (str, optional): Defaults to 'data/images'.
        split_days_into_folders (bool, optional): Defaults to False.
    """

    for filename in os.listdir(img_folder_path):
        img = load_img(  # added needed arguments (franchesoni)
            meta_path=meta_path,
            img_name=filename,
            mk_folder_path=mk_folder_path,
            img_folder_path=img_folder_path,
        )

        img = np.asarray(img)

        # sets pixel over 100 to 100
        img = np.clip(img, 0, 100)

        if split_days_into_folders:
            day = re.sub("[^0-9]", "", filename)[4:7].lstrip("0")
            try:
                os.makedirs(os.path.join(os.getcwd(), destintation_path, "dia_" + day))
            except:
                pass
            path = os.path.join(
                destintation_path, "day_" + day, os.path.splitext(filename)[0] + ".npy"
            )

        else:
            try:
                os.makedirs(
                    os.path.join(os.getcwd(), destintation_path, "loaded_images")
                )
            except:
                pass
            path = os.path.join(
                destintation_path,
                "loaded_images",
                os.path.splitext(filename)[0] + ".npy",
            )

        np.save(path, img)


def save_imgs_list_2npy(
    imgs_list=[],
    meta_path="data/meta",
    mk_folder_path="data/C02-MK/2020",
    img_folder_path="data/C02-FR/2020",
    destintation_path="data/images",
    split_days_into_folders=True,
    region=None,
):
    """Saves images as Numpy arrays to folders

    Args:
        imgs_list[] (list): List containing the names of the images to be saved. ie: days.
        meta_path (str, optional): Defaults to 'data/meta'.
        mk_folder_path (str, optional): Defaults to 'data/C02-MK/2020'.
        img_folder_path (str, optional): Defaults to 'data/C02-FR/2020'.
        destintation_path (str, optional): Defaults to 'data/images'.
        split_days_into_folders (bool, optional): Defaults to False.
        region (str, optional): Select cropping region.
    """

    for filename in imgs_list:
        img = load_img(  # added needed arguments (franchesoni)
            meta_path=meta_path,
            img_name=filename,
            mk_folder_path=mk_folder_path,
            img_folder_path=img_folder_path,
        )

        if region == "mvd":
            img = img[1550 : 1550 + 256, 1600 : 1600 + 256]
        elif region == "uru":
            img = img[1205 : 1205 + 512, 1450 : 1450 + 512]
        elif region == "region3":
            img = img[800 : 800 + 1024, 1250 : 1250 + 1024]

        # image clipping: sets pixel over 100 to 100
        img = np.clip(img, 0, 100)

        if split_days_into_folders:
            day = re.sub("[^0-9]", "", filename)[:7]
            try:
                os.makedirs(os.path.join(os.getcwd(), destintation_path, day))
            except:
                pass
            path = os.path.join(
                destintation_path, day, os.path.splitext(filename)[0] + ".npy"
            )

        else:
            try:
                os.makedirs(
                    os.path.join(os.getcwd(), destintation_path, "loaded_images")
                )
            except:
                pass
            path = os.path.join(
                destintation_path,
                "loaded_images",
                os.path.splitext(filename)[0] + ".npy",
            )

        np.save(path, img)


def sequence_df_generator_moving_mnist(
    path: str,
    in_channel: int = 3,
    use_previous_sequence: bool = False,
):
    """Generates DataFrame that contains all possible sequences for images separated by day

    Args:
        path (str): path to folder containing images '/train'
        in_channel (int): Quantity of input images in sequence
        use_previous_sequence (bool): If True, uses the last images in the previous sequence as input

    Returns:
        [pd.DataFrame]: Rows contain all sequences
    """

    folders = os.listdir(path)
    folders = sorted(
        [folder for folder in folders if folder[0] != "." and folder[0] != "_"]
    )

    sequences_df = []

    for i, folder in enumerate(folders):
        sequence_folder = os.path.join(path, folder)
        folderfiles = sorted(os.listdir(sequence_folder))

        if use_previous_sequence:
            previous_sequence_folder = os.path.join(path, folders[i - 1])
            previous_sequence_folderfiles = sorted(os.listdir(previous_sequence_folder))
            folderfiles = previous_sequence_folderfiles[-in_channel:] + folderfiles

        for i in range(
            len(folderfiles) - in_channel
        ):  # check if a set of in_channel images can be completed
            sequence = []
            for n, filename in enumerate(folderfiles[i : i + in_channel + 1]):
                if use_previous_sequence and i + n < in_channel:
                    sequence.append(os.path.join(previous_sequence_folder, filename))
                else:
                    sequence.append(os.path.join(sequence_folder, filename))
            sequences_df.append(sequence)

    sequences_df = pd.DataFrame(sequences_df)
    return sequences_df


def create_moving_mnist_dataset():
    moving_mnist_dataset = np.load("datasets/mnist_test_seq.npy")

    for n in range(moving_mnist_dataset.shape[1]):
        print(n)
        # create directory for each sequence of the moving mnist dataset
        os.mkdir(f"datasets/moving_mnist_dataset/{str(n).zfill(4)}")
        for i in range(moving_mnist_dataset.shape[0]):
            # save the image of the moving mnist dataset as a npy file
            moving_mnist_dataset[i, n, :, :].dump(
                f"datasets/moving_mnist_dataset/{str(n).zfill(4)}/{str(i).zfill(3)}.npy"
            )


def classify_array_in_bins(input_array: np.ndarray, num_bins: int):
    """Classify an array of continous values with one-hot encoding"""

    if np.max(input_array) > 1 or np.min(input_array) < 0:
        raise ValueError("Input array must be in the range [0, 1]")

    # Compute bin indices for each element
    bin_indices = np.digitize(input_array, np.linspace(0, 1, num_bins + 1))

    # Create binary array with 1s in the corresponding bin positions
    bin_array = np.zeros((num_bins, *input_array.shape), dtype=int)

    for bin_idx in range(1, num_bins + 1):
        bin_mask = bin_indices == bin_idx
        bin_array[bin_idx - 1][bin_mask] = 1

    return bin_array


def classify_array_in_integer_classes(input_array: np.ndarray, num_bins: int):
    """Classify an array of continous values into integer classes"""
    if np.max(input_array) > 1 or np.min(input_array) < 0:
        raise ValueError("Input array must be in the range [0, 1]")

    # Compute bin indices for each element
    bin_indices = np.digitize(input_array, np.linspace(0, 1, num_bins)) - 1

    return bin_indices


def nc_name_to_npy(filename: str) -> str:
    t_coverage = filename.split("/")[-1].split("_")[3][1:]

    yl = t_coverage[0:4]
    day_of_year = t_coverage[4:7]
    hh = t_coverage[7:9]
    mm = t_coverage[9:11]
    ss = t_coverage[11:13]

    out_path = f"{yl}_{day_of_year}"
    crop_filename = f"{yl}_{day_of_year}_UTC_{hh}{mm}{ss}"
    path_to_img = out_path + f"/{crop_filename}.npy"
    return path_to_img


def filter_df_by_inpaint_pct(
    df: pd.DataFrame, inpaint_pct: float, dataset_path: str
) -> pd.DataFrame:
    latest_folder = ""
    columns = df.columns.to_list()
    rows_to_drop = []
    for index, row in df.iterrows():
        for n in range(len(row) - 1):
            if row[columns[n]].split("/")[0] != row[columns[n + 1]].split("/")[0]:
                raise ValueError("Images in row are from different days")
            folder = row[columns[0]].split("/")[0]
            if folder != latest_folder:
                data_df = pd.read_csv(
                    os.path.join(dataset_path, folder, f"data_{folder}.csv")
                )
                data_df = data_df.dropna()
                data_df["npy_filename"] = data_df["filenames"].apply(
                    lambda x: nc_name_to_npy(x)
                )
                latest_folder = folder
        for col in columns:
            if (
                data_df.loc[data_df["npy_filename"] == row[col]][
                    "inpaint_pct"
                ].to_numpy()
                > inpaint_pct
            ):
                # an image in the sequence is too broken
                rows_to_drop.append(index)
                break
    df = df.drop(rows_to_drop)
    return df


def filter_df_by_black_images(df: pd.DataFrame, path: str) -> pd.DataFrame:
    dataset = path.split("/")[-2]
    if dataset not in ["train", "val", "test"]:
        raise ValueError(
            f"Dataset {dataset} not accepted, must be 'train', 'val' or 'test'"
        )

    if path.startswith(".."):
        path_to_json = "../black_images.json"
    else:
        path_to_json = "black_images.json"
    with open(path_to_json, "r") as file:
        black_images = json.load(file)

    black_images = black_images[dataset]

    rows_to_drop = []
    for index, row in df.iterrows():
        for col in df.columns:
            if row[col].split("/")[-1] in black_images:
                rows_to_drop.append(index)
                break

    df = df.drop(rows_to_drop)
    return df
