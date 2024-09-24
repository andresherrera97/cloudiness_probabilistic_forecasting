# USAGE:
#   Utility functions and functions that perform part of the computation
#   for functions in other modules. Should be used as little as possible.
#

import re
import csv
import datetime
import os
import cv2
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import pandas as pd
from .preprocessing_functions import read_meta, get_dtime, get_cosangs


def datetime2str(datetime_obj):
    """
    Receives a datetime object and returns a string
    in format 'day/month/year hr:mins:secs'
    """

    return datetime_obj.strftime("%d/%m/%Y %H:%M:%S")


def str2datetime(date_str):
    """
    Receives a string with format 'day/month/year hr:mins:secs'
    and returns a datetime object
    """
    date, time = date_str.split()
    day, month, year = date.split("/")
    hr, mins, secs = time.split(":")
    return datetime.datetime(
        int(year),
        int(month),
        int(day),
        int(hr),
        int(mins),
        int(secs),
    )


def save_errorarray_as_csv(error_array, time_stamp, filename):
    """Generates a CSV file with the error of the predictions at the different times of the day

    Args:
        error_array (array): Array containing the values of the error of a prediction
        time_stamp (list): Contains the diferent timestamps of the day
        filename (string): path and name of the generated file
    """

    M, N = error_array.shape
    fieldnames = []
    fieldnames.append("timestamp")
    for i in range(N):
        # fieldnames.append(str(10*(i+1)) + 'min')
        fieldnames.append(str(10 * (i)) + "min")

    with open(filename + ".csv", "w", newline="") as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(M):
            row_dict = {}
            row_dict["timestamp"] = time_stamp[i]
            for j in range(N):
                # row_dict[str(10*(j+1)) + 'min']  = error_array[i,j]
                row_dict[str(10 * (j)) + "min"] = error_array[i, j]

            writer.writerow(row_dict)


def get_cosangs_mask(meta_path: str = "data/meta", img_name: str = "ART_2020020_111017.FR"):
    """Returns zenithal cos from img_name, with and whitout threshold

    Args:
        meta_path (str, optional): Defaults to 'data/meta'.
        img_name (str, optional): 'ART....FR' or 'dd/mm/yyyy hh:mm:ss'
    """

    lats, lons = read_meta(meta_path)

    if img_name[0:3] == "ART":
        dtime = get_dtime(img_name)
    else:
        dtime = datetime.datetime(
            year=int(img_name[6:10]),
            month=int(img_name[3:5]),
            day=int(img_name[0:2]),
            hour=int(img_name[11:13]),
            minute=int(img_name[14:16]),
            second=int(img_name[17:]),
        )

    cosangs, _ = get_cosangs(dtime, lats, lons)

    cosangs_thresh = cosangs.copy()

    cosangs_thresh[(0 < cosangs) & (cosangs <= 0.15)] = 0.5
    cosangs_thresh[0.15 < cosangs] = 1

    return cosangs, cosangs_thresh


def get_last_checkpoint(path):
    checkpoints = os.listdir(path)
    epochs = [int(epoch) for cp in checkpoints for epoch in re.findall(r"\d+", cp)]
    last_epoch = max(epochs)
    return last_epoch


def sequence_df_generator_folders(
    path: str, in_channel: int, output_index: int, min_time_diff: int, max_time_diff: int
):
    """Generates DataFrame that contains all possible sequences for images separated by day

    Args:
        path (str): path to folder containing images '/train'
        in_channel (int): Quantity of input images in sequence
        output_index (int): Quantity of output images in sequence
        min_time_diff (int): Images separated by less than this time cannot be a sequence
        max_time_diff (int): Images separated by more than this time cannot be a sequence

    Returns:
        [pd.DataFrame]: Rows contain all sequences
    """
    # folder names
    folders = sorted(os.listdir(path))
    folders = [f for f in folders if f.replace("_", "").isdigit()]
    dt_min = timedelta(minutes=min_time_diff)
    dt_max = timedelta(minutes=max_time_diff)

    sequences_df = []

    for folder in folders:
        folderfiles = sorted(
            [f for f in listdir(join(path, folder)) if isfile(join(path, folder, f)) and f.endswith(".npy")]
        )
        len_day = len(folderfiles)
        for i in range(
            len_day - (in_channel + output_index - 1)
        ):  # check if it can complete a whole sequence
            complete_seq = True
            image_sequence = []
            for j in range(
                in_channel + output_index - 1
            ):  # check if time difference is acceptable
                if complete_seq:
                    dt_i = datetime(
                        1997,
                        5,
                        28,
                        hour=int(folderfiles[i + j][13:15]),
                        minute=int(folderfiles[i + j][15:17]),
                        second=int(folderfiles[i + j][17:19]),
                    )
                    dt_f = datetime(
                        1997,
                        5,
                        28,
                        hour=int(folderfiles[i + j + 1][13:15]),
                        minute=int(folderfiles[i + j + 1][15:17]),
                        second=int(folderfiles[i + j + 1][17:19]),
                    )

                    time_diff = dt_f - dt_i

                    if (
                        dt_min < time_diff < dt_max
                    ):  # images are correctly spaced in time
                        if j == 0:
                            image_sequence.append(f"{folder}/{folderfiles[i + j]}")
                            image_sequence.append(f"{folder}/{folderfiles[i + j + 1]}")
                        if j > 0:
                            image_sequence.append(f"{folder}/{folderfiles[i + j + 1]}")
                    else:
                        complete_seq = False

            if complete_seq:
                sequences_df.append(image_sequence)

    sequences_df = pd.DataFrame(sequences_df)
    return sequences_df


def save_pickle_dict(path="reports/model_training", name="", dict_=None):
    with open(os.path.join(path, name + ".pkl"), "wb") as f:
        pickle.dump(dict_, f, pickle.HIGHEST_PROTOCOL)


def load_pickle_dict(path="reports/model_training", name=""):
    with open(os.path.join(path, name + ".pkl"), "rb") as f:
        return pickle.load(f)


def create_video(source_folder, dest_folder=None, video_name=None, filter=None, fps=1):
    """
    create avi video from png images from "source_folder", filtering using
    "filter" and saves it in "dest_folder" as "video_name".avi
    """

    if video_name is not None:
        video_name = f"{video_name}.avi"
    else:
        video_name = "video.avi"

    if dest_folder is not None:
        dest_path = os.path.join(dest_folder, video_name)
    else:
        dest_path = os.path.join(source_folder, video_name)

    if filter is not None:
        images = [
            img
            for img in os.listdir(source_folder)
            if filter in img and img.endswith(".png")
        ]
    else:
        images = [img for img in os.listdir(source_folder)]

    frame = cv2.imread(os.path.join(source_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(dest_path, 0, fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(source_folder, image)))
    cv2.destroyAllWindows()
    video.release()

    return


def out_channel_calculator(time_horizon: str):

    number = int(re.findall(r"\d+", time_horizon)[0])
    return number // 10
