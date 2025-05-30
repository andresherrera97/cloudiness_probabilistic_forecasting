import os
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from torch.utils.data import Dataset
import logging
import utils.utils as utils
from typing import Optional
from .utils import (
    classify_array_in_bins,
    classify_array_in_integer_classes,
    filter_df_by_inpaint_pct,
    filter_df_by_black_images,
)


# Configure logging
logging.basicConfig(level=logging.INFO)


class GOES16Dataset(Dataset):
    """Dataset for GOES16 Dataset separated by folders named YYYY_DOY"""

    def __init__(
        self,
        path: str,
        num_in_images: int = 3,
        minutes_forward: int = 60,
        spatial_context: int = 0,
        num_bins: Optional[int] = None,
        binarization_method: Optional[str] = None,
        expected_time_diff: int = 10,
        inpaint_pct_threshold: Optional[float] = None,
        crop_or_downsample: Optional[str] = None,
    ):
        super().__init__()

        if binarization_method is not None and binarization_method not in [
            "one_hot_encoding",
            "integer_classes",
            "both",
        ]:
            raise ValueError(
                "binarization_method must be either 'one_hot_encoding' ,'integer_classes' or 'both'"
            )

        self.expected_time_diff = expected_time_diff
        self.minutes_forward = minutes_forward
        if self.minutes_forward % self.expected_time_diff != 0:
            raise ValueError(
                f"minutes_forward ({minutes_forward}) must be divisible "
                f"by expected_time_diff ({expected_time_diff})"
            )

        if self.expected_time_diff == 10:
            # dataset is generated from FULL DISK images
            min_time_diff = 5
            max_time_diff = 15
        else:
            raise ValueError(
                f"Expected time difference {self.expected_time_diff} not supported"
            )

        self._logger = logging.getLogger(self.__class__.__name__)
        self.path = path
        self.num_in_images = num_in_images
        self.minutes_forward = minutes_forward
        self.spatial_context = spatial_context
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff
        self.num_bins = num_bins
        self.binarization_method = binarization_method
        self.crop_or_downsample = crop_or_downsample

        output_index = minutes_forward // expected_time_diff

        self.sequence_df = utils.sequence_df_generator_folders(
            path=path,
            in_channel=num_in_images,
            output_index=output_index,
            min_time_diff=min_time_diff,
            max_time_diff=max_time_diff,
        )

        # Keep only the first num_in_images columns and the last one
        self.sequence_df = self.sequence_df.iloc[:, list(range(num_in_images)) + [-1]]

        if inpaint_pct_threshold is not None:
            # filter sequences which contain images with an inpaint pct over threshold
            num_seq_before = len(self.sequence_df)
            self.sequence_df = filter_df_by_inpaint_pct(
                self.sequence_df, inpaint_pct_threshold, path
            )
            self._logger.info(
                f"Number of sequences filtered: {num_seq_before - len(self.sequence_df)}"
            )

        # Check if there are sequences with black images
        num_seq_before = len(self.sequence_df)
        self.sequence_df = filter_df_by_black_images(self.sequence_df, path)
        self.sequence_df = self.sequence_df.sort_values(by=self.sequence_df.columns[0])
        self._logger.info(
            f"Number of sequences filtered by black images: {num_seq_before - len(self.sequence_df)}"
        )

    def __getitem__(self, index):
        # images loading

        for i in range(self.num_in_images + 1):
            if i == 0:  # first image in in_frames
                in_frames = np.load(
                    os.path.join(self.path, self.sequence_df.values[index][i])
                )
                in_frames = in_frames[np.newaxis]
            if 0 < i < self.num_in_images:  # next images in in_frames
                aux = np.load(
                    os.path.join(
                        self.path,
                        self.sequence_df.values[index][i],
                    )
                )
                aux = aux[np.newaxis]
                in_frames = np.concatenate((in_frames, aux), axis=0)

            if i == self.num_in_images:
                # output image
                out_frames = np.load(
                    os.path.join(
                        self.path,
                        self.sequence_df.values[index][i],
                    )
                )
                if self.crop_or_downsample is not None:
                    if (
                        "crop" in self.crop_or_downsample
                        and "down" in self.crop_or_downsample
                    ):
                        # crop_X_down_Y
                        if self.crop_or_downsample.split("_")[0] != "crop":
                            raise ValueError(
                                "Invalid crop_or_downsample value, first crop"
                            )
                        crop_value = int(self.crop_or_downsample.split("_")[1])
                        down_value = int(self.crop_or_downsample.split("_")[3])
                        border_value = (1024 - crop_value) // 2
                        out_frames = out_frames[
                            border_value:-border_value, border_value:-border_value
                        ]
                        out_frames = out_frames[::down_value, ::down_value]
                    elif (
                        "crop" in self.crop_or_downsample
                        and "down" not in self.crop_or_downsample
                    ):
                        crop_value = int(self.crop_or_downsample.split("_")[-1])
                        border_value = (1024 - crop_value) // 2
                        out_frames = out_frames[
                            border_value:-border_value, border_value:-border_value
                        ]
                    elif (
                        "down" in self.crop_or_downsample
                        and "crop" not in self.crop_or_downsample
                    ):
                        down_value = int(self.crop_or_downsample.split("_")[-1])
                        out_frames = out_frames[::down_value, ::down_value]
                    else:
                        raise ValueError("Invalid crop_or_downsample value")
                if self.spatial_context > 0:
                    out_frames = out_frames[
                        self.spatial_context : -self.spatial_context,
                        self.spatial_context : -self.spatial_context,
                    ]
                out_frames = out_frames[np.newaxis]

        if self.crop_or_downsample is not None:
            if "crop" in self.crop_or_downsample and "down" in self.crop_or_downsample:
                # crop_X_down_Y
                if self.crop_or_downsample.split("_")[0] != "crop":
                    raise ValueError("Invalid crop_or_downsample value, first crop")
                crop_value = int(self.crop_or_downsample.split("_")[1])
                down_value = int(self.crop_or_downsample.split("_")[3])
                border_value = (1024 - crop_value) // 2
                in_frames = in_frames[
                    :, border_value:-border_value, border_value:-border_value
                ]
                in_frames = in_frames[:, ::down_value, ::down_value]
            elif (
                "crop" in self.crop_or_downsample
                and "down" not in self.crop_or_downsample
            ):
                crop_value = int(self.crop_or_downsample.split("_")[-1])
                border_value = (1024 - crop_value) // 2
                in_frames = in_frames[
                    :, border_value:-border_value, border_value:-border_value
                ]
            elif (
                "down" in self.crop_or_downsample
                and "crop" not in self.crop_or_downsample
            ):
                down_value = int(self.crop_or_downsample.split("_")[-1])
                in_frames = in_frames[:, ::down_value, ::down_value]
            else:
                raise ValueError("Invalid crop_or_downsample value")

        # check if images are uint8
        if in_frames.dtype == np.uint8:
            in_frames = in_frames.astype(np.float32)
            in_frames /= 255.0
            in_frames = in_frames.astype(np.float16)
            out_frames = out_frames.astype(np.float32)
            out_frames /= 255.0
            out_frames = out_frames.astype(np.float16)

        # pixel encoding for bin classification
        if (
            self.binarization_method is not None
            and self.num_bins is not None
            and self.num_bins > 0
        ):
            if self.binarization_method == "one_hot_encoding":
                bin_output = classify_array_in_bins(out_frames[0], self.num_bins)
            elif self.binarization_method == "integer_classes":
                bin_output = classify_array_in_integer_classes(
                    out_frames[0], self.num_bins
                )
            elif self.binarization_method == "both":
                out_frames_one_hot = classify_array_in_bins(
                    out_frames[0], self.num_bins
                )
                out_frames_integer = classify_array_in_integer_classes(
                    out_frames[0], self.num_bins
                )
                bin_output = (out_frames_one_hot, out_frames_integer)

            out_frames = (out_frames, bin_output)

        return in_frames, out_frames

    def __len__(self):
        return len(self.sequence_df)


class SatelliteDataset(Dataset):
    """Dataset for Satellite Dataset separated by folders named 2020XXX"""

    def __init__(
        self,
        path: str,
        in_channel: int = 3,
        out_channel: int = 1,
        min_time_diff: int = 5,
        max_time_diff: int = 15,
        cosangs_csv_path: Optional[str] = None,
        transform=None,
        output_last: bool = True,
        data_aug: bool = False,
        day_pct: int = 1,
        num_bins: Optional[int] = None,
        binarization_method: Optional[str] = None,
    ):
        super(SatelliteDataset, self).__init__()

        if out_channel == 0:
            raise ValueError("out_channel must be greater than 0")
        if binarization_method is not None and binarization_method not in [
            "one_hot_encoding",
            "integer_classes",
            "both",
        ]:
            raise ValueError(
                "binarization_method must be either 'one_hot_encoding' ,'integer_classes' or 'both'"
            )

        self.path = path
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff
        self.transform = transform
        self.output_last = output_last
        self.day_pct = day_pct
        self.num_bins = num_bins
        self.binarization_method = binarization_method

        if cosangs_csv_path is not None:
            cosangs_df = pd.read_csv(cosangs_csv_path, header=None)
            cosangs_df = cosangs_df.loc[cosangs_df[1] >= self.day_pct]
            self.sequence_df = utils.sequence_df_generator_w_cosangs_folders(
                path=path,
                in_channel=in_channel,
                out_channel=out_channel,
                min_time_diff=min_time_diff,
                max_time_diff=max_time_diff,
                cosangs_df=cosangs_df,
            )
        else:
            self.sequence_df = utils.sequence_df_generator_folders(
                path=path,
                in_channel=in_channel,
                output_index=out_channel,
                min_time_diff=min_time_diff,
                max_time_diff=max_time_diff,
            )

        self.data_aug = data_aug

    def __getitem__(self, index):
        # images loading

        for i in range(self.in_channel + self.out_channel):
            if i == 0:  # first image in in_frames
                in_frames = np.load(
                    os.path.join(
                        self.path,
                        self.sequence_df.values[index][i][4:11],
                        self.sequence_df.values[index][i],
                    )
                )
                in_frames = in_frames[np.newaxis]
            if i > 0 and i < self.in_channel:  # next images in in_frames
                aux = np.load(
                    os.path.join(
                        self.path,
                        self.sequence_df.values[index][i][4:11],
                        self.sequence_df.values[index][i],
                    )
                )
                aux = aux[np.newaxis]
                in_frames = np.concatenate((in_frames, aux), axis=0)
            if self.output_last:
                if i == (
                    self.in_channel + self.out_channel - 1
                ):  # first image in out_frames
                    out_frames = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )
                    out_frames = out_frames[np.newaxis]
            else:
                if i == self.in_channel:  # first image in out_frames
                    out_frames = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )
                    out_frames = out_frames[np.newaxis]
                if i > self.in_channel:
                    aux = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )
                    aux = aux[np.newaxis]
                    out_frames = np.concatenate((out_frames, aux), axis=0)

        if self.transform:
            if isinstance(self.transform, list):
                for function in self.transform:
                    in_frames, out_frames = function(in_frames, out_frames)
            else:
                in_frames, out_frames = self.transform(in_frames, out_frames)

        if self.data_aug:
            rot_angle = np.random.randint(0, 4) * 90
            in_frames = rotate(in_frames, angle=rot_angle, axes=(1, 2))
            out_frames = rotate(out_frames, angle=rot_angle, axes=(1, 2))

        # pixel encoding for bin classification
        if self.num_bins is not None and self.num_bins > 0:
            if self.binarization_method == "one_hot_encoding":
                out_frames = classify_array_in_bins(out_frames[0], self.num_bins)
            elif self.binarization_method == "integer_classes":
                out_frames = classify_array_in_integer_classes(
                    out_frames[0], self.num_bins
                )
            elif self.binarization_method == "both":
                out_frames_one_hot = classify_array_in_bins(
                    out_frames[0], self.num_bins
                )
                out_frames_integer = classify_array_in_integer_classes(
                    out_frames[0], self.num_bins
                )
                out_frames = (out_frames_one_hot, out_frames_integer)

        return in_frames, out_frames

    def __len__(self):
        return len(self.sequence_df)


class MontevideoFoldersDataset(Dataset):
    """Dataset for Montevideo Dataset separated by folders named 2020XXX"""

    def __init__(
        self,
        path,
        in_channel=3,
        out_channel=1,
        min_time_diff=5,
        max_time_diff=15,
        csv_path=None,
        transform=None,
        output_last=False,
        data_aug=False,
        day_pct=1,
    ):
        super(MontevideoFoldersDataset, self).__init__()

        self.path = path
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff
        self.transform = transform
        self.output_last = output_last
        self.day_pct = day_pct

        if csv_path:
            cosangs_df = pd.read_csv(csv_path, header=None)
            cosangs_df = cosangs_df.loc[cosangs_df[1] >= self.day_pct]
            self.sequence_df = utils.sequence_df_generator_w_cosangs_folders(
                path=path,
                in_channel=in_channel,
                out_channel=out_channel,
                min_time_diff=min_time_diff,
                max_time_diff=max_time_diff,
                cosangs_df=cosangs_df,
            )
        else:
            self.sequence_df = utils.sequence_df_generator_folders(
                path=path,
                in_channel=in_channel,
                output_index=out_channel,
                min_time_diff=min_time_diff,
                max_time_diff=max_time_diff,
            )

        self.data_aug = data_aug

    def __getitem__(self, index):
        # images loading

        for i in range(self.in_channel + self.out_channel):
            if i == 0:  # first image in in_frames
                in_frames = np.load(
                    os.path.join(
                        self.path,
                        self.sequence_df.values[index][i][4:11],
                        self.sequence_df.values[index][i],
                    )
                )
                in_frames = in_frames[np.newaxis]
            if i > 0 and i < self.in_channel:  # next images in in_frames
                aux = np.load(
                    os.path.join(
                        self.path,
                        self.sequence_df.values[index][i][4:11],
                        self.sequence_df.values[index][i],
                    )
                )
                aux = aux[np.newaxis]
                in_frames = np.concatenate((in_frames, aux), axis=0)
            if self.output_last:
                if i == (
                    self.in_channel + self.out_channel - 1
                ):  # first image in out_frames
                    out_frames = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )
                    out_frames = out_frames[np.newaxis]
            else:
                if i == self.in_channel:  # first image in out_frames
                    out_frames = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )
                    out_frames = out_frames[np.newaxis]
                if i > self.in_channel:
                    aux = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )
                    aux = aux[np.newaxis]
                    out_frames = np.concatenate((out_frames, aux), axis=0)

        if self.transform:
            if type(self.transform) == list:
                for function in self.transform:
                    in_frames, out_frames = function(in_frames, out_frames)
            else:
                in_frames, out_frames = self.transform(in_frames, out_frames)

        if self.data_aug:
            rot_angle = np.random.randint(0, 4) * 90
            in_frames = rotate(in_frames, angle=rot_angle, axes=(1, 2))
            out_frames = rotate(out_frames, angle=rot_angle, axes=(1, 2))

        return in_frames, out_frames

    def __len__(self):
        return len(self.sequence_df)
