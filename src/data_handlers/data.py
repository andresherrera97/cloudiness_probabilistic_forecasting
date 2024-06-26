import os
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from torch.utils.data import Dataset
import src.utils.utils as utils
from typing import Optional


class SatelliteDataset(Dataset):
    """Dataset for Satellite Dataset separated by folders named 2020XXX"""

    def __init__(
        self,
        path: str,
        in_channel: int = 3,
        out_channel: int = 1,
        min_time_diff: int = 5,
        max_time_diff: int = 15,
        csv_path: Optional[str] = None,
        transform=None,
        output_last: bool = True,
        data_aug: bool = False,
        day_pct: int = 1,
    ):
        super(SatelliteDataset, self).__init__()

        if out_channel == 0:
            raise ValueError("out_channel must be greater than 0")
        self.path = path
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff
        self.transform = transform
        self.output_last = output_last
        self.day_pct = day_pct

        if csv_path is not None:
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
                out_channel=out_channel,
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


class MontevideoDataset(Dataset):
    def __init__(
        self,
        path,
        in_channel=3,
        out_channel=1,
        min_time_diff=5,
        max_time_diff=15,
        csv_path=None,
    ):
        super(MontevideoDataset, self).__init__()

        self.path = path
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff

        if csv_path is None:
            self.sequence_df = utils.sequence_df_generator(
                path=path,
                in_channel=in_channel,
                out_channel=out_channel,
                min_time_diff=min_time_diff,
                max_time_diff=max_time_diff,
            )
        else:
            self.sequence_df = pd.read_csv(csv_path, header=None)

    def __getitem__(self, index):

        # images loading

        for i in range(self.in_channel + self.out_channel):
            if i == 0:  # first image in in_frames
                in_frames = np.load(
                    os.path.join(self.path, self.sequence_df.values[index][i])
                )
                in_frames = in_frames[np.newaxis]
            if i > 0 and i < self.in_channel:  # next images in in_frames
                aux = np.load(
                    os.path.join(self.path, self.sequence_df.values[index][i])
                )
                aux = aux[np.newaxis]
                in_frames = np.concatenate((in_frames, aux), axis=0)
            if i == self.in_channel:  # first image in out_frames
                out_frames = np.load(
                    os.path.join(self.path, self.sequence_df.values[index][i])
                )
                out_frames = out_frames[np.newaxis]
            if i > self.in_channel:
                aux = np.load(
                    os.path.join(self.path, self.sequence_df.values[index][i])
                )
                aux = aux[np.newaxis]
                out_frames = np.concatenate((out_frames, aux), axis=0)

        if self.transform:
            in_frames, out_frames = self.transform(in_frames, out_frames)

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
                out_channel=out_channel,
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


class MontevideoFoldersDataset_w_CMV(Dataset):
    """Dataset for Montevideo Dataset separated by folders named 2020XXX. It also loads the predictions done by CMV."""

    def __init__(
        self,
        path,
        cmv_path,
        in_channel=3,
        out_channel=1,
        min_time_diff=5,
        max_time_diff=15,
        csv_path=None,
        transform=None,
        output_last=True,
        nan_value=0,
        day_pct=1,
    ):
        super(MontevideoFoldersDataset_w_CMV, self).__init__()

        self.path = path
        self.cmv_path = cmv_path
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff
        self.transform = transform
        self.output_last = output_last
        self.nan_value = nan_value
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
                out_channel=out_channel,
                min_time_diff=min_time_diff,
                max_time_diff=max_time_diff,
            )

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

            if i > 0 and i == self.in_channel:
                # load CMV prediction
                output_index = self.in_channel + self.out_channel - 1
                aux = np.load(
                    os.path.join(
                        self.cmv_path,
                        self.sequence_df.values[index][output_index][4:11],
                        self.sequence_df.values[index][output_index],
                    )
                )

                aux[np.isnan(aux)] = self.nan_value

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

        return in_frames, out_frames

    def __len__(self):
        return len(self.sequence_df)


class MontevideoFoldersDataset_w_time(Dataset):
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
        day_pct=1,
    ):
        super(MontevideoFoldersDataset_w_time, self).__init__()

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
                out_channel=out_channel,
                min_time_diff=min_time_diff,
                max_time_diff=max_time_diff,
            )

    def __getitem__(self, index):
        # images loading
        in_time = np.zeros((self.in_channel, 3))
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
                in_time[i, 0] = int(self.sequence_df.values[index][i][8:11])  # day
                in_time[i, 1] = int(self.sequence_df.values[index][i][12:14])  # hh
                in_time[i, 2] = int(self.sequence_df.values[index][i][14:16])  # mm
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
                in_time[i, 0] = int(self.sequence_df.values[index][i][8:11])  # day
                in_time[i, 1] = int(self.sequence_df.values[index][i][12:14])  # hh
                in_time[i, 2] = int(self.sequence_df.values[index][i][14:16])  # mm

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
                    out_time = np.zeros((1, 3))
                    out_time[0, 0] = self.sequence_df.values[index][i][8:11]  # day
                    out_time[0, 1] = self.sequence_df.values[index][i][12:14]  # hh
                    out_time[0, 2] = self.sequence_df.values[index][i][14:16]  # mm

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
                    out_time = np.zeros((self.out_channel, 3))
                    out_time[0, 0] = self.sequence_df.values[index][i][8:11]  # day
                    out_time[0, 1] = self.sequence_df.values[index][i][12:14]  # hh
                    out_time[0, 2] = self.sequence_df.values[index][i][14:16]  # mm

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
                    out_time[i - self.in_channel, 0] = self.sequence_df.values[index][
                        i
                    ][
                        8:11
                    ]  # day
                    out_time[i - self.in_channel, 1] = self.sequence_df.values[index][
                        i
                    ][
                        12:14
                    ]  # hh
                    out_time[i - self.in_channel, 2] = self.sequence_df.values[index][
                        i
                    ][
                        14:16
                    ]  # mm
                # ART_2020xxx_hhmmss.npy

                # out_time.append(self.sequence_df.values[index][i][12:18])

        if self.transform:
            if type(self.transform) == list:
                for function in self.transform:
                    in_frames, out_frames = function(in_frames, out_frames)
            else:
                in_frames, out_frames = self.transform(in_frames, out_frames)

        return in_frames, out_frames, in_time, out_time

    def __len__(self):
        return len(self.sequence_df)


class MontevideoFoldersDataset_w_name(Dataset):
    """Dataset for Montevideo Dataset separated by folders named 2020XXX and returns output name"""

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
        day_pct=1,
    ):
        super(MontevideoFoldersDataset_w_name, self).__init__()

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
                out_channel=out_channel,
                min_time_diff=min_time_diff,
                max_time_diff=max_time_diff,
            )

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
                    out_time = self.sequence_df.values[index][i]

        if self.transform:
            if type(self.transform) == list:
                for function in self.transform:
                    in_frames, out_frames = function(in_frames, out_frames)
            else:
                in_frames, out_frames = self.transform(in_frames, out_frames)

        return in_frames, out_frames, out_time

    def __len__(self):
        return len(self.sequence_df)


class PatchesFoldersDataset(Dataset):
    """Dataset for patches in R3 Dataset, separated by folders named 2020XXX"""

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
        output_30min=False,
        img_size=512,
        patch_size=128,
        day_pct=1,
        train=True,
    ):

        super(PatchesFoldersDataset, self).__init__()

        self.path = path
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff
        self.transform = transform
        self.output_last = output_last
        self.output_30min = output_30min
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
                out_channel=out_channel,
                min_time_diff=min_time_diff,
                max_time_diff=max_time_diff,
            )
        self.img_size = img_size
        self.pred_size = patch_size
        self.train = train

        if output_30min and output_last:
            raise ValueError("Both output_30min and output_last are set True")

        if output_30min and out_channel % 3 != 0:
            raise ValueError(
                "out_channel must be multiple of 3 when output_30min set True"
            )

    def __getitem__(self, index):

        # images loading
        if self.train:
            top = np.random.randint(0, self.img_size - self.pred_size)
            left = np.random.randint(0, self.img_size - self.pred_size)

            for i in range(self.in_channel + self.out_channel):
                if i == 0:  # first image in in_frames
                    in_frames = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )[top : top + self.pred_size, left : left + self.pred_size]
                    in_frames = in_frames[np.newaxis]
                elif i > 0 and i < self.in_channel:  # next images in in_frames
                    aux = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )[top : top + self.pred_size, left : left + self.pred_size]
                    aux = aux[np.newaxis]
                    in_frames = np.concatenate((in_frames, aux), axis=0)
                elif i >= self.in_channel:
                    if self.output_last:
                        if i == (
                            self.in_channel + self.out_channel - 1
                        ):  # last image in out_frames
                            out_frames = np.load(
                                os.path.join(
                                    self.path,
                                    self.sequence_df.values[index][i][4:11],
                                    self.sequence_df.values[index][i],
                                )
                            )[top : top + self.pred_size, left : left + self.pred_size]
                            out_frames = out_frames[np.newaxis]

                    elif self.output_30min:
                        if (i - self.in_channel + 1) % 3 == 0 and (
                            i - self.in_channel + 1
                        ) == 3:  # only 30min period images
                            out_frames = np.load(
                                os.path.join(
                                    self.path,
                                    self.sequence_df.values[index][i][4:11],
                                    self.sequence_df.values[index][i],
                                )
                            )[top : top + self.pred_size, left : left + self.pred_size]
                            out_frames = out_frames[np.newaxis]
                        if (i - self.in_channel + 1) % 3 == 0 and (
                            i - self.in_channel + 1
                        ) > 3:  # only 30min period images
                            aux = np.load(
                                os.path.join(
                                    self.path,
                                    self.sequence_df.values[index][i][4:11],
                                    self.sequence_df.values[index][i],
                                )
                            )[top : top + self.pred_size, left : left + self.pred_size]
                            aux = aux[np.newaxis]
                            out_frames = np.concatenate((out_frames, aux), axis=0)
                    else:
                        if i == self.in_channel:  # first image in out_frames
                            out_frames = np.load(
                                os.path.join(
                                    self.path,
                                    self.sequence_df.values[index][i][4:11],
                                    self.sequence_df.values[index][i],
                                )
                            )[top : top + self.pred_size, left : left + self.pred_size]
                            out_frames = out_frames[np.newaxis]
                        if i > self.in_channel:
                            aux = np.load(
                                os.path.join(
                                    self.path,
                                    self.sequence_df.values[index][i][4:11],
                                    self.sequence_df.values[index][i],
                                )
                            )[top : top + self.pred_size, left : left + self.pred_size]
                            aux = aux[np.newaxis]
                            out_frames = np.concatenate((out_frames, aux), axis=0)

        else:
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
                elif i > 0 and i < self.in_channel:  # next images in in_frames
                    aux = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )
                    aux = aux[np.newaxis]
                    in_frames = np.concatenate((in_frames, aux), axis=0)

                elif i >= self.in_channel:
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
                    elif self.output_30min:
                        if (i - self.in_channel + 1) % 3 == 0 and (
                            i - self.in_channel + 1
                        ) == 3:  # only 30min period images
                            out_frames = np.load(
                                os.path.join(
                                    self.path,
                                    self.sequence_df.values[index][i][4:11],
                                    self.sequence_df.values[index][i],
                                )
                            )
                            out_frames = out_frames[np.newaxis]
                        if (i - self.in_channel + 1) % 3 == 0 and (
                            i - self.in_channel + 1
                        ) > 3:  # only 30min period images
                            aux = np.load(
                                os.path.join(
                                    self.path,
                                    self.sequence_df.values[index][i][4:11],
                                    self.sequence_df.values[index][i],
                                )
                            )
                            aux = aux[np.newaxis]
                            out_frames = np.concatenate((out_frames, aux), axis=0)
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

        return in_frames, out_frames

    def __len__(self):
        return len(self.sequence_df)


class PatchesFoldersDataset_w_geodata(Dataset):
    """Dataset for patches in R3 Dataset, separated by folders named 2020XXX"""

    def __init__(
        self,
        path,
        in_channel=3,
        out_channel=1,
        min_time_diff=5,
        max_time_diff=15,
        csv_path=None,
        output_last=False,
        img_size=512,
        patch_size=128,
        geo_data_path=None,
        day_pct=1,
        train=True,
    ):

        super(PatchesFoldersDataset_w_geodata, self).__init__()

        self.path = path
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff

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
                out_channel=out_channel,
                min_time_diff=min_time_diff,
                max_time_diff=max_time_diff,
            )

        self.img_size = img_size
        self.pred_size = patch_size

        if not geo_data_path:
            raise ValueError("GEO data path needed.")

        self.elevation = np.load(os.path.join(geo_data_path, "elevation.npy"))
        self.lats_lons_array = np.load(
            os.path.join(geo_data_path, "lats_lons_array.npy")
        )

        self.lats_lons_array = abs(self.lats_lons_array)

        if img_size == 256:
            self.elevation = self.elevation[1550 : 1550 + 256, 1600 : 1600 + 256]
            self.elevation = self.elevation / np.max(abs(self.elevation))
            self.elevation = self.elevation[np.newaxis]

            self.lats_lons_array = self.lats_lons_array[
                :, 1550 : 1550 + 256, 1600 : 1600 + 256
            ]
            self.lats_lons_array[0] = (
                self.lats_lons_array[0] - np.min(self.lats_lons_array[0])
            ) / (np.max(self.lats_lons_array[0]) - np.min(self.lats_lons_array[0]))
            self.lats_lons_array[1] = (
                self.lats_lons_array[1] - np.min(self.lats_lons_array[1])
            ) / (np.max(self.lats_lons_array[1]) - np.min(self.lats_lons_array[1]))

        elif img_size == 512:
            self.elevation = self.elevation[1205 : 1205 + 512, 1450 : 1450 + 512]
            self.elevation = self.elevation / np.max(abs(self.elevation))
            self.elevation = self.elevation[np.newaxis]

            self.lats_lons_array = self.lats_lons_array[
                :, 1205 : 1205 + 512, 1450 : 1450 + 512
            ]
            self.lats_lons_array[0] = (
                self.lats_lons_array[0] - np.min(self.lats_lons_array[0])
            ) / (np.max(self.lats_lons_array[0]) - np.min(self.lats_lons_array[0]))
            self.lats_lons_array[1] = (
                self.lats_lons_array[1] - np.min(self.lats_lons_array[1])
            ) / (np.max(self.lats_lons_array[1]) - np.min(self.lats_lons_array[1]))

        elif img_size == 1024:
            self.elevation = self.elevation[800 : 800 + 1024, 1250 : 1250 + 1024]
            self.elevation = self.elevation / np.max(abs(self.elevation))
            self.elevation = self.elevation[np.newaxis]

            self.lats_lons_array = self.lats_lons_array[
                :, 800 : 800 + 1024, 1250 : 1250 + 1024
            ]
            self.lats_lons_array[0] = (
                self.lats_lons_array[0] - np.min(self.lats_lons_array[0])
            ) / (np.max(self.lats_lons_array[0]) - np.min(self.lats_lons_array[0]))
            self.lats_lons_array[1] = (
                self.lats_lons_array[1] - np.min(self.lats_lons_array[1])
            ) / (np.max(self.lats_lons_array[1]) - np.min(self.lats_lons_array[1]))

        else:
            raise ValueError("Img size must correspond to MVD, URU or R3.")

        self.train = train

    def __getitem__(self, index):

        # images loading
        if self.train:
            top = np.random.randint(0, self.img_size - self.pred_size)
            left = np.random.randint(0, self.img_size - self.pred_size)

            patch_elevation = self.elevation[
                :, top : top + self.pred_size, left : left + self.pred_size
            ]  # 1,H,W
            patch_lats_lons = self.lats_lons_array[
                :, top : top + self.pred_size, left : left + self.pred_size
            ]  # 2,H,W

            for i in range(self.in_channel + self.out_channel):
                if i == 0:  # first image in in_frames
                    in_frames = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )[top : top + self.pred_size, left : left + self.pred_size]
                    in_frames = in_frames / 100
                    in_frames = in_frames[np.newaxis]  # 1, H, W
                    in_frames = np.concatenate(
                        (in_frames, patch_lats_lons, patch_elevation), axis=0
                    )
                    in_frames = in_frames[np.newaxis]  # 1, 4, H, W

                if i > 0 and i < self.in_channel:  # next images in in_frames
                    aux = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )[top : top + self.pred_size, left : left + self.pred_size]
                    aux = aux / 100
                    aux = aux[np.newaxis]  # 1, H, W
                    aux = np.concatenate(
                        (aux, patch_lats_lons, patch_elevation), axis=0
                    )
                    aux = aux[np.newaxis]  # 1, 4, H, W

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
                        )[top : top + self.pred_size, left : left + self.pred_size]
                        out_frames = out_frames / 100
                        out_frames = out_frames[np.newaxis]
                else:
                    if i == self.in_channel:  # first image in out_frames
                        out_frames = np.load(
                            os.path.join(
                                self.path,
                                self.sequence_df.values[index][i][4:11],
                                self.sequence_df.values[index][i],
                            )
                        )[top : top + self.pred_size, left : left + self.pred_size]
                        out_frames = out_frames / 100
                        out_frames = out_frames[np.newaxis]
                    if i > self.in_channel:
                        aux = np.load(
                            os.path.join(
                                self.path,
                                self.sequence_df.values[index][i][4:11],
                                self.sequence_df.values[index][i],
                            )
                        )[top : top + self.pred_size, left : left + self.pred_size]
                        aux = aux / 100
                        aux = aux[np.newaxis]
                        out_frames = np.concatenate((out_frames, aux), axis=0)

        else:
            for i in range(self.in_channel + self.out_channel):
                if i == 0:  # first image in in_frames
                    in_frames = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )
                    in_frames = in_frames / 100

                    in_frames = in_frames[np.newaxis]  # 1, H, W
                    in_frames = np.concatenate(
                        (in_frames, self.lats_lons_array, self.elevation), axis=0
                    )
                    in_frames = in_frames[np.newaxis]  # 1, 4, H, W

                if i > 0 and i < self.in_channel:  # next images in in_frames
                    aux = np.load(
                        os.path.join(
                            self.path,
                            self.sequence_df.values[index][i][4:11],
                            self.sequence_df.values[index][i],
                        )
                    )
                    aux = aux / 100
                    aux = aux[np.newaxis]  # 1, H, W
                    aux = np.concatenate(
                        (aux, self.lats_lons_array, self.elevation), axis=0
                    )
                    aux = aux[np.newaxis]  # 1, 4, H, W

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
                        out_frames = out_frames / 100
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
                        out_frames = out_frames / 100
                        out_frames = out_frames[np.newaxis]
                    if i > self.in_channel:
                        aux = np.load(
                            os.path.join(
                                self.path,
                                self.sequence_df.values[index][i][4:11],
                                self.sequence_df.values[index][i],
                            )
                        )
                        aux = aux / 100
                        aux = aux[np.newaxis]
                        out_frames = np.concatenate((out_frames, aux), axis=0)

        return in_frames, out_frames

    def __len__(self):
        return len(self.sequence_df)


class MontevideoFoldersDataset_input_time(Dataset):
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
        super(MontevideoFoldersDataset_input_time, self).__init__()

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
                out_channel=out_channel,
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
                shape = in_frames.shape
                in_frames_times = np.full(
                    shape,
                    int(self.sequence_df.values[index][i][14:16]) / 60,
                    dtype=np.float32,
                )
                in_frames_times = np.concatenate(
                    (
                        in_frames_times,
                        np.full(
                            shape,
                            int(self.sequence_df.values[index][i][12:14]) / 24,
                            dtype=np.float32,
                        ),
                    )
                )
                in_frames_times = np.concatenate(
                    (
                        in_frames_times,
                        np.full(
                            shape,
                            int(self.sequence_df.values[index][i][8:11]) / 356,
                            dtype=np.float32,
                        ),
                    )
                )  # 8:11
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
                in_frames_times = np.concatenate(
                    (
                        in_frames_times,
                        np.full(
                            shape,
                            int(self.sequence_df.values[index][i][14:16]) / 60,
                            dtype=np.float32,
                        ),
                    )
                )
                in_frames_times = np.concatenate(
                    (
                        in_frames_times,
                        np.full(
                            shape,
                            int(self.sequence_df.values[index][i][12:14]) / 24,
                            dtype=np.float32,
                        ),
                    )
                )
                in_frames_times = np.concatenate(
                    (
                        in_frames_times,
                        np.full(
                            shape,
                            int(self.sequence_df.values[index][i][8:11]) / 356,
                            dtype=np.float32,
                        ),
                    )
                )
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

        # in_frames = np.concatenate((in_frames[0:1], in_frames_times[0:3], in_frames[1:2], in_frames_times[3:6], in_frames[2:3], in_frames_times[6:9]), axis=0)
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

        in_frames = np.concatenate(
            (
                in_frames[0:1],
                in_frames_times[0:3],
                in_frames[1:2],
                in_frames_times[3:6],
                in_frames[2:3],
                in_frames_times[6:9],
            ),
            axis=0,
        )
        return in_frames, out_frames

    def __len__(self):
        return len(self.sequence_df)


class MontevideoFoldersDataset_output_time(Dataset):
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
        output_last=True,
        data_aug=False,
        day_pct=1,
    ):
        super(MontevideoFoldersDataset_output_time, self).__init__()

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
                out_channel=out_channel,
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
                    shape = out_frames.shape
                    # out_frames_times = np.full(shape, int(self.sequence_df.values[index][i][14:16])/60, dtype=np.float32)
                    # out_frames_times = np.concatenate((out_frames_times, np.full(shape, int(self.sequence_df.values[index][i][12:14])/24, dtype=np.float32)))
                    out_frames_times = np.full(
                        shape,
                        int(self.sequence_df.values[index][i][12:14]) / 24,
                        dtype=np.float32,
                    )
                    out_frames_times = np.concatenate(
                        (
                            out_frames_times,
                            np.full(
                                shape,
                                int(self.sequence_df.values[index][i][8:11]) / 356,
                                dtype=np.float32,
                            ),
                        )
                    )
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

        in_frames = np.concatenate((in_frames, out_frames_times), axis=0)
        return in_frames, out_frames

    def __len__(self):
        return len(self.sequence_df)
