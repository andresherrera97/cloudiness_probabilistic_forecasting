import datetime
import os
import re
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from torch.utils.data import Dataset
import src.utils.utils as utils


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


class MontevideoFoldersDataset_v2(Dataset):
    """
    Dataset for Montevideo Dataset separated by folders named 2020XXX
    """

    def __init__(
        self,
        path,
        in_frames=3,
        out_frame=1,
        min_time_diff=5,
        max_time_diff=15,
        csv_path=None,
        transform=None,
    ):
        super(MontevideoFoldersDataset, self).__init__()

        self.path = path
        self.in_frames = in_frames
        self.out_frame = out_frame
        self.min_time_diff = min_time_diff
        self.max_time_diff = max_time_diff
        self.transform = transform
        if csv_path is None:
            self.sequence_df = utils.sequence_df_generator_folders(
                path=path,
                in_channel=in_channel,
                out_channel=out_frame,
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


class SatelliteImagesDatasetSW(Dataset):
    """South America Satellite Images Dataset

    Args:
        root_dir (string): Directory with all images from day n.
        window (int): Size of the moving window to load the images.
        transform (callable, optional): Optional transform to be applied on a sample.


    Returns:
        [dict]: {'images': images, 'time_stamps': time_stamps}
    """

    dia_ref = datetime.datetime(2019, 12, 31)

    def __init__(
        self,
        root_dir,
        window=1,
        transform=None,
        fading_window=True,
        load_cosangs=False,
        meta_path="data/meta",
    ):
        self.root_dir = root_dir
        self.images_list = np.sort(os.listdir(self.root_dir))
        self.transform = transform
        self.window = window
        self.fading_window = fading_window
        self.load_cosangs = load_cosangs

        if load_cosangs:
            self.meta_path = meta_path

        # Load the first "window" images to mem
        img_names = [
            os.path.join(self.root_dir, self.images_list[idx])  # esto es todo el path
            for idx in range(self.window)
        ]

        images = np.array(
            [np.load(img_name) for img_name in img_names]
        )  # cargo imagenes

        if self.transform:
            images = np.array([self.transform(image) for image in images])

        parsed_img_names = [
            re.sub("[^0-9]", "", self.images_list[idx]) for idx in range(self.window)
        ]

        time_stamps = [
            self.dia_ref
            + datetime.timedelta(
                days=int(img_name[4:7]),
                hours=int(img_name[7:9]),
                minutes=int(img_name[9:11]),
                seconds=int(img_name[11:]),
            )
            for img_name in parsed_img_names
        ]

        self.__samples = {
            "images": images,
            "time_stamps": [utils.datetime2str(ts) for ts in time_stamps],
        }

        if self.load_cosangs:
            cosangs_masks = np.array(
                [
                    utils.get_cosangs_mask(meta_path=self.meta_path, img_name=img_name)[
                        1
                    ]
                    for img_name in self.images_list
                ]
            )

            if self.transform:
                cosangs_masks = np.array(
                    [self.transform(mask) for mask in cosangs_masks]
                )

            self.__samples["cosangs_masks"] = cosangs_masks

    def __len__(self):
        if self.fading_window:
            return len(self.images_list)
        else:
            return len(self.images_list) - self.window + 1

    def __getitem__(self, idx):
        if idx == 0:
            return self.__samples
        else:
            # 1) Delete whats left out of the window
            self.__samples["images"] = np.delete(
                self.__samples["images"], obj=0, axis=0
            )
            del self.__samples["time_stamps"][0]

            if self.load_cosangs:
                self.__samples["cosangs_masks"] = np.delete(
                    self.__samples["cosangs_masks"], obj=0, axis=0
                )

            # If i have images left to load:
            #   a) Load images, ts, cosangs (if load_cosangs)
            #   b) Append to dictionary
            if idx < len(self.images_list) - self.window + 1:
                next_image = os.path.join(
                    self.root_dir, self.images_list[idx + self.window - 1]
                )

                image = np.load(next_image)

                if self.transform:
                    image = np.array(self.transform(image))

                self.__samples["images"] = np.append(
                    self.__samples["images"], values=image[np.newaxis, ...], axis=0
                )

                img_name = re.sub("[^0-9]", "", self.images_list[idx + self.window - 1])

                time_stamp = self.dia_ref + datetime.timedelta(
                    days=int(img_name[4:7]),
                    hours=int(img_name[7:9]),
                    minutes=int(img_name[9:11]),
                    seconds=int(img_name[11:]),
                )

                self.__samples["time_stamps"].append(utils.datetime2str(time_stamp))

                if self.load_cosangs:
                    cosangs_mask = utils.get_cosangs_mask(
                        meta_path=self.meta_path,
                        img_name=self.images_list[idx + self.window - 1],
                    )[1]

                    if self.transform:
                        cosangs_mask = np.array(self.transform(cosangs_mask))

                    self.__samples["cosangs_masks"] = np.append(
                        self.__samples["cosangs_masks"],
                        values=cosangs_mask[np.newaxis, ...],
                        axis=0,
                    )
            # If i dont have images left:
            else:
                if self.fading_window:
                    self.window -= 1

            return self.__samples


class SatelliteImagesDatasetSW_NoMasks(Dataset):
    """South America Satellite Images Dataset
    Args:
        root_dir (string): Directory with all images from day n.
        window (int): Size of the moving window to load the images.
        transform (callable, optional): Optional transform to be applied on a sample.


    Returns:
        [dict]: {'images': images, 'time_stamps': time_stamps}
    """

    dia_ref = datetime.datetime(2019, 12, 31)

    def __init__(self, root_dir, window=1, transform=None):
        self.root_dir = root_dir
        self.images_list = np.sort(os.listdir(self.root_dir))
        self.transform = transform
        self.window = window

        # Load the first "window" images to mem
        img_names = [
            os.path.join(self.root_dir, self.images_list[idx])
            for idx in range(self.window)
        ]

        images = np.array([np.load(img_name) for img_name in img_names])

        if self.transform:
            images = np.array([self.transform(image) for image in images])

        img_names = [
            re.sub("[^0-9]", "", self.images_list[idx]) for idx in range(self.window)
        ]

        time_stamps = [
            self.dia_ref
            + datetime.timedelta(
                days=int(img_name[4:7]),
                hours=int(img_name[7:9]),
                minutes=int(img_name[9:11]),
                seconds=int(img_name[11:]),
            )
            for img_name in img_names
        ]

        self.__samples = {
            "images": images,
            "time_stamps": [utils.datetime2str(ts) for ts in time_stamps],
        }

    def __len__(self):
        return len(self.images_list) - self.window + 1

    def __getitem__(self, idx):
        if idx == 0:
            return self.__samples
        else:

            next_image = os.path.join(
                self.root_dir, self.images_list[idx + self.window - 1]
            )

            image = np.load(next_image)

            if self.transform:
                image = np.array(self.transform(image))

            img_name = re.sub("[^0-9]", "", self.images_list[idx + self.window - 1])

            time_stamp = self.dia_ref + datetime.timedelta(
                days=int(img_name[4:7]),
                hours=int(img_name[7:9]),
                minutes=int(img_name[9:11]),
                seconds=int(img_name[11:]),
            )

            self.__samples["images"] = np.append(
                np.delete(self.__samples["images"], obj=0, axis=0),
                values=image[np.newaxis, ...],
                axis=0,
            )

            del self.__samples["time_stamps"][0]
            self.__samples["time_stamps"].append(utils.datetime2str(time_stamp))

            return self.__samples


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
