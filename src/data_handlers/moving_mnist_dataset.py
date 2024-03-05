import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import (
    sequence_df_generator_moving_mnist,
    classify_array_in_bins,
    classify_array_in_integer_classes,
)
from typing import Optional


class MovingMnistDataset(Dataset):
    def __init__(
        self,
        path: str,
        input_frames: int = 3,
        num_bins: Optional[int] = None,
        shuffle: bool = False,
    ):
        super(MovingMnistDataset, self).__init__()

        self.path = path
        self.shuffle = shuffle
        self.input_frames = input_frames
        self.num_bins = num_bins

        # list all of the folders in path
        self.sequence_df = sequence_df_generator_moving_mnist(
            path=path,
            in_channel=input_frames,
        )

    def __getitem__(self, idx):
        sequence = self.sequence_df.values[idx].tolist()
        # load the numpy arrays from the list in sequence and concatenate them
        in_frames = np.load(sequence[0], allow_pickle=True)
        in_frames = in_frames[np.newaxis]
        for n in range(self.input_frames - 1):
            next_input_frame = np.load(sequence[n + 1], allow_pickle=True)[np.newaxis]
            in_frames = np.concatenate((in_frames, next_input_frame), axis=0)

        in_frames = in_frames / 255

        out_frames = np.load(sequence[-1], allow_pickle=True)[np.newaxis]
        out_frames = out_frames / 255

        if self.num_bins is not None and self.num_bins > 0:
            # out_frames = classify_array_in_integer_classes(out_frames[0], self.num_bins)
            out_frames = classify_array_in_bins(out_frames[0], self.num_bins)

        return in_frames, out_frames

    def __len__(self):
        return self.sequence_df.shape[0]
