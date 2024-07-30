import numpy as np
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
        binarization_method: Optional[str] = None,
        use_previous_sequence: bool = False,
    ):
        super(MovingMnistDataset, self).__init__()
        if binarization_method is not None and binarization_method not in [
            "one_hot_encoding",
            "integer_classes",
            "both",
        ]:
            raise ValueError(
                "binarization_method must be either 'one_hot_encoding' ,'integer_classes' or 'both'"
            )

        self.path = path
        self.input_frames = input_frames
        self.num_bins = num_bins
        self.binarization_method = binarization_method
        # list all of the folders in path
        self.sequence_df = sequence_df_generator_moving_mnist(
            path=path,
            in_channel=input_frames,
            use_previous_sequence=use_previous_sequence,
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
                bin_output = (
                    classify_array_in_bins(out_frames[0], self.num_bins),
                    classify_array_in_integer_classes(out_frames[0], self.num_bins),
                )
            out_frames = (out_frames, bin_output)

        return in_frames, out_frames

    def __len__(self):
        return self.sequence_df.shape[0]
