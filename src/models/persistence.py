import datetime as datetime
import numpy as np
import cv2 as cv
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from data_handlers import GOES16Dataset
from metrics.deterministic_metrics import relative_rmse, relative_mae


class Persistence:
    """
    Class that predicts the next images using naive prediction.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.train_loader = None
        self.val_loader = None
        self.time_horizon: int = None
        self.dataset_path: str = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def create_dataloaders(
        self,
        dataset: str,
        path: str,
        time_horizon: int,
    ):
        self.time_horizon = time_horizon
        self.dataset_path = path

        if dataset.lower() in ["goes16", "satellite", "sat"]:
            train_dataset = GOES16Dataset(
                path=os.path.join(path, "train/"),
                num_in_images=1,
                minutes_forward=time_horizon,
                num_bins=None,
                binarization_method=None,
                expected_time_diff=10,
                inpaint_pct_threshold=1.0,
            )

            val_dataset = GOES16Dataset(
                path=os.path.join(path, "val/"),
                num_in_images=1,
                minutes_forward=time_horizon,
                num_bins=None,
                binarization_method=None,
                expected_time_diff=10,
                inpaint_pct_threshold=1.0,
            )

        else:
            raise ValueError(f"Dataset {dataset} not recognized.")

        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Get one sample from train_loader and val_loader to check they have the same size
        train_input_sample, train_output_sample = next(iter(self.train_loader))
        val_input_sample, val_output_sample = next(iter(self.val_loader))

        assert (
            train_input_sample[0].shape == val_input_sample[0].shape
        ), "Train and validation input samples have different sizes"
        assert (
            train_output_sample[0].shape == val_output_sample[0].shape
        ), "Train and validation output samples have different sizes"

        self.height = train_input_sample.shape[2]
        self.width = train_input_sample.shape[3]

        self._logger.info(f"Train loader size: {len(self.train_loader)}")
        self._logger.info(f"Val loader size: {len(self.val_loader)}")
        self._logger.info(f"Samples height: {self.height}, Samples width: {self.width}")

    def evaluate(self, dataset: str = "val"):
        """
        Evaluate the model on the validation set.
        """
        self._logger.info("Evaluating model on validation set")
        metrics = {
            "mae": [],
            "rmse": [],
            "r_mae": [],
            "r_mae_pw": [],
            "r_rmse": [],
            "r_rmse_pw": [],
        }

        loader_to_use = self.val_loader if dataset == "val" else self.train_loader
        mae_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()

        for batch_idx, (in_frames, out_frames) in enumerate(loader_to_use):
            in_frames = in_frames.to(device=self.device)
            out_frames = out_frames.to(device=self.device)

            metrics["mae"].append(mae_loss(in_frames, out_frames).item())
            metrics["rmse"].append(torch.sqrt(mse_loss(in_frames, out_frames)).item())
            metrics["r_mae"].append(
                relative_mae(out_frames, in_frames, pixel_wise=False)
            )
            metrics["r_mae_pw"].append(
                relative_mae(out_frames, in_frames, pixel_wise=True)
            )
            metrics["r_rmse"].append(
                relative_rmse(out_frames, in_frames, pixel_wise=False)
            )
            metrics["r_rmse_pw"].append(
                relative_rmse(out_frames, in_frames, pixel_wise=True)
            )

        for key, value in metrics.items():
            metrics[key] = np.mean(value)

        return metrics


class NoisyPersistence(Persistence):
    """Sub class of Persistence, adds white noise to predictions.

    Args:
        Persistence ([type]): [description]
    """

    def __init__(self, sigma: int):
        # sigma (int): standard deviation of the gauss noise
        super().__init__()
        self.sigma = sigma

    def generate_prediction(self, image, i: int = 0):
        return np.clip(image + np.random.normal(0, self.sigma, image.shape), 0, 1)


class BlurredPersistence(Persistence):
    """
    Sub class of Persistence, returns predictions after passign through a gauss filter.

    Args:
        Persistence ([type]): [description]
    """

    def __init__(self, kernel_size=(0, 0), kernel_size_list=None):
        super().__init__()
        # kernel_size (tuple): size of kernel
        self.kernel_size = kernel_size
        self.kernel_size_list = kernel_size_list

    def generate_prediction(self, image, i: int = 0):
        if self.kernel_size_list:
            kernel_size = self.kernel_size_list[i]
        else:
            kernel_size = self.kernel_size
        return cv.GaussianBlur(image, kernel_size, 0)
