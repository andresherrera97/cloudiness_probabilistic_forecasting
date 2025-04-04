# Standard library imports
import os

# Related third-party imports
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging

# Local application/library specific imports
from metrics import CRPSLoss
from data_handlers import MovingMnistDataset, GOES16Dataset


class PersistenceEnsemble:
    def __init__(self, n_quantiles: int, device: str = "cpu"):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.n_quantiles = n_quantiles
        self.quantiles = list(np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1])
        self.train_loader = None
        self.val_loader = None
        self.device = device
        self.crps_loss = CRPSLoss(quantiles=self.quantiles, device=device)

    def create_dataloaders(
        self,
        dataset: str,
        path: str,
        batch_size: int,
        time_horizon: int,
    ):
        self.batch_size = batch_size
        self.time_horizon = time_horizon
        self.dataset_path = path

        if dataset.lower() in ["moving_mnist", "mnist", "mmnist"]:
            train_dataset = MovingMnistDataset(
                path=os.path.join(path, "train/"),
                input_frames=self.n_quantiles,
            )
            val_dataset = MovingMnistDataset(
                path=os.path.join(path, "validation/"), input_frames=self.n_quantiles
            )

        elif dataset.lower() in [
            "goes16", "satellite", "sat", "debug_salto", "debug", "downsample",
            "salto_down", "salto_512"
        ]:
            train_dataset = GOES16Dataset(
                path=os.path.join(path, "train/"),
                num_in_images=self.n_quantiles,
                minutes_forward=time_horizon,
                expected_time_diff=10,
                inpaint_pct_threshold=1.0,
            )

            val_dataset = GOES16Dataset(
                path=os.path.join(path, "val/"),
                num_in_images=self.n_quantiles,
                minutes_forward=time_horizon,
                expected_time_diff=10,
                inpaint_pct_threshold=1.0,
            )

        else:
            raise ValueError(f"Dataset {dataset} not recognized.")

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

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

    def predict(self, X, allow_equal_quantile_values=True):
        sorted_X = torch.sort(X, dim=1)[0]
        if not allow_equal_quantile_values:
            sorted_X = self.modify_consecutive_equals(sorted_X)
        return sorted_X

    def crps_quantile(self, predictions, y_target):
        return self.crps_loss.crps_loss(
            pred=predictions,
            y=y_target,
        )

    def modify_consecutive_equals(self, tensor, base_value=1e-5):
        B, N, H, W = tensor.shape

        # Create a tensor of increasing values for each slice in N
        n_values = torch.arange(1, N + 1, dtype=tensor.dtype, device=tensor.device)
        n_values = n_values.view(1, N, 1, 1)

        # Compare each slice with the next one
        differences = tensor[:, 1:] - tensor[:, :-1]

        # Find where the difference is zero (i.e., consecutive slices are equal)
        mask = differences == 0

        # Create a tensor to add, with the same shape as the input tensor
        to_add = torch.zeros_like(tensor)

        # Set the values to add where the mask is True
        # The added value is proportional to the slice index
        to_add[:, 1:][mask] = base_value * n_values[:, 1:].expand_as(mask)[mask]

        # Add the values
        modified_tensor = tensor + to_add
        modified_tensor = torch.sort(modified_tensor, dim=1)[0]

        zero_mask = modified_tensor[:, 0, :, :] == 0
        modified_tensor[:, 0, :, :] = (
            modified_tensor[:, 0, :, :] + modified_tensor[:, 1, :, :] / 2 * zero_mask
        )

        return modified_tensor

    def random_example(self):
        for _, (in_frames, out_frames) in enumerate(self.train_loader):
            in_frames = in_frames.to(self.device)
            out_frames = out_frames.to(self.device)
            predictions = self.predict(in_frames)
            crps = self.crps_loss.crps_loss(
                pred=predictions,
                y=out_frames,
            )
            break
        return in_frames, out_frames, predictions, crps

    def predict_on_dataset(self, dataset="validation"):
        data_loader = (
            self.val_loader if dataset in ["validation", "val"] else self.train_loader
        )
        crps_per_batch = []
        for batch_idx, (in_frames, out_frames) in enumerate(data_loader):
            in_frames = in_frames.to(self.device)
            out_frames = out_frames.to(self.device)
            predictions = self.predict(in_frames)
            predictions = predictions.to(self.device)
            crps_per_batch.append(
                self.crps_loss.crps_loss(
                    pred=predictions,
                    y=out_frames,
                )
            )

        dataset_crps = torch.mean(torch.tensor(crps_per_batch))
        return dataset_crps
