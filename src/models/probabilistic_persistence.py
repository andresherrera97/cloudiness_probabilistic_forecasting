# Standard library imports
import os

# Related third-party imports
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional

# Local application/library specific imports
from metrics import CRPSLoss
from data_handlers import MovingMnistDataset, SatelliteDataset, normalize_pixels


class PersistenceEnsemble:
    def __init__(self, n_quantiles: int, device: str = "cpu"):
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
        time_horizon: Optional[int],
        cosangs_csv_path: Optional[str] = None,
    ):
        if dataset.lower() in ["moving_mnist", "mnist", "mmnist"]:
            train_dataset = MovingMnistDataset(
                path=os.path.join(path, "train/"),
                input_frames=self.n_quantiles,
                use_previous_sequence=True,
            )
            val_dataset = MovingMnistDataset(
                path=os.path.join(path, "validation/"),
                input_frames=self.n_quantiles,
                use_previous_sequence=True,
            )

        elif dataset.lower() in ["goes16", "satellite"]:
            train_dataset = SatelliteDataset(
                path=os.path.join(path, "train/"),
                cosangs_csv_path=f"{cosangs_csv_path}train.csv",
                in_channel=self.n_quantiles,
                out_channel=time_horizon,
                transform=normalize_pixels(mean0=False),
                output_last=True,
                day_pct=1,
            )
            val_dataset = SatelliteDataset(
                path=os.path.join(path, "validation/"),
                cosangs_csv_path=f"{cosangs_csv_path}validation.csv",
                in_channel=self.n_quantiles,
                out_channel=time_horizon,
                transform=normalize_pixels(mean0=False),
                output_last=True,
                day_pct=1,
            )

        else:
            raise ValueError(f"Dataset {dataset} not recognized.")

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

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
            predictions = self.predict(in_frames)
            crps_per_batch.append(
                self.crps_loss.crps_loss(
                    pred=predictions,
                    y=out_frames,
                )
            )

        dataset_crps = np.mean(crps_per_batch)
        return dataset_crps
