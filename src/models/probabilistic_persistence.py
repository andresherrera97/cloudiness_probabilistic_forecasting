# Standard library imports
import os

# Related third-party imports
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional

# Local application/library specific imports
from metrics import crps_quantile
from data_handlers import MovingMnistDataset, SatelliteDataset, normalize_pixels


class PersistenceEnsemble:
    def __init__(
        self,
        n_quantiles: int,
        device: str = "cpu"
    ):
        self.n_quantiles = n_quantiles
        self.quantiles = list(np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1])
        self.train_loader = None
        self.val_loader = None
        self.device = device

    def create_dataloaders(
        self,
        dataset: str,
        path: str,
        batch_size: int,
        time_horizon: int,
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

    def predict(self, X):
        return torch.sort(X, dim=1)[0]

    def crps_quantile(self, predictions, y_target, device):
        return crps_quantile(predictions, y_target, self.quantiles, device)

    def random_example(self):
        for _, (in_frames, out_frames) in enumerate(self.train_loader):
            predictions = self.predict(in_frames)
            crps = self.crps_quantile(predictions, out_frames, self.device)
            break
        return in_frames, out_frames, predictions, crps

    def predict_on_dataset(self, dataset="validation"):
        data_loader = self.val_loader if dataset in ["validation", "val"] else self.train_loader
        crps_per_batch = []
        for batch_idx, (in_frames, out_frames) in enumerate(data_loader):
            predictions = self.predict(in_frames)
            crps_per_batch.append(self.crps_quantile(predictions, out_frames, self.device))

        dataset_crps = np.mean(crps_per_batch)
        return dataset_crps
