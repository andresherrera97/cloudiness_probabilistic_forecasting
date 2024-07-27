# Standard library imports
import os
import time
import copy
import datetime
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np

# Related third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_handlers import MovingMnistDataset, SatelliteDataset, normalize_pixels
from .unet import UNet
from .model_initialization import weights_init, optimizer_init, scheduler_init
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)


class UNetPipeline():
    def __init__(
        self,
        in_frames: int = 3,
        filters: int = 16,
        output_activation: str = "sigmoid",
    ):
        self.in_frames = in_frames
        self.filters = filters
        self.output_activation = output_activation
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=1,
            filters=self.filters,
            output_activation=self.output_activation,
        )
        self.best_model_dict = None

        self.loss_fn = nn.L1Loss()  # Use MAE as train loss
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def initialize_weights(self):
        self.model.apply(weights_init)

    def initialize_optimizer(self, method: str, lr: float):
        self.optimizer = optimizer_init(self.model, method, lr)

    def initialize_scheduler(
        self,
        method: str,
        step_size: int,
        gamma: float,
        patience: int,
        min_lr: float,
    ):
        self.scheduler = scheduler_init(
            self.optimizer, method, step_size, gamma, patience, min_lr
        )

    def create_dataloaders(
        self,
        dataset: str,
        path: str,
        batch_size: int,
        time_horizon: int,
        cosangs_csv_path: Optional[str] = None,
        binarization_method=None,
    ):
        if dataset.lower() in ["moving_mnist", "mnist", "mmnist"]:
            train_dataset = MovingMnistDataset(
                path=os.path.join(path, "train/"),
                input_frames=self.in_frames,
                num_bins=None,
            )
            val_dataset = MovingMnistDataset(
                path=os.path.join(path, "validation/"),
                input_frames=self.in_frames,
                num_bins=None,
            )

        elif dataset.lower() in ["goes16", "satellite"]:
            train_dataset = SatelliteDataset(
                path=os.path.join(path, "train/"),
                cosangs_csv_path=f"{cosangs_csv_path}train.csv",
                in_channel=self.in_frames,
                out_channel=time_horizon,
                transform=normalize_pixels(mean0=False),
                output_last=True,
                day_pct=1,
            )
            val_dataset = SatelliteDataset(
                path=os.path.join(path, "validation/"),
                cosangs_csv_path=f"{cosangs_csv_path}validation.csv",
                in_channel=self.in_frames,
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

    def fit(
        self,
        n_epochs=1,
        num_train_samples: int = 1000,
        print_train_every_n_batch: Optional[int] = 500,
        num_val_samples: int = 1000,
        device: str = "cpu",
        run=None,
        verbose: bool = True,
        model_name: str = "unet",
        checkpoint_metric: str = "val_loss",
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:

        # create checkpoint directory if it does not exist
        if checkpoint_path is not None:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # perists through epochs, stores the mean of each epoch
        train_loss_per_epoch = []
        val_loss_per_epoch = []

        best_val_loss = 1e5

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
            val_loss_in_epoch = []  # stores values inside the current epoch
            self.model.train()

            for batch_idx, (in_frames, out_frames) in enumerate(self.train_loader):

                start_batch = time.time()

                # data to cuda if possible
                in_frames = in_frames.to(device=device).float()
                out_frames = out_frames.to(device=device).float()

                # forward
                frames_pred = self.model(in_frames.float())
                loss = self.calculate_loss(frames_pred, out_frames)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()

                train_loss_in_epoch_list.append(loss.detach().item())
                end_batch = time.time()

                if (
                    verbose
                    and print_train_every_n_batch is not None
                    and batch_idx % print_train_every_n_batch == 0
                ):
                    self._logger.info(
                        f"BATCH({batch_idx + 1}/{len(self.train_loader)}) | "
                        f"Train loss({loss.detach().item():.4f}) | "
                        f"Time Batch({(end_batch - start_batch):.2f}) | "
                    )

                if num_train_samples is not None and batch_idx >= num_train_samples:
                    break

            train_loss_in_epoch = sum(train_loss_in_epoch_list) / len(
                train_loss_in_epoch_list
            )

            self.model.eval()
            val_loss_per_batch = []  # stores values for this validation run

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device).float()

                    frames_pred = self.model(in_frames.float())

                    val_loss = self.calculate_loss(frames_pred, out_frames)

                    val_loss_per_batch.append(val_loss.detach().item())

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            val_loss_in_epoch = sum(val_loss_per_batch) / len(val_loss_per_batch)
            
            if self.scheduler is not None:
                self.scheduler.step(val_loss_in_epoch)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log(
                    {"lr": self.optimizer.state_dict()["param_groups"][0]["lr"]},
                    step=epoch,
                )

            end_epoch = time.time()

            if verbose:
                self._logger.info(
                    f"Epoch({epoch + 1}/{n_epochs}) | "
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | "
                    f"Val_loss({val_loss_in_epoch:.4f}) | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)

            if val_loss_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best val loss: {val_loss_in_epoch:.4f}"
                )
                best_val_loss = val_loss_in_epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "num_filters": self.filters,
                    "epoch": epoch + 1,
                    "ts": datetime.datetime.now().strftime("%d-%m-%Y_%H:%M"),
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
                    "train_loss_per_epoch": train_loss_per_epoch,
                    "val_loss_per_epoch": val_loss_per_epoch,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                }
                if checkpoint_path is not None:
                    checkpoint_name = (
                        f"{model_name}_{str(epoch + 1).zfill(3)}_"
                        f"{datetime.datetime.now().strftime('%Y-%m-%d')}.pt"
                    )
                    torch.save(
                        self.best_model_dict,
                        os.path.join(checkpoint_path, checkpoint_name),
                    )

        return train_loss_per_epoch, val_loss_per_epoch

    def predict(self, X: torch.Tensor):
        return self.model(X.float())

    def calculate_loss(self, predictions: torch.Tensor, y_target: torch.Tensor):
        return self.loss_fn(predictions, y_target)

    def load_checkpoint(self, checkpoint_path: str, device: str):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=1,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)

    @property
    def name(self):
        return f"UNet_{self.in_frames}frames_{self.filters}filters"
