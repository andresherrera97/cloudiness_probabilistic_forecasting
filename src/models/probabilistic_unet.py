# Standard library imports
import os
import time
import copy
import datetime
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

# Related third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision

# Local application/library specific imports
from metrics import (
    QuantileLoss,
    mean_std_loss,
    crps_gaussian,
    crps_bin_classification,
    crps_quantile,
)
from data_handlers import MovingMnistDataset, SatelliteDataset, normalize_pixels
from .unet import UNet
from .model_initialization import weights_init, optimizer_init, scheduler_init
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)


__all__ = [
    "QuantileRegressorUNet",
    "BinClassifierUNet",
    "MonteCarloDropoutUNet",
    "MeanStdUNet",
]


class ProbabilisticUNet(ABC):
    """
    Abstract base class for probabilistic U-Net models.

    Subclasses must implement the abstract methods to provide specific
    implementations for initializing weights, optimizers, dataloaders,
    calculating loss, and cumulative distribution function (CDF).
    """

    def fit(
        self,
        n_epochs: int,
        num_train_samples: int,
        print_train_every_n_batch: Optional[int],
        num_val_samples: int,
        device: str,
        run,
        verbose: bool,
        model_name: str,
        checkpoint_metric: str,
        checkpoint_path: Optional[str],
    ):
        """Train the model on the given input data and labels for a specified number of epochs."""
        pass

    def compute_extra_params(self, X, y):
        """Compute and return any extra parameters needed during training."""
        return None

    def predict(self, X, iterations: int):
        """Generate predictions for the input data."""
        pass

    @abstractmethod
    def initialize_weights(self):
        """Abstract method to initialize the weights of the model."""
        pass

    @abstractmethod
    def initialize_optimizer(self, method: str, lr: float):
        """Abstract method to initialize the optimizer for training the model."""
        pass

    @abstractmethod
    def initialize_scheduler(
        self,
        method: str,
        step_size: int,
        gamma: float,
        patience: int,
        min_lr: float,
    ):
        """Abstract method to initialize the learning rate scheduler for training the model."""
        pass

    @abstractmethod
    def create_dataloaders(
        self,
        dataset: str,
        path: str,
        batch_size: int,
        time_horizon: int,
        binarization_method: Optional[str],
    ):
        """Abstract method to create dataloaders for training and validation data."""
        pass

    @abstractmethod
    def calculate_loss(self, predictions, y_target):
        """Abstract method to calculate the loss between predicted and target values."""
        pass

    @abstractmethod
    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        """Abstract method to compute the cumulative distribution function."""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str, device: str):
        """Abstract method to load a trained checkpoint of the model."""
        pass

    @property
    @abstractmethod
    def name(self):
        """Abstract property to get a unique identifier or name for the model."""
        pass


class BinClassifierUNet(ProbabilisticUNet):
    def __init__(self, n_bins=10, in_frames=3, filters=16, device="cpu"):
        self.n_bins = n_bins
        self.in_frames = in_frames
        self.filters = filters
        self.model = UNet(
            in_frames=self.in_frames, n_classes=self.n_bins, filters=self.filters
        )
        self.loss_fn = nn.CrossEntropyLoss().to(device=device)
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.multiclass_precision_metric = MulticlassPrecision(
            num_classes=n_bins, average="macro", top_k=1, multidim_average="global"
        ).to(device=device)
        self.best_model_dict = None
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
        binarization_method: str,
    ):
        train_dataset = MovingMnistDataset(
            path=os.path.join(path, "train/"),
            input_frames=self.in_frames,
            num_bins=self.n_bins,
            binarization_method=binarization_method,
        )
        val_dataset = MovingMnistDataset(
            path=os.path.join(path, "validation/"),
            input_frames=self.in_frames,
            num_bins=self.n_bins,
            binarization_method=binarization_method,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    def fit(
        self,
        n_epochs: int,
        num_train_samples: int,
        print_train_every_n_batch: Optional[int],
        num_val_samples: int,
        device: str,
        run,
        verbose: bool,
        model_name: str,
        checkpoint_metric: str = "crps",
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:
        # create checkpoint directory if it does not exist
        if checkpoint_path is not None:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # perists through epochs, stores the mean of each epoch
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        crps_per_epoch = []

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
                out_frames = out_frames.to(device=device)

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
            crps_bin_list = []
            precision_list = []

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device)

                    frames_pred = self.model(in_frames.float())

                    val_loss = self.calculate_loss(frames_pred, out_frames)

                    val_loss_per_batch.append(val_loss.detach().item())

                    # calculate auxiliary metrics
                    crps_bin_list.append(
                        crps_bin_classification(frames_pred, out_frames.unsqueeze(1))
                    )
                    precision_list.append(
                        self.multiclass_precision_metric(frames_pred, out_frames)
                    )

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            val_loss_in_epoch = sum(val_loss_per_batch) / len(val_loss_per_batch)
            crps_in_epoch = sum(crps_bin_list) / len(crps_bin_list)
            precision_in_epoch = sum(precision_list) / len(precision_list)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"crps_bin": crps_in_epoch}, step=epoch)
                run.log({"crps": crps_in_epoch}, step=epoch)
                run.log({"precision": precision_in_epoch}, step=epoch)
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
                    f"CRPS({crps_in_epoch:.4f} | "
                    f"precision({precision_in_epoch:.4f} | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            crps_per_epoch.append(crps_in_epoch)

            if checkpoint_metric == "crps":
                checkpoint_metric_in_epoch = crps_in_epoch
            elif checkpoint_metric == "precision":
                checkpoint_metric_in_epoch = 1 - precision_in_epoch
            else:
                checkpoint_metric_in_epoch = val_loss_in_epoch

            if checkpoint_metric_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best {checkpoint_metric}: "
                    f"{checkpoint_metric_in_epoch:.4f}"
                )
                best_val_loss = checkpoint_metric_in_epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "num_filters": self.filters,
                    "num_bins": self.n_bins,
                    "epoch": epoch + 1,
                    "ts": datetime.datetime.now().strftime("%d-%m-%Y_%H:%M"),
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
                    "train_loss_per_epoch": train_loss_per_epoch,
                    "val_loss_per_epoch": val_loss_per_epoch,
                    "crps_per_epoch": crps_per_epoch,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                    "checkpoint_metric": checkpoint_metric,
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

    def predict(self, X, iterations: Optional[int] = None):
        return self.model(X.float())

    def calculate_loss(self, predictions, y_target):
        return self.loss_fn(predictions, y_target)

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        pass

    def load_checkpoint(self, checkpoint_path: str, device: str):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.n_bins = checkpoint["num_bins"]

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=self.n_bins,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)

    @property
    def name(self):
        return f"BinClassifierUNet_{self.n_bins}bins_{self.in_frames}frames_{self.filters}filters"


class QuantileRegressorUNet(ProbabilisticUNet):
    def __init__(
        self,
        in_frames: int = 3,
        filters: int = 16,
        quantiles: Optional[List[float]] = None,
    ):
        self.in_frames = in_frames
        self.filters = filters
        if quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]
        else:
            self.quantiles = quantiles
        self.n_bins = len(self.quantiles)
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=self.n_bins,
            filters=self.filters,
        )
        self.loss_fn = QuantileLoss(quantiles=self.quantiles)
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.best_model_dict = None
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
                min_time_diff=5,
                max_time_diff=15,
                transform=normalize_pixels(mean0=False),
                output_last=True,
                day_pct=1,
            )

            val_dataset = SatelliteDataset(
                path=os.path.join(path, "validation/"),
                cosangs_csv_path=f"{cosangs_csv_path}validation.csv",
                in_channel=self.in_frames,
                out_channel=time_horizon,
                min_time_diff=5,
                max_time_diff=15,
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
        n_epochs: int,
        num_train_samples: int,
        print_train_every_n_batch: Optional[int],
        num_val_samples: int,
        device: str,
        run,
        verbose: bool,
        model_name: str,
        checkpoint_metric: str = "crps",
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:

        # create checkpoint directory if it does not exist
        if checkpoint_path is not None:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # perists through epochs, stores the mean of each epoch
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        crps_per_epoch = []

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
            crps_quantile_list = []

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device)

                    frames_pred = self.model(in_frames.float())

                    val_loss = self.calculate_loss(frames_pred, out_frames)

                    val_loss_per_batch.append(val_loss.detach().item())

                    # calculate auxiliary metrics
                    crps_quantile_list.append(
                        crps_quantile(frames_pred, out_frames, self.quantiles, device)
                    )

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            val_loss_in_epoch = sum(val_loss_per_batch) / len(val_loss_per_batch)
            crps_in_epoch = sum(crps_quantile_list) / len(crps_quantile_list)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"crps_quantile": crps_in_epoch}, step=epoch)
                run.log({"crps": crps_in_epoch}, step=epoch)
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
                    f"CRPS({crps_in_epoch:.4f}) | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            crps_per_epoch.append(crps_in_epoch)

            if checkpoint_metric == "crps":
                checkpoint_metric_in_epoch = crps_in_epoch
            else:
                checkpoint_metric_in_epoch = val_loss_in_epoch

            if checkpoint_metric_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best {checkpoint_metric}: {checkpoint_metric_in_epoch:.4f}"
                )
                best_val_loss = checkpoint_metric_in_epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "num_filters": self.filters,
                    "quantiles": self.quantiles,
                    "epoch": epoch + 1,
                    "ts": datetime.datetime.now().strftime("%d-%m-%Y_%H:%M"),
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
                    "train_loss_per_epoch": train_loss_per_epoch,
                    "val_loss_per_epoch": val_loss_per_epoch,
                    "crps_per_epoch": crps_per_epoch,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                    "checkpoint_metric": checkpoint_metric,
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

    def predict(self, X, iterations: Optional[int] = None):
        return self.model(X.float())

    def calculate_loss(self, predictions, y_target):
        y_target = y_target.repeat(1, predictions.shape[1], 1, 1)
        return self.loss_fn(predictions, y_target)

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        pass

    def load_checkpoint(self, checkpoint_path: str, device: str):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.quantiles = list(checkpoint["quantiles"])
        self.n_bins = len(self.quantiles)

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=self.n_bins,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)

    @property
    def name(self):
        return f"QuantileRegressorUNet_{self.n_bins}bins_{self.in_frames}frames_{self.filters}filters"


class MeanStdUNet(ProbabilisticUNet):
    def __init__(
        self,
        in_frames: int = 3,
        filters: int = 16,
    ):
        self.in_frames = in_frames
        self.filters = filters
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=2,
            filters=self.filters,
        )
        self.best_model_dict = None

        self.loss_fn = mean_std_loss
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
        binarization_method=None,
    ):
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
        model_name: str = "mean_std_unet",
        checkpoint_metric: str = "crps",
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:

        # create checkpoint directory if it does not exist
        if checkpoint_path is not None:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # perists through epochs, stores the mean of each epoch
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        val_mae_per_epoch = []
        crps_per_epoch = []

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
            mae_loss_mean_pred = []
            crps_gaussian_list = []

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device).float()

                    frames_pred = self.model(in_frames.float())

                    val_loss = self.calculate_loss(frames_pred, out_frames)

                    val_loss_per_batch.append(val_loss.detach().item())

                    # calculate auxiliary metrics
                    mae_loss_mean_pred.append(
                        nn.L1Loss()(frames_pred[:, 0, :, :], out_frames[:, 0, :, :])
                    )
                    crps_gaussian_list.append(
                        crps_gaussian(
                            out_frames[:, 0, :, :],
                            frames_pred[:, 0, :, :],
                            frames_pred[:, 1, :, :],
                        )
                    )

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            val_loss_in_epoch = sum(val_loss_per_batch) / len(val_loss_per_batch)
            val_mae_mean_pred_in_epoch = sum(mae_loss_mean_pred) / len(
                mae_loss_mean_pred
            )
            crps_in_epoch = sum(crps_gaussian_list) / len(crps_gaussian_list)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"val_mae_mean_pred": val_mae_mean_pred_in_epoch}, step=epoch)
                run.log({"crps_gaussian": crps_in_epoch}, step=epoch)
                run.log({"crps": crps_in_epoch}, step=epoch)
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
                    f"MAE({val_mae_mean_pred_in_epoch:.4f}) | "
                    f"CRPS({crps_in_epoch:.4f} | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            crps_per_epoch.append(crps_in_epoch)
            val_mae_per_epoch.append(val_mae_mean_pred_in_epoch)

            if checkpoint_metric == "crps":
                checkpoint_metric_in_epoch = crps_in_epoch
            elif checkpoint_metric == "mae":
                checkpoint_metric_in_epoch = val_mae_mean_pred_in_epoch
            else:
                checkpoint_metric_in_epoch = val_loss_in_epoch

            if checkpoint_metric_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best val loss: {checkpoint_metric_in_epoch:.4f}"
                )
                best_val_loss = checkpoint_metric_in_epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "num_filters": self.filters,
                    "epoch": epoch + 1,
                    "ts": datetime.datetime.now().strftime("%d-%m-%Y_%H:%M"),
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
                    "train_loss_per_epoch": train_loss_per_epoch,
                    "val_loss_per_epoch": val_loss_per_epoch,
                    "crps_per_epoch": crps_per_epoch,
                    "val_mae_per_epoch": val_mae_per_epoch,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                    "checkpoint_metric": checkpoint_metric,
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

    def predict(self, X, iterations: int):
        return self.model(X.float())

    def calculate_loss(self, predictions, y_target):
        return self.loss_fn(predictions, y_target)

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        mu, sigma2 = predicted_params[:, 0, :, :], nn.functional.softplus(
            predicted_params[:, 1, :, :]
        )
        dist = torch.distributions.Normal(mu, torch.sqrt(sigma2))
        return dist.cdf(points_to_evaluate)

    def load_checkpoint(self, checkpoint_path: str, device: str):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=2,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)

    @property
    def name(self):
        return f"MeanStdUNet_{self.in_frames}frames_{self.filters}filters"


class MonteCarloDropoutUNet(ProbabilisticUNet):
    def __init__(
        self,
        in_frames: int = 3,
        filters: int = 16,
        n_quantiles: int = 5,
        dropout_p: float = 0.5,
    ):
        if dropout_p is None:
            raise ValueError("Dropout probability must be specified.")

        self.in_frames = in_frames
        self.filters = filters
        self.dropout_p = dropout_p
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=1,
            dropout_p=self.dropout_p,
            filters=self.filters,
        )
        self.n_quantiles = n_quantiles
        self.quantiles = list(np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1])

        self.loss_fn = nn.L1Loss()
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.best_model_dict = None
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
        binarization_method=None,
    ):
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

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    def fit(
        self,
        n_epochs: int,
        num_train_samples: int,
        print_train_every_n_batch: Optional[int],
        num_val_samples: int,
        device: str,
        run,
        verbose: bool,
        model_name: str,
        checkpoint_metric: str = "crps",
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:

        # create checkpoint directory if it does not exist
        if checkpoint_path is not None:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # perists through epochs, stores the mean of each epoch
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        crps_per_epoch = []

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

            # by using F.droput in the UNet, the model can still be set to eval mode
            self.model.eval()

            val_loss_per_batch = []  # stores values for this validation run
            crps_quantile_list = []

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device).float()

                    frames_pred = self.predict(
                        in_frames.float(), iterations=self.n_quantiles
                    )

                    # validation loss is calculated as the mean of the quantiles predictions
                    val_loss = self.calculate_loss(
                        torch.mean(frames_pred, dim=1), out_frames
                    )

                    val_loss_per_batch.append(val_loss.detach().item())

                    # calculate auxiliary metrics
                    crps_quantile_list.append(
                        crps_quantile(frames_pred, out_frames, self.quantiles, device)
                    )

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            val_loss_in_epoch = sum(val_loss_per_batch) / len(val_loss_per_batch)
            crps_in_epoch = sum(crps_quantile_list) / len(crps_quantile_list)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"crps_quantile": crps_in_epoch}, step=epoch)
                run.log({"crps": crps_in_epoch}, step=epoch)
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
                    f"CRPS({crps_in_epoch:.4f}) | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            crps_per_epoch.append(crps_in_epoch)

            if checkpoint_metric == "crps":
                checkpoint_metric_in_epoch = crps_in_epoch
            else:
                checkpoint_metric_in_epoch = val_loss_in_epoch

            if checkpoint_metric_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best {checkpoint_metric}: {checkpoint_metric_in_epoch:.4f}"
                )
                best_val_loss = checkpoint_metric_in_epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "num_filters": self.filters,
                    "dropout_p": self.dropout_p,
                    "n_quantiles": self.n_quantiles,
                    "quantiles": self.quantiles,
                    "epoch": epoch + 1,
                    "ts": datetime.datetime.now().strftime("%d-%m-%Y_%H:%M"),
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
                    "train_loss_per_epoch": train_loss_per_epoch,
                    "val_loss_per_epoch": val_loss_per_epoch,
                    "crps_per_epoch": crps_per_epoch,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                    "checkpoint_metric": checkpoint_metric,
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

    def predict(self, X, iterations: int):
        # as images get bigger the computational cost can be too high, find better way to do this
        predictions = self.model(X.float())

        for _ in range(iterations - 1):
            predictions = torch.cat((predictions, self.model(X.float())), dim=1)

        return torch.sort(predictions, dim=1)[0]

    def calculate_loss(self, predictions, y_target):
        return self.loss_fn(predictions, y_target)

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        # TODO: fix to match quantile regressor method
        mu, sigma2 = predicted_params[:, 0, :, :], nn.functional.softplus(
            predicted_params[:, 1, :, :]
        )
        dist = torch.distributions.Normal(mu, torch.sqrt(sigma2))
        return dist.cdf(points_to_evaluate)

    def load_checkpoint(self, checkpoint_path: str, device: str):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.dropout_p = checkpoint["dropout_p"]
        self.n_quantiles = checkpoint["n_quantiles"]
        self.quantiles = checkpoint["quantiles"]

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=1,
            dropout_p=self.dropout_p,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)

    @property
    def name(self):
        return f"MCDUNet_{self.in_frames}frames_{self.filters}filters_{self.dropout_p}dropoutp_{self.n_quantiles}quantiles"
