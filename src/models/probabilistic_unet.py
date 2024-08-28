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
    median_scale_loss,
    crps_gaussian,
    CRPSLoss,
)
from data_handlers import MovingMnistDataset, SatelliteDataset, normalize_pixels
from .unet import UNet
from .model_initialization import weights_init, optimizer_init, scheduler_init
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)


class ProbabilisticUNet(ABC):
    """
    Abstract base class for probabilistic U-Net models.

    Subclasses must implement the abstract methods to provide specific
    implementations for initializing weights, optimizers, dataloaders,
    calculating loss, and cumulative distribution function (CDF).
    """

    def __init__(
        self,
        in_frames: int = 3,
        filters: int = 16,
        output_activation="sigmoid",
        device="cpu",
    ):
        self.in_frames = in_frames
        self.filters = filters
        self.output_activation = output_activation
        self.device = device
        self._logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.best_model_dict = None
        self.loss_fn = None
        self.n_bins = None

    @abstractmethod
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
        train_metric: str,
        val_metric: str,
        checkpoint_path: Optional[str],
    ):
        """Train the model on the given input data and labels for a specified number of epochs."""
        pass

    def predict(self, X, iterations: Optional[int] = None):
        return self.model(X.float())

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
        binarization_method: Optional[str] = None,
    ):
        if dataset.lower() in ["moving_mnist", "mnist", "mmnist"]:
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

        elif dataset.lower() in ["goes16", "satellite", "sat"]:
            train_dataset = SatelliteDataset(
                path=os.path.join(path, "train/"),
                cosangs_csv_path=f"{cosangs_csv_path}train.csv",
                in_channel=self.in_frames,
                out_channel=time_horizon,
                transform=normalize_pixels(mean0=False),
                output_last=True,
                day_pct=1,
                num_bins=self.n_bins,
                binarization_method=binarization_method,
            )
            val_dataset = SatelliteDataset(
                path=os.path.join(path, "validation/"),
                cosangs_csv_path=f"{cosangs_csv_path}validation.csv",
                in_channel=self.in_frames,
                out_channel=time_horizon,
                transform=normalize_pixels(mean0=False),
                output_last=True,
                day_pct=1,
                num_bins=self.n_bins,
                binarization_method=binarization_method,
            )

        else:
            raise ValueError(f"Dataset {dataset} not recognized.")

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    def calculate_loss(self, predictions, y_target):
        return self.loss_fn(predictions, y_target)

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
    def __init__(
        self,
        n_bins=10,
        in_frames=3,
        filters=16,
        device="cpu",
        output_activation="sigmoid",
    ):
        super().__init__(in_frames, filters, output_activation, device)
        self.n_bins = n_bins
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=self.n_bins,
            filters=self.filters,
            output_activation=self.output_activation,
        )
        self.loss_fn = nn.CrossEntropyLoss().to(device=self.device)
        self.crps_loss = CRPSLoss(num_bins=self.n_bins + 1, device=self.device)
        self.multiclass_precision_metric = MulticlassPrecision(
            num_classes=n_bins, average="macro", top_k=1, multidim_average="global"
        ).to(device=self.device)

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
        train_metric: Optional[str] = None,
        val_metric: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:
        # create checkpoint directory if it does not exist
        if checkpoint_path is not None:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # perists through epochs, stores the mean of each epoch
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        crps_per_epoch = []
        cross_entropy_per_epoch = []
        precision_per_epoch = []

        best_val_loss = 1e5

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
            self.model.train()

            for batch_idx, (in_frames, (out_frames, bin_output)) in enumerate(
                self.train_loader
            ):

                start_batch = time.time()

                # data to cuda if possible
                in_frames = in_frames.to(device=device).float()

                # forward
                frames_pred = self.model(in_frames.float())

                if train_metric is None or train_metric in ["cross_entropy", "ce"]:
                    bin_output = bin_output.to(device=device)
                    loss = self.calculate_loss(frames_pred, bin_output)
                elif train_metric == "crps":
                    out_frames = out_frames.to(device=device)
                    loss = self.crps_loss.crps_loss(frames_pred, out_frames)
                else:
                    raise ValueError(f"Training loss {train_metric} not recognized.")

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
            cross_entropy_loss_per_batch = []  # stores values for this validation run
            crps_bin_list = []
            precision_list = []

            with torch.no_grad():
                for val_batch_idx, (in_frames, (out_frames, bin_output)) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device)
                    bin_output = bin_output.to(device=device)

                    frames_pred = self.model(in_frames.float())

                    cross_entropy_loss = self.calculate_loss(frames_pred, bin_output)
                    cross_entropy_loss_per_batch.append(
                        cross_entropy_loss.detach().item()
                    )

                    crps_bin_list.append(
                        self.crps_loss.crps_loss(frames_pred, out_frames)
                    )
                    precision_list.append(
                        self.multiclass_precision_metric(frames_pred, bin_output)
                    )

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            cross_entropy_in_epoch = sum(cross_entropy_loss_per_batch) / len(
                cross_entropy_loss_per_batch
            )
            crps_in_epoch = sum(crps_bin_list) / len(crps_bin_list)
            precision_in_epoch = sum(precision_list) / len(precision_list)

            if val_metric is None or val_metric in ["cross_entropy", "ce"]:
                val_loss_in_epoch = cross_entropy_in_epoch
            elif val_metric == "crps":
                val_loss_in_epoch = crps_in_epoch
            elif val_metric == "precision":
                val_loss_in_epoch = 1 - precision_in_epoch
            else:
                raise ValueError(f"Validation loss {val_metric} not recognized.")

            if self.scheduler is not None:
                self.scheduler.step(val_loss_in_epoch)

            if run is not None:
                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"cross_entropy": cross_entropy_in_epoch}, step=epoch)
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
                    f"Val_loss({(val_loss_in_epoch):06.4f}) | "
                    f"Cross Entropy({cross_entropy_in_epoch:.4f}) | "
                    f"CRPS({crps_in_epoch:.4f} | "
                    f"precision({precision_in_epoch:.4f} | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            cross_entropy_per_epoch.append(cross_entropy_in_epoch)
            crps_per_epoch.append(crps_in_epoch)
            precision_per_epoch.append(precision_in_epoch)

            if val_loss_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best val loss: " f"{val_loss_in_epoch:.4f}"
                )
                best_val_loss = val_loss_in_epoch
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
                    "cross_entropy_per_epoch": cross_entropy_per_epoch,
                    "precision_per_epoch": precision_per_epoch,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                    "train_metric": train_metric,
                    "val_metric": val_metric,
                }

        if checkpoint_path is not None:
            self._logger.info(f"Saving best model to {checkpoint_path}")
            checkpoint_name = (
                f"{model_name}_" f"{datetime.datetime.now().strftime('%Y-%m-%d')}.pt"
            )
            torch.save(
                self.best_model_dict,
                os.path.join(checkpoint_path, checkpoint_name),
            )

        return train_loss_per_epoch, val_loss_per_epoch

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
        device: str = "cpu",
        output_activation="sigmoid",
        quantiles: Optional[List[float]] = None,
        predict_diff: bool = False,
    ):
        super().__init__(in_frames, filters, output_activation, device)
        self.predict_diff = predict_diff
        if quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]
        else:
            self.quantiles = quantiles
        self.n_bins = len(self.quantiles)
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=self.n_bins,
            filters=self.filters,
            output_activation=self.output_activation,
        )
        self.loss_fn = QuantileLoss(quantiles=self.quantiles)
        self.crps_loss = CRPSLoss(quantiles=self.quantiles, device=self.device)

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
        train_metric: Optional[str] = None,
        val_metric: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:

        # create checkpoint directory if it does not exist
        if checkpoint_path is not None:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # perists through epochs, stores the mean of each epoch
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        quantile_loss_per_epoch = []
        crps_per_epoch = []

        best_val_loss = 1e5

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
            self.model.train()

            for batch_idx, (in_frames, out_frames) in enumerate(self.train_loader):

                start_batch = time.time()

                # data to cuda if possible
                in_frames = in_frames.to(device=device).float()
                out_frames = out_frames.to(device=device).float()

                # forward
                frames_pred = self.model(in_frames.float())
                if self.predict_diff:
                    frames_pred = torch.cumsum(frames_pred, dim=1)

                if train_metric is None or train_metric in [
                    "quantile",
                    "pinball",
                    "quant",
                ]:
                    loss = self.calculate_loss(frames_pred, out_frames)
                elif train_metric == "crps":
                    loss = self.crps_loss.crps_loss(
                        pred=frames_pred,
                        y=out_frames,
                    )
                else:
                    raise ValueError(f"Training loss {train_metric} not recognized.")

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
            crps_loss_per_batch = []  # stores values for this validation run
            quantile_loss_per_batch = []  # stores values for this validation run

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device)

                    frames_pred = self.model(in_frames.float())
                    if self.predict_diff:
                        frames_pred = torch.cumsum(frames_pred, dim=1)

                    quantile_loss = self.calculate_loss(frames_pred, out_frames)
                    quantile_loss_per_batch.append(quantile_loss.detach().item())

                    crps_loss = self.crps_loss.crps_loss(
                        pred=frames_pred,
                        y=out_frames,
                    )

                    crps_loss_per_batch.append(crps_loss.detach().item())

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            quantile_loss_in_epoch = sum(quantile_loss_per_batch) / len(
                quantile_loss_per_batch
            )
            crps_in_epoch = sum(crps_loss_per_batch) / len(crps_loss_per_batch)

            if val_metric is None or val_metric.lower() in [
                "quantile",
                "pinball",
                "quant",
            ]:
                val_loss_in_epoch = quantile_loss_in_epoch
            elif val_metric.lower() == "crps":
                val_loss_in_epoch = crps_in_epoch
            else:
                raise ValueError(f"Validation loss {val_metric} not recognized.")

            if self.scheduler is not None:
                self.scheduler.step(val_loss_in_epoch)

            if run is not None:
                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"quantile_loss": quantile_loss_in_epoch}, step=epoch)
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
                    f"Quantile_loss({quantile_loss_in_epoch:.4f}) | "
                    f"CRPS({crps_in_epoch:.4f}) | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            crps_per_epoch.append(crps_in_epoch)
            quantile_loss_per_epoch.append(quantile_loss_in_epoch)

            if val_loss_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best val loss: {val_loss_in_epoch:.4f}"
                )
                best_val_loss = val_loss_in_epoch
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
                    "quantile_loss_per_epoch": quantile_loss_per_epoch,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                    "predict_diff": self.predict_diff,
                    "train_metric": train_metric,
                    "val_metric": val_metric,
                }

        if checkpoint_path is not None:
            self._logger.info(f"Saving best model to {checkpoint_path}")
            checkpoint_name = (
                f"{model_name}_" f"{datetime.datetime.now().strftime('%Y-%m-%d')}.pt"
            )
            torch.save(
                self.best_model_dict,
                os.path.join(checkpoint_path, checkpoint_name),
            )

        return train_loss_per_epoch, val_loss_per_epoch

    def predict(self, X, iterations: Optional[int] = None):
        frames_pred = self.model(X.float())
        if self.predict_diff:
            frames_pred = torch.cumsum(frames_pred, dim=1)
        return frames_pred

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
        output_activation: str = "sigmoid",
        device: str = "cpu",
    ):
        super().__init__(in_frames, filters, output_activation, device)
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=2,
            filters=self.filters,
            output_activation=self.output_activation,
        )

        self.loss_fn = mean_std_loss

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
        train_metric: str = "mean_std",
        val_metric: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:

        # create checkpoint directory if it does not exist
        if checkpoint_path is not None:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # perists through epochs, stores the mean of each epoch
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        mse_per_epoch = []
        crps_per_epoch = []
        mean_std_per_epoch = []

        best_val_loss = 1e5

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
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
            mean_std_loss_per_batch = []  # stores values for this validation run
            mse_loss_mean_pred = []
            crps_gaussian_list = []

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device).float()

                    frames_pred = self.model(in_frames.float())

                    mean_std_loss_per_batch.append(
                        self.calculate_loss(frames_pred, out_frames).detach().item()
                    )

                    # calculate auxiliary metrics
                    mse_loss_mean_pred.append(
                        nn.MSELoss()(frames_pred[:, 0, :, :], out_frames[:, 0, :, :])
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

            mean_std_loss_in_epoch = sum(mean_std_loss_per_batch) / len(
                mean_std_loss_per_batch
            )
            mse_mean_pred_in_epoch = sum(mse_loss_mean_pred) / len(mse_loss_mean_pred)
            crps_in_epoch = sum(crps_gaussian_list) / len(crps_gaussian_list)

            if val_metric is None or val_metric.lower() in ["mean_std", "meanstd"]:
                val_loss_in_epoch = mean_std_loss_in_epoch
            elif val_metric.lower() == "mse":
                val_loss_in_epoch = mse_mean_pred_in_epoch
            elif val_metric.lower() == "crps":
                val_loss_in_epoch = crps_in_epoch
            else:
                raise ValueError(f"Validation loss {val_metric} not recognized.")

            if self.scheduler is not None:
                self.scheduler.step(val_loss_in_epoch)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"mean_std_loss": mean_std_loss_in_epoch}, step=epoch)
                run.log({"mse_mean_pred": mse_mean_pred_in_epoch}, step=epoch)
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
                    f"mean_std_loss: {mean_std_loss_in_epoch:.4f}) | "
                    f"MSE({mse_mean_pred_in_epoch:.4f}) | "
                    f"CRPS({crps_in_epoch:.4f} | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            crps_per_epoch.append(crps_in_epoch)
            mse_per_epoch.append(mse_mean_pred_in_epoch)
            mean_std_per_epoch.append(mean_std_loss_in_epoch)

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
                    "crps_per_epoch": crps_per_epoch,
                    "mse_per_epoch": mse_per_epoch,
                    "mean_std_per_epoch": mean_std_per_epoch,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                    "val_metric": val_metric,
                    "train_metric": train_metric,
                }
        if checkpoint_path is not None:
            self._logger.info(f"Saving best model to {checkpoint_path}")
            checkpoint_name = (
                f"{model_name}_" f"{datetime.datetime.now().strftime('%Y-%m-%d')}.pt"
            )
            torch.save(
                self.best_model_dict,
                os.path.join(checkpoint_path, checkpoint_name),
            )

        return train_loss_per_epoch, val_loss_per_epoch

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


class MedianScaleUNet(ProbabilisticUNet):
    def __init__(
        self,
        in_frames: int = 3,
        filters: int = 16,
        output_activation: str = "sigmoid",
        device: str = "cpu",
    ):
        super().__init__(in_frames, filters, output_activation, device)
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=2,
            filters=self.filters,
            output_activation=self.output_activation,
        )

        self.loss_fn = median_scale_loss

    def fit(
        self,
        n_epochs=1,
        num_train_samples: int = 1000,
        print_train_every_n_batch: Optional[int] = 500,
        num_val_samples: int = 1000,
        device: str = "cpu",
        run=None,
        verbose: bool = True,
        model_name: str = "median_scale_unet",
        train_metric: str = "median_scale",
        val_metric: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:

        # create checkpoint directory if it does not exist
        if checkpoint_path is not None:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # perists through epochs, stores the mean of each epoch
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        mae_per_epoch = []
        crps_per_epoch = []
        median_scale_per_epoch = []

        best_val_loss = 1e5

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
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
            median_scale_loss_per_batch = []  # stores values for this validation run
            mae_loss_mean_pred = []
            crps_gaussian_list = []

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device).float()

                    frames_pred = self.model(in_frames.float())

                    median_scale_loss_per_batch.append(
                        self.calculate_loss(frames_pred, out_frames).detach().item()
                    )

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

            median_scale_loss_in_epoch = sum(median_scale_loss_per_batch) / len(
                median_scale_loss_per_batch
            )
            mae_mean_pred_in_epoch = sum(mae_loss_mean_pred) / len(mae_loss_mean_pred)
            crps_in_epoch = sum(crps_gaussian_list) / len(crps_gaussian_list)

            if val_metric is None or val_metric.lower() in ["median_scale"]:
                val_loss_in_epoch = median_scale_loss_in_epoch
            elif val_metric.lower() == "mae":
                val_loss_in_epoch = mae_mean_pred_in_epoch
            elif val_metric.lower() == "crps":
                val_loss_in_epoch = crps_in_epoch
            else:
                raise ValueError(f"Validation loss {val_metric} not recognized.")

            if self.scheduler is not None:
                self.scheduler.step(val_loss_in_epoch)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"median_sclae_loss": median_scale_loss_in_epoch}, step=epoch)
                run.log({"mae_mean_pred": mae_mean_pred_in_epoch}, step=epoch)
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
                    f"MedianScale({median_scale_loss_in_epoch:.4f}) | "
                    f"MAE({mae_mean_pred_in_epoch:.4f}) | "
                    f"CRPS({crps_in_epoch:.4f} | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            crps_per_epoch.append(crps_in_epoch)
            mae_per_epoch.append(mae_mean_pred_in_epoch)
            median_scale_per_epoch.append(median_scale_loss_in_epoch)

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
                    "crps_per_epoch": crps_per_epoch,
                    "val_mae_per_epoch": mae_per_epoch,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                    "val_metric": val_metric,
                    "train_metric": train_metric,
                }
        if checkpoint_path is not None:
            self._logger.info(f"Saving best model to {checkpoint_path}")
            checkpoint_name = (
                f"{model_name}_" f"{datetime.datetime.now().strftime('%Y-%m-%d')}.pt"
            )
            torch.save(
                self.best_model_dict,
                os.path.join(checkpoint_path, checkpoint_name),
            )

        return train_loss_per_epoch, val_loss_per_epoch

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
        return f"MedianScaleUNet_{self.in_frames}frames_{self.filters}filters"


class MonteCarloDropoutUNet(ProbabilisticUNet):
    def __init__(
        self,
        in_frames: int = 3,
        filters: int = 16,
        output_activation="sigmoid",
        device: str = "cpu",
        n_quantiles: int = 5,
        dropout_p: float = 0.5,
    ):
        super().__init__(in_frames, filters, output_activation, device)
        if dropout_p is None:
            raise ValueError("Dropout probability must be specified.")

        self.dropout_p = dropout_p
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=1,
            dropout_p=self.dropout_p,
            filters=self.filters,
            output_activation=self.output_activation,
        )
        self.n_quantiles = n_quantiles
        self.quantiles = list(np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1])
        self.crps_loss = CRPSLoss(quantiles=self.quantiles, device=self.device)
        self.crps_loss_bin = CRPSLoss(num_bins=101, device=self.device)
        self.loss_fn = nn.L1Loss()

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
        train_metric: Optional[str] = None,
        val_metric: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[float], List[float]]:

        # create checkpoint directory if it does not exist
        if checkpoint_path is not None:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # perists through epochs, stores the mean of each epoch
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        crps_per_epoch = []
        mae_per_epoch = []

        best_val_loss = 1e5

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
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

            mae_loss_per_batch = []  # stores values for this validation run
            crps_ranked_list = []
            crps_quantile_list = []
            crps_bin_list = []

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
                    # mae_loss = self.calculate_loss(frames_pred[:, :1, :, :], out_frames)
                    mae_loss = self.calculate_loss(frames_pred, out_frames)
                    mae_loss_per_batch.append(mae_loss.detach().item())

                    # calculate auxiliary metrics
                    # ranked predictions takes the predictions and sorts them from lowest to highest to
                    # calcualte an aproximation to the values of the quantiles. While quantile predictions
                    # takes the predictions and from the distribution it calculates the value of the quantiles.
                    ranked_predictions = self.rank_predictions(frames_pred)
                    crps_ranked_list.append(
                        self.crps_loss.crps_loss(
                            pred=ranked_predictions,
                            y=out_frames,
                        )
                    )

                    # extra_frames_pred = self.predict(
                    #     in_frames.float(), iterations=self.n_quantiles * 2
                    # )
                    # extra_frames_pred = torch.cat(
                    #     (extra_frames_pred, frames_pred), dim=1
                    # )

                    # quantile_predictions = self.quantile_predictions(
                    #     extra_frames_pred, torch.tensor(self.quantiles, device=device, dtype=torch.float32)
                    # )
                    # crps_quantile_list.append(
                    #     self.crps_loss.crps_loss(
                    #         pred=quantile_predictions,
                    #         y=out_frames,
                    #     )
                    # )
                    crps_quantile_list.append(-1)

                    # bin_predictions = self.bin_predictions(extra_frames_pred, bins=100)
                    # crps_bin_list.append(
                    #     self.crps_loss_bin.crps_loss(
                    #         pred=bin_predictions,
                    #         y=out_frames,
                    #     )
                    # )
                    crps_bin_list.append(-1)

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            mae_loss_in_epoch = sum(mae_loss_per_batch) / len(mae_loss_per_batch)
            crps_in_epoch = sum(crps_ranked_list) / len(crps_ranked_list)
            crps_quantile_in_epoch = sum(crps_quantile_list) / len(crps_quantile_list)
            crps_bin_in_epoch = sum(crps_bin_list) / len(crps_bin_list)

            if val_metric is None or val_metric == "mae":
                val_loss_in_epoch = mae_loss_in_epoch
            elif val_metric == "crps":
                val_loss_in_epoch = crps_in_epoch
            else:
                raise ValueError(f"Validation metric {val_metric} not recognized.")

            if self.scheduler is not None:
                self.scheduler.step(val_loss_in_epoch)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"crps_quantile": crps_in_epoch}, step=epoch)
                run.log({"crps": crps_in_epoch}, step=epoch)
                run.log({"crps_torch_quant": crps_quantile_in_epoch}, step=epoch)
                run.log({"crps_bin": crps_bin_in_epoch}, step=epoch)
                run.log({"mae": mae_loss_in_epoch}, step=epoch)
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
                    f"MAE({mae_loss_in_epoch:.4f}) | "
                    f"CRPS({crps_in_epoch:.4f}) | "
                    f"CRPS_Torch({crps_quantile_in_epoch:.4f}) | "
                    f"CRPS_Bin({crps_bin_in_epoch:.4f}) | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            crps_per_epoch.append(crps_in_epoch)
            mae_per_epoch.append(mae_loss_in_epoch)
            # add torch quantile crps

            if val_loss_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best val loss: {val_loss_in_epoch:.4f}"
                )
                best_val_loss = val_loss_in_epoch
                best_epoch = epoch
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
                    "val_metric": val_metric,
                    "train_metric": train_metric,
                }
        if checkpoint_path is not None:
            checkpoint_name = (
                f"{model_name}_E{best_epoch}_VM{val_metric}_BVM{str(best_val_loss).replace('0.', '')[:4]}_"
                f"D{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}.pt"
            )
            self._logger.info(
                f"Saving best model to {checkpoint_path}/{checkpoint_name}"
            )
            torch.save(
                self.best_model_dict,
                os.path.join(checkpoint_path, checkpoint_name),
            )

        return train_loss_per_epoch, val_loss_per_epoch

    def predict(self, X: torch.Tensor, iterations: int) -> torch.Tensor:
        """
        Returns an ensemble of predictions for the given input tensor X.
        The number of predictions is determined by the iterations parameter.

        Returns:
        --------
        torch.Tensor: A tensor of shape (B, iterations, H, W)
        """
        predictions = self.model(X.float())

        for _ in range(iterations - 1):
            predictions = torch.cat((predictions, self.model(X.float())), dim=1)

        return predictions

    def rank_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Ranks the predictions along the batch dimension.

        Returns:
        --------
        torch.Tensor: A tensor of shape (B, iterations, H, W)
        """
        return torch.sort(predictions, dim=1)[0]

    def quantile_predictions(
        self, predictions: torch.Tensor, quantiles: List[float]
    ) -> torch.Tensor:
        """
        Bins the predictions along the batch dimension.

        Returns:
        --------
        torch.Tensor: A tensor of shape (B, len(quantiles), H, W)
        """
        return torch.quantile(predictions, quantiles, dim=1).transpose(1, 0)

    def bin_predictions(self, predictions: torch.Tensor, bins: int) -> torch.Tensor:
        """
        Bins the predictions along the batch dimension.

        Returns:
        --------
        torch.Tensor: A tensor of shape (B, bins, H, W)
        """
        B, _, H, W = predictions.shape
        prob_prediction = torch.zeros((B, bins, H, W), device=predictions.device)

        # Iterate over each spatial location
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    # Get all predictions for this spatial location
                    # location_preds = predictions[..., h, w].view(-1)

                    # Calculate histogram
                    hist = torch.histc(predictions[b, :, h, w], bins=bins, min=0, max=1)

                    # Normalize to get probabilities
                    probs = hist / hist.sum()

                    # Assign probabilities to the output tensor
                    prob_prediction[b, :, h, w] = probs

        return prob_prediction

    def calculate_loss(self, predictions, y_target):
        y_target = y_target.repeat(1, predictions.shape[1], 1, 1)
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
        return f"MCD_IN{self.in_frames}_F{self.filters}_DP{str(self.dropout_p).replace('0.', '')}_NQ{self.n_quantiles}"
