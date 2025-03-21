# Standard library imports
import os
import time
import copy
import datetime
import random
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import stats

# Related third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision

# Local application/library specific imports
from metrics import (
    QuantileLoss,
    mean_std_loss,
    median_scale_loss,
    crps_gaussian,
    crps_laplace,
    CRPSLoss,
    MixtureDensityLoss,
)
from data_handlers import MovingMnistDataset, GOES16Dataset, PrefetchLoader
from .unet import UNet
from .model_initialization import weights_init, optimizer_init, scheduler_init
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


@dataclass
class UNetConfig:
    in_frames: int = 3
    spatial_context: int = 0
    filters: int = 16
    output_activation: str = "sigmoid"
    device: str = "cpu"


class UNetPipeline(ABC):
    def __init__(
        self,
        config: UNetConfig,
    ):
        super().__init__()
        self.in_frames = config.in_frames
        self.spatial_context = config.spatial_context
        self.filters = config.filters
        self.output_activation = config.output_activation
        self.device = config.device
        self._logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.best_model_dict = None
        self.loss_fn = None
        self.n_bins: int = None
        self.height: int = None
        self.width: int = None
        self.batch_size: int = None
        self.time_horizon: int = None
        self.dataset_path: str = None
        self.torch_dtype = torch.float16

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
        max_lr: float = 0.1,
        epochs: int = 50,
        steps_per_epoch: int = 20000,
        warmup_start: float = 0.3,
    ):
        self.scheduler = scheduler_init(
            self.optimizer,
            method,
            step_size,
            gamma,
            patience,
            min_lr,
            max_lr,
            epochs,
            steps_per_epoch,
            warmup_start,
        )

    def create_dataloaders(
        self,
        dataset: str,
        path: str,
        batch_size: int,
        time_horizon: int,
        binarization_method: Optional[str] = None,
        crop_or_downsample: Optional[str] = None,
        shuffle: bool = True,
        create_test_loader: bool = False,
        prefetch_loader: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        self.batch_size = batch_size
        self.time_horizon = time_horizon
        self.dataset_path = path

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

        elif dataset.lower() in [
            "goes16",
            "satellite",
            "sat",
            "salto",
            "downsample",
            "salto_down",
            "salto_512",
            "debug",
            "debug_salto",
        ]:
            train_dataset = GOES16Dataset(
                path=os.path.join(path, "train/"),
                num_in_images=self.in_frames,
                minutes_forward=time_horizon,
                spatial_context=self.spatial_context,
                num_bins=self.n_bins,
                binarization_method=binarization_method,
                expected_time_diff=10,
                inpaint_pct_threshold=1.0,
                crop_or_downsample=crop_or_downsample,
            )

            val_dataset = GOES16Dataset(
                path=os.path.join(path, "val/"),
                num_in_images=self.in_frames,
                minutes_forward=time_horizon,
                spatial_context=self.spatial_context,
                num_bins=self.n_bins,
                binarization_method=binarization_method,
                expected_time_diff=10,
                inpaint_pct_threshold=1.0,
                crop_or_downsample=crop_or_downsample,
            )

            if create_test_loader:
                test_dataset = GOES16Dataset(
                    path=os.path.join(path, "test/"),
                    num_in_images=self.in_frames,
                    minutes_forward=time_horizon,
                    spatial_context=self.spatial_context,
                    num_bins=self.n_bins,
                    binarization_method=binarization_method,
                    expected_time_diff=10,
                    inpaint_pct_threshold=1.0,
                    crop_or_downsample=crop_or_downsample,
                )

        else:
            raise ValueError(f"Dataset {dataset} not recognized.")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if create_test_loader:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        if prefetch_loader:
            self.train_loader = PrefetchLoader(self.train_loader, self.device)
            self.val_loader = PrefetchLoader(self.val_loader, self.device)
            if create_test_loader:
                self.test_loader = PrefetchLoader(self.test_loader, self.device)

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

    def remove_spatial_context(self, tensor):
        """
        Removes the spatial context from an arbitrary number of input tensors.
        Optimized for speed, works with both 3D and 4D tensors.
        """
        if self.spatial_context == 0:
            return tensor

        if tensor.dim() == 3:
            tensor = tensor[
                :,
                self.spatial_context : -self.spatial_context,
                self.spatial_context : -self.spatial_context,
            ]
        elif tensor.dim() == 4:
            tensor = tensor[
                :,
                :,
                self.spatial_context : -self.spatial_context,
                self.spatial_context : -self.spatial_context,
            ]
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")

        return tensor

    def calculate_loss(self, predictions: torch.Tensor, y_target: torch.Tensor):
        return self.loss_fn(predictions, y_target)

    def save_checkpoint(
        self,
        model_name: str,
        best_epoch: int,
        best_val_loss: float,
        checkpoint_path: str,
    ):
        checkpoint_name = (
            f"{model_name}_BS_{self.batch_size}_TH{self.time_horizon}_"
            f"E{best_epoch}_BVM{str(best_val_loss).replace('.', '_')[:4]}_"
            f"D{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}.pt"
        )
        self._logger.info(f"Saving best model to {checkpoint_path}/{checkpoint_name}")
        torch.save(
            self.best_model_dict,
            os.path.join(checkpoint_path, checkpoint_name),
        )

    @abstractmethod
    def load_checkpoint(
        self, checkpoint_path: str, device: str, eval_mode: bool = True
    ):
        """Abstract method to load a trained checkpoint of the model."""
        pass

    @property
    @abstractmethod
    def name(self):
        """Abstract property to get a unique identifier or name for the model."""
        pass


class ProbabilisticUNet(UNetPipeline):
    """
    Abstract base class for probabilistic U-Net models.

    Subclasses must implement the abstract methods to provide specific
    implementations for initializing weights, optimizers, dataloaders,
    calculating loss, and cumulative distribution function (CDF).
    """

    def __init__(
        self,
        config: UNetConfig,
    ):
        super().__init__(config)
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
        """Train the model on the given input data and labels for a
        specified number of epochs."""
        pass

    @abstractmethod
    def get_F_at_points(self, points, pred_params):
        pass

    def get_numerical_CRPS(
        self,
        y: torch.Tensor,
        pred: torch.Tensor,
        lower: float,
        upper: float,
        count: int,
    ):
        dys = torch.linspace(lower, upper, count).to(self.device)
        dys = dys.view(1, count, 1, 1)
        dys = dys.expand(1, count, pred.shape[2], pred.shape[3])

        Fy = self.get_F_at_points(dys, pred)  # (B, count, H, W)
        heavyside = 1 * (y <= dys)  # (1, count, H, W)
        integrant = (Fy - heavyside) ** 2  # (B, count, H, W)
        crps = (dys[0, 1] - dys[0, 0]) * (
            integrant[:, 0] / 2 + integrant[:, 1:].sum(dim=1) + integrant[:, -1] / 2
        )
        return torch.mean(crps)

    def predict(self, X, iterations: Optional[int] = None):
        return self.model(X.float())


class BinClassifierUNet(ProbabilisticUNet):
    def __init__(
        self,
        config: UNetConfig,
        n_bins: int = 10,
    ):
        super().__init__(config)
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

    def get_F_at_points(self, points, pred_params):
        # TODO: implement if want to use NUMERICAL CRPS
        pass

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

        best_val_loss = float("inf")
        device_type = "cpu" if device == torch.device("cpu") else "cuda"
        self._logger.info(f"device type: {device_type}")

        scaler = torch.amp.GradScaler(device)  # For mixed precision training

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
            self.model.train()

            for batch_idx, (in_frames, (out_frames, bin_output)) in enumerate(
                self.train_loader
            ):

                start_batch = time.time()

                # data to cuda if possible
                in_frames = in_frames.to(device=device, dtype=self.torch_dtype)

                # forward
                with torch.autocast(
                    device_type=device_type, dtype=self.torch_dtype
                ):  # Enable mixed precision
                    frames_pred = self.model(in_frames)
                    frames_pred = self.remove_spatial_context(frames_pred)

                    if train_metric is None or train_metric in ["cross_entropy", "ce"]:
                        bin_output = bin_output.to(device=device, dtype=torch.long)
                        loss = self.calculate_loss(frames_pred, bin_output)
                    elif train_metric == "crps":
                        out_frames = out_frames.to(
                            device=device, dtype=self.torch_dtype
                        )
                        loss = self.crps_loss.crps_loss(frames_pred, out_frames)
                    else:
                        raise ValueError(
                            f"Training loss {train_metric} not recognized."
                        )

                # backward
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient than zero_grad()

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

                    in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                    out_frames = out_frames.to(device=device, dtype=self.torch_dtype)
                    bin_output = bin_output.to(device=device, dtype=torch.long)

                    with torch.autocast(
                        device_type=device_type, dtype=self.torch_dtype
                    ):
                        frames_pred = self.model(in_frames)
                        frames_pred = self.remove_spatial_context(frames_pred)
                        cross_entropy_loss = self.calculate_loss(
                            frames_pred, bin_output
                        )

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

            if verbose:
                self._logger.info(
                    f"Epoch({epoch + 1}/{n_epochs}) | "
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | "
                    f"Val_loss({(val_loss_in_epoch):06.4f}) | "
                    f"Cross Entropy({cross_entropy_in_epoch:.4f}) | "
                    f"CRPS({crps_in_epoch:.4f} | "
                    f"precision({precision_in_epoch:.4f} | "
                    f"Time_Epoch({(time.time() - start_epoch):.2f}s) |"
                )

            # epoch end
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
                best_epoch = epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "num_filters": self.filters,
                    "num_bins": self.n_bins,
                    "spatial_context": self.spatial_context,
                    "output_activation": self.output_activation,
                    "time_horizon": self.time_horizon,
                    "dataset": self.dataset_path,
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
                    self.save_checkpoint(
                        model_name, best_epoch, best_val_loss, checkpoint_path
                    )

        return train_loss_per_epoch, val_loss_per_epoch

    def load_checkpoint(
        self, checkpoint_path: str, device: str, eval_mode: bool = True
    ):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.n_bins = checkpoint["num_bins"]
        self.spatial_context = checkpoint["spatial_context"]
        self.time_horizon = checkpoint["time_horizon"]

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=self.n_bins,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()

    @property
    def name(self):
        return f"BinUNet_IN{self.in_frames}_NB{self.n_bins}_F{self.filters}_SC{self.spatial_context}"


class QuantileRegressorUNet(ProbabilisticUNet):
    def __init__(
        self,
        config: UNetConfig,
        quantiles: Optional[List[float]] = None,
        predict_diff: bool = False,
    ):
        super().__init__(config)
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

    def get_F_at_points(self, points, pred_params):
        # TODO: implement if want to use NUMERICAL CRPS
        pass

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

        best_val_loss = float("inf")

        device_type = "cpu" if device == torch.device("cpu") else "cuda"
        self._logger.info(f"device type: {device_type}")

        scaler = torch.amp.GradScaler(device)  # For mixed precision training

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
            self.model.train()

            for batch_idx, (in_frames, out_frames) in enumerate(self.train_loader):

                start_batch = time.time()

                # data to cuda if possible
                in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                # forward
                with torch.autocast(
                    device_type=device_type, dtype=self.torch_dtype
                ):  # Enable mixed precision
                    frames_pred = self.model(in_frames)
                    if self.predict_diff:
                        frames_pred = torch.cumsum(frames_pred, dim=1)

                    frames_pred = self.remove_spatial_context(frames_pred)

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
                        raise ValueError(
                            f"Training loss {train_metric} not recognized."
                        )

                # backward
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient than zero_grad()

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

                    in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                    out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                    with torch.autocast(
                        device_type=device_type, dtype=self.torch_dtype
                    ):
                        frames_pred = self.model(in_frames)
                        if self.predict_diff:
                            frames_pred = torch.cumsum(frames_pred, dim=1)

                        frames_pred = self.remove_spatial_context(frames_pred)

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

            if verbose:
                self._logger.info(
                    f"Epoch({epoch + 1}/{n_epochs}) | "
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | "
                    f"Val_loss({val_loss_in_epoch:.4f}) | "
                    f"Quantile_loss({quantile_loss_in_epoch:.4f}) | "
                    f"CRPS({crps_in_epoch:.4f}) | "
                    f"Time_Epoch({(time.time() - start_epoch):.2f}s) |"
                )

            # epoch end
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            crps_per_epoch.append(crps_in_epoch)
            quantile_loss_per_epoch.append(quantile_loss_in_epoch)

            if val_loss_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best val loss: {val_loss_in_epoch:.4f}"
                )
                best_val_loss = val_loss_in_epoch
                best_epoch = epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "num_filters": self.filters,
                    "quantiles": self.quantiles,
                    "spatial_context": self.spatial_context,
                    "output_activation": self.output_activation,
                    "time_horizon": self.time_horizon,
                    "dataset": self.dataset_path,
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
                    self.save_checkpoint(
                        model_name, best_epoch, best_val_loss, checkpoint_path
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

    def load_checkpoint(
        self, checkpoint_path: str, device: str, eval_mode: bool = True
    ):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.quantiles = list(checkpoint["quantiles"])
        self.n_bins = len(self.quantiles)
        self.spatial_context = checkpoint["spatial_context"]
        self.time_horizon = checkpoint["time_horizon"]

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=self.n_bins,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()

    @property
    def name(self):
        return (
            f"QRUNet_IN{self.in_frames}_NB{self.n_bins}_F{self.filters}_"
            f"SC{self.spatial_context}_PD{self.predict_diff}"
        )


class MeanStdUNet(ProbabilisticUNet):
    def __init__(self, config):
        super().__init__(config)
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=2,
            filters=self.filters,
            output_activation=self.output_activation,
        )

        # self.loss_fn = mean_std_loss
        self.loss_fn = torch.nn.GaussianNLLLoss(eps=1e-6, reduction="mean")

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

        best_val_loss = float("inf")
        device_type = "cpu" if device == torch.device("cpu") else "cuda"
        self._logger.info(f"device type: {device_type}")

        scaler = torch.amp.GradScaler(device)  # For mixed precision training

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
            self.model.train()

            for batch_idx, (in_frames, out_frames) in enumerate(self.train_loader):

                start_batch = time.time()

                in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                # forward
                with torch.autocast(
                    device_type=device_type, dtype=self.torch_dtype
                ):  # Enable mixed precision
                    frames_pred = self.model(in_frames)
                    frames_pred = self.remove_spatial_context(frames_pred)
                    loss = self.calculate_loss(frames_pred, out_frames)

                # backward
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient than zero_grad()

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
            # crps_gaussian_list = []
            # numeric_crps = []

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):
                    in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                    out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                    with torch.autocast(
                        device_type=device_type, dtype=self.torch_dtype
                    ):
                        frames_pred = self.model(in_frames)

                        frames_pred = self.remove_spatial_context(frames_pred)

                        mean_std_loss_per_batch.append(
                            # self.calculate_loss(frames_pred, out_frames).detach().item()
                            self.calculate_loss(frames_pred, out_frames)
                        )

                        # calculate auxiliary metrics
                        mse_loss_mean_pred.append(
                            nn.MSELoss()(
                                frames_pred[:, 0, :, :], out_frames[:, 0, :, :]
                            )
                        )
                        # crps_gaussian_list.append(
                        #     crps_gaussian(
                        #         out_frames[:, 0, :, :],
                        #         frames_pred[:, 0, :, :],
                        #         frames_pred[:, 1, :, :],
                        #     )
                        # )

                    # numeric_crps.append(
                    #     self.get_numerical_CRPS(
                    #         y=out_frames, pred=frames_pred, lower=0., upper=1., count=100
                    #     )
                    # )

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            mean_std_loss_in_epoch = torch.mean(torch.tensor(mean_std_loss_per_batch))
            mse_mean_pred_in_epoch = torch.mean(torch.tensor(mse_loss_mean_pred))
            # mse_mean_pred_in_epoch = sum(mse_loss_mean_pred) / len(mse_loss_mean_pred)
            # crps_in_epoch = sum(crps_gaussian_list) / len(crps_gaussian_list)
            # numeric_crps_in_epoch = sum(numeric_crps) / len(numeric_crps)

            if val_metric is None or val_metric.lower() in ["mean_std", "meanstd"]:
                val_loss_in_epoch = mean_std_loss_in_epoch
            elif val_metric.lower() == "mse":
                val_loss_in_epoch = mse_mean_pred_in_epoch
            # elif val_metric.lower() == "crps":
            #     val_loss_in_epoch = crps_in_epoch
            else:
                raise ValueError(f"Validation loss {val_metric} not recognized.")

            if self.scheduler is not None:
                self.scheduler.step(val_loss_in_epoch)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"mean_std_loss": mean_std_loss_in_epoch}, step=epoch)
                run.log({"mse_mean_pred": mse_mean_pred_in_epoch}, step=epoch)
                # run.log({"crps_gaussian": crps_in_epoch}, step=epoch)
                # run.log({"crps": crps_in_epoch}, step=epoch)
                run.log(
                    {"lr": self.optimizer.state_dict()["param_groups"][0]["lr"]},
                    step=epoch,
                )

            if verbose:
                self._logger.info(
                    f"Epoch({epoch + 1}/{n_epochs}) | "
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | "
                    f"Val_loss({val_loss_in_epoch:.4f}) | "
                    f"mean_std_loss: {mean_std_loss_in_epoch:.4f}) | "
                    f"MSE({mse_mean_pred_in_epoch:.4f}) | "
                    # f"CRPS({crps_in_epoch:.4f} | "
                    f"Time_Epoch({(time.time() - start_epoch):.2f}s) |"
                )

            # epoch end
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            # crps_per_epoch.append(crps_in_epoch)
            mse_per_epoch.append(mse_mean_pred_in_epoch)
            mean_std_per_epoch.append(mean_std_loss_in_epoch)

            if val_loss_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best val loss: {val_loss_in_epoch:.4f}"
                )
                best_val_loss = val_loss_in_epoch
                best_epoch = epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "num_filters": self.filters,
                    "spatial_context": self.spatial_context,
                    "time_horizon": self.time_horizon,
                    "output_activation": self.output_activation,
                    "dataset": self.dataset_path,
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
                    self.save_checkpoint(
                        model_name,
                        best_epoch,
                        best_val_loss.detach().item(),
                        checkpoint_path,
                    )

        return train_loss_per_epoch, val_loss_per_epoch

    def calculate_loss(self, predictions: torch.Tensor, y_target: torch.Tensor):
        # Extract mean and variance from predictions
        mu = predictions[:, 0:1, :, :]  # Keep dimension: [batch_size, 1, H, W]
        # Convert to variance and ensure positive
        var = (
            F.softplus(predictions[:, 1:, :, :]) + 1e-6
        )  # Keep dimension: [batch_size, 1, H, W]
        # targets is already [batch_size, 1, H, W] so no need to modify
        return self.loss_fn(mu, y_target, var)

    def load_checkpoint(
        self, checkpoint_path: str, device: str, eval_mode: bool = True
    ):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.spatial_context = checkpoint["spatial_context"]
        self.time_horizon = checkpoint["time_horizon"]

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=2,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()

    def get_F_at_points(self, points: torch.Tensor, pred_params: torch.Tensor):
        mus, sig = pred_params[:, 0:1, :, :], pred_params[:, 1:, :, :]
        sx = (points - mus) / sig
        cdf = stats.norm.cdf(sx)
        cdf = torch.tensor(cdf).to(device=self.device)

        return cdf

    @property
    def name(self):
        return (
            f"MeanStdUNet_IN{self.in_frames}_F{self.filters}_SC{self.spatial_context}"
        )


class MedianScaleUNet(ProbabilisticUNet):
    def __init__(
        self,
        config: UNetConfig,
    ):
        super().__init__(config)
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

        best_val_loss = float("inf")
        device_type = "cpu" if device == torch.device("cpu") else "cuda"
        self._logger.info(f"device type: {device_type}")

        scaler = torch.amp.GradScaler(device)  # For mixed precision training

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
            self.model.train()

            for batch_idx, (in_frames, out_frames) in enumerate(self.train_loader):

                start_batch = time.time()

                # data to cuda if possible
                in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                # forward
                with torch.autocast(
                    device_type=device_type, dtype=self.torch_dtype
                ):  # Enable mixed precision
                    frames_pred = self.model(in_frames)
                    frames_pred = self.remove_spatial_context(frames_pred)
                    loss = self.calculate_loss(frames_pred, out_frames)

                # backward
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient than zero_grad()

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
            crps_laplace_list = []
            # numeric_crps = []

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                    out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                    with torch.autocast(
                        device_type=device_type, dtype=self.torch_dtype
                    ):
                        frames_pred = self.model(in_frames)
                        frames_pred = self.remove_spatial_context(frames_pred)

                        median_scale_loss_per_batch.append(
                            self.calculate_loss(frames_pred, out_frames).detach().item()
                        )

                        # calculate auxiliary metrics
                        mae_loss_mean_pred.append(
                            nn.L1Loss()(frames_pred[:, 0, :, :], out_frames[:, 0, :, :])
                        )
                        crps_laplace_list.append(
                            crps_laplace(
                                out_frames,
                                frames_pred,
                            )
                        )

                        # numeric_crps.append(
                        #     self.get_numerical_CRPS(
                        #         y=out_frames, pred=frames_pred, lower=0., upper=1., count=100
                        #     )
                        # )

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            median_scale_loss_in_epoch = sum(median_scale_loss_per_batch) / len(
                median_scale_loss_per_batch
            )
            mae_mean_pred_in_epoch = sum(mae_loss_mean_pred) / len(mae_loss_mean_pred)
            crps_in_epoch = sum(crps_laplace_list) / len(crps_laplace_list)
            # numeric_crps_in_epoch = sum(numeric_crps) / len(numeric_crps)

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

            if verbose:
                self._logger.info(
                    f"Epoch({epoch + 1}/{n_epochs}) | "
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | "
                    f"Val_loss({val_loss_in_epoch:.4f}) | "
                    f"MedianScale({median_scale_loss_in_epoch:.4f}) | "
                    f"MAE({mae_mean_pred_in_epoch:.4f}) | "
                    f"CRPS({crps_in_epoch:.4f} | "
                    f"Time_Epoch({(time.time() - start_epoch):.2f}s) |"
                )

            # epoch end
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
                best_epoch = epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "num_filters": self.filters,
                    "spatial_context": self.spatial_context,
                    "time_horizon": self.time_horizon,
                    "output_activation": self.output_activation,
                    "dataset": self.dataset_path,
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
                    self.save_checkpoint(
                        model_name, best_epoch, best_val_loss, checkpoint_path
                    )

        return train_loss_per_epoch, val_loss_per_epoch

    def load_checkpoint(
        self, checkpoint_path: str, device: str, eval_mode: bool = True
    ):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.spatial_context = checkpoint["spatial_context"]
        self.time_horizon = checkpoint["time_horizon"]

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=2,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()

    def get_F_at_points(self, points: torch.Tensor, pred_params: torch.Tensor):
        mus, bs = pred_params[:, 0:1, :, :], pred_params[:, 1:, :, :]
        return 0.5 + 0.5 * (2 * (mus < points) - 1) * (
            1 - torch.exp(-torch.abs(mus - points) / bs)
        )

    @property
    def name(self):
        return (
            f"MedianScaleUNet_IN{self.in_frames}_F{self.filters}"
            f"_SC{self.spatial_context}"
        )


class MixtureDensityUNet(ProbabilisticUNet):
    def __init__(
        self,
        config: UNetConfig,
        n_components: int = 5,
    ):
        super().__init__(config)
        self.n_components = n_components
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=3 * self.n_components,
            filters=self.filters,
            output_activation=self.output_activation,
        )
        self.loss_fn = MixtureDensityLoss(n_components=n_components)

    def get_F_at_points(self, points, pred_params):
        pis, mus, sigmas = (
            pred_params[:, : self.n_components, :, :],
            pred_params[:, self.n_components : 2 * self.n_components, :, :],
            pred_params[:, 2 * self.n_components :, :, :],
        )

        F = torch.zeros_like(points)
        for i in range(points.shape[1]):
            # F[:, i] = torch.sum(
            F[:, i] = torch.mean(
                pis
                * (0.5 * (1 + torch.erf((points[:, i] - mus) / (sigmas * np.sqrt(2)))))
            )
        return F

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

        best_val_loss = float("inf")

        device_type = "cpu" if device == torch.device("cpu") else "cuda"
        self._logger.info(f"device type: {device_type}")

        scaler = torch.amp.GradScaler(device)  # For mixed precision training

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []
            self.model.train()

            for batch_idx, (in_frames, out_frames) in enumerate(self.train_loader):
                start_batch = time.time()

                # data to cuda if possible
                in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                # forward
                with torch.autocast(
                    device_type=device_type, dtype=self.torch_dtype
                ):  # Enable mixed precision
                    frames_pred = self.model(in_frames)
                    frames_pred = self.remove_spatial_context(frames_pred)
                    frames_pred = self.mdn_forward(frames_pred)
                    loss = self.calculate_loss(frames_pred, out_frames)

                # backward
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient than zero_grad()

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
                    in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                    out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                    with torch.autocast(
                        device_type=device_type, dtype=self.torch_dtype
                    ):
                        frames_pred = self.model(in_frames)
                        frames_pred = self.remove_spatial_context(frames_pred)
                        frames_pred = self.mdn_forward(frames_pred)
                        val_loss_per_batch.append(
                            self.calculate_loss(frames_pred, out_frames).detach().item()
                        )

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

            if verbose:
                self._logger.info(
                    f"Epoch({epoch + 1}/{n_epochs}) | "
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | "
                    f"Val_loss({val_loss_in_epoch:.4f}) | "
                    f"Time_Epoch({(time.time() - start_epoch):.2f}s) |"
                )

            # epoch end
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)

            if val_loss_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best val loss: {val_loss_in_epoch:.4f}"
                )
                best_val_loss = val_loss_in_epoch
                best_epoch = epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "num_filters": self.filters,
                    "n_components": self.n_components,
                    "spatial_context": self.spatial_context,
                    "time_horizon": self.time_horizon,
                    "output_activation": self.output_activation,
                    "dataset": self.dataset_path,
                    "epoch": epoch + 1,
                    "ts": datetime.datetime.now().strftime("%d-%m-%Y_%H:%M"),
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
                    "train_loss_per_epoch": train_loss_per_epoch,
                    "val_loss_per_epoch": val_loss_per_epoch,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                    "val_metric": val_metric,
                    "train_metric": train_metric,
                }

                if checkpoint_path is not None:
                    self.save_checkpoint(
                        model_name, best_epoch, best_val_loss, checkpoint_path
                    )

        return train_loss_per_epoch, val_loss_per_epoch

    def mdn_forward(self, frames_pred: torch.Tensor) -> torch.Tensor:
        pis, mus, sigmas = (
            frames_pred[:, : self.n_components, :, :],
            frames_pred[:, self.n_components : 2 * self.n_components, :, :],
            frames_pred[:, 2 * self.n_components :, :, :],
        )
        pis = nn.functional.softmax(pis, dim=1)  # pis must be positive and sum to 1
        sigmas = nn.functional.softplus(sigmas)  # w_b must be positive
        return torch.concatenate([pis, mus, sigmas], dim=1)

    def load_checkpoint(
        self, checkpoint_path: str, device: str, eval_mode: bool = True
    ):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.n_components = checkpoint["n_components"]
        self.spatial_context = checkpoint["spatial_context"]
        self.time_horizon = checkpoint["time_horizon"]

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=3 * self.n_components,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()

    @property
    def name(self):
        return (
            f"MixDensityUNet_IN{self.in_frames}_F{self.filters}"
            f"_NC{self.n_components}_SC{self.spatial_context}"
        )


class MonteCarloDropoutUNet(ProbabilisticUNet):
    def __init__(
        self,
        config: UNetConfig,
        n_quantiles: int = 5,
        dropout_p: float = 0.5,
    ):
        super().__init__(config)
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

    def get_F_at_points(self, points, pred_params):
        # TODO: implement if want to use NUMERICAL CRPS
        pass

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

        best_val_loss = float("inf")
        device_type = "cpu" if device == torch.device("cpu") else "cuda"
        self._logger.info(f"device type: {device_type}")

        scaler = torch.amp.GradScaler(device)  # For mixed precision training

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
            self.model.train()

            for batch_idx, (in_frames, out_frames) in enumerate(self.train_loader):

                start_batch = time.time()

                # data to cuda if possible
                in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                # forward
                with torch.autocast(
                    device_type=device_type, dtype=self.torch_dtype
                ):  # Enable mixed precision
                    frames_pred = self.model(in_frames)
                    frames_pred = self.remove_spatial_context(frames_pred)
                    loss = self.calculate_loss(frames_pred, out_frames)

                # backward
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient than zero_grad()

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

                    in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                    out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                    with torch.autocast(
                        device_type=device_type, dtype=self.torch_dtype
                    ):

                        frames_pred = self.predict(
                            in_frames, iterations=self.n_quantiles
                        )

                        frames_pred = self.remove_spatial_context(frames_pred)

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

            if verbose:
                self._logger.info(
                    f"Epoch({epoch + 1}/{n_epochs}) | "
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | "
                    f"Val_loss({val_loss_in_epoch:.4f}) | "
                    f"MAE({mae_loss_in_epoch:.4f}) | "
                    f"CRPS({crps_in_epoch:.4f}) | "
                    f"CRPS_Torch({crps_quantile_in_epoch:.4f}) | "
                    f"CRPS_Bin({crps_bin_in_epoch:.4f}) | "
                    f"Time_Epoch({(time.time() - start_epoch):.2f}s) |"
                )

            # epoch end
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
                    "spatial_context": self.spatial_context,
                    "output_activation": self.output_activation,
                    "time_horizon": self.time_horizon,
                    "dataset": self.dataset_path,
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
                    self.save_checkpoint(
                        model_name, best_epoch, best_val_loss, checkpoint_path
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
        predictions = self.model(X)

        for _ in range(iterations - 1):
            predictions = torch.cat((predictions, self.model(X)), dim=1)

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

    def load_checkpoint(
        self, checkpoint_path: str, device: str, eval_mode: bool = True
    ):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.dropout_p = checkpoint["dropout_p"]
        self.n_quantiles = checkpoint["n_quantiles"]
        self.quantiles = checkpoint["quantiles"]
        self.spatial_context = checkpoint["spatial_context"]
        self.time_horizon = checkpoint["time_horizon"]

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=1,
            dropout_p=self.dropout_p,
            filters=self.filters,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()

    @property
    def name(self):
        return (
            f"MCD_IN{self.in_frames}_F{self.filters}"
            f"_DP{str(self.dropout_p).replace('0.', '')}"
            f"_NQ{self.n_quantiles}_SC{self.spatial_context}"
        )
