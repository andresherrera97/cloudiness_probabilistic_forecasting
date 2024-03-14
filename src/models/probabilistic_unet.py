# Standard library imports
import os
import time
import copy
from typing import List, Optional
from abc import ABC, abstractmethod

# Related third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision
import wandb

# Local application/library specific imports
from metrics import QuantileLoss, mean_std_loss, crps_gaussian, crps_bin_classification
from data_handlers import MovingMnistDataset
from .unet import UNet
from .model_initialization import weights_init, optimizer_init


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
    def create_dataloaders(
        self, path: str, batch_size: int, binarization_method: Optional[str]
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

    @property
    @abstractmethod
    def name(self):
        """Abstract property to get a unique identifier or name for the model."""
        pass


class BinClassifierUNet(ProbabilisticUNet):
    def __init__(self, n_bins=10, in_frames=3, filters=16):
        self.n_bins = n_bins
        self.in_frames = in_frames
        self.filters = filters
        self.model = UNet(
            in_frames=self.in_frames, n_classes=self.n_bins, filters=self.filters
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.multiclass_precision_metric = MulticlassPrecision(
            num_classes=n_bins, average="macro", top_k=1, multidim_average="global"
        )

    def initialize_weights(self):
        self.model.apply(weights_init)

    def initialize_optimizer(self, method: str, lr: float):
        self.optimizer = optimizer_init(self.model, method, lr)

    def create_dataloaders(self, path: str, batch_size: int, binarization_method: str):
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
    ):
        TRAIN_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch
        VAL_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch

        BEST_VAL_ACC = 1e5

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
                    print(
                        f"BATCH({batch_idx + 1}/{len(self.train_loader)}) | ",
                        end="",
                    )
                    print(f"Train loss({loss.detach().item():.4f}) | ", end="")
                    print(f"Time Batch({(end_batch - start_batch):.2f}) | ")

                if num_train_samples is not None and batch_idx >= num_train_samples:
                    break

            train_loss_in_epoch = sum(train_loss_in_epoch_list) / len(
                train_loss_in_epoch_list
            )

            self.model.eval()
            VAL_LOSS_LOCAL = []  # stores values for this validation run
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

                    VAL_LOSS_LOCAL.append(val_loss.detach().item())

                    # calculate auxiliary metrics
                    crps_bin_list.append(
                        crps_bin_classification(frames_pred, out_frames.unsqueeze(1))
                    )
                    precision_list.append(
                        self.multiclass_precision_metric(frames_pred, out_frames)
                    )

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            val_loss_in_epoch = sum(VAL_LOSS_LOCAL) / len(VAL_LOSS_LOCAL)
            crps_in_epoch = sum(crps_bin_list) / len(crps_bin_list)
            precision_in_epoch = sum(precision_list) / len(precision_list)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"crps_bin": crps_in_epoch}, step=epoch)
                run.log({"precision": precision_in_epoch}, step=epoch)

            end_epoch = time.time()

            if verbose:
                print(f"Epoch({epoch + 1}/{n_epochs}) | ", end="")
                print(
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | Val_loss({val_loss_in_epoch:.4f}) | CRPS({crps_in_epoch:.4f} | precision({precision_in_epoch:.4f} | ",
                    end="",
                )
                print(f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |")

            # epoch end
            end_epoch = time.time()
            TRAIN_LOSS_GLOBAL.append(train_loss_in_epoch)
            VAL_LOSS_GLOBAL.append(val_loss_in_epoch)

            if val_loss_in_epoch < BEST_VAL_ACC:
                print(f"Saving best model. Best val loss: {val_loss_in_epoch:.4f}")
                BEST_VAL_ACC = val_loss_in_epoch
                self.best_model_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
                    "train_loss_per_batch": train_loss_in_epoch_list,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                }

        return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL

    def predict(self, X, iterations: int):
        return self.model(X.float())

    def calculate_loss(self, predictions, y_target):
        return self.loss_fn(predictions, y_target)

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        pass

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

    def initialize_weights(self):
        self.model.apply(weights_init)

    def initialize_optimizer(self, method: str, lr: float):
        self.optimizer = optimizer_init(self.model, method, lr)

    def create_dataloaders(self, path: str, batch_size: int, binarization_method=None):
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
    ):
        TRAIN_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch
        VAL_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch

        BEST_VAL_ACC = 1e5

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
                    print(
                        f"BATCH({batch_idx + 1}/{len(self.train_loader)}) | ",
                        end="",
                    )
                    print(f"Train loss({loss.detach().item():.4f}) | ", end="")
                    print(f"Time Batch({(end_batch - start_batch):.2f}) | ")

                if num_train_samples is not None and batch_idx >= num_train_samples:
                    break

            train_loss_in_epoch = sum(train_loss_in_epoch_list) / len(
                train_loss_in_epoch_list
            )

            self.model.eval()
            VAL_LOSS_LOCAL = []  # stores values for this validation run
            # crps_bin_list = []
            # precision_list = []

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device)

                    frames_pred = self.model(in_frames.float())

                    val_loss = self.calculate_loss(frames_pred, out_frames)

                    VAL_LOSS_LOCAL.append(val_loss.detach().item())

                    # calculate auxiliary metrics
                    # crps_bin_list.append(
                    #     crps_bin_classification(frames_pred, out_frames.unsqueeze(1))
                    # )

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            val_loss_in_epoch = sum(VAL_LOSS_LOCAL) / len(VAL_LOSS_LOCAL)
            # crps_in_epoch = sum(crps_bin_list) / len(crps_bin_list)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                # run.log({"crps_bin": crps_in_epoch}, step=epoch)

            end_epoch = time.time()

            if verbose:
                print(f"Epoch({epoch + 1}/{n_epochs}) | ", end="")
                print(
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | Val_loss({val_loss_in_epoch:.4f}) | ",
                    end="",
                )
                print(f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |")

            # epoch end
            end_epoch = time.time()
            TRAIN_LOSS_GLOBAL.append(train_loss_in_epoch)
            VAL_LOSS_GLOBAL.append(val_loss_in_epoch)

            if val_loss_in_epoch < BEST_VAL_ACC:
                print(f"Saving best model. Best val loss: {val_loss_in_epoch:.4f}")
                BEST_VAL_ACC = val_loss_in_epoch
                self.best_model_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
                    "train_loss_per_batch": train_loss_in_epoch_list,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                }

        return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL

    def predict(self, X, iterations: int):
        return self.model(X.float())

    def calculate_loss(self, predictions, y_target):
        y_target = y_target.repeat(1, predictions.shape[1], 1, 1)
        return self.loss_fn(predictions, y_target)

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        pass

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

    def initialize_weights(self):
        self.model.apply(weights_init)

    def initialize_optimizer(self, method: str, lr: float):
        self.optimizer = optimizer_init(self.model, method, lr)

    def create_dataloaders(self, path: str, batch_size: int, binarization_method=None):
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
    ):
        TRAIN_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch
        VAL_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch

        BEST_VAL_ACC = 1e5

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
                    print(
                        f"BATCH({batch_idx + 1}/{len(self.train_loader)}) | ",
                        end="",
                    )
                    print(f"Train loss({loss.detach().item():.4f}) | ", end="")
                    print(f"Time Batch({(end_batch - start_batch):.2f}) | ")

                if num_train_samples is not None and batch_idx >= num_train_samples:
                    break

            train_loss_in_epoch = sum(train_loss_in_epoch_list) / len(
                train_loss_in_epoch_list
            )

            self.model.eval()
            VAL_LOSS_LOCAL = []  # stores values for this validation run
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

                    VAL_LOSS_LOCAL.append(val_loss.detach().item())

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
            # print(f"val_crps: {np.mean(val_crps_local)}")
            val_loss_in_epoch = sum(VAL_LOSS_LOCAL) / len(VAL_LOSS_LOCAL)
            mae_loss_mean_pred_in_epoch = sum(mae_loss_mean_pred) / len(
                mae_loss_mean_pred
            )
            crps_in_epoch = sum(crps_gaussian_list) / len(crps_gaussian_list)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"mae_loss_mean_pred": mae_loss_mean_pred_in_epoch}, step=epoch)
                run.log({"crps_gaussian": crps_in_epoch}, step=epoch)

                wandb_target_img = wandb.Image(
                    out_frames[0, 0, :, :].unsqueeze(-1).cpu().numpy()
                )
                wandb_mean_img = wandb.Image(
                    frames_pred[0, 0, :, :].unsqueeze(-1).cpu().numpy(),
                    caption=f"min: {frames_pred[0, 0, :, :].min():.2f}, max: {frames_pred[0, 0, :, :].max():.2f}",
                )
                wandb_std_img = wandb.Image(
                    frames_pred[0, 1, :, :].unsqueeze(-1).cpu().numpy(),
                    caption=f"min: {frames_pred[0, 0, :, :].min():.2f}, max: {frames_pred[0, 0, :, :].max():.2f}, mean: {frames_pred[0, 0, :, :].mean():.2f}",
                )

                my_table = wandb.Table(
                    columns=["target", "mean", "std"],
                    data=[[wandb_target_img, wandb_mean_img, wandb_std_img]],
                )

                # Log your Table to W&B
                run.log({f"epoch_{epoch}": my_table}, step=epoch)

            end_epoch = time.time()

            if verbose:
                print(f"Epoch({epoch + 1}/{n_epochs}) | ", end="")
                print(
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | Val_loss({val_loss_in_epoch:.4f}) | MAE({mae_loss_mean_pred_in_epoch:.4f}) | CRPS({crps_in_epoch:.4f} | ",
                    end="",
                )
                print(f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |")

            # epoch end
            end_epoch = time.time()
            TRAIN_LOSS_GLOBAL.append(train_loss_in_epoch)
            VAL_LOSS_GLOBAL.append(val_loss_in_epoch)

            if val_loss_in_epoch < BEST_VAL_ACC:
                BEST_VAL_ACC = val_loss_in_epoch
                self.best_model_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
                    "train_loss_per_batch": train_loss_in_epoch_list,
                    "train_loss_epoch_mean": train_loss_in_epoch,
                }

        return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL

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

    @property
    def name(self):
        return f"MeanStdUNet_{self.in_frames}frames_{self.filters}filters"


class MonteCarloDropoutUNet(ProbabilisticUNet):
    def __init__(
        self,
        in_frames: int = 3,
        filters: int = 16,
        dropout_p: float = 0.5,
    ):
        self.in_frames = in_frames
        self.filters = filters
        self.dropout_p = dropout_p
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=1,
            dropout_p=dropout_p,
            filters=self.filters,
        )

        self.loss_fn = nn.L1Loss()
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None

    def initialize_weights(self):
        self.model.apply(weights_init)

    def initialize_optimizer(self, method: str, lr: float):
        self.optimizer = optimizer_init(self.model, method, lr)

    def create_dataloaders(self, path: str, batch_size: int, binarization_method=None):
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
    ):
        pass

    def predict(self, X, iterations: int):
        # as images get bigger the computational cost can be too high, find better way to do this
        predictions = self.model(X.float())

        for _ in range(iterations - 1):
            predictions = torch.cat((predictions, self.model(X.float())), dim=1)

        return (
            torch.std_mean(predictions, dim=1, keepdim=True),
            torch.min(predictions, dim=1, keepdim=True),
            torch.max(predictions, dim=1, keepdim=True),
        )

    def calculate_loss(self, predictions, y_target):
        return self.loss_fn(predictions, y_target)

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        mu, sigma2 = predicted_params[:, 0, :, :], nn.functional.softplus(
            predicted_params[:, 1, :, :]
        )
        dist = torch.distributions.Normal(mu, torch.sqrt(sigma2))
        return dist.cdf(points_to_evaluate)

    @property
    def name(self):
        return f"MCDropoutUNet_{self.in_frames}frames_{self.filters}filters_{self.dropout_p}dropout_p"
