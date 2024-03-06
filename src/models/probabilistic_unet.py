import torch
from .unet import UNet
from abc import ABC, abstractmethod
import torch.nn as nn
from metrics import QuantileLoss, mean_std_loss
from typing import List, Optional
from data_handlers import MovingMnistDataset
from torch.utils.data import DataLoader
from .weight_initialization import weights_init
import os


__all__ = [
    "QuantileRegressorUNet",
    "BinClassifierUNet",
    "MonteCarloDropoutUNet",
    "MeanStdUNet",
]


class ProbabilisticUNet(ABC):
    def fit(self, X, y, n_epochs=1):
        pass

    def compute_extra_params(self, X, y):
        return None

    def predict(self, X, iterations: int):
        pass

    @abstractmethod
    def initialize_weights(self):
        pass

    @abstractmethod
    def create_dataloaders(self, path: str, batch_size: int):
        pass

    @abstractmethod
    def calculate_loss(self, predictions, y_target):
        pass

    @abstractmethod
    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        pass

    @property
    @abstractmethod
    def name(self):
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

    def initialize_weights(self):
        self.model.apply(weights_init)

    def create_dataloaders(self, path: str, batch_size: int):
        train_dataset = MovingMnistDataset(
            path=os.path.join(path, "train/"),
            input_frames=self.in_frames,
            num_bins=self.n_bins,
            shuffle=False,
        )
        val_dataset = MovingMnistDataset(
            path=os.path.join(path, "validation/"),
            input_frames=self.in_frames,
            num_bins=self.n_bins,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    def fit(self, X, y, n_epochs=1):
        pass

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

    def initialize_weights(self):
        self.model.apply(weights_init)

    def create_dataloaders(self, path: str, batch_size: int):
        train_dataset = MovingMnistDataset(
            path=os.path.join(path, "train/"),
            input_frames=self.in_frames,
            num_bins=None,
            shuffle=False,
        )
        val_dataset = MovingMnistDataset(
            path=os.path.join(path, "validation/"),
            input_frames=self.in_frames,
            num_bins=None,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    def fit(self, X, y, n_epochs=1):
        pass

    def predict(self, X, iterations: int):
        return self.model(X.float())

    def calculate_loss(self, predictions, y_target):
        y_target = y_target.repeat(1, predictions.shape[1], 1, 1)
        return self.loss_fn(predictions, y_target)

    def cdf(self, predicted_params, extra_params, points_to_evaluate):
        pass

    @property
    def name(self):
        return f"QuantileRegressorUNet_{self.n_bins}bins_{self.in_frames}frames_{self.n_classes}classes_{self.filters}filters"


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

        self.loss_fn = mean_std_loss
        self.train_loader = None
        self.val_loader = None

    def initialize_weights(self):
        self.model.apply(weights_init)

    def create_dataloaders(self, path: str, batch_size: int):
        train_dataset = MovingMnistDataset(
            path=os.path.join(path, "train/"),
            input_frames=self.in_frames,
            num_bins=None,
            shuffle=False,
        )
        val_dataset = MovingMnistDataset(
            path=os.path.join(path, "validation/"),
            input_frames=self.in_frames,
            num_bins=None,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    def fit(self, X, y, n_epochs=1):
        pass

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

    def initialize_weights(self):
        self.model.apply(weights_init)

    def create_dataloaders(self, path: str, batch_size: int):
        train_dataset = MovingMnistDataset(
            path=os.path.join(path, "train/"),
            input_frames=self.in_frames,
            num_bins=None,
            shuffle=False,
        )
        val_dataset = MovingMnistDataset(
            path=os.path.join(path, "validation/"),
            input_frames=self.in_frames,
            num_bins=None,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    def fit(self, X, y, n_epochs=1):
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
