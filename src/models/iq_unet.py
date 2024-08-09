# Standard library imports
import os
import time
import copy
import datetime
from typing import List, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from metrics import QuantileLoss

from .model_initialization import weights_init, optimizer_init, scheduler_init
import logging

from data_handlers import MovingMnistDataset, SatelliteDataset, normalize_pixels

# Configure logging
logging.basicConfig(level=logging.INFO)


class QuantileEmbedding(nn.Module):
    def __init__(self, cosine_embedding_dimension, feature_dimension):
        super().__init__()
        self.cosine_embedding_dimension = cosine_embedding_dimension
        self.feature_dimension = feature_dimension
        self.embedding = nn.Linear(cosine_embedding_dimension, feature_dimension)

    def forward(self, tau):
        x = tau.unsqueeze(-1)  # Add feature dimension
        x = torch.cos(
            torch.pi * torch.arange(self.cosine_embedding_dimension)[None] * x
        )
        x = self.embedding(x)
        x = F.relu(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels  # In Down mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, bias=bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, bias=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, bias=bias
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, bias=bias)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class IQUNet(nn.Module):
    def __init__(
        self,
        in_frames: int = 3,
        n_classes: int = 1,
        bilinear: bool = True,
        filters: int = 64,
        bias: bool = False,
        cosine_embedding_dimension: int = 64,
        # embedding_dim: int = 32,
    ):
        super().__init__()
        self.description = f"IQN_UNet_IN{in_frames}_OUT{n_classes}"
        self.n_channels = in_frames
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.cosine_embedding_dimension = cosine_embedding_dimension

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_frames, filters)
        self.down1 = Down(filters, 2 * filters, bias=bias)
        self.down2 = Down(2 * filters, 4 * filters, bias=bias)
        self.down3 = Down(4 * filters, 8 * filters, bias=bias)
        self.down4 = Down(8 * filters, 16 * filters // factor, bias=bias)

        self.up1 = Up(16 * filters, 8 * filters // factor, bilinear, bias=bias)
        self.up2 = Up(8 * filters, 4 * filters // factor, bilinear, bias=bias)
        self.up3 = Up(4 * filters, 2 * filters // factor, bilinear, bias=bias)
        self.up4 = Up(2 * filters, filters, bilinear, bias=bias)
        self.outc = OutConv(filters, n_classes)

        # the embedding dimension is the number of filters times 8, and
        # multiplied by 4*4 because of image size downscaling
        embedding_dim = filters * 8 * 4 * 4
        self.quantile_embedding = QuantileEmbedding(
            cosine_embedding_dimension, embedding_dim
        )

    def forward(self, x, tau):
        tau_embedding = self.quantile_embedding(tau)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Integrate tau embedding
        
        x5_flat = torch.flatten(x5, start_dim=1)
        x5_flat = x5_flat * tau_embedding

        x5 = x5_flat.reshape(x5.shape)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

    def quantile_huber_loss(self, predicted, target, tau, kappa=1.0):
        diff = target - predicted
        loss = torch.where(
            torch.abs(diff) <= kappa,
            0.5 * diff.pow(2),
            kappa * (torch.abs(diff) - 0.5 * kappa),
        )
        weight = torch.abs(tau - (diff < 0).float())
        return (weight * loss).mean()


class IQUNetPipeline:
    def __init__(
        self,
        in_frames: int = 3,
        n_classes: int = 1,
        filters: int = 16,
        cosine_embedding_dimension: int = 64,
        device: str = "cpu",
    ):
        self.in_frames = in_frames
        self.n_classes = n_classes
        self.filters = filters
        self.cosine_embedding_dimension = cosine_embedding_dimension
        self.device = device

        self.model = IQUNet(
            in_frames=in_frames,
            n_classes=n_classes,
            filters=filters,
            cosine_embedding_dimension=cosine_embedding_dimension,
        ).to(device)

        self.quantiles = [0.5]
        self.loss_fn = QuantileLoss(quantiles=self.quantiles)

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
                taus = torch.rand(1).to(device=device)
                frames_pred = self.model(in_frames.float(), taus)

                loss = QuantileLoss(quantiles=taus)(frames_pred, out_frames)

                # loss = self.calculate_loss(frames_pred, out_frames)

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
            quantile_loss_per_batch = []  # stores values for this validation run

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device).float()
                    out_frames = out_frames.to(device=device)

                    taus = torch.rand(1).to(device=device)
                    frames_pred = self.model(in_frames.float(), taus)

                    quantile_loss = QuantileLoss(quantiles=taus)(frames_pred, out_frames)

                    quantile_loss_per_batch.append(quantile_loss.detach().item())

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            quantile_loss_in_epoch = sum(quantile_loss_per_batch) / len(
                quantile_loss_per_batch
            )

            val_loss_in_epoch = quantile_loss_in_epoch

            if self.scheduler is not None:
                self.scheduler.step(val_loss_in_epoch)
                
            if run is not None:
                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"quantile_loss": quantile_loss_in_epoch}, step=epoch)
                run.log(
                    {"lr": self.optimizer.state_dict()["param_groups"][0]["lr"]},
                    step=epoch,
                )

            end_epoch = time.time()
            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            quantile_loss_per_epoch.append(quantile_loss_in_epoch)

            if verbose:
                self._logger.info(
                    f"Epoch({epoch + 1}/{n_epochs}) | "
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | "
                    f"Val_loss({val_loss_in_epoch:.4f}) | "
                    f"Quantile_loss({quantile_loss_in_epoch:.4f}) | "
                    f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |"
                )

            # epoch end
            end_epoch = time.time()
            
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
                    "quantile_loss_per_epoch": quantile_loss_per_epoch,
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

    def calculate_loss(self, predictions, y_target):
        y_target = y_target.repeat(1, predictions.shape[1], 1, 1)
        return self.loss_fn(predictions, y_target)

    @property
    def name(self):
        return f"IQUNet_IN{self.in_frames}_F{self.filters}"
