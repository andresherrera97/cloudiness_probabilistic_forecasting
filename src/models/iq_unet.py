# Standard library imports
import time
import copy
import datetime
from typing import List, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import QuantileLoss
from .probabilistic_unet import ProbabilisticUNet, UNetConfig
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)


class QuantileEmbedding(nn.Module):
    def __init__(
        self,
        cosine_embedding_dimension,
        feature_dimension,
        device,
        sort_taus: bool = True,
    ):
        super().__init__()
        self.device = device
        self.cosine_embedding_dimension = cosine_embedding_dimension
        self.feature_dimension = feature_dimension
        self.embedding = nn.Linear(cosine_embedding_dimension, feature_dimension)
        self.pis = (
            torch.pi
            * torch.arange(self.cosine_embedding_dimension, device=self.device)[None]
        )
        self.sort_taus = sort_taus

    def forward(self, taus):
        if self.sort_taus:
            taus, _ = torch.sort(taus)
        x = taus.unsqueeze(-1)  # Add feature dimension
        x = torch.cos(self.pis * x)
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


class ReductionConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReductionConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class IQUNet(nn.Module):
    def __init__(
        self,
        in_frames: int = 3,
        n_classes: int = 1,
        output_activation: Optional[str] = "sigmoid",
        bilinear: bool = True,
        filters: int = 64,
        bias: bool = False,
        cosine_embedding_dimension: int = 64,
        num_taus: int = 10,
        image_size: int = 1024,
        device: str = "cpu",
    ):
        super().__init__()
        self.description = f"IQN_UNet_IN{in_frames}_OUT{n_classes}_Q{num_taus}_CED{cosine_embedding_dimension}_NT{num_taus}"
        self.n_channels = in_frames
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.cosine_embedding_dimension = cosine_embedding_dimension
        self.num_taus = num_taus
        self.image_size = image_size

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
        embedding_dim = filters * 8 * (image_size//16) * (image_size//16)
        embedding_dim = int(embedding_dim)
        self.quantile_embedding = QuantileEmbedding(
            cosine_embedding_dimension, embedding_dim, device, sort_taus=True
        ).to(device)

        self.dim_reduction_conv = ReductionConv(filters * 8 * num_taus, filters * 8)
        output_activation = output_activation.lower()
        if output_activation is None or output_activation in ["none", ""]:
            self.out_activation = nn.Identity()
        elif output_activation in ["sigmoid", "sigmoide", "sig"]:
            self.out_activation = nn.Sigmoid()
        elif output_activation in ["relu"]:
            self.out_activation = nn.Hardtanh(
                min_val=0, max_val=1.0
            )  # works as relu clip between [0,1]
        elif output_activation in ["tanh"]:
            self.out_activation = nn.Tanh()
        elif output_activation in ["softmax"]:
            self.out_activation = nn.Softmax(dim=1)
        elif output_activation in ["softplus"]:
            self.out_activation = nn.Softplus()
        else:
            raise ValueError(f"Activation function {output_activation} not recognized")

    def forward(self, x, taus):
        taus_embedding = self.quantile_embedding(taus).unsqueeze(0)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Integrate tau embedding
        x5_flat = torch.flatten(x5, start_dim=1).unsqueeze(1)
        x5_flat = x5_flat * taus_embedding
        x5 = x5_flat.reshape(
            (
                x5.shape[0],
                x5.shape[1] * taus_embedding.shape[1],
                x5.shape[2],
                x5.shape[3],
            )
        )
        x5 = self.dim_reduction_conv(x5)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = self.out_activation(out)
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


class IQUNetPipeline(ProbabilisticUNet):
    def __init__(
        self,
        config: UNetConfig,
        cosine_embedding_dimension: int = 64,
        num_taus: int = 9,
        predict_diff: bool = False,
        min_value: float = 0,
        max_value: float = 1,
        image_size: int = 512,
    ):
        super().__init__(config)
        self.cosine_embedding_dimension = cosine_embedding_dimension
        self.predict_diff = predict_diff
        self.num_taus = num_taus
        self.min_value = min_value
        self.max_value = max_value
        self.val_quantiles = torch.linspace(min_value, max_value, self.num_taus + 2)[
            1:-1
        ].to(device=self.device)
        self.image_size = image_size

        self.model = IQUNet(
            in_frames=self.in_frames,
            n_classes=num_taus,
            output_activation=self.output_activation,
            filters=self.filters,
            cosine_embedding_dimension=cosine_embedding_dimension,
            num_taus=num_taus,
            image_size=self.image_size,
            device=self.device,
        ).to(self.device)

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

        best_val_loss = float('inf')
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
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient than zero_grad()

                # forward
                with torch.autocast(device_type=device_type, dtype=self.torch_dtype):  # Enable mixed precision
                    taus = torch.rand(self.num_taus).to(device=device)
                    frames_pred = self.model(in_frames, taus)
                    frames_pred = self.remove_spatial_context(frames_pred)
                    if self.predict_diff:
                        frames_pred = torch.cumsum(frames_pred, dim=1)
                    loss = self.calculate_loss(frames_pred, out_frames, taus)

                # backward
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

                train_loss_in_epoch_list.append(loss.detach().item())

                if (
                    verbose
                    and print_train_every_n_batch is not None
                    and batch_idx % print_train_every_n_batch == 0
                ):
                    self._logger.info(
                        f"BATCH({batch_idx + 1}/{len(self.train_loader)}) | "
                        f"Train loss({loss.detach().item():.4f}) | "
                        f"Time Batch({(time.time() - start_batch):.2f}) | "
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

                    in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                    out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                    with torch.autocast(device_type=device_type, dtype=self.torch_dtype):
                        frames_pred = self.model(in_frames, self.val_quantiles)
                        frames_pred = self.remove_spatial_context(frames_pred)
                        if self.predict_diff:
                            frames_pred = torch.cumsum(frames_pred, dim=1)
                        quantile_loss = self.calculate_loss(
                            frames_pred, out_frames, self.val_quantiles
                        )
                        quantile_loss_per_batch.append(quantile_loss.detach().item())

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            quantile_loss_in_epoch = sum(quantile_loss_per_batch) / len(
                quantile_loss_per_batch
            )

            val_loss_in_epoch = quantile_loss_in_epoch

            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step(val_loss_in_epoch)

            if run is not None:
                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log({"quantile_loss": quantile_loss_in_epoch}, step=epoch)
                run.log(
                    {"lr": self.optimizer.state_dict()["param_groups"][0]["lr"]},
                    step=epoch,
                )

            train_loss_per_epoch.append(train_loss_in_epoch)
            val_loss_per_epoch.append(val_loss_in_epoch)
            quantile_loss_per_epoch.append(quantile_loss_in_epoch)

            if verbose:
                self._logger.info(
                    f"Epoch({epoch + 1}/{n_epochs}) | "
                    f"Train_loss({(train_loss_in_epoch):06.4f}) | "
                    f"Val_loss({val_loss_in_epoch:.4f}) | "
                    f"Quantile_loss({quantile_loss_in_epoch:.4f}) | "
                    f"Time_Epoch({(time.time() - start_epoch):.2f}s) |"
                )

            if val_loss_in_epoch < best_val_loss:
                self._logger.info(
                    f"Saving best model. Best val loss: {val_loss_in_epoch:.4f}"
                )
                best_val_loss = val_loss_in_epoch
                best_epoch = epoch
                self.best_model_dict = {
                    "num_input_frames": self.in_frames,
                    "spatial_context": self.spatial_context,
                    "time_horizon": self.time_horizon,
                    "image_size": self.image_size,
                    "num_filters": self.filters,
                    "quantiles": self.val_quantiles,
                    "output_activation": self.output_activation,
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
                    "num_taus": self.num_taus,
                    "cosine_embedding_dimension": self.cosine_embedding_dimension,
                }

                if checkpoint_path is not None:
                    self.save_checkpoint(
                        model_name, best_epoch, best_val_loss, checkpoint_path
                    )

        return train_loss_per_epoch, val_loss_per_epoch

    def calculate_loss(self, predictions, y_target, taus):
        loss_fn = QuantileLoss(quantiles=taus)
        y_target = y_target.repeat(1, predictions.shape[1], 1, 1)
        return loss_fn(predictions, y_target)

    def get_F_at_points(self, points, pred_params):
        pass

    def load_checkpoint(self, checkpoint_path: str, device: str, eval_mode: bool = True):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.spatial_context = checkpoint.get("spatial_context", 0)
        self.output_activation = checkpoint.get("output_activation", None)
        self.time_horizon = checkpoint.get("time_horizon", None)
        self.num_taus = checkpoint.get("num_taus", 9)
        self.cosine_embedding_dimension = checkpoint.get("cosine_embedding_dimension", 64)
        self.image_size = checkpoint.get("image_size", 512)

        # Generate same architecture
        self.model = IQUNet(
            in_frames=self.in_frames,
            n_classes=self.num_taus,
            filters=self.filters,
            cosine_embedding_dimension=self.cosine_embedding_dimension,
            num_taus=self.num_taus,
            image_size=self.image_size,
            device=self.device,
            output_activation=self.output_activation,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=device)
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()

    @property
    def name(self):
        return f"IQUNet_IN{self.in_frames}_F{self.filters}_NT{self.num_taus}_CED{self.cosine_embedding_dimension}_PD{int(self.predict_diff)}"
