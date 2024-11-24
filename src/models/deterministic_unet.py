# Standard library imports
import time
import copy
import datetime
from typing import List, Optional, Tuple
from pathlib import Path

# Related third-party imports
import torch
import torch.nn as nn
from .probabilistic_unet import UNetPipeline, UNetConfig
from .unet import UNet
import logging
from metrics.deterministic_metrics import DeterministicMetrics


# Configure logging
logging.basicConfig(level=logging.INFO)


class DeterministicUNet(UNetPipeline):
    def __init__(
        self,
        config: UNetConfig,
    ):
        super().__init__(config)
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=1,
            filters=self.filters,
            output_activation=self.output_activation,
        )

        self.loss_fn = nn.L1Loss()  # Use MAE as train loss
        self.deterministic_metrics = DeterministicMetrics()

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

        best_val_loss = float('inf')

        device_type = "cpu" if device == torch.device("cpu") else "cuda"
        self._logger.info(f"device type: {device_type}")

        scaler = torch.amp.GradScaler(device)  # For mixed precision training

        for epoch in range(n_epochs):
            start_epoch = time.time()
            train_loss_in_epoch_list = []  # stores values inside the current epoch
            val_loss_in_epoch = []  # stores values inside the current epoch
            self.model.train()

            for batch_idx, (in_frames, out_frames) in enumerate(self.train_loader):

                start_batch = time.time()

                # Use float16 for mixed precision
                in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                # forward
                with torch.autocast(device_type=device_type, dtype=self.torch_dtype):  # Enable mixed precision
                    frames_pred = self.model(in_frames)
                    frames_pred = self.remove_spatial_context(frames_pred)
                    loss = self.calculate_loss(frames_pred, out_frames)

                # backward
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

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
            self.deterministic_metrics.start_epoch()

            with torch.no_grad():
                for val_batch_idx, (in_frames, out_frames) in enumerate(
                    self.val_loader
                ):

                    in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                    out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                    with torch.autocast(device_type=device_type, dtype=self.torch_dtype):
                        frames_pred = self.model(in_frames)
                        frames_pred = self.remove_spatial_context(frames_pred)
                        persistence_pred = self.remove_spatial_context(in_frames[:, self.in_frames-1:, :, :])
                        val_loss = self.calculate_loss(frames_pred, out_frames)

                        self.deterministic_metrics.run_per_batch_metrics(
                            y_true=out_frames,
                            y_pred=frames_pred,
                            y_persistence=persistence_pred,
                            pixel_wise=False,
                            eps=1e-5,
                        )

                    val_loss_per_batch.append(val_loss.detach().item())

                    if num_val_samples is not None and val_batch_idx >= num_val_samples:
                        break

            val_loss_in_epoch = sum(val_loss_per_batch) / len(val_loss_per_batch)
            forecasting_metrics = self.deterministic_metrics.end_epoch()

            if self.scheduler is not None:
                self.scheduler.step(val_loss_in_epoch)

            if run is not None:

                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log(
                    {"lr": self.optimizer.state_dict()["param_groups"][0]["lr"]},
                    step=epoch,
                )
                for key, value in forecasting_metrics.items():
                    run.log({key: value}, step=epoch)

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
                    "train_loss_epoch_mean": train_loss_in_epoch,
                }
                if checkpoint_path is not None:
                    self.save_checkpoint(
                        model_name, best_epoch, best_val_loss, checkpoint_path
                    )

        return train_loss_per_epoch, val_loss_per_epoch

    def predict(self, X: torch.Tensor):
        return self.model(X.float())

    def load_checkpoint(self, checkpoint_path: str, device: str, eval_mode: bool = True):
        """Abstract method to load a trained checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.in_frames = checkpoint["num_input_frames"]
        self.filters = checkpoint["num_filters"]
        self.spatial_context = checkpoint["spatial_context"]
        self.output_activation = checkpoint.get("output_activation", None)
        self.time_horizon = checkpoint.get("time_horizon", None)

        # Generate same architecture
        self.model = UNet(
            in_frames=self.in_frames,
            n_classes=1,
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
        return f"UNet_IN{self.in_frames}_F{self.filters}_SC{self.spatial_context}"
