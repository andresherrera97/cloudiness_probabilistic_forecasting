# Standard library imports
import wandb
import time
import copy
import datetime
from typing import List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt

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
        predict_background: bool = True,  # Changed default to True
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

        # Initialize background tensor if prediction is enabled
        if predict_background:
            # Start with zeros - during training this will learn the background pattern
            # Single channel background
            self.background = torch.nn.Parameter(
                torch.zeros(
                    (1, 1, self.height, self.width),
                    device=device,
                    dtype=torch.float32,
                ),
                requires_grad=True,
            )
            # Add to optimizer
            
            # Add to optimizer with the same learning rate as other parameters
            # Get the learning rate from the first parameter group
            initial_lr = self.optimizer.param_groups[0]['lr']
            
            # Copy all optimizer settings from the first group to ensure compatibility with scheduler
            param_group = {'params': [self.background]}
            for key in self.optimizer.param_groups[0]:
                if key != 'params':
                    param_group[key] = self.optimizer.param_groups[0][key]
            
            self.optimizer.add_param_group(param_group)
            self._logger.info("Single-channel background prediction enabled")

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
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient than zero_grad()

                # forward
                with torch.autocast(
                    device_type=device_type, dtype=self.torch_dtype
                ):  # Enable mixed precision
                    frames_pred = self.model(in_frames)
                    frames_pred = self.remove_spatial_context(frames_pred)

                    # Apply background prediction if enabled
                    if predict_background:
                        # Expand background to match batch size
                        batch_size = frames_pred.shape[0]
                        expanded_background = self.background.expand(
                            batch_size, -1, -1, -1
                        )

                        # Take maximum between prediction and background
                        frames_pred = torch.maximum(frames_pred, expanded_background)

                    loss = self.calculate_loss(frames_pred, out_frames)

                # backward
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
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

            val_loss_in_epoch, forecasting_metrics = self.run_validation(
                device, device_type, num_val_samples, predict_background
            )

            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step(val_loss_in_epoch)

            if run is not None:
                run.log({"train_loss": train_loss_in_epoch}, step=epoch)
                run.log({"val_loss": val_loss_in_epoch}, step=epoch)
                run.log(
                    {"lr": self.optimizer.state_dict()["param_groups"][0]["lr"]},
                    step=epoch,
                )

                # Log background tensor to wandb if enabled
                if predict_background:
                    # Log background stats
                    background_cpu = self.background.detach().cpu()
                    run.log(
                        {"background_mean": float(background_cpu.mean())}, step=epoch
                    )
                    run.log({"background_max": float(background_cpu.max())}, step=epoch)
                    run.log({"background_min": float(background_cpu.min())}, step=epoch)

                    # Log background as image - single channel
                    try:
                        # Get the background data
                        bg_data = background_cpu[
                            0, 0
                        ].numpy()  # Remove batch and channel dimensions

                        # Normalize to [0, 1] for visualization
                        bg_min, bg_max = bg_data.min(), bg_data.max()
                        if bg_max > bg_min:  # Avoid division by zero
                            bg_normalized = (bg_data - bg_min) / (bg_max - bg_min)
                        else:
                            bg_normalized = bg_data

                        # Create figure with colorbar
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(bg_normalized, cmap="viridis")
                        ax.set_title(f"Background - Epoch {epoch+1}")
                        plt.colorbar(im, ax=ax, label="Normalized Value")

                        # Log to wandb
                        run.log({"background_heatmap": wandb.Image(fig)}, step=epoch)
                        plt.close(fig)

                        # Also log as a simple image
                        run.log(
                            {"background_raw": wandb.Image(bg_normalized)}, step=epoch
                        )

                    except Exception as e:
                        self._logger.warning(
                            f"Failed to log background visualization: {e}"
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
                if predict_background:
                    self._logger.info(
                        f"Background stats - Mean: {self.background.detach().mean():.4f}, "
                        f"Min: {self.background.detach().min():.4f}, "
                        f"Max: {self.background.detach().max():.4f}"
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
                    "predict_background": predict_background,
                }

                # Save background tensor if using background prediction
                if predict_background:
                    self.best_model_dict["background"] = copy.deepcopy(
                        self.background.data
                    )

                if checkpoint_path is not None:
                    self.save_checkpoint(
                        model_name, best_epoch, best_val_loss, checkpoint_path
                    )

        # At the end of training, log background evolution
        if run is not None and predict_background:
            try:
                # Create a more detailed final visualization

                # Get the final background
                final_bg = self.background.detach().cpu().numpy()[0, 0]  # [H, W]

                # Create a figure with detailed visualization
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(final_bg, cmap="viridis")
                ax.set_title("Final Learned Background")
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Value")

                plt.tight_layout()
                run.log({"final_background_analysis": wandb.Image(fig)})
                plt.close(fig)

            except Exception as e:
                self._logger.warning(
                    f"Failed to log final background visualization: {e}"
                )

        return train_loss_per_epoch, val_loss_per_epoch

    def run_validation(
        self, device: str, device_type: str, num_val_samples: int, dataset: str = "val"
    ):
        self.model.eval()
        val_loss_per_batch = []  # stores values for this validation run
        self.deterministic_metrics.start_epoch()

        if dataset == "val":
            data_loader = self.val_loader
        elif dataset == "test":
            data_loader = self.test_loader

        with torch.no_grad():
            for val_batch_idx, (in_frames, out_frames) in enumerate(data_loader):

                in_frames = in_frames.to(device=device, dtype=self.torch_dtype)
                out_frames = out_frames.to(device=device, dtype=self.torch_dtype)

                with torch.autocast(device_type=device_type, dtype=self.torch_dtype):
                    frames_pred = self.model(in_frames)
                    frames_pred = self.remove_spatial_context(frames_pred)
                    persistence_pred = self.remove_spatial_context(
                        in_frames[:, self.in_frames - 1 :, :, :]
                    )
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
        return val_loss_in_epoch, forecasting_metrics

    def predict(self, X: torch.Tensor):
        return self.model(X.float())

    def load_checkpoint(
        self, checkpoint_path: str, device: str, eval_mode: bool = True
    ):
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
