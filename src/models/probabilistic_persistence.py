# Standard library imports
import os

# Related third-party imports
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging

# Local application/library specific imports,
from metrics import (
    CRPSLoss,
    collect_reliability_diagram_data,
    plot_reliability_diagram,
    collect_reliability_diagram_data,
    calculate_reliability_diagram_coordinates,
    plot_reliability_diagram_CDF,
    logscore_bin_fn,
    calculate_reliability_diagram_data,
)
from data_handlers import MovingMnistDataset, GOES16Dataset
from data_handlers.utils import classify_array_in_integer_classes
from postprocessing.transform import quantile_2_bin
from postprocessing.cdf_bin_preds import get_cdf_from_binned_probabilities_numpy


class PersistenceEnsemble:
    def __init__(self, n_quantiles: int, device: str = "cpu"):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.n_quantiles = n_quantiles
        self.quantiles = list(np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1])
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.device = device
        self.crps_loss = CRPSLoss(quantiles=self.quantiles, device=device)

    def create_dataloaders(
        self,
        dataset: str,
        path: str,
        batch_size: int,
        time_horizon: int,
        create_test_loader: bool = False,
    ):
        self.batch_size = batch_size
        self.time_horizon = time_horizon
        self.dataset_path = path

        if dataset.lower() in ["moving_mnist", "mnist", "mmnist"]:
            train_dataset = MovingMnistDataset(
                path=os.path.join(path, "train/"),
                input_frames=self.n_quantiles,
            )
            val_dataset = MovingMnistDataset(
                path=os.path.join(path, "validation/"), input_frames=self.n_quantiles
            )

        elif dataset.lower() in [
            "goes16",
            "satellite",
            "sat",
            "debug_salto",
            "debug",
            "downsample",
            "salto_down",
            "salto_512",
        ]:
            train_dataset = GOES16Dataset(
                path=os.path.join(path, "train/"),
                num_in_images=self.n_quantiles,
                minutes_forward=time_horizon,
                expected_time_diff=10,
                inpaint_pct_threshold=1.0,
            )

            val_dataset = GOES16Dataset(
                path=os.path.join(path, "val/"),
                num_in_images=self.n_quantiles,
                minutes_forward=time_horizon,
                expected_time_diff=10,
                inpaint_pct_threshold=1.0,
            )
            if create_test_loader:
                test_dataset = GOES16Dataset(
                    path=os.path.join(path, "test/"),
                    num_in_images=self.n_quantiles,
                    minutes_forward=time_horizon,
                    expected_time_diff=10,
                    inpaint_pct_threshold=1.0,
                )

        else:
            raise ValueError(f"Dataset {dataset} not recognized.")

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        if create_test_loader:
            self.test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=True
            )

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

    def predict(self, X, allow_equal_quantile_values=True):
        sorted_X = torch.sort(X, dim=1)[0]
        if not allow_equal_quantile_values:
            sorted_X = self.modify_consecutive_equals(sorted_X)
        return sorted_X

    def crps_quantile(self, predictions, y_target):
        return self.crps_loss.crps_loss(
            pred=predictions,
            y=y_target,
        )

    def modify_consecutive_equals(self, tensor, base_value=1e-5):
        B, N, H, W = tensor.shape

        # Create a tensor of increasing values for each slice in N
        n_values = torch.arange(1, N + 1, dtype=tensor.dtype, device=tensor.device)
        n_values = n_values.view(1, N, 1, 1)

        # Compare each slice with the next one
        differences = tensor[:, 1:] - tensor[:, :-1]

        # Find where the difference is zero (i.e., consecutive slices are equal)
        mask = differences == 0

        # Create a tensor to add, with the same shape as the input tensor
        to_add = torch.zeros_like(tensor)

        # Set the values to add where the mask is True
        # The added value is proportional to the slice index
        to_add[:, 1:][mask] = base_value * n_values[:, 1:].expand_as(mask)[mask]

        # Add the values
        modified_tensor = tensor + to_add
        modified_tensor = torch.sort(modified_tensor, dim=1)[0]

        zero_mask = modified_tensor[:, 0, :, :] == 0
        modified_tensor[:, 0, :, :] = (
            modified_tensor[:, 0, :, :] + modified_tensor[:, 1, :, :] / 2 * zero_mask
        )

        return modified_tensor

    def random_example(self):
        for _, (in_frames, out_frames) in enumerate(self.train_loader):
            in_frames = in_frames.to(self.device)
            out_frames = out_frames.to(self.device)
            predictions = self.predict(in_frames)
            crps = self.crps_loss.crps_loss(
                pred=predictions,
                y=out_frames,
            )
            break
        return in_frames, out_frames, predictions, crps

    def predict_on_dataset(
        self,
        subset: str = "validation",
        debug: bool = False,
        time_horizon: int = 60,
        reliability_diagram_bins: int = 20,
    ):
        if subset in ["validation", "val"]:
            self._logger.info("Predicting on validation set")
            data_loader = self.val_loader
        elif subset in ["train", "training"]:
            self._logger.info("Predicting on training set")
            data_loader = self.train_loader
        elif subset in ["test", "testing"]:
            self._logger.info("Predicting on test set")
            data_loader = self.test_loader
        else:
            raise ValueError(f"Dataset {subset} not recognized.")

        crps_per_batch = []
        logscore_per_batch = []
        logscore_dividing_per_batch = []

        reliability_diagram = {
            "ensemble_persistence": {
                "predicted_probs": [],
                "actual_outcomes": [],
                "cdf_values": [],
            }
        }

        for batch_idx, (in_frames, out_frames) in enumerate(data_loader):
            in_frames = in_frames.to(self.device)
            out_frames = out_frames.to(self.device)
            predictions = self.predict(in_frames)
            predictions = predictions.to(self.device)
            crps_per_batch.append(
                self.crps_loss.crps_loss(
                    pred=predictions,
                    y=out_frames,
                )
            )
            # logscore
            preds_bin = quantile_2_bin(
                quantiles=self.quantiles,
                quantiles_values=predictions,
                num_bins=self.n_quantiles + 1,
            )

            bin_output = classify_array_in_integer_classes(
                out_frames[0, 0].cpu().numpy(), num_bins=self.n_quantiles + 1
            )

            logscore_per_batch.append(
                logscore_bin_fn(
                    torch.tensor(preds_bin).to(self.device),
                    torch.tensor(bin_output).to(self.device).unsqueeze(0),
                )
                .detach()
                .item()
            )

            logscore_dividing_per_batch.append(
                logscore_bin_fn(
                    torch.tensor(preds_bin).to(self.device),
                    torch.tensor(bin_output).to(self.device).unsqueeze(0),
                    divide_by_bin_width=True,
                )
                .detach()
                .item()
            )

            reliability_diagram = collect_reliability_diagram_data(
                model_name="ensemble_persistence",
                predicted_probs=preds_bin,
                actual_outcomes=np.expand_dims(bin_output, axis=0),
                reliability_diagram=reliability_diagram,
            )

            reliability_diagram["ensemble_persistence"]["cdf_values"].append(
                get_cdf_from_binned_probabilities_numpy(
                    y_value=out_frames[0, 0, 256, 256].cpu().numpy(),
                    probabilities=[1 / (self.n_quantiles + 1)] * (self.n_quantiles + 1),
                    bin_edges=[0]
                    + predictions[0, :, 256, 256].cpu().numpy().tolist()
                    + [1.0],
                )
            )

            if debug and batch_idx > 2:
                break

        dataset_crps = torch.mean(torch.tensor(crps_per_batch)).detach().item()
        dataset_logscore = torch.mean(torch.tensor(logscore_per_batch)).detach().item()
        dataset_logscore_dividing = (
            torch.mean(torch.tensor(logscore_dividing_per_batch)).detach().item()
        )

        # --- Calculate and Plot Reliability Diagrams ---
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        rd_filename_suffix = f"{subset}_{time_horizon}min"
        crop_info = "_central_pixel"
        rd_filename_suffix += crop_info
        if debug:
            rd_filename_suffix += "_debug"
        rd_filename = os.path.join(
            results_dir,
            f"reliability_diagram_ensemble_persistende_{rd_filename_suffix}.png",
        )

        curve_mean_preds, curve_obs_freqs, hist_centers, hist_counts = (
            calculate_reliability_diagram_data(
                reliability_diagram["ensemble_persistence"]["predicted_probs"],
                reliability_diagram["ensemble_persistence"]["actual_outcomes"],
                n_reliability_bins=reliability_diagram_bins,
            )
        )
        plot_reliability_diagram(
            curve_mean_preds,
            curve_obs_freqs,
            hist_centers,
            hist_counts,
            model_name=(
                f"Ensemble Persistence central pixel, {subset} subset, "
                f"{time_horizon} min time horizon"
            ),
            filename=rd_filename,
        )

        sorted_tau_values, empirical_cdf_values = (
            calculate_reliability_diagram_coordinates(
                reliability_diagram["ensemble_persistence"]["cdf_values"]
            )
        )
        title = f"Reliability Diagram CDF for Ensemble Persistence, {subset} subset, {time_horizon} min time horizon"
        if debug:
            title += " (Debug Mode)"
        plot_reliability_diagram_CDF(
            sorted_tau_values,
            empirical_cdf_values,
            title=title,
        )

        return dataset_crps, dataset_logscore, dataset_logscore_dividing
