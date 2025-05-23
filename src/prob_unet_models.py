import fire
import torch
import logging
import json
from typing import Dict
import numpy as np
from torchmetrics.classification import MulticlassPrecision
import os

from models import (
    QuantileRegressorUNet,
    UNetConfig,
    BinClassifierUNet,
    MedianScaleUNet,
    IQUNetPipeline,
)
from metrics import (
    crps_laplace,
    logscore_bin_fn,
    calculate_reliability_diagram_data,
    plot_reliability_diagram,
    collect_reliability_diagram_data,
    calculate_reliability_diagram_coordinates,
    plot_reliability_diagram_CDF,
)
from postprocessing.cdf_bin_preds import (
    get_cdf_from_binned_probabilities_numpy
)
from postprocessing.transform import quantile_2_bin
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Eval Script")


# TODO: move this to a separate file
def laplace_cdf(x, mu, b):
    """
    Calculates the Cumulative Distribution Function (CDF) of a Laplace distribution.

    Args:
        x (float or np.ndarray): The point(s) at which to evaluate the CDF.
        mu (float): The location parameter of the Laplace distribution.
        b (float): The scale parameter of the Laplace distribution (b > 0).

    Returns:
        float or np.ndarray: The CDF value(s).
    """
    if b <= 0:
        raise ValueError("Scale parameter 'b' must be positive.")
    
    # Ensure x can be compared with mu element-wise if x is an array
    if isinstance(x, np.ndarray):
        res = np.zeros_like(x, dtype=float)
        idx_less = (x < mu)
        idx_ge = (x >= mu)
        
        res[idx_less] = 0.5 * np.exp((x[idx_less] - mu) / b)
        res[idx_ge] = 1.0 - 0.5 * np.exp(-(x[idx_ge] - mu) / b)
        return res
    else: # x is a scalar
        if x < mu:
            return 0.5 * np.exp((x - mu) / b)
        else:
            return 1.0 - 0.5 * np.exp(-(x - mu) / b)


def calculate_laplace_bin_probabilities(mu, b, num_bins):
    """
    Calculates the probability of a Laplace-distributed variable falling into
    equally sized bins within the [0, 1] range for multiple predictions.

    Args:
        mus (np.ndarray or list): Array of location parameters (mu) for each Laplace prediction.
        bs (np.ndarray or list): Array of scale parameters (b) for each Laplace prediction.
                                Must be positive.
        num_bins (int): The number of equally sized bins to divide the [0, 1] range into.

    Returns:
        np.ndarray: A 2D array where rows correspond to predictions and columns
                    correspond to bins. Element (i, j) is the probability
                    assigned by the i-th Laplace prediction to the j-th bin.
                    Shape: (len(mus), num_bins).
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    bin_probabilities = np.zeros(num_bins)
    
    # Define bin edges for the [0, 1] interval
    # Example: num_bins = 5 -> edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_edges = np.linspace(0, 1, num_bins + 1)

    if b <= 0:
        # Or handle this by assigning uniform probabilities, or NaN, or skip
        print(f"Warning: Scale parameter b <= 0 for prediction {i}. Probabilities for this prediction might be ill-defined.")
        # For this example, let's assign NaN if b is invalid for a prediction
        bin_probabilities[:] = np.nan
        return bin_probabilities

    for j in range(num_bins):
        lower_bound = bin_edges[j]
        upper_bound = bin_edges[j+1]
        
        prob_in_bin = laplace_cdf(upper_bound, mu, b) - laplace_cdf(lower_bound, mu, b)
        bin_probabilities[j] = prob_in_bin

    return bin_probabilities


def get_checkpoint_path(time_horizon: int) -> Dict[str, str]:
    if time_horizon == 60:
        return {
            "qr": "checkpoints/salto_down/prob_60min_salto_512/qr/QRUNet_IN3_NB9_F32_SC0_PDTrue_BS_8_TH60_E6_BVM0_04_D2025-04-13_21:52.pt",
            "bin": "checkpoints/salto_down/prob_60min_salto_512/bin/BinUNet_IN3_NB10_F32_SC0_BS_8_TH60_E8_BVM1_83_D2025-05-07_09:53.pt",
            "laplace": "checkpoints/salto_down/prob_60min_salto_512/laplace//MedianScaleUNet_IN3_F32_SC0_BS_8_TH60_E7_BVMtens_D2025-05-15_05:42.pt",
            "iqn": "checkpoints/salto_down/prob_60min_salto_512/iqn/IQUNet_IN3_F32_NT9_CED64_PD0_BS_8_TH60_E7_BVM0_23_D2025-05-08_22:10.pt",
        }
    elif time_horizon == 120:
        return {
            "qr": "checkpoints/salto_down/prob_120min_salto_512/qr/QRUNet_IN3_NB9_F32_SC0_PDTrue_BS_8_TH120_E8_BVM0_04_D2025-04-13_18:26.pt",
            "bin": "checkpoints/salto_down/prob_120min_salto_512/bin/BinUNet_IN3_NB10_F32_SC0_BS_8_TH120_E9_BVM1_88_D2025-05-07_07:44.pt",
            "laplace": "checkpoints/salto_down/prob_120min_salto_512/laplace/MedianScaleUNet_IN3_F32_SC0_BS_8_TH120_E9_BVMtens_D2025-05-16_09:00.pt",
            "iqn": "checkpoints/salto_down/prob_120min_salto_512/iqn/IQUNet_IN3_F32_NT9_CED64_PD0_BS_8_TH120_E9_BVM0_29_D2025-05-08_22:48.pt",
        }
    else:
        raise ValueError("Invalid time horizon. Supported values are 60 or 120.")


def main(
    time_horizon: int = 60,
    dataset: str = "salto_down",
    dataset_path: str = "datasets/salto_downsample/",
    subset: str = "val",
    debug: bool = False,
    reliability_diagram_bins: int = 20,
):
    logger.info("Starting evaluation script...")
    logger.info(f"Time horizon: {time_horizon} min")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Subset: {subset}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Number of bins for reliability diagram: {reliability_diagram_bins}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"using device: {device}")

    torch.set_grad_enabled(False)

    num_bins = 10
    quantiles = np.linspace(0, 1, num_bins + 1)[1:-1]

    multiclass_precision_metric = MulticlassPrecision(
        num_classes=num_bins, average="macro", top_k=1, multidim_average="global"
    ).to(device=device)

    models_paths = get_checkpoint_path(time_horizon)

    unet_config = UNetConfig(
        in_frames=3,
        spatial_context=0,
        filters=32,
        output_activation="sigmoid",
        device=device,
    )

    qr_unet = QuantileRegressorUNet(
        config=unet_config, quantiles=quantiles, predict_diff=True
    )
    qr_unet.load_checkpoint(checkpoint_path=models_paths["qr"], device=device)

    bin_unet = BinClassifierUNet(config=unet_config, n_bins=num_bins)
    bin_unet.load_checkpoint(checkpoint_path=models_paths["bin"], device=device)

    laplace_unet = MedianScaleUNet(unet_config)
    laplace_unet.load_checkpoint(checkpoint_path=models_paths["laplace"], device=device)

    iqn_unet = IQUNetPipeline(config=unet_config)
    iqn_unet.load_checkpoint(checkpoint_path=models_paths["iqn"], device=device)

    bin_unet.create_dataloaders(
        dataset=dataset,
        path=dataset_path,
        batch_size=1,
        time_horizon=time_horizon,
        binarization_method="integer_classes",
        create_test_loader=(subset == "test"),
        shuffle=False,
        drop_last=False,
    )

    if subset == "train":
        data_loader = bin_unet.train_loader
    elif subset == "val":
        data_loader = bin_unet.val_loader
    elif subset == "test":
        data_loader = bin_unet.test_loader
    else:
        raise ValueError("Invalid subset.")

    metrics = {
        "qr": {"crps": [], "logscore": [], "logscore_dividing": [], "precision": []},
        "bin": {"crps": [], "logscore": [], "logscore_dividing": [], "precision": []},
        "laplace": {
            "crps": [],
            "logscore": [],
            "logscore_dividing": [],
            "precision": [],
        },
        "iqn": {"crps": [], "logscore": [], "logscore_dividing": [], "precision": []},
    }

    # Add laplace to reliability diagram
    reliability_diagram = {
        "qr": {
            "predicted_probs": [],
            "actual_outcomes": [],
            "cdf_values": [],
        },
        "bin": {
            "predicted_probs": [],
            "actual_outcomes": [],
            "cdf_values": [],
        },
        "iqn": {
            "predicted_probs": [],
            "actual_outcomes": [],
            "cdf_values": [],
        },
        "laplace": {
            "predicted_probs": [],
            "actual_outcomes": [],
            "cdf_values": [],
        },
    }

    for batch_idx, (in_frames, (out_frames, bin_output)) in enumerate(data_loader):
        in_frames = in_frames.to(device)
        out_frames = out_frames.to(device)
        bin_output = bin_output.to(device)

        # --- Quantile Regression UNet ---
        qr_unet_preds = qr_unet.predict(in_frames)
        metrics["qr"]["crps"].append(
            qr_unet.crps_loss.crps_loss(pred=qr_unet_preds, y=out_frames)
            .detach()
            .item()
        )
        qr_binarized_preds_np = quantile_2_bin(
            quantiles=quantiles,
            quantiles_values=qr_unet_preds.cpu().numpy(),
            num_bins=num_bins,
        )
        qr_binarized_preds_tensor = (
            torch.from_numpy(qr_binarized_preds_np).float().to(device)
        )
        metrics["qr"]["logscore"].append(
            logscore_bin_fn(qr_binarized_preds_tensor, bin_output).detach().item()
        )
        metrics["qr"]["logscore_dividing"].append(
            logscore_bin_fn(
                qr_binarized_preds_tensor, bin_output, divide_by_bin_width=True
            )
            .detach()
            .item()
        )
        metrics["qr"]["precision"].append(
            multiclass_precision_metric(qr_binarized_preds_tensor, bin_output)
            .detach()
            .item()
        )

        reliability_diagram = collect_reliability_diagram_data(
            model_name="qr",
            predicted_probs=qr_binarized_preds_np,
            actual_outcomes=bin_output,
            reliability_diagram=reliability_diagram,
        )

        reliability_diagram["qr"]["cdf_values"].append(
            get_cdf_from_binned_probabilities_numpy(
                y_value=out_frames[0, 0, 256, 256].cpu().numpy(),
                probabilities=[1/num_bins] * num_bins,
                bin_edges=[0] + qr_unet_preds[0, :, 256, 256].cpu().numpy().tolist() + [1.0],
            )
        )

        # --- Bin Classifier UNet ---
        bin_unet_preds = bin_unet.predict(in_frames.float())
        metrics["bin"]["crps"].append(
            bin_unet.crps_loss.crps_loss(bin_unet_preds, out_frames).detach().item()
        )
        metrics["bin"]["logscore"].append(
            logscore_bin_fn(bin_unet_preds, bin_output).detach().item()
        )
        metrics["bin"]["logscore_dividing"].append(
            logscore_bin_fn(bin_unet_preds, bin_output, divide_by_bin_width=True)
            .detach()
            .item()
        )
        metrics["bin"]["precision"].append(
            multiclass_precision_metric(bin_unet_preds, bin_output).detach().item()
        )

        reliability_diagram = collect_reliability_diagram_data(
            model_name="bin",
            predicted_probs=bin_unet_preds,
            actual_outcomes=bin_output,
            reliability_diagram=reliability_diagram,
        )

        reliability_diagram["bin"]["cdf_values"].append(
            get_cdf_from_binned_probabilities_numpy(
                y_value=out_frames[0, 0, 256, 256].cpu().numpy(),
                probabilities=bin_unet_preds[0, :, 256, 256].cpu().numpy(),
                bin_edges=[i/num_bins for i in range(num_bins + 1)],
            )
        )

        # --- Median Scale UNet ---
        laplace_unet_preds = laplace_unet.model(in_frames.float())
        metrics["laplace"]["crps"].append(
            crps_laplace(out_frames, laplace_unet_preds).detach().item()
        )
        laplace_logscore = (
            laplace_unet.loss_fn(laplace_unet_preds, out_frames).detach().item()
        )
        metrics["laplace"]["logscore"].append(laplace_logscore)
        metrics["laplace"]["logscore_dividing"].append(laplace_logscore)
        metrics["laplace"]["precision"].append(0)

        reliability_diagram["laplace"]["cdf_values"].append(
            laplace_unet.get_F_at_points(
                points=out_frames[0, 0, 256:257, 256:257].unsqueeze(0).unsqueeze(0),
                pred_params=laplace_unet_preds[0, :, 256:257, 256:257].unsqueeze(0),
            )[0, 0, 0, 0].cpu().numpy()
        )

        laplace_pred_bin_probs = calculate_laplace_bin_probabilities(
            laplace_unet_preds[0, 0, 256, 256].cpu().numpy(),
            laplace_unet_preds[0, 1, 256, 256].cpu().numpy(),
            num_bins
        )

        laplace_pred_bin_probs = torch.from_numpy(laplace_pred_bin_probs).float().to(device)
        laplace_pred_bin_probs = laplace_pred_bin_probs.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        reliability_diagram = collect_reliability_diagram_data(
            model_name="laplace",
            predicted_probs=laplace_pred_bin_probs,
            actual_outcomes=bin_output,
            reliability_diagram=reliability_diagram,
        )

        # --- IQN UNet ---
        iqn_unet_pred = iqn_unet.model(in_frames.float(), iqn_unet.val_quantiles)
        iqn_unet_pred = torch.sort(iqn_unet_pred, dim=1)[0]
        metrics["iqn"]["crps"].append(
            iqn_unet.crps_loss.crps_loss(pred=iqn_unet_pred, y=out_frames)
            .detach()
            .item()
        )
        iqn_binarized_preds_np = quantile_2_bin(
            quantiles=quantiles,
            quantiles_values=iqn_unet_pred.cpu().numpy(),
            num_bins=num_bins,
        )
        iqn_binarized_preds_tensor = (
            torch.from_numpy(iqn_binarized_preds_np).float().to(device)
        )
        metrics["iqn"]["logscore"].append(
            logscore_bin_fn(iqn_binarized_preds_tensor, bin_output).detach().item()
        )
        metrics["iqn"]["logscore_dividing"].append(
            logscore_bin_fn(
                iqn_binarized_preds_tensor, bin_output, divide_by_bin_width=True
            )
            .detach()
            .item()
        )
        metrics["iqn"]["precision"].append(
            multiclass_precision_metric(iqn_binarized_preds_tensor, bin_output)
            .detach()
            .item()
        )
        reliability_diagram = collect_reliability_diagram_data(
            model_name="iqn",
            predicted_probs=iqn_binarized_preds_np,
            actual_outcomes=bin_output,
            reliability_diagram=reliability_diagram,
        )

        reliability_diagram["iqn"]["cdf_values"].append(
            get_cdf_from_binned_probabilities_numpy(
                y_value=out_frames[0, 0, 256, 256].cpu().numpy(),
                probabilities=[1/num_bins] * num_bins,
                bin_edges=[0] + iqn_unet_pred[0, :, 256, 256].cpu().numpy().tolist() + [1.0],
            )
        )

        if debug and batch_idx >= 3:  # Ensure a few batches run for debug
            logger.info(f"Debug mode: stopping after {batch_idx} batches.")
            break

    # --- Calculate and Plot Reliability Diagrams ---
    for model_name, model_metrics in reliability_diagram.items():
        if len(model_metrics["predicted_probs"]) > 0:

            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)

            rd_filename_suffix = f"{subset}_{time_horizon}min"
            crop_info = "_central_pixel"
            rd_filename_suffix += crop_info
            if debug:
                rd_filename_suffix += "_debug"
            rd_filename = os.path.join(
                results_dir,
                f"reliability_diagram_{model_name}_unet_{rd_filename_suffix}.png",
            )

            curve_mean_preds, curve_obs_freqs, hist_centers, hist_counts = (
                calculate_reliability_diagram_data(
                    model_metrics["predicted_probs"],
                    model_metrics["actual_outcomes"],
                    n_reliability_bins=reliability_diagram_bins,
                )
            )

            plot_reliability_diagram(
                curve_mean_preds,
                curve_obs_freqs,
                hist_centers,
                hist_counts,
                model_name=(
                    f"{model_name} UNet central pixel, {subset} subset, "
                    f"{time_horizon} min time horizon"
                ),
                filename=rd_filename,
            )
        else:
            logger.warning(
                f"No data collected for {model_name} UNet reliability diagram. Skipping plot."
            )
        # Plot CDF
        if len(model_metrics["cdf_values"]) > 0:
            sorted_tau_values, empirical_cdf_values = calculate_reliability_diagram_coordinates(model_metrics["cdf_values"])
            title = f"Reliability Diagram CDF for {model_name} UNet, {subset} subset, {time_horizon} min time horizon"
            if debug:
                title += " (Debug Mode)"
            plot_reliability_diagram_CDF(
                sorted_tau_values,
                empirical_cdf_values,
                title=title,
            )

    metrics_filename_suffix = f"{subset}_{time_horizon}min"
    if debug:
        metrics_filename_suffix += "_debug"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename_suffix += f"_{timestamp}"
    metrics_path = os.path.join("results", f"metrics_{metrics_filename_suffix}.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("Calculating mean metrics...")
    logger.info(f"Mean metrics for {subset} subset, time horizon: {time_horizon} min")
    mean_metrics = {}
    for model_name, model_metrics in metrics.items():
        logger.info(f"--- {model_name} ---")
        mean_metrics[model_name] = {}
        for metric_name, metric_values in model_metrics.items():
            if metric_values:
                mean_value = np.mean(metric_values)
                logger.info(f"  {metric_name}: {mean_value:.4f}")
                mean_metrics[model_name][metric_name] = mean_value
            else:
                logger.info(f"  {metric_name}: N/A (no values)")
                mean_metrics[model_name][metric_name] = None
        # Add checkpoint path for this model
        mean_metrics[model_name]["checkpoint_path"] = models_paths.get(model_name, "N/A")

    mean_metrics_path = os.path.join("results", f"mean_metrics_{metrics_filename_suffix}.json")
    with open(mean_metrics_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
    logger.info(f"Mean metrics saved to {mean_metrics_path}")


if __name__ == "__main__":
    fire.Fire(main)
