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
)
from postprocessing.transform import quantile_2_bin


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Eval Script")


def get_checkpoint_path(time_horizon: int) -> Dict[str, str]:
    if time_horizon == 60:
        return {
            "qr": "checkpoints/salto_down/prob_60min_salto_512/qr/QRUNet_IN3_NB9_F32_SC0_PDTrue_BS_8_TH60_E6_BVM0_04_D2025-04-13_21:52.pt",
            "bin": "checkpoints/salto_down/prob_60min_salto_512/bin/BinUNet_IN3_NB10_F32_SC0_BS_8_TH60_E8_BVM1_83_D2025-05-07_09:53.pt",
            "laplace": "checkpoints/salto_down/prob_60min_salto_512/laplace/MedianScaleUNet_IN3_F32_SC0_BS_8_TH60_E7_BVM0_40_D2025-04-25_13:36.pt",
            "iqn": "checkpoints/salto_down/prob_60min_salto_512/iqn/IQUNet_IN3_F32_NT9_CED64_PD0_BS_8_TH60_E7_BVM0_23_D2025-05-08_22:10.pt",
        }
    elif time_horizon == 120:
        return {
            "qr": "checkpoints/salto_down/prob_120min_salto_512/qr/QRUNet_IN3_NB9_F32_SC0_PDTrue_BS_8_TH120_E8_BVM0_04_D2025-04-13_18:26.pt",
            "bin": "checkpoints/salto_down/prob_120min_salto_512/bin/BinUNet_IN3_NB10_F32_SC0_BS_8_TH120_E9_BVM1_88_D2025-05-07_07:44.pt",
            "laplace": "checkpoints/salto_down/prob_120min_salto_512/laplace/MedianScaleUNet_IN3_F32_SC0_BS_8_TH120_E9_BVM0_42_D2025-04-25_09:25.pt",
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
        },
        "bin": {
            "predicted_probs": [],
            "actual_outcomes": [],
        },
        "iqn": {
            "predicted_probs": [],
            "actual_outcomes": [],
        },
    }

    for batch_idx, (in_frames, (out_frames, bin_output)) in enumerate(data_loader):
        in_frames = in_frames.to(device)
        out_frames = out_frames.to(device)
        bin_output = bin_output.to(device)

        # --- Quantile Regression UNet ---
        qr_unet_preds = qr_unet.model(in_frames.float())
        qr_unet_preds = torch.cumsum(qr_unet_preds, dim=1)
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

    metrics_filename_suffix = f"{subset}_{time_horizon}min"
    if debug:
        metrics_filename_suffix += "_debug"
    metrics_path = os.path.join("results", f"metrics_{metrics_filename_suffix}.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("Calculating mean metrics...")
    logger.info(f"Mean metrics for {subset} subset, time horizon: {time_horizon} min")
    for model_name, model_metrics in metrics.items():
        logger.info(f"--- {model_name} ---")
        for metric_name, metric_values in model_metrics.items():
            if metric_values:
                mean_value = np.mean(metric_values)
                logger.info(f"  {metric_name}: {mean_value:.4f}")
            else:
                logger.info(f"  {metric_name}: N/A (no values)")


if __name__ == "__main__":
    fire.Fire(main)
