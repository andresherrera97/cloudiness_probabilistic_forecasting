import fire
import torch
import logging
import json
from typing import Dict, Optional
import numpy as np

from models import (
    QuantileRegressorUNet,
    UNetConfig,
    BinClassifierUNet,
    MedianScaleUNet,
    IQUNetPipeline,
)
from metrics import crps_laplace
from postprocessing.transform import quantile_2_bin


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Train Script")


def logscore_bin_fn(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epsilon: Optional[float] = 1e-12,
    # divide_by_bin_width: bool = True,
    divide_by_bin_width: bool = False,
    bin_width: float = 0.1,
) -> torch.Tensor:
    """
    Calculates the element-wise log score for binarized predictions using PyTorch.

    Log Score = log(probability assigned by the model to the correct bin).

    Args:
        predictions (torch.Tensor): Tensor of predicted probabilities.
                                     Shape: [BS, NUM_BINS, H, W].
                                     Values should represent probabilities for each bin.
                                     Requires float dtype.
        targets (torch.Tensor): Tensor of ground truth target bin indices.
                                 Shape: [BS, H, W]. Values must be integers
                                 in the range [0, NUM_BINS-1], indicating the
                                 correct bin index for each pixel.
                                 Requires integer dtype (e.g., torch.long).
        epsilon (Optional[float]): A small value added to the selected probabilities
                                   before taking the logarithm. This helps avoid
                                   log(0) = -infinity if a probability is exactly zero.
                                   Set to None to disable. Defaults to 1e-9.

    Returns:
        torch.Tensor: Tensor containing the log score for each pixel.
                      Shape: [BS, H, W]. Values will be <= 0.
                      Higher values (closer to 0) indicate better predictions.

    Raises:
        ValueError: If input shapes are incompatible or dtypes are incorrect.
    """
    # --- Input Validation ---
    if predictions.dim() != 4:
        raise ValueError(f"predictions must be 4D [BS, NUM_BINS, H, W], but got shape {predictions.shape}")
    if targets.dim() != 3:
        raise ValueError(f"targets must be 3D [BS, H, W], but got shape {targets.shape}")
    if not torch.is_floating_point(predictions):
        raise ValueError(f"predictions tensor must have a floating-point dtype, got {predictions.dtype}")
    elif targets.dtype != torch.long:
        # Ensure it's long for gather compatibility if it's another int type
        targets = targets.long()

    bs, num_bins, h, w = predictions.shape
    if targets.shape[0] != bs or targets.shape[1] != h or targets.shape[2] != w:
        raise ValueError(f"Shape mismatch: predictions shape {predictions.shape} "
                         f"and targets shape {targets.shape} are incompatible.")

    # Check if target values are within the valid range
    if torch.min(targets) < 0 or torch.max(targets) >= num_bins:
         raise ValueError(f"Target values must be in the range [0, {num_bins-1}]")

    # --- Log Score Calculation ---

    # Reshape targets to align with predictions for gather
    # We need targets to be [BS, 1, H, W] to select along the NUM_BINS dimension (dim=1)
    # The values in targets_expanded will indicate *which* bin index to pick.
    targets_expanded = targets.unsqueeze(1)  # Shape: [BS, 1, H, W]

    # Use torch.gather to select the probability of the correct bin for each pixel
    # Input: predictions [BS, NUM_BINS, H, W]
    # Dim: 1 (the NUM_BINS dimension)
    # Index: targets_expanded [BS, 1, H, W]
    # Output: selected_probs [BS, 1, H, W]
    selected_probs = torch.gather(predictions, 1, targets_expanded)

    # Remove the singleton dimension (dim=1)
    selected_probs = selected_probs.squeeze(1) # Shape: [BS, H, W]

    # Add epsilon for numerical stability (avoid log(0))
    if epsilon is not None:
        # Clamp probability first to avoid potential issues if prob > 1
        # Although probabilities shouldn't exceed 1, numerical precision might cause it.
        # This step is optional but can add robustness.
        # selected_probs = torch.clamp(selected_probs, min=0.0, max=1.0)
        probs_for_log = selected_probs + epsilon
    else:
        probs_for_log = selected_probs

    if divide_by_bin_width:
        # Divide by bin width if specified
        # This is done to normalize the log score
        # The bin width is assumed to be 0.1 as per the original code
        # Adjust this value if your bin width is different
        if bin_width <= 0:
            raise ValueError("bin_width must be positive.")
        probs_for_log = probs_for_log / bin_width

    # Calculate the log score
    # log(p) where p is the probability assigned to the true outcome
    log_scores = torch.mean(-torch.log(probs_for_log))

    return log_scores


def get_checkpoint_path(time_horizon: int) -> Dict[str, str]:
    if time_horizon == 60:
        return {
            "qr": "checkpoints/salto_down/prob_60min_salto_512/qr/QRUNet_IN3_NB9_F32_SC0_PDTrue_BS_8_TH60_E6_BVM0_04_D2025-04-13_21:52.pt",
            "bin": "checkpoints/salto_down/prob_60min_salto_512/bin/BinUNet_IN3_NB10_F32_SC0_BS_8_TH60_E8_BVM1_78_D2025-04-24_20:13.pt",
            "laplace": "checkpoints/salto_down/prob_60min_salto_512/laplace/MedianScaleUNet_IN3_F32_SC0_BS_8_TH60_E7_BVM0_40_D2025-04-25_13:36.pt",
            "iqn": "checkpoints/salto_down/prob_60min_salto_512/iqn/IQUNet_IN3_F32_NT9_CED64_PD0_BS_8_TH60_E7_BVM0_23_D2025-04-26_01:28.pt",
        }
    elif time_horizon == 120:
        return {
            "qr": "checkpoints/salto_down/prob_120min_salto_512/qr/QRUNet_IN3_NB9_F32_SC0_PDTrue_BS_8_TH120_E8_BVM0_04_D2025-04-13_18:26.pt",
            "bin": "checkpoints/salto_down/prob_120min_salto_512/bin/BinUNet_IN3_NB10_F32_SC0_BS_8_TH120_E9_BVM1_82_D2025-04-24_20:35.pt",
            "laplace": "checkpoints/salto_down/prob_120min_salto_512/laplace/MedianScaleUNet_IN3_F32_SC0_BS_8_TH120_E9_BVM0_42_D2025-04-25_09:25.pt",
            "iqn": "checkpoints/salto_down/prob_120min_salto_512/iqn/IQUNet_IN3_F32_NT9_CED64_PD0_BS_8_TH120_E9_BVM0_29_D2025-04-25_22:11.pt",
        }
    else:
        raise ValueError("Invalid time horizon. Supported values are 60 or 120.")


def main(
    time_horizon: int = 60,
    dataset: str = "salto_down",
    dataset_path: str = "datasets/salto_downsample/",
    subset: str = "val",
    debug: bool = False,
):
    logger.info("Starting evaluation script...")
    logger.info(f"Time horizon: {time_horizon} min")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Subset: {subset}")
    logger.info(f"Debug mode: {debug}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"using device: {device}")

    torch.set_grad_enabled(False)

    num_bins = 10
    quantiles = np.linspace(0, 1, num_bins + 1)[1:-1]

    models_paths = get_checkpoint_path(time_horizon)

    # Load models
    unet_config = UNetConfig(
        in_frames=3,
        spatial_context=0,
        filters=32,
        output_activation="sigmoid",
        device=device,
    )

    qr_unet = QuantileRegressorUNet(
        config=unet_config,
        quantiles=quantiles,
        predict_diff=True,
    )

    qr_unet.load_checkpoint(
        checkpoint_path=models_paths["qr"],
        device=device,
    )

    bin_unet = BinClassifierUNet(
        config=unet_config,
        n_bins=num_bins,
    )

    bin_unet.load_checkpoint(
        checkpoint_path=models_paths["bin"],
        device=device,
    )

    laplace_unet = MedianScaleUNet(unet_config)

    laplace_unet.load_checkpoint(
        checkpoint_path=models_paths["laplace"],
        device=device,
    )

    iqn_unet = IQUNetPipeline(
        config=unet_config,
    )

    iqn_unet.load_checkpoint(
        checkpoint_path=models_paths["iqn"],
        device=device,
    )

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

    # run evaluation
    if subset == "train":
        data_loader = bin_unet.train_loader
    elif subset == "val":
        data_loader = bin_unet.val_loader
    elif subset == "test":
        data_loader = bin_unet.test_loader
    else:
        raise ValueError(
            "Invalid subset. Supported values are 'train', 'val', or 'test'."
        )

    metrics = {
        "qr": {
            "crps": [],
            "logscore": [],
        },
        "bin": {
            "crps": [],
            "logscore": [],
        },
        "laplace": {
            "crps": [],
            "logscore": [],
        },
        "iqn": {
            "crps": [],
            "logscore": [],
        },
    }

    for batch_idx, (in_frames, (out_frames, bin_output)) in enumerate(data_loader):
        in_frames = in_frames.to(device)
        out_frames = out_frames.to(device)
        bin_output = bin_output.to(device)

        # Quantile Regression UNet
        qr_unet_preds = qr_unet.model(in_frames.float())
        qr_unet_preds = torch.cumsum(qr_unet_preds, dim=1)
        metrics["qr"]["crps"].append(
            qr_unet.crps_loss.crps_loss(
                pred=qr_unet_preds,
                y=out_frames,
            )
            .detach()
            .item()
        )
        qr_binarized_preds = quantile_2_bin(
            quantiles=quantiles,
            quantiles_values=qr_unet_preds,
            num_bins=10,
        )

        metrics["qr"]["logscore"].append(
            logscore_bin_fn(
                torch.tensor(qr_binarized_preds).to(device),
                torch.tensor(bin_output).to(device)
            ).detach().item()
        )

        # Bin Classifier UNet
        bin_unet_preds = bin_unet.model(in_frames.float())

        metrics["bin"]["crps"].append(
            bin_unet.crps_loss.crps_loss(bin_unet_preds, out_frames).detach().item()
        )

        metrics["bin"]["logscore"].append(
            logscore_bin_fn(bin_unet_preds, bin_output).detach().item()
        )

        # Median Scale UNet
        laplace_unet_preds = laplace_unet.model(in_frames.float())
        metrics["laplace"]["crps"].append(
            crps_laplace(out_frames, laplace_unet_preds).detach().item()
        )
        laplace_logscore = laplace_unet.loss_fn(laplace_unet_preds, out_frames).detach().item()
        metrics["laplace"]["logscore"].append(laplace_logscore)

        # IQN UNet
        iqn_unet_pred = iqn_unet.model(in_frames.float(), iqn_unet.val_quantiles)
        iqn_unet_pred = torch.sort(iqn_unet_pred, dim=1)[
            0
        ]  # check how necessary this is

        metrics["iqn"]["crps"].append(
            iqn_unet.crps_loss.crps_loss(
                pred=iqn_unet_pred,
                y=out_frames,
            ).detach().item()
        )
        iqn_binarized_preds = quantile_2_bin(
            quantiles=quantiles,
            quantiles_values=iqn_unet_pred,
            num_bins=10,
        )
        metrics["iqn"]["logscore"].append(
            logscore_bin_fn(
                torch.tensor(iqn_binarized_preds).to(device),
                torch.tensor(bin_output)
            ).detach().item()
        )

        if debug and batch_idx >= 2:
            break

    # Save metrics to JSON file
    if debug:
        metrics_path = f"results/metrics_{subset}_{time_horizon}min_debug.json"
    else:
        metrics_path = f"results/metrics_{subset}_{time_horizon}min.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("Metrics calculated and saved.")
    logger.info("Calculating mean metrics...")
    logger.info("Mean metrics:")
    logger.info(f"Metrics for {subset} subset, time horizon: {time_horizon} min")

    # Calculate mean metrics
    for model_name, model_metrics in metrics.items():
        for metric_name, metric_values in model_metrics.items():
            mean_value = np.mean(metric_values)
            logger.info(f"{model_name} {metric_name}: {mean_value:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
