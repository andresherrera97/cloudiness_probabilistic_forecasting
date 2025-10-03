import os
import torch
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from models import BinClassifierUNet, UNetConfig
from sklearn.metrics import brier_score_loss
import logging
import joblib  # For saving the model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Bin Classifier Isotonic Regression")


def calculate_ece(y_true, y_pred, n_bins=15):
    """
    Calculates the Expected Calibration Error of a model.
    """
    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    binids = np.digitize(y_pred, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        bin_mask = binids == i
        if np.any(bin_mask):
            bin_size = np.sum(bin_mask)

            # Calculate accuracy and confidence for the bin
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_confidence = np.mean(y_pred[bin_mask])

            # ECE contribution from this bin
            ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

    return ece


def main(checkpoint_path: str, crop_size: int, output_path: str):
    output_path = os.path.join(
        output_path, checkpoint_path.split("/")[-1].replace(".pt", "")
    )
    os.makedirs(output_path, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"using device: {device}")
    torch.set_grad_enabled(False)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    logger.info(f"in_frames: {checkpoint['num_input_frames']}")
    logger.info(f"filters: {checkpoint['num_filters']}")
    logger.info(f"n_bins: {checkpoint['num_bins']}")
    logger.info(f"spatial_context: {checkpoint['spatial_context']}")
    logger.info(f"time_horizon: {checkpoint['time_horizon']}")
    logger.info(f"Output activation: {checkpoint['output_activation']}")

    unet_config = UNetConfig(
        in_frames=3,
        spatial_context=0,
        filters=32,
        output_activation=checkpoint["output_activation"],
        device=device,
    )

    bin_unet = BinClassifierUNet(
        config=unet_config,
    )

    bin_unet.load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    bin_unet.create_dataloaders(
        dataset="salto_down",
        path="datasets/salto_downsample",
        batch_size=1,
        time_horizon=checkpoint["time_horizon"],
        binarization_method="one_hot_encoding",
        create_test_loader=False,
        shuffle=False,
        drop_last=False,
    )

    crop_y_start = (512 - crop_size) // 2
    crop_y_end = crop_y_start + crop_size
    crop_x_start = (512 - crop_size) // 2
    crop_x_end = crop_x_start + crop_size
    logger.info(
        f"Using crop: y({crop_y_start}:{crop_y_end}), x({crop_x_start}:{crop_x_end})"
    )

    predictions = []
    ground_truths = []

    for val_batch_idx, (in_frames, (out_frames, bin_output)) in enumerate(
        bin_unet.train_loader
    ):
        in_frames = in_frames.to(device)
        bin_output = bin_output.to(device)
        bin_unet_preds = bin_unet.predict(in_frames[:, -3:, :, :].float())

        out_frames_crop = bin_output[
            :, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end
        ]
        bin_unet_preds_crop = bin_unet_preds[
            :, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end
        ]
        predictions.append(bin_unet_preds_crop.flatten().cpu().numpy())
        ground_truths.append(out_frames_crop.flatten().cpu().numpy())

    ground_truths_flatten = np.asarray(ground_truths).flatten()
    predictions_flatten = np.asarray(predictions).flatten()

    # --- 2. Train the Isotonic Regression Calibrator ---
    # This is the key step. We fit the calibrator on our raw predictions and true labels.

    logger.info("\nStep 2: Training the Isotonic Regression calibrator...")
    # We use y_min and y_max to ensure output is always a valid probability.
    # `out_of_bounds='clip'` handles cases where test data is outside the range seen in training.
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")

    # Fit the model: iso_reg learns the mapping from raw_predictions -> true_labels
    iso_reg.fit(predictions_flatten, ground_truths_flatten)
    logger.info("Calibrator trained successfully.")
    joblib.dump(iso_reg, f"{output_path}/isotonic_regressor_model.joblib")

    # --- 3. Apply the Calibrator to the Predictions ---
    # Use the trained calibrator to get corrected probabilities.
    predictions_test = []
    ground_truths_test = []

    for val_batch_idx, (in_frames, (out_frames, bin_output)) in enumerate(
        bin_unet.val_loader
    ):
        in_frames = in_frames.to(device)
        bin_output = bin_output.to(device)
        bin_unet_preds = bin_unet.predict(in_frames[:, -3:, :, :].float())

        out_frames_crop = bin_output[
            :, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end
        ]
        bin_unet_preds_crop = bin_unet_preds[
            :, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end
        ]
        predictions_test.append(bin_unet_preds_crop.flatten().cpu().numpy())
        ground_truths_test.append(out_frames_crop.flatten().cpu().numpy())

    ground_truths_test_flatten = np.asarray(ground_truths_test).flatten()
    predictions_test_flatten = np.asarray(predictions_test).flatten()

    logger.info("\nStep 3: Applying calibration to the raw predictions...")
    calibrated_predictions = iso_reg.predict(predictions_test_flatten)

    # --- 4. Visualize the Calibration Improvement (Reliability Diagram) ---
    # A reliability diagram plots the actual frequency of positives against the predicted probability.
    # For a perfectly calibrated model, the plot should be on the y=x diagonal.

    logger.info("\nStep 4: Generating reliability diagrams to show improvement...")

    # Calculate calibration curve for the raw, uncalibrated model
    prob_true_raw, prob_pred_raw = calibration_curve(
        ground_truths_test_flatten, predictions_test_flatten, n_bins=15
    )

    # Calculate calibration curve for the calibrated model
    prob_true_cal, prob_pred_cal = calibration_curve(
        ground_truths_test_flatten, calibrated_predictions, n_bins=15
    )

    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Plot the raw calibration curve
    ax.plot(
        prob_pred_raw,
        prob_true_raw,
        "s-",
        label="Uncalibrated U-Net",
        color="red",
        alpha=0.8,
    )

    # Plot the calibrated calibration curve
    ax.plot(
        prob_pred_cal,
        prob_true_cal,
        "s-",
        label="Calibrated (Isotonic)",
        color="blue",
        alpha=0.8,
    )

    # Plot the perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

    ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=16)
    ax.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax.legend(loc="best")
    ax.grid(True)
    # plt.show()
    plt.savefig(
        os.path.join(output_path, "calibration_plot.png"), dpi=200, bbox_inches="tight"
    )
    plt.close()

    logger.info(
        "Done! The plot shows how Isotonic Regression corrected the model's calibration."
    )

    # --- 1. Calculate metrics for the UNCALIBRATED model ---
    brier_raw = brier_score_loss(ground_truths_test_flatten, predictions_test_flatten)
    ece_raw = calculate_ece(ground_truths_test_flatten, predictions_test_flatten)

    logger.info("--- Uncalibrated Model Metrics ---")
    logger.info(f"Brier Score: {brier_raw:.4f}")
    logger.info(f"Expected Calibration Error (ECE): {ece_raw:.4f}")

    # --- 2. Calculate metrics for the CALIBRATED model ---
    brier_calibrated = brier_score_loss(
        ground_truths_test_flatten, calibrated_predictions
    )
    ece_calibrated = calculate_ece(ground_truths_test_flatten, calibrated_predictions)

    logger.info("\n--- Calibrated Model Metrics ---")
    logger.info(f"Brier Score: {brier_calibrated:.4f}")
    logger.info(f"Expected Calibration Error (ECE): {ece_calibrated:.4f}")

    # --- 3. Print the improvement ---
    logger.info("\n--- Improvement Summary ✅ ---")
    logger.info(
        f"Brier Score Improvement: {brier_raw - brier_calibrated:.4f} (Lower is better)"
    )
    logger.info(f"ECE Improvement: {ece_raw - ece_calibrated:.4f} (Lower is better)")


if __name__ == "__main__":
    main(
        checkpoint_path="checkpoints/salto_down/prob_60min_salto_512/bin/BinUNet_IN3_NB10_F32_SC0_BS_8_TH60_E8_BVM1_83_D2025-05-07_09:53.pt",
        crop_size=4,
        output_path="bin_isotonic_regression/",
    )
