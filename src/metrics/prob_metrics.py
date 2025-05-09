import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict


def logscore_bin_fn(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epsilon: Optional[float] = 1e-12,
    divide_by_bin_width: bool = False,
    bin_width: float = 0.1,
    nan_mask: Optional[torch.Tensor] = None,
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

    if nan_mask is not None and nan_mask.dim() == 2:
        nan_mask = nan_mask.unsqueeze(0)

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
    selected_probs = selected_probs.squeeze(1)  # Shape: [BS, H, W]

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

    if nan_mask is None:
        log_scores = torch.mean(-torch.log(probs_for_log))
    else:
        log_scores = torch.mean(-torch.log(probs_for_log)[~nan_mask])

    return log_scores


def calculate_reliability_diagram_data(
    all_predicted_probs, all_actual_outcomes, n_reliability_bins=10
):
    all_predicted_probs = np.asarray(all_predicted_probs)
    all_actual_outcomes = np.asarray(all_actual_outcomes)

    bin_boundaries = np.linspace(0, 1, n_reliability_bins + 1)

    mean_predicted_probs_curve = []
    observed_frequencies_curve = []
    hist_bin_centers = []
    hist_bin_counts = np.zeros(n_reliability_bins, dtype=int)

    for i in range(n_reliability_bins):
        lower_bound = bin_boundaries[i]
        upper_bound = bin_boundaries[i + 1]

        hist_bin_centers.append((lower_bound + upper_bound) / 2.0)

        if i == n_reliability_bins - 1:
            in_bin_mask = (all_predicted_probs >= lower_bound) & (
                all_predicted_probs <= upper_bound
            )
        else:
            in_bin_mask = (all_predicted_probs >= lower_bound) & (
                all_predicted_probs < upper_bound
            )

        probs_this_bin = all_predicted_probs[in_bin_mask]
        outcomes_this_bin = all_actual_outcomes[in_bin_mask]

        current_bin_count = len(probs_this_bin)
        hist_bin_counts[i] = current_bin_count

        if current_bin_count > 0:
            mean_predicted_probs_curve.append(np.mean(probs_this_bin))
            observed_frequencies_curve.append(np.mean(outcomes_this_bin))

    return (
        mean_predicted_probs_curve,
        observed_frequencies_curve,
        hist_bin_centers,
        hist_bin_counts,
    )


def plot_reliability_diagram(
    mean_predicted_probs,
    observed_frequencies,
    hist_bin_centers,
    hist_bin_counts,
    model_name="Model",
    filename="reliability_diagram.png",
):
    plt.figure(figsize=(10, 10))

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    if mean_predicted_probs:
        ax1.plot(
            mean_predicted_probs,
            observed_frequencies,
            "s-",
            label=model_name,
            color="blue",
        )
    ax1.set_xlabel("Mean Predicted Probability (Confidence)")
    ax1.set_ylabel("Observed Frequency (Accuracy)")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f"Reliability Diagram for {model_name}")
    ax1.grid(True)

    ax2 = plt.subplot2grid((3, 1), (2, 0))
    if len(hist_bin_centers) > 0:
        bar_width = (
            (hist_bin_centers[0] - 0.0) * 2 * 0.9
            if len(hist_bin_centers) == 1
            else (hist_bin_centers[1] - hist_bin_centers[0]) * 0.9
        )
        ax2.bar(
            hist_bin_centers,
            hist_bin_counts,
            width=bar_width,
            edgecolor="black",
            color="lightblue",
        )

    ax2.set_xlabel("Predicted Probability Bins")
    ax2.set_ylabel("Count")
    if np.any(hist_bin_counts > 0):
        ax2.set_yscale("log")

    if len(hist_bin_centers) > 20:
        tick_skip = max(1, len(hist_bin_centers) // 10)
        ax2.set_xticks(hist_bin_centers[::tick_skip])
        ax2.set_xticklabels(
            [f"{c:.2f}" for c in hist_bin_centers[::tick_skip]], rotation=45, ha="right"
        )
    elif len(hist_bin_centers) > 0:
        ax2.set_xticks(hist_bin_centers)
        ax2.set_xticklabels(
            [f"{c:.2f}" for c in hist_bin_centers], rotation=45, ha="right"
        )

    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Reliability diagram saved to {filename}")
    plt.close()


def collect_reliability_diagram_data(
    model_name: str,
    predicted_probs: torch.Tensor,
    actual_outcomes: torch.Tensor,
    reliability_diagram: Dict[str, Dict[str, list]],
):
    if isinstance(predicted_probs, torch.Tensor):
        predicted_probs_np = predicted_probs.cpu().numpy()
    else:
        predicted_probs_np = predicted_probs

    actual_outcomes_np = actual_outcomes.cpu().numpy()
    B, C, H_img, W_img = predicted_probs_np.shape

    actual_bin_idx = int(actual_outcomes_np[0, H_img//2, W_img//2])

    for c_idx in range(C):
        predicted_prob = predicted_probs_np[0, c_idx, H_img//2, W_img//2]
        actual_outcome = 1 if c_idx == actual_bin_idx else 0
        reliability_diagram[model_name]["predicted_probs"].append(predicted_prob)
        reliability_diagram[model_name]["actual_outcomes"].append(actual_outcome)

    return reliability_diagram
