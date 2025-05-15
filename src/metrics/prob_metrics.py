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


def calculate_reliability_diagram_coordinates(tau_values):
    """
    Calculates the coordinates for a reliability diagram.

    The reliability diagram is a visual tool to assess the calibration of
    probabilistic predictions. It plots the sorted probabilistic predictions (τ)
    against their empirical cumulative distribution function (CDF) values,
    which under perfect calibration should follow the identity line.

    According to the Probability Integral Transform (PIT) theorem, if F is the
    true cumulative distribution function (CDF) of a random variable Y, then
    τ = F(Y) should be uniformly distributed on [0, 1]. The reliability
    diagram leverages this by plotting the sorted τ values against their expected
    cumulative probabilities under uniformity (i/n).

    Args:
        tau_values (list or np.ndarray): A collection of τ_i values, where
                                         τ_i = P(Y <= y_i | x_i). These are
                                         the probabilistic predictions from
                                         the model, representing the CDF
                                         evaluated at the observed y_i.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - sorted_tau_values (np.ndarray): The τ values sorted in ascending order.
                                             These are the x-coordinates for the plot.
            - empirical_cdf_values (np.ndarray): The corresponding empirical CDF values (i/n).
                                                These are the y-coordinates for the plot.
                                                Returns (None, None) if input is empty.
    """
    if not isinstance(tau_values, (list, np.ndarray)):
        raise TypeError("Input tau_values must be a list or NumPy array.")

    if len(tau_values) == 0:
        print("Warning: tau_values is empty. Returning (None, None).")
        return None, None

    # Convert to NumPy array for efficient computation
    tau_values = np.asarray(tau_values)

    # Check if values are within [0, 1] - typical for probabilities/CDFs
    if np.any(tau_values < 0) or np.any(tau_values > 1):
        print("Warning: Some tau_values are outside the [0, 1] interval. "
              "Ensure these are valid CDF values.")

    # Sort the τ values in ascending order
    # These are the τ(i) from the description
    sorted_tau_values = np.sort(tau_values)

    # Get the number of data points
    n = len(sorted_tau_values)

    # Calculate the empirical CDF values (i/n for i=1 to n)
    # These are the i/n from the description, where i is the rank
    empirical_cdf_values = np.arange(1, n + 1) / n

    return sorted_tau_values, empirical_cdf_values


def plot_reliability_diagram_CDF(sorted_tau_values, empirical_cdf_values, title="Reliability Diagram"):
    """
    Plots the reliability diagram.

    Args:
        sorted_tau_values (np.ndarray): The sorted τ values (x-coordinates).
        empirical_cdf_values (np.ndarray): The empirical CDF values (y-coordinates).
        title (str): The title of the plot.
    """
    if sorted_tau_values is None or empirical_cdf_values is None:
        print("Cannot plot: input data is None.")
        return

    plt.figure(figsize=(7, 7))
    # Plot the reliability curve (scatter plot of (τ(i), i/n))
    # plt.scatter(sorted_tau_values, empirical_cdf_values, label='Model Reliability', color='blue', s=10)
    plt.scatter(sorted_tau_values, empirical_cdf_values, label='Model Reliability', color='blue', s=10)

    # Plot the diagonal line for perfect calibration (y=x)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfect Calibration (Identity)')

    plt.xlabel(r'Sorted Probabilistic Predictions ($\tau_{(i)}$)')
    plt.ylabel(r'Empirical CDF ($i/n$)')
    plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"results/{title.replace(' ', '_').lower()}.png")
    print(f"Reliability diagram saved to {title.replace(' ', '_').lower()}.png")
    plt.close()


if __name__ == '__main__':
    # --- Example Usage ---

    # Simulate some τ values (e.g., from a model's probabilistic predictions)
    # A perfectly calibrated model would have τ values uniformly distributed.
    np.random.seed(42) # for reproducibility
    # Example 1: Well-calibrated model (τ values are roughly uniform)
    tau_values_well_calibrated = np.random.uniform(0, 1, 100)

    # Example 2: Poorly-calibrated model (e.g., overconfident, τ values skewed towards 0 or 1)
    # Simulating overconfidence towards lower probabilities
    tau_values_overconfident_low = np.random.beta(0.5, 2, 100)

    # Simulating overconfidence towards higher probabilities
    tau_values_overconfident_high = np.random.beta(2, 0.5, 100)

    # Example 3: Underconfident model (τ values clustered around 0.5)
    tau_values_underconfident = np.random.beta(5, 5, 100)


    print("--- Well-Calibrated Model Example ---")
    sorted_tau_wc, ecdf_wc = calculate_reliability_diagram_coordinates(tau_values_well_calibrated)
    if sorted_tau_wc is not None:
        print(f"First 5 sorted tau values: {sorted_tau_wc[:5]}")
        print(f"First 5 empirical CDF values: {ecdf_wc[:5]}")
        plot_reliability_diagram_CDF(sorted_tau_wc, ecdf_wc, title="Reliability Diagram (Well-Calibrated Example)")

    print("\n--- Overconfident (Low) Model Example ---")
    sorted_tau_ol, ecdf_ol = calculate_reliability_diagram_coordinates(tau_values_overconfident_low)
    if sorted_tau_ol is not None:
        plot_reliability_diagram_CDF(sorted_tau_ol, ecdf_ol, title="Reliability Diagram (Overconfident - Low Bias)")

    print("\n--- Overconfident (High) Model Example ---")
    sorted_tau_oh, ecdf_oh = calculate_reliability_diagram_coordinates(tau_values_overconfident_high)
    if sorted_tau_oh is not None:
        plot_reliability_diagram_CDF(sorted_tau_oh, ecdf_oh, title="Reliability Diagram (Overconfident - High Bias)")

    print("\n--- Underconfident Model Example ---")
    sorted_tau_uc, ecdf_uc = calculate_reliability_diagram_coordinates(tau_values_underconfident)
    if sorted_tau_uc is not None:
        plot_reliability_diagram_CDF(sorted_tau_uc, ecdf_uc, title="Reliability Diagram (Underconfident Example)")

    # --- Example with a small number of points as in the thought process ---
    print("\n--- Small Example from Description ---")
    tau_values_small = [0.1, 0.8, 0.4, 0.6]
    sorted_tau_small, ecdf_small = calculate_reliability_diagram_coordinates(tau_values_small)
    if sorted_tau_small is not None:
        print(f"Input taus: {tau_values_small}")
        print(f"Sorted taus (x-coordinates): {sorted_tau_small}")
        print(f"Empirical CDF (y-coordinates): {ecdf_small}")
        # Expected: ([0.1, 0.4, 0.6, 0.8], [0.25, 0.50, 0.75, 1.00])
        plot_reliability_diagram_CDF(sorted_tau_small, ecdf_small, title="Reliability Diagram (Small Example)")

    # --- Example with empty input ---
    print("\n--- Empty Input Example ---")
    tau_empty = []
    sorted_tau_empty, ecdf_empty = calculate_reliability_diagram_coordinates(tau_empty)
    # This should print a warning and return (None, None)

    # --- Example with invalid input type ---
    print("\n--- Invalid Input Type Example ---")
    try:
        calculate_reliability_diagram_coordinates("not a list")
    except TypeError as e:
        print(f"Caught expected error: {e}")
