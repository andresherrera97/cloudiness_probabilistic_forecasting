import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Optional


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
        nan_mask = nan_mask.unsqueeze(0).unsqueeze(0)

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


def calculate_reliability_diagram(
    predictions,
    observations,
    quantile_levels=None,
    plot=True,
    surrogate_method=False,
    n_bootstraps=1000,
    truncation_point=None,
    confidence_level=0.9,
):
    """
    Calculate and optionally plot the reliability diagram for a quantile regressor model.

    Parameters:
    -----------
    predictions : dict or np.ndarray
        If dict: Keys are quantile levels (0-1), values are arrays of quantile predictions
        If array: Shape (n_samples, n_quantiles) with quantile predictions
    observations : np.ndarray
        Target observations with values in range [0, 1]
    quantile_levels : list or np.ndarray, optional
        Levels of the quantiles if predictions is an array. If None and predictions is an array,
        quantile levels are assumed to be evenly spaced from 0.05 to 0.95
    plot : bool, default=True
        Whether to plot the reliability diagram
    surrogate_method : bool, default=False
        Whether to generate consistency bars using surrogate method from the paper
    n_bootstraps : int, default=1000
        Number of bootstrap samples for consistency bars
    truncation_point : int, optional
        Truncation point M for smooth spectrum estimation in surrogate method.
        If None, uses 2*sqrt(N) as suggested in the paper
    confidence_level : float, default=0.9
        Confidence level for consistency bars (1-beta in the paper)

    Returns:
    --------
    dict:
        'nominal': Nominal quantile levels
        'observed': Observed proportions for each quantile level
        'consistency_bars': If surrogate_method=True, tuple of (lower, upper) bounds
    """
    # Convert predictions to dictionary if provided as array
    if not isinstance(predictions, dict):
        if quantile_levels is None:
            # Default evenly spaced quantiles from 0.05 to 0.95
            quantile_levels = np.linspace(0.05, 0.95, predictions.shape[1])

        predictions_dict = {
            alpha: predictions[:, i] for i, alpha in enumerate(quantile_levels)
        }
    else:
        predictions_dict = predictions
        quantile_levels = sorted(predictions_dict.keys())

    n_samples = len(observations)
    observed_proportions = []

    # Calculate observed proportions for each quantile level
    for alpha in quantile_levels:
        # Indicator variable: 1 if observation is below quantile prediction
        indicators = (observations < predictions_dict[alpha]).astype(int)
        # Observed proportion is the mean of indicators
        observed_prop = np.mean(indicators)
        observed_proportions.append(observed_prop)

    # Calculate consistency bars if required
    consistency_bars = None
    if surrogate_method and n_samples > 2:
        # Compute probability integral transforms (PIT)
        pit_values = np.zeros(n_samples)

        # For each observation, find its position in the predictive distribution
        # This is an approximation as we only have discrete quantile levels
        for i, obs in enumerate(observations):
            # Find where observation falls in predicted quantiles
            quantile_preds = [predictions_dict[q][i] for q in quantile_levels]
            if obs <= quantile_preds[0]:
                pit_values[i] = 0
            elif obs >= quantile_preds[-1]:
                pit_values[i] = 1
            else:
                # Linear interpolation between quantile levels
                for j in range(len(quantile_levels) - 1):
                    if quantile_preds[j] <= obs < quantile_preds[j + 1]:
                        q_low, q_high = quantile_levels[j], quantile_levels[j + 1]
                        pred_low, pred_high = quantile_preds[j], quantile_preds[j + 1]

                        # Linear interpolation
                        t = (
                            (obs - pred_low) / (pred_high - pred_low)
                            if pred_high > pred_low
                            else 0
                        )
                        pit_values[i] = q_low + t * (q_high - q_low)
                        break

        # Convert to standard Gaussian using inverse normal CDF
        # Avoid boundary values (0 or 1) which would give -inf or inf
        pit_values = np.clip(pit_values, 0.001, 0.999)
        z_values = norm.ppf(pit_values)

        # Set default truncation point if not provided
        if truncation_point is None:
            truncation_point = int(2 * np.sqrt(n_samples))

        # Generate consistency bars using surrogate method
        lower_bounds, upper_bounds = surrogate_consistency_bars(
            z_values, quantile_levels, n_bootstraps, truncation_point, confidence_level
        )

        consistency_bars = (lower_bounds, upper_bounds)

    # Create the plot if requested
    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        plt.plot(
            quantile_levels, observed_proportions, "bo-", label="Observed proportion"
        )

        if consistency_bars is not None:
            lower, upper = consistency_bars
            plt.fill_between(
                quantile_levels,
                lower,
                upper,
                alpha=0.2,
                color="b",
                label=f"{confidence_level*100:.0f}% consistency bars",
            )

        plt.xlabel("Nominal proportion")
        plt.ylabel("Observed proportion")
        plt.title("Reliability Diagram")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    # Return the results
    result = {
        "nominal": np.array(quantile_levels),
        "observed": np.array(observed_proportions),
    }

    if consistency_bars is not None:
        result["consistency_bars"] = consistency_bars

    return result


def surrogate_consistency_bars(
    z_values,
    quantile_levels,
    n_bootstraps=1000,
    truncation_point=None,
    confidence_level=0.9,
):
    """
    Generate consistency bars using the surrogate consistency resampling method
    described in the paper.

    Parameters:
    -----------
    z_values : np.ndarray
        Inverse normal transformed PIT values
    quantile_levels : list or np.ndarray
        Nominal quantile levels
    n_bootstraps : int
        Number of bootstrap samples
    truncation_point : int
        Truncation point M for smooth spectrum estimation
    confidence_level : float
        Confidence level (1-beta in the paper)

    Returns:
    --------
    tuple:
        (lower_bounds, upper_bounds) for each quantile level
    """
    from scipy import signal

    n_samples = len(z_values)
    beta = 1 - confidence_level

    # Estimate smooth spectrum with lag window
    # Compute autocorrelation
    acf = (
        np.correlate(
            z_values - np.mean(z_values), z_values - np.mean(z_values), mode="full"
        )
        / np.var(z_values)
        / n_samples
    )

    # Keep only the second half (positive lags)
    acf = acf[n_samples - 1 :]

    # Apply Tukey-Hanning window (a=0.25)
    window = np.zeros_like(acf)
    for k in range(min(truncation_point, len(acf))):
        window[k] = 1 - 0.5 + 0.5 * np.cos(np.pi * k / truncation_point)

    windowed_acf = acf * window

    # Compute smooth spectrum
    freqs = np.fft.rfftfreq(2 * n_samples - 1)
    smooth_spectrum = np.fft.rfft(windowed_acf)

    # Generate surrogate time series and compute observed proportions
    surrogate_proportions = []

    for _ in range(n_bootstraps):
        # Generate surrogate periodogram
        surrogate_spectrum = np.zeros_like(smooth_spectrum, dtype=complex)

        # Generate phases randomly
        phases = np.random.uniform(0, 2 * np.pi, len(smooth_spectrum))

        # Set amplitude from smooth spectrum, with random phase
        surrogate_spectrum = np.abs(smooth_spectrum) * np.exp(1j * phases)

        # Generate surrogate time series
        surrogate_acf = np.fft.irfft(surrogate_spectrum)

        # Create AR model coefficients from ACF (Yule-Walker equations)
        ar_order = min(20, truncation_point // 2)  # Practical limit for AR order
        ar_coeffs = np.zeros(ar_order)
        for i in range(ar_order):
            ar_coeffs[i] = surrogate_acf[i + 1]  # Skip lag 0

        # Generate surrogate time series using AR model
        surrogate_z = np.random.normal(0, 1, n_samples)
        for i in range(ar_order, n_samples):
            for j in range(ar_order):
                surrogate_z[i] -= ar_coeffs[j] * surrogate_z[i - j - 1]

        # Normalize
        surrogate_z = (surrogate_z - np.mean(surrogate_z)) / np.std(surrogate_z)

        # Transform back to U[0,1]
        surrogate_pit = norm.cdf(surrogate_z)

        # Calculate observed proportions for each quantile level
        obs_props = []
        for alpha in quantile_levels:
            indicators = (surrogate_pit < alpha).astype(int)
            obs_props.append(np.mean(indicators))

        surrogate_proportions.append(obs_props)

    # Calculate bounds
    surrogate_proportions = np.array(surrogate_proportions)
    lower_bounds = np.percentile(surrogate_proportions, beta / 2 * 100, axis=0)
    upper_bounds = np.percentile(surrogate_proportions, (1 - beta / 2) * 100, axis=0)

    return lower_bounds, upper_bounds


if __name__ == "__main__":
    # Example with synthetic data
    n_samples = 500
    true_observations = np.random.uniform(0, 1, n_samples)

    # Simulate quantile predictions from a well-calibrated model with some noise
    quantile_levels = np.arange(0.05, 1.0, 0.05)
    predictions = {}

    for q in quantile_levels:
        # Perfect calibration would be the q-th quantile of the true distribution
        predictions[q] = np.random.normal(q, 0.05, n_samples)
        predictions[q] = np.clip(predictions[q], 0, 1)  # Keep in [0,1] range

    # Calculate and plot reliability diagram
    result = calculate_reliability_diagram(
        predictions, true_observations,
        surrogate_method=True,
        n_bootstraps=1000,
        truncation_point=40
    )
