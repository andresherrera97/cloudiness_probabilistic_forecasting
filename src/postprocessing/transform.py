import torch
import numpy as np
from typing import List, Union
import scipy.stats as stats


def bin_2_quantile(
    bin_probabilities: np.ndarray,
    quantiles: List,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.ndarray:
    """
    Calculate the quantiles based on the given 3D bin probabilities for each pixel.

    Parameters:
    - bin_probabilities: 3D numpy array of probabilities with shape (height, width, num_bins).
    - quantiles: list of quantiles to find.

    Returns:
    - 3D numpy array of quantile values with shape (height, width, len(quantiles)).
    """
    # TODO: make this function more efficient.
    # Shape of the input bin probabilities
    if len(bin_probabilities.shape) != 4:
        raise ValueError(
            "The input bin probabilities should have shape (batch_size, num_bins, height, width)"
        )

    batch_size, num_bins, height, width = bin_probabilities.shape

    # Initialize the array to store the quantile values
    quantile_values = np.zeros((batch_size, len(quantiles), height, width))

    # Calculate the cumulative sum of the probabilities along the bins axis
    cumulative_probabilities = np.cumsum(bin_probabilities, axis=1)

    # The bin edges, assuming bins are equally spaced between 0 and 1
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    # Iterate over each pixel to calculate the quantiles
    for n in range(batch_size):
        for i in range(height):
            for j in range(width):
                quantile_values[n, :, i, j] = np.interp(
                    quantiles, cumulative_probabilities[n, :, i, j], bin_edges[1:]
                )

    return quantile_values


def quantile_2_bin_slow(
    quantiles: List,
    quantiles_values: np.ndarray,
    num_bins: int,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.ndarray:
    """
    Calculate the probability of values falling inside specified bins based
    on quantiles.
    """
    if isinstance(quantiles_values, torch.Tensor):
        quantiles_values = quantiles_values.detach().cpu().numpy()

    if len(quantiles_values.shape) == 3:
        quantiles_values = np.expand_dims(quantiles_values, axis=0)

    if len(quantiles_values.shape) != 4:
        raise ValueError(
            "The input quantile values must have shape (batch_size, "
            "num_quantiles, height, width)"
        )

    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    batch_size, _, height, width = quantiles_values.shape

    # Sort quantiles by probability
    sorted_quantiles = sorted(quantiles)

    # Add CDF boundaries (0 and 1) with corresponding min and max values
    cdf_probs = [0] + sorted_quantiles + [1]

    bin_probs = np.zeros((batch_size, len(bin_edges) - 1, height, width))

    # bin_probs = []
    for n in range(batch_size):
        for i in range(height):
            for j in range(width):
                cdf_values = np.array(
                    ([min_value] + list(quantiles_values[n, :, i, j]) + [max_value])
                )
                bin_edges_prob = [0]
                for bin_n, (bin_edge) in enumerate(bin_edges[1:]):
                    if min_value < bin_edge < max_value:
                        range_idx = np.searchsorted(cdf_values, bin_edge, side="left")
                        pend = (cdf_probs[range_idx] - cdf_probs[range_idx - 1]) / (
                            cdf_values[range_idx] - cdf_values[range_idx - 1]
                        )
                        bin_edge_prob = cdf_probs[range_idx - 1] + pend * (
                            bin_edge - cdf_values[range_idx - 1]
                        )
                    elif bin_edge == min_value:
                        bin_edge_prob = 0
                    elif bin_edge == max_value:
                        bin_edge_prob = 1
                    bin_edges_prob.append(bin_edge_prob)

                    # Calculate the probability within the bin range
                    bin_prob = bin_edges_prob[bin_n + 1] - bin_edges_prob[bin_n]
                    bin_probs[n, bin_n, i, j] = bin_prob

    return bin_probs


def quantile_2_bin(
    quantiles: List,
    quantiles_values: Union[np.ndarray, torch.Tensor],
    num_bins: int,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.ndarray:
    """
    Calculate the probability of values falling inside specified bins based
    on quantiles (Optimized Version).

    Args:
        quantiles: List of quantile levels (e.g., [0.1, 0.5, 0.9]). Must be sorted.
        quantiles_values: Array or Tensor of quantile values with shape
                          (batch_size, num_quantiles, height, width) or
                          (num_quantiles, height, width).
        num_bins: Number of output bins.
        min_value: Minimum value for the bins range.
        max_value: Maximum value for the bins range.

    Returns:
        np.ndarray: Probabilities per bin, shape (batch_size, num_bins, height, width).
    """
    if isinstance(quantiles_values, torch.Tensor):
        quantiles_values = quantiles_values.detach().cpu().numpy()

    # Add batch dimension if missing (e.g., shape was Nq, H, W)
    if quantiles_values.ndim == 3:
        quantiles_values = np.expand_dims(quantiles_values, axis=0)
    elif quantiles_values.ndim != 4:
        raise ValueError(
            "The input quantile values must have shape (batch_size, "
            "num_quantiles, height, width) or (num_quantiles, height, width)"
        )

    batch_size, num_quantiles_in, height, width = quantiles_values.shape

    if len(quantiles) != num_quantiles_in:
        raise ValueError(
            f"Length of quantiles ({len(quantiles)}) must match "
            f"quantiles_values.shape[1] ({num_quantiles_in})"
        )

    # --- Precomputation ---
    # Ensure quantiles are sorted (important for interpolation logic)
    # If they might not be, sort them and sort quantiles_values accordingly
    sort_indices = np.argsort(quantiles)
    sorted_quantiles = np.array(quantiles)[sort_indices]
    sorted_quantiles_values = quantiles_values[:, sort_indices, :, :]

    # Add CDF boundaries (0 and 1) with corresponding min and max values
    # cdf_probs: [0, q1, q2, ..., qN, 1]
    cdf_probs = np.concatenate(
        ([0.0], sorted_quantiles, [1.0])
    )  # Shape: (num_quantiles + 2,)

    # Create the full array of CDF values including min/max
    # full_cdf_values: shape (batch_size, num_quantiles + 2, height, width)
    full_cdf_values = np.full(
        (batch_size, num_quantiles_in + 2, height, width),
        min_value,  # Initialize with min_value might not be strictly necessary but ok
        dtype=quantiles_values.dtype,
    )
    full_cdf_values[:, 0, :, :] = min_value
    full_cdf_values[:, 1:-1, :, :] = sorted_quantiles_values
    full_cdf_values[:, -1, :, :] = max_value

    # Ensure monotonicity of CDF values (clip if necessary, handle potential duplicates)
    # This prevents issues in interpolation if input quantiles are weird
    # np.maximum.accumulate ensures values are non-decreasing along the quantile axis
    full_cdf_values = np.maximum.accumulate(full_cdf_values, axis=1)
    # Ensure the last value is at least max_value (can happen if max quantile value < max_value)
    full_cdf_values[:, -1, :, :] = np.maximum(full_cdf_values[:, -1, :, :], max_value)
    # Ensure the first value is at most min_value
    full_cdf_values[:, 0, :, :] = np.minimum(full_cdf_values[:, 0, :, :], min_value)

    # Bin edges: [min_val, edge1, edge2, ..., max_val]
    bin_edges = np.linspace(
        min_value, max_value, num_bins + 1
    )  # Shape: (num_bins + 1,)

    # --- Vectorized Interpolation ---
    # We want to find the probability P(X <= bin_edge) for each bin edge and each pixel.
    # Target shape for interpolated probabilities: (batch_size, num_bins + 1, height, width)
    interpolated_cdf_probs = np.zeros(
        (batch_size, num_bins + 1, height, width),
        dtype=np.float64,  # Use float64 for precision
    )

    # Use broadcasting and np.searchsorted to find the interval indices for all bin edges and pixels
    # Reshape bin_edges for broadcasting: (1, num_bins + 1, 1, 1)
    bin_edges_b = bin_edges.reshape(1, -1, 1, 1)

    # Find indices: For each pixel (b, h, w) and each bin_edge k, find index `idx` such that
    # full_cdf_values[b, idx-1, h, w] <= bin_edges[k] < full_cdf_values[b, idx, h, w]
    # `np.searchsorted` works on the *last* axis if applied correctly, but our data is on axis 1.
    # Alternative: Use broadcasting comparison and sum. This is generally quite efficient.
    # indices shape: (batch_size, num_bins + 1, height, width)
    # Clamp indices to be within the valid range [1, num_quantiles + 1] for accessing cdf_probs and full_cdf_values
    indices = np.sum(full_cdf_values[:, :, np.newaxis, :, :] < bin_edges_b, axis=1)
    indices = np.clip(indices, 1, len(cdf_probs) - 1)

    # Gather the lower and upper bounds for interpolation using the calculated indices
    # Need to reshape indices to use with take_along_axis

    # Gather lower values (at index - 1)
    idx_lower = (indices - 1)[:, :, :]  # (B, 1, num_bins+1, H, W) for axis=1 gather

    # cdf_v_lower = np.take_along_axis(full_cdf_values, idx_lower, axis=1).squeeze(axis=1) # (B, num_bins+1, H, W)
    cdf_v_lower = np.take_along_axis(
        full_cdf_values, idx_lower, axis=1
    )  # (B, num_bins+1, H, W)
    cdf_p_lower = cdf_probs[
        indices - 1
    ]  # Broadcasting works: (Nq+2,)[(B,Nb+1,H,W)] -> (B,Nb+1,H,W)

    # Gather upper values (at index)
    idx_upper = indices[:, :, :]  # (B, 1, num_bins+1, H, W) for axis=1 gather
    cdf_v_upper = np.take_along_axis(
        full_cdf_values, idx_upper, axis=1
    )  # (B, num_bins+1, H, W)
    cdf_p_upper = cdf_probs[indices]  # Broadcasting works

    # --- Perform Linear Interpolation ---
    delta_v = cdf_v_upper - cdf_v_lower
    delta_p = cdf_p_upper - cdf_p_lower

    # Handle division by zero: where delta_v is 0, the slope is 0 if delta_p is also 0,
    # or undefined (can treat as 0 or infinity depending on context, usually 0 is safe here if values are clamped).
    # If delta_v is 0, it means bin_edge falls exactly on a quantile value or between duplicate values.
    # In this case, the interpolated probability should be cdf_p_lower.
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.divide(
            delta_p, delta_v, out=np.zeros_like(delta_p), where=delta_v > 1e-9
        )  # Add epsilon for float safety

    # Calculate interpolated probability: P(X <= bin_edge)
    # interpolated_cdf_probs shape: (batch_size, num_bins + 1, height, width)
    interpolated_cdf_probs = cdf_p_lower + slope * (
        bin_edges_b.squeeze(axis=0) - cdf_v_lower
    )  # Use broadcasted bin_edges

    # Correct probabilities where delta_v was zero (or near zero)
    interpolated_cdf_probs[delta_v <= 1e-9] = cdf_p_lower[delta_v <= 1e-9]

    # Ensure probabilities are within [0, 1] and monotonically increasing (due to float errors)
    interpolated_cdf_probs = np.maximum.accumulate(
        interpolated_cdf_probs, axis=1
    )  # Enforce monotonicity along bin axis
    interpolated_cdf_probs = np.clip(interpolated_cdf_probs, 0.0, 1.0)

    # --- Calculate Bin Probabilities ---
    # Probability for bin k = P(X <= edge_{k+1}) - P(X <= edge_k)
    # Use np.diff along the bin dimension (axis=1)
    bin_probs = np.diff(
        interpolated_cdf_probs, axis=1
    )  # Shape: (batch_size, num_bins, height, width)

    # Final clipping for safety, although diff should preserve range if input is monotonic [0,1]
    bin_probs = np.clip(bin_probs, 0.0, 1.0)

    return bin_probs


def gaussian_2_quantile(
    mean_array: np.ndarray, std_dev_array: np.ndarray, quantiles: List
):
    """
    Compute quantiles for a Gaussian distribution.

    Parameters:
    mean (float): The mean of the Gaussian distribution.
    std_dev (float): The standard deviation of the Gaussian distribution.
    quantiles (list of float): A list of desired quantiles (between 0 and 1).

    Returns:
    list of float: The values corresponding to the given quantiles.
    """
    if len(mean_array.shape) != 4:
        raise ValueError(
            "The input quantile values must have shape (batch_size, "
            "1, height, width)"
        )

    if len(std_dev_array.shape) != 4:
        raise ValueError(
            "The input quantile values must have shape (batch_size, "
            "1, height, width)"
        )

    batch_size, _, height, width = mean_array.shape
    quantiles_values = np.zeros((batch_size, len(quantiles), height, width))

    for n in range(batch_size):
        for i in range(height):
            for j in range(width):
                mean = mean_array[n, 0, i, j]
                std_dev = std_dev_array[n, 0, i, j]
                # Create a normal distribution with the given mean and standard deviation
                distribution = stats.norm(loc=mean, scale=std_dev)

                # Compute the quantiles
                quantiles_values[n, :, i, j] = distribution.ppf(quantiles)

    return quantiles_values


def gaussian_2_bin(
    mean_array: np.ndarray,
    std_dev_array: np.ndarray,
    num_bins: int,
    min_value: float = 0.0,
    max_value: float = 1.0,
):
    """
    Compute quantiles for a Gaussian distribution.

    Parameters:
    mean (float): The mean of the Gaussian distribution.
    std_dev (float): The standard deviation of the Gaussian distribution.
    quantiles (list of float): A list of desired quantiles (between 0 and 1).

    Returns:
    list of float: The values corresponding to the given quantiles.
    """
    if len(mean_array.shape) != 4:
        raise ValueError(
            "The input quantile values must have shape (batch_size, "
            "1, height, width)"
        )

    if len(std_dev_array.shape) != 4:
        raise ValueError(
            "The input quantile values must have shape (batch_size, "
            "1, height, width)"
        )

    batch_size, _, height, width = mean_array.shape
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    bin_probs = np.zeros((batch_size, len(bin_edges) - 1, height, width))

    for n in range(batch_size):
        for i in range(height):
            for j in range(width):
                mean = mean_array[n, 0, i, j]
                std_dev = std_dev_array[n, 0, i, j]
                # Create a normal distribution with the given mean and standard deviation
                distribution = stats.norm(loc=mean, scale=std_dev)

                # Compute the quantiles
                cdf_values = distribution.cdf(bin_edges)
                # Compute the bin probabilities as differences of successive CDF values
                bin_probs[n, :, i, j] = np.diff(cdf_values)

    return bin_probs


if __name__ == "__main__":

    # Example usage:
    quantiles = [0.25, 0.5, 0.75]  # Desired quantiles

    # batch_size = 5
    # num_bins = 4
    # height, width = 3, 3
    bin_probabilities_array = np.ones((5, 4, 3, 3))
    bin_probabilities_array[:, 0, :, :] = 0.2
    bin_probabilities_array[:, 1, :, :] = 0.1
    bin_probabilities_array[:, 2, :, :] = 0.3
    bin_probabilities_array[:, 3, :, :] = 0.4

    # print(bin_probabilities_array.shape)

    quantile_values_array = bin_2_quantile(bin_probabilities_array, quantiles)
    # print(quantile_values_array)

    # Example usage:
    quantiles = [0.1, 0.5, 0.9]
    quantiles_values = [0.13, 0.48, 0.80]

    quantiles_values_array = np.ones((1, 3, 5, 5))
    quantiles_values_array[:, 0, :, :] = 0.13
    quantiles_values_array[:, 1, :, :] = 0.48
    quantiles_values_array[:, 2, :, :] = 0.90

    bin_probabilities = quantile_2_bin(quantiles, quantiles_values_array, num_bins=5)

    # second quantile example
    quantiles = [0.25, 0.5, 0.75]

    quantiles_values_array = np.ones((1, 3, 5, 5))
    quantiles_values_array[:, 0, :, :] = 0.1
    quantiles_values_array[:, 1, :, :] = 0.33
    quantiles_values_array[:, 2, :, :] = 0.72

    bin_probabilities = quantile_2_bin(quantiles, quantiles_values_array, num_bins=5)
    print(f"Example 2: bin_probabilities: {bin_probabilities}")

    # gaussina to quantiles
    # Example usage:
    mean = 0.5
    mean_array = np.ones((1, 1, 1, 1))
    mean_array[:, 0, :, :] = 0.50
    std_dev = 0.02
    std_array = np.ones((1, 1, 1, 1))
    std_array[:, 0, :, :] = 0.2

    quantiles = [0.10, 0.25, 0.5, 0.75, 0.9]  # 25th, 50th, and 75th percentiles

    quantile_values = gaussian_2_quantile(mean_array, std_array, quantiles)

    bin_probs_gaussian = gaussian_2_bin(mean_array, std_array, num_bins=5)
