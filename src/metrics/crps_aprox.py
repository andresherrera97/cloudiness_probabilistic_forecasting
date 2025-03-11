import numpy as np
import torch


def calculate_crps_grid(bin_probabilities, target_values):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a grid of forecasts
    using vectorized operations.

    Parameters:
    -----------
    bin_probabilities : numpy.ndarray (3D)
        A 3D array with shape (n_bins, height, width) containing probability
        distributions at each pixel. For each pixel, the probabilities should sum to 1.
    target_values : numpy.ndarray (2D)
        A 2D array with shape (height, width) containing the observed values,
        with each value between 0 and 1.

    Returns:
    --------
    numpy.ndarray (2D)
        A 2D array with the CRPS score at each pixel. Lower values indicate better forecasts.
    """
    # Get dimensions
    n_bins, height, width = bin_probabilities.shape

    # Check that target_values has compatible dimensions
    if target_values.shape != (height, width):
        raise ValueError(
            f"Target shape {target_values.shape} doesn't match bin_probabilities shape {bin_probabilities.shape[1:]}"
        )

    # Create equally spaced bin edges between 0 and 1
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_width = 1.0 / n_bins

    # Calculate the cumulative distribution function for each pixel
    cdf = np.cumsum(bin_probabilities, axis=0)

    # Find which bin contains each target value
    target_bins = np.searchsorted(bin_edges, target_values, side="right") - 1

    # Clip to valid bin indices (in case of numerical precision issues)
    target_bins = np.clip(target_bins, 0, n_bins - 1)

    # Create a mask for each bin index
    bin_indices = np.arange(n_bins)[:, np.newaxis, np.newaxis]

    # Create the Heaviside function (0 before target bin, 1 after target bin)
    heaviside = (bin_indices >= target_bins).astype(float)

    # Calculate the fraction for the bin containing the target
    fractions = (target_values - bin_edges[target_bins]) / bin_width

    # Create a mask for the target bins to apply fractions
    target_bin_mask = bin_indices == target_bins

    # Update the Heaviside function with interpolated values in the target bins
    heaviside = np.where(target_bin_mask, fractions, heaviside)

    # Calculate the squared difference between CDF and Heaviside
    squared_diff = (cdf - heaviside) ** 2

    # Sum over bins (axis=0) and multiply by bin width
    crps_grid = np.sum(squared_diff, axis=0) * bin_width

    return crps_grid


def calculate_mean_crps(bin_probabilities, target_values):
    """
    Calculate the mean CRPS across all pixels.

    Parameters:
    -----------
    bin_probabilities : numpy.ndarray (3D)
        A 3D array with shape (n_bins, height, width).
    target_values : numpy.ndarray (2D)
        A 2D array with shape (height, width).

    Returns:
    --------
    float
        The mean CRPS across all pixels.
    """
    crps_grid = calculate_crps_grid(bin_probabilities, target_values)
    return np.mean(crps_grid)


def calculate_crps_grid_torch(bin_probabilities, target_values):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a grid of forecasts
    using PyTorch tensors on GPU.

    Parameters:
    -----------
    bin_probabilities : torch.Tensor (3D)
        A 3D tensor with shape (n_bins, height, width) containing probability
        distributions at each pixel. For each pixel, the probabilities should sum to 1.
        Should be on the same device as target_values.
    target_values : torch.Tensor (2D)
        A 2D tensor with shape (height, width) containing the observed values,
        with each value between 0 and 1.
        Should be on the same device as bin_probabilities.

    Returns:
    --------
    torch.Tensor (2D)
        A 2D tensor with the CRPS score at each pixel. Lower values indicate better forecasts.
    """
    # Get dimensions and device
    n_bins, height, width = bin_probabilities.shape
    device = bin_probabilities.device

    # Check that target_values has compatible dimensions
    if target_values.shape != (height, width):
        raise ValueError(
            f"Target shape {target_values.shape} doesn't match bin_probabilities shape {bin_probabilities.shape[1:]}"
        )

    # Create equally spaced bin edges between 0 and 1
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=device)
    bin_width = 1.0 / n_bins

    # Calculate the cumulative distribution function for each pixel
    cdf = torch.cumsum(bin_probabilities, dim=0)

    # Find which bin contains each target value
    # Create a tensor of shape (height, width, n_bins+1) with bin edges repeated for each pixel
    expanded_bin_edges = bin_edges.unsqueeze(1).unsqueeze(1).expand(-1, height, width)

    # Create a tensor of shape (height, width, n_bins+1) with target values repeated for each bin edge
    expanded_targets = target_values.unsqueeze(0).expand(n_bins + 1, -1, -1)

    # Compare targets with bin edges to create a mask
    comparison = (expanded_targets >= expanded_bin_edges).to(torch.float)

    # Sum along the bins dimension to get the index of the bin containing each target
    target_bins = torch.sum(comparison, dim=0) - 1

    # Clip to valid bin indices
    target_bins = torch.clamp(target_bins, 0, n_bins - 1)

    # Create bin indices tensor
    bin_indices = (
        torch.arange(n_bins, device=device)
        .unsqueeze(1)
        .unsqueeze(1)
        .expand(-1, height, width)
    )

    # Create the Heaviside function (0 before target bin, 1 after/at target bin)
    heaviside = (bin_indices >= target_bins).to(torch.float)

    # Get the fraction for the bin containing the target
    # First, get the lower bin edge for each target
    lower_bin_edges = torch.gather(
        expanded_bin_edges, 0, target_bins.unsqueeze(0).to(torch.long)
    ).squeeze(0)

    # Calculate fractions
    fractions = (target_values - lower_bin_edges) / bin_width

    # Apply fractions only to the target bins
    target_bin_mask = bin_indices == target_bins
    heaviside = torch.where(target_bin_mask, fractions, heaviside)

    # Calculate squared difference and sum
    squared_diff = (cdf - heaviside) ** 2
    crps_grid = torch.sum(squared_diff, dim=0) * bin_width

    return crps_grid


def calculate_mean_crps_torch(bin_probabilities, target_values):
    """
    Calculate the mean CRPS across all pixels.

    Parameters:
    -----------
    bin_probabilities : torch.Tensor (3D)
        A 3D tensor with shape (n_bins, height, width).
    target_values : torch.Tensor (2D)
        A 2D tensor with shape (height, width).

    Returns:
    --------
    torch.Tensor (0D)
        The mean CRPS across all pixels.
    """
    crps_grid = calculate_crps_grid_torch(bin_probabilities, target_values)
    return torch.mean(crps_grid)
