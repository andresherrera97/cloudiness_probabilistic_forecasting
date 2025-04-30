import torch
import numpy as np
from typing import List
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


def quantile_2_bin(
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
