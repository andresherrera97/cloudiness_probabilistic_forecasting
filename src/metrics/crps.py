import torch
import numpy as np
from scipy import stats


def crps_batch(predictions, ground_truth):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a batch of images.

    Parameters:
    - predictions: Numpy array with shape (batch_size, height, width, num_bins)
                   representing the predicted probabilities for each pixel.
    - ground_truth: Numpy array with shape (batch_size, height, width)
                   representing the ground truth values for each pixel.
                   It can be either categorized values or continuous real values.

    Returns:
    - crps: Numpy array with shape (batch_size,) representing the CRPS for each image in the batch.
    """

    # Calculate cumulative probabilities
    pred_cum_probs = torch.cumsum(predictions, axis=1)
    gt_cum_probs = torch.cumsum(ground_truth, axis=1)

    # Calculate squared difference and integrate using trapezoidal rule
    cdf_difference = (pred_cum_probs - gt_cum_probs) ** 2
    crps = torch.sum(cdf_difference, dim=1)
    crps = crps.float().mean()

    return crps


def crps_gaussian(target, mu, sig, sample_weight=None):
    sx = (target - mu) / sig
    pdf = stats.norm.pdf(sx)
    cdf = stats.norm.cdf(sx)
    per_obs_crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - 1.0 / np.sqrt(np.pi))
    return np.average(per_obs_crps, weights=sample_weight)


if __name__ == "__main__":
    # test crps calculation

    # TODO: test crps_batch

    # create random array with size (batch_size, height, width)
    target = np.random.rand(8, 32, 32)
    moved_mean = target + np.random.rand(8, 32, 32)
    big_std_array = np.ones_like(target) * 3
    small_std_array = np.ones_like(target) * 0.01
    if crps_gaussian(target, target, big_std_array) > crps_gaussian(
        target, target, small_std_array
    ):
        print("CORRECT: CRPS increases with std deviation")
    else:
        print("INCORRECT: CRPS does not increase with std deviation")

    if crps_gaussian(target, moved_mean, small_std_array) > crps_gaussian(
        target, target, small_std_array
    ):
        print("CORRECT: CRPS increases with mu distance from target")
    else:
        print("INCORRECT: CRPS does not increases with mu distance from target")
