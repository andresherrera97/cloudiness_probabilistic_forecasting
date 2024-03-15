import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import properscoring as ps
from math import isclose


def crps_bin_classification(predictions, ground_truth):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a batch of images.

    Parameters:
    - predictions: Numpy array with shape (batch_size, num_bins, height, width)
                   representing the predicted probabilities for each pixel.
    - ground_truth: Numpy array with shape (batch_size, num_bins, height, width) or (batch_size, 1, height, width)
                   representing the ground truth values for each pixel. If dim 1 has shape 1, then the heavy
                   side function is calculated.

    Returns:
    - crps: Numpy array with shape (batch_size,) representing the CRPS for each image in the batch.
    """
    if predictions.shape != ground_truth.shape and ground_truth.shape[1] != 1:
        raise ValueError(
            "The shape of the preds and gt must be the same unless the gt has len 1 in dim 1."
        )

    if ground_truth.shape[1] == 1:
        # Create binary array with 1s in the corresponding bin positions
        ground_truth = nn.functional.one_hot(
            ground_truth.to(torch.int64), num_classes=predictions.shape[1]
        )
        ground_truth = ground_truth.squeeze(1)
        ground_truth = ground_truth.permute(0, 3, 1, 2)

    n_bins = predictions.shape[1]
    # Calculate cumulative probabilities
    pred_cum_probs = torch.cumsum(predictions, axis=1)
    gt_cum_probs = torch.cumsum(ground_truth, axis=1)

    # Calculate squared difference and integrate using trapezoidal rule
    cdf_difference = (pred_cum_probs - gt_cum_probs) ** 2
    crps = torch.sum(cdf_difference, dim=1) / n_bins
    crps = torch.mean(crps)

    return crps


def crps_gaussian(target, mu, sig, sample_weight=None):
    # TODO: it is detected that ensemble predictions might come with some
    # sig values equal to 0 which causes the CRPS to be NaN.
    sx = (target - mu) / sig
    pdf = stats.norm.pdf(sx)
    cdf = stats.norm.cdf(sx)
    per_obs_crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - 1.0 / np.sqrt(np.pi))
    return np.average(per_obs_crps, weights=sample_weight)


if __name__ == "__main__":
    N_BINS = 10
    BATCH_SIZE = 8
    IMG_SIZE = 32

    print("=== TESTING: CRPS_BATCH ===")
    bin_target = torch.rand(
        BATCH_SIZE, N_BINS, IMG_SIZE, IMG_SIZE
    )  # (batch_size, num_bins, height, width)
    if crps_bin_classification(bin_target, bin_target) == 0:
        print("CORRECT: CRPS is 0 for identical predictions and ground truth")
    else:
        print("INCORRECT: CRPS is not 0 for identical predictions and ground truth")

    step_target = torch.zeros(
        (BATCH_SIZE, N_BINS, IMG_SIZE, IMG_SIZE)
    )  # (batch_size, num_bins, height, width)
    step_prediction_lower = torch.zeros(
        (BATCH_SIZE, N_BINS, IMG_SIZE, IMG_SIZE)
    )  # (batch_size, num_bins, height, width)
    step_prediction_higher = torch.zeros(
        (BATCH_SIZE, N_BINS, IMG_SIZE, IMG_SIZE)
    )  # (batch_size, num_bins, height, width)
    step_target[:, 6, :, :] = 1
    step_prediction_lower[:, 5, :, :] = 1
    step_prediction_higher[:, 7, :, :] = 1
    integer_target = torch.ones((BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)) * 6

    if crps_bin_classification(
        step_prediction_lower, step_target
    ) == crps_bin_classification(step_prediction_higher, step_target):
        print("CORRECT: CRPS is equal for predictions with same distance from target")
    else:
        print(
            "INCORRECT: CRPS is not equal for predictions with same distance from target"
        )

    if (
        round(crps_bin_classification(step_prediction_higher, step_target).item(), 3)
        == 1 / N_BINS
    ):
        print(
            f"CORRECT: CRPS is 1/N_BINS ({1/N_BINS:.2f}) for predictions with one bin higher than target"
        )
    else:
        print(
            f"INCORRECT: CRPS is not 1/N_BINS ({1/N_BINS:.2f}) for predictions with one bin higher than target"
        )
    if crps_bin_classification(
        step_prediction_higher, step_target
    ) == crps_bin_classification(step_prediction_higher, integer_target):
        print("CORRECT: CRPS is equal for bin target and integer target")
    else:
        print("INCORRECT: CRPS is not equal for bin target and integer target")

    print("=== TESTING: CRPS_GAUSSIAN ===")
    # create random array with size (batch_size, height, width)
    target = np.random.rand(BATCH_SIZE, IMG_SIZE, IMG_SIZE)
    moved_mean = target + np.random.rand(BATCH_SIZE, IMG_SIZE, IMG_SIZE)
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

    one_target_array = np.ones_like(target)
    mean_target_array = np.ones_like(target) * 0.3
    std_target_array = np.ones_like(target) * 0.5

    if isclose(
        crps_gaussian(one_target_array, mean_target_array, std_target_array),
        ps.crps_gaussian(1, mu=0.3, sig=0.5),
    ):
        print("CORRECT: CRPS is equal to properscoring package")
    else:
        print("INCORRECT: CRPS is not equal to properscoring package")
