import torch


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
