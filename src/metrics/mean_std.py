import torch
import torch.nn as nn


log2pi = torch.log(torch.tensor(2 * torch.pi))


def mean_std_loss(predictions, y_target):
    mu = predictions[:, 0, :, :]
    sigma2 = nn.functional.softplus(predictions[:, 1, :, :])
    y_target = y_target.unsqueeze(1)

    return (
        0.5 * (torch.log(2 * torch.pi * sigma2) + (mu - y_target) ** 2 / sigma2)
    ).mean()


def median_scale_loss(predictions, y_target):
    median = predictions[:, 0, :, :]
    scale = nn.functional.softplus(predictions[:, 1, :, :])
    y_target = y_target.unsqueeze(1)

    return ((torch.log(2 * scale) + torch.abs(median - y_target) / scale)).mean()


def laplace_nll_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    use_softplus: bool = True,
    reduction: str = "mean",
    epsilon: float = 1e-6,
    use_scale_regularization: bool = False,
    penalty_weight: float = 0.01,
) -> torch.Tensor:
    """
    Calculates the Negative Log-Likelihood (NLL) loss for a Laplace distribution.

    Assumes the prediction tensor contains parameters for a Laplace distribution
    for each target element.

    Args:
        prediction (torch.Tensor): The predicted parameters tensor with shape
                                   [BS, 2, H, W]. Channel 0 is predicted mu (location),
                                   Channel 1 is the raw predicted scale parameter
                                   (e.g., log(b) or input to softplus).
        target (torch.Tensor): The ground truth tensor with shape [BS, 1, H, W].
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. Default: 'mean'.
        epsilon (float): A small value added to the scale parameter for
                         numerical stability (to avoid log(0) or division by zero).
                         Default: 1e-6.

    Returns:
        torch.Tensor: The calculated Laplace NLL loss. Shape depends on reduction:
                      - 'none': [BS, 1, H, W]
                      - 'mean' or 'sum': scalar
    """
    if prediction.shape[1] != 2:
        raise ValueError(
            f"Prediction tensor must have 2 channels (mu, raw_scale), but got shape {prediction.shape}"
        )
    if (
        prediction.shape[0] != target.shape[0]
        or prediction.shape[2:] != target.shape[2:]
    ):
        raise ValueError(
            f"Prediction shape {prediction.shape} and target shape {target.shape} mismatch (excluding channels)."
        )
    if target.shape[1] != 1:
        raise ValueError(
            f"Target tensor must have 1 channel, but got shape {target.shape}"
        )

    # Extract parameters
    # Shape: [BS, 1, H, W]
    pred_mu = prediction[:, 0:1, :, :]
    pred_raw_scale = prediction[:, 1:2, :, :]  # Input to softplus/exp

    # --- Ensure scale 'b' is positive ---
    # Option 1: Using softplus (often more stable)
    if use_softplus:
        pred_scale = nn.functional.softplus(pred_raw_scale) + epsilon
    else:
        pred_scale = pred_raw_scale

    # Option 2: Using exp (uncomment if you prefer this)
    # pred_scale = torch.exp(pred_raw_scale) + epsilon
    # ------------------------------------

    # Calculate absolute error |x - mu|
    abs_error = torch.abs(target - pred_mu)

    # Calculate NLL components: log(2) + log(b) + |x - mu| / b
    # Note: log(2) is a constant and doesn't affect optimization minimum,
    # but we include it for the correct NLL value.
    log_2 = torch.log(torch.tensor(2.0, device=prediction.device))
    log_b = torch.log(pred_scale)
    loss_per_pixel = log_2 + log_b + (abs_error / pred_scale)  # Shape: [BS, 1, H, W]

    # Apply reduction
    if reduction == "mean":
        loss = torch.mean(loss_per_pixel)
    elif reduction == "sum":
        loss = torch.sum(loss_per_pixel)
    elif reduction == "none":
        loss = loss_per_pixel
    else:
        raise ValueError(
            f"Unknown reduction: {reduction}. Choose 'none', 'mean', or 'sum'."
        )

    # Optional scale regularization
    if use_scale_regularization:
        # Regularization term: encourage scale to be close to 1
        # This is optional and can be adjusted based on your needs.
        scale_penalty = penalty_weight * torch.mean(
            torch.relu(-pred_scale - 0.01)
        )
        loss += scale_penalty

    return loss


class MixtureDensityLoss:
    def __init__(self, n_components):
        self.n_components = n_components

    def get_logscore_at_y(self, pred_params, batch_y):
        pis, mus, sigmas = (
            pred_params[:, : self.n_components, :, :],
            pred_params[:, self.n_components : 2 * self.n_components, :, :],
            pred_params[:, 2 * self.n_components :, :, :],
        )
        assert pis.shape[0] == batch_y.shape[0]
        assert pis.shape[1] == self.n_components
        assert (
            len(batch_y.shape) == 4
        ), f"batch_y.shape is {batch_y.shape} but should be (B, 1, H, W)"

        log_prob_per_component = (
            -0.5 * (((batch_y - mus) / sigmas) ** 2) - torch.log(sigmas) - 0.5 * log2pi
        )  # (B, NUM_COMP, H, W)
        weighted_log_prob = torch.log(pis) + log_prob_per_component
        neg_log_likelihood = -torch.logsumexp(weighted_log_prob, dim=1)
        return torch.mean(neg_log_likelihood)

    def __call__(self, pred_params, batch_y):
        return self.get_logscore_at_y(pred_params, batch_y)
