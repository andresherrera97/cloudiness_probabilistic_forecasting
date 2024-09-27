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
