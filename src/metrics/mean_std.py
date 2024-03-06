import torch
import torch.nn as nn


def mean_std_loss(predictions, y_target):
    mu = predictions[:, 0, :, :]
    sigma2 = nn.functional.softplus(predictions[:, 1, :, :])
    y_target = y_target.unsqueeze(1)

    return (
        0.5 * (torch.log(2 * torch.pi * sigma2) + (mu - y_target) ** 2 / sigma2)
    ).mean()
