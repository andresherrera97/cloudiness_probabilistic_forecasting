import torch
import torch.nn as nn


def optimizer_init(model, method: str, lr: float):
    '''Initialize optimizer for the model'''
    if method.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif method.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
    elif method.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
            amsgrad=False,
        )
    else:
        raise ValueError(f"Optimizer {method} not recognized.")


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_normal_(model.weight)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)


# how to apply

# 1) load model , ex: model = UNet(...)
# 2) model.apply(weights_init)
