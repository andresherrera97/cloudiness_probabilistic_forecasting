import torch
import torch.nn as nn


def optimizer_init(model, method: str, lr: float):
    if method.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_normal_(model.weight)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)


# how to apply

# 1) load model , ex: model = UNet(...)
# 2) model.apply(weights_init)
