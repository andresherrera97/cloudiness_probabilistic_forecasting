import torch
import torch.nn as nn


def optimizer_init(model, method: str, lr: float):
    """Initialize optimizer for the model"""
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
            weight_decay=0.001,
            amsgrad=False,
        )
    else:
        raise ValueError(f"Optimizer {method} not recognized.")


def scheduler_init(
    optimizer,
    method: str,
    step_size: int = 20,
    gamma: float = 0.3,
    patience: int = 15,
    min_lr: float = 1e-7,
    max_lr: float = 0.1,
    epochs: int = 50,
    steps_per_epoch: int = 20000,
    warmup_start: float = 0.3,
):
    """Initialize scheduler for the optimizer"""
    if method.lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=gamma,
            patience=patience,
            min_lr=min_lr,
        )
    elif method.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif method.lower() == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=step_size, gamma=gamma
        )
    elif method.lower() == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif method.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step_size)
    elif method.lower() == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warmup_start,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0,
        )
    else:
        raise ValueError(f"Scheduler {method} not recognized.")


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_normal_(model.weight)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)


# how to apply

# 1) load model , ex: model = UNet(...)
# 2) model.apply(weights_init)
