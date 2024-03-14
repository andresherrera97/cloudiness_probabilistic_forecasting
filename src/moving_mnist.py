import datetime
import wandb
import torch
import models.probabilistic_unet as p_unet
from models import (
    MeanStdUNet,
    BinClassifierUNet,
    QuantileRegressorUNet,
    MonteCarloDropoutUNet,
)
import matplotlib.pyplot as plt
import numpy as np
import random


if __name__ == "__main__":

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    NUM_BINS = 10
    INPUT_FRAMES = 3
    EPOCHS = 5
    NUM_TRAIN_SAMPLES = 1
    PRINT_EVERY_N_BATCHES = 500
    NUM_VAL_SAMPLES = 1
    BATCH_SIZE = 32
    FILTERS = 2
    LEARNING_RATE = 1e-3
    SAVE_EXPERIMENT = False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"using device: {device}")

    for cls in map(p_unet.__dict__.get, p_unet.__all__):
        if cls.__name__ == MeanStdUNet.__name__:
            continue
            print("=" * 3, "MeanStdUNet", "=" * 3)
            probabilistic_unet = cls(in_frames=INPUT_FRAMES, filters=FILTERS)
        elif cls.__name__ == BinClassifierUNet.__name__:
            continue
            print("=" * 3, "BinClassifierUNet", "=" * 3)
            probabilistic_unet = cls(
                n_bins=NUM_BINS, in_frames=INPUT_FRAMES, filters=FILTERS
            )
        elif cls.__name__ == QuantileRegressorUNet.__name__:
            print("=" * 3, "QuantileRegressorUNet", "=" * 3)
            probabilistic_unet = cls(
                quantiles=[0.1, 0.5, 0.9], in_frames=INPUT_FRAMES, filters=FILTERS
            )
        elif cls.__name__ == MonteCarloDropoutUNet.__name__:
            continue
            print("=" * 3, "MonteCarloDropoutUNet", "=" * 3)
            probabilistic_unet = cls(
                dropout_p=0.5, in_frames=INPUT_FRAMES, filters=FILTERS
            )
        else:
            print("Wrong class type!")

        probabilistic_unet.model.to(device)
        probabilistic_unet.initialize_weights()
        probabilistic_unet.initialize_optimizer(method="SGD", lr=LEARNING_RATE)
        probabilistic_unet.create_dataloaders(
            path="datasets/moving_mnist_dataset/",
            batch_size=BATCH_SIZE,
            binarization_method="integer_classes",  # needed for BinClassifierUNet
        )

        # start a new wandb run to track this script
        if SAVE_EXPERIMENT:
            run = wandb.init(
                project="cloudiness_probabilistic_forecasting",
                name=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                config={
                    "optimizer": "SGD",
                    "learning_rate": LEARNING_RATE,
                    "architecture": probabilistic_unet.name,
                    "dataset": "moving_mnist",
                    "epochs": EPOCHS,
                },
            )

            wandb.watch(probabilistic_unet.model, log_freq=100)
        else:
            run = None

        train_loss, val_loss = probabilistic_unet.fit(
            n_epochs=EPOCHS,
            num_train_samples=NUM_TRAIN_SAMPLES,
            print_train_every_n_batch=PRINT_EVERY_N_BATCHES,
            num_val_samples=NUM_VAL_SAMPLES,
            device=device,
            run=run,
            verbose=True,
        )

        if SAVE_EXPERIMENT:
            wandb.finish()
