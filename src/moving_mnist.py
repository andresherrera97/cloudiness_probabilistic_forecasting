import time
import copy
import torch
from data_handlers import MovingMnistDataset
from torch.utils.data import DataLoader
from models import Persistence, UNet, weights_init
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Optional
from metrics import crps_batch, QuantileLoss
import numpy as np


def train_model_moving_mnist(
    model,
    criterion,
    optimizer,
    device,
    train_loader,
    val_loader,
    epochs: int = 1,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
    verbose: bool = True,
):

    TRAIN_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch

    BEST_VAL_ACC = 1e5

    for epoch in range(epochs):
        start_epoch = time.time()
        train_loss_in_epoch_list = []  # stores values inside the current epoch
        val_loss_in_epoch = []  # stores values inside the current epoch
        model.train()

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):

            start_batch = time.time()

            # data to cuda if possible
            in_frames = in_frames.to(device=device).float()
            out_frames = out_frames.to(device=device).float()

            # forward
            frames_pred = model(in_frames.float())
            out_frames = out_frames.repeat(1, frames_pred.shape[1], 1, 1)

            print(f"frames_pred: {frames_pred[0, :, 32, 32]}")
            print(f"out_frames: {out_frames[0, :, 32, 32]}")

            loss = criterion(frames_pred, out_frames)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            end_batch = time.time()

            train_loss_in_epoch_list.append(loss.detach().item())

            if verbose and batch_idx % 500 == 0:
                print(f"BATCH({batch_idx + 1}/{len(train_loader)}) | ", end="")
                print(f"Train loss({loss.detach().item():.4f}) | ", end="")
                print(f"Time Batch({(end_batch - start_batch):.2f}) | ")

            if batch_idx == num_train_samples:
                break

        train_loss_in_epoch = sum(train_loss_in_epoch_list) / len(
            train_loss_in_epoch_list
        )

        model.eval()
        VAL_LOSS_LOCAL = []  # stores values for this validation run
        # val_crps_local = []
        with torch.no_grad():
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):

                in_frames = in_frames.to(device=device).float()
                out_frames = out_frames.to(device=device).float()

                frames_pred = model(in_frames.float())
                out_frames = out_frames.repeat(1, frames_pred.shape[1], 1, 1)

                val_loss = criterion(frames_pred, out_frames)
                # val_crps_local.append(crps_batch(frames_pred, out_frames))

                VAL_LOSS_LOCAL.append(val_loss.detach().item())

                if val_batch_idx == num_val_samples:
                    break
        # print(f"val_crps: {np.mean(val_crps_local)}")
        val_loss_in_epoch = sum(VAL_LOSS_LOCAL) / len(VAL_LOSS_LOCAL)

        end_epoch = time.time()

        if verbose:
            print(f"Epoch({epoch + 1}/{epochs}) | ", end="")
            print(
                f"Train_loss({(train_loss_in_epoch):06.4f}) | Val_loss({val_loss_in_epoch:.4f}) | ",
                end="",
            )
            print(f"Time_Epoch({(end_epoch - start_epoch):.2f}s) |")

        # epoch end
        end_epoch = time.time()
        TRAIN_LOSS_GLOBAL.append(train_loss_in_epoch)
        VAL_LOSS_GLOBAL.append(val_loss_in_epoch)

        if val_loss_in_epoch < BEST_VAL_ACC:
            BEST_VAL_ACC = val_loss_in_epoch
            best_model_dict = {
                "epoch": epoch + 1,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "train_loss_per_batch": train_loss_in_epoch_list,
                "train_loss_epoch_mean": train_loss_in_epoch,
            }

    return best_model_dict, TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


if __name__ == "__main__":

    NUM_BINS = 10
    INPUT_FRAMES = 3
    EPOCHS = 100
    NUM_TRAIN_SAMPLES = 1000
    NUM_VAL_SAMPLES = None
    BATCH_SIZE = 32

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"using device: {device}")

    model = UNet(
        in_frames=INPUT_FRAMES,
        n_classes=NUM_BINS,
        filters=2,
    ).to(device)

    model.apply(weights_init)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_dataset = MovingMnistDataset(
        path="datasets/moving_mnist_dataset/train/",
        input_frames=INPUT_FRAMES,
        # num_bins=NUM_BINS,
        num_bins=None,
        shuffle=False,
    )
    val_dataset = MovingMnistDataset(
        path="datasets/moving_mnist_dataset/validation/",
        input_frames=INPUT_FRAMES,
        # num_bins=NUM_BINS,
        num_bins=None,
        shuffle=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    mae_loss = nn.L1Loss()
    cross_entropy_loss = nn.CrossEntropyLoss()
    quantile_loss = QuantileLoss(
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    )

    best_model_dict, train_loss, val_loss = train_model_moving_mnist(
        model=model,
        criterion=quantile_loss,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        num_train_samples=NUM_TRAIN_SAMPLES,
        num_val_samples=NUM_VAL_SAMPLES,
        verbose=True,
    )
