import time
import copy
import torch
import models.probabilistic_unet as p_unet
from models import (
    MeanStdUNet,
    BinClassifierUNet,
    QuantileRegressorUNet,
    MonteCarloDropoutUNet,
)
import matplotlib.pyplot as plt
from typing import Optional
from metrics import crps_batch


def train_model_moving_mnist(
    probabilistic_model,
    optimizer,
    device,
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
        probabilistic_model.model.train()

        for batch_idx, (in_frames, out_frames) in enumerate(
            probabilistic_model.train_loader
        ):

            start_batch = time.time()

            # data to cuda if possible
            in_frames = in_frames.to(device=device).float()
            out_frames = out_frames.to(device=device).float()

            # forward
            frames_pred = probabilistic_model.model(in_frames.float())
            loss = probabilistic_model.calculate_loss(frames_pred, out_frames)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            end_batch = time.time()

            train_loss_in_epoch_list.append(loss.detach().item())

            if verbose and batch_idx % 500 == 0:
                print(
                    f"BATCH({batch_idx + 1}/{len(probabilistic_model.train_loader)}) | ",
                    end="",
                )
                print(f"Train loss({loss.detach().item():.4f}) | ", end="")
                print(f"Time Batch({(end_batch - start_batch):.2f}) | ")

            if batch_idx == num_train_samples:
                break

        train_loss_in_epoch = sum(train_loss_in_epoch_list) / len(
            train_loss_in_epoch_list
        )

        probabilistic_model.model.eval()
        VAL_LOSS_LOCAL = []  # stores values for this validation run
        # val_crps_local = []
        with torch.no_grad():
            for val_batch_idx, (in_frames, out_frames) in enumerate(
                probabilistic_model.val_loader
            ):

                in_frames = in_frames.to(device=device).float()
                out_frames = out_frames.to(device=device).float()

                frames_pred = probabilistic_model.model(in_frames.float())

                val_loss = probabilistic_model.calculate_loss(frames_pred, out_frames)
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
                "model_state_dict": copy.deepcopy(
                    probabilistic_model.model.state_dict()
                ),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "train_loss_per_batch": train_loss_in_epoch_list,
                "train_loss_epoch_mean": train_loss_in_epoch,
            }

    return best_model_dict, TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


if __name__ == "__main__":

    NUM_BINS = 10
    INPUT_FRAMES = 3
    EPOCHS = 3
    NUM_TRAIN_SAMPLES = 100
    NUM_VAL_SAMPLES = 100
    BATCH_SIZE = 16
    FILTERS = 2

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"using device: {device}")

    for cls in map(p_unet.__dict__.get, p_unet.__all__):
        if cls.__name__ == MeanStdUNet.__name__:
            print("=" * 3, "MeanStdUNet", "=" * 3)
            probabilistic_unet = cls(in_frames=INPUT_FRAMES, filters=FILTERS)
        elif cls.__name__ == BinClassifierUNet.__name__:
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
            print("=" * 3, "MonteCarloDropoutUNet", "=" * 3)
            probabilistic_unet = cls(
                dropout_p=0.5, in_frames=INPUT_FRAMES, filters=FILTERS
            )
        else:
            print("Wrong class type!")

        probabilistic_unet.model.to(device)
        probabilistic_unet.initialize_weights()

        optimizer = torch.optim.SGD(
            probabilistic_unet.model.parameters(), lr=0.001, momentum=0.9
        )
        probabilistic_unet.create_dataloaders(
            path="datasets/moving_mnist_dataset/", batch_size=BATCH_SIZE
        )

        best_model_dict, train_loss, val_loss = train_model_moving_mnist(
            probabilistic_model=probabilistic_unet,
            optimizer=optimizer,
            device=device,
            epochs=EPOCHS,
            num_train_samples=NUM_TRAIN_SAMPLES,
            num_val_samples=NUM_VAL_SAMPLES,
            verbose=True,
        )
