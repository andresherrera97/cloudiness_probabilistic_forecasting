# USAGE:
#   Training loops and checkpoint saving
#
import datetime
import time
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def train_model(
    model,
    criterion,
    optimizer,
    device,
    train_loader,
    val_loader,
    epochs: int = 1,
    num_val_samples: int = 10,
    checkpoint_every: Optional[int] = None,
    verbose: bool = True,
    eval_every: int = 100,
    writer=None,
    scheduler=None,
    model_name: Optional[str] = None,
):

    if model_name is None:
        model_name = "model"

    TRAIN_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch

    TIME = []

    BEST_VAL_ACC = 1e5

    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)

    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = []  # stores values inside the current epoch
        VAL_LOSS_EPOCH = []  # stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

            start_batch = time.time()

            # data to cuda if possible
            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            loss = criterion(frames_pred, out_frames)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            end_batch = time.time()
            TIME.append(end_batch - start_batch)

            TRAIN_LOSS_EPOCH.append(loss.detach().item())

            if (batch_idx > 0 and batch_idx % eval_every == 0) or (
                batch_idx == len(train_loader) - 1
            ):
                model.eval()
                VAL_LOSS_LOCAL = []  # stores values for this validation run
                start_val = time.time()
                with torch.no_grad():
                    for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):

                        in_frames = in_frames.to(device=device)
                        out_frames = out_frames.to(device=device)

                        frames_pred = model(in_frames)
                        val_loss = criterion(frames_pred, out_frames)

                        VAL_LOSS_LOCAL.append(val_loss.detach().item())

                        if val_batch_idx == num_val_samples:
                            if (batch_idx == len(train_loader) - 1) and writer:
                                # enter if last batch of the epoch and there is a writer
                                writer.add_images(
                                    "groundtruth_batch", out_frames, epoch
                                )
                                writer.add_images(
                                    "predictions_batch", frames_pred, epoch
                                )
                            break

                end_val = time.time()
                val_time = end_val - start_val
                CURRENT_VAL_ACC = sum(VAL_LOSS_LOCAL) / len(VAL_LOSS_LOCAL)
                VAL_LOSS_EPOCH.append(CURRENT_VAL_ACC)

                CURRENT_TRAIN_ACC = sum(
                    TRAIN_LOSS_EPOCH[batch_idx - eval_every :]
                ) / len(TRAIN_LOSS_EPOCH[batch_idx - eval_every :])

                if verbose:
                    # print statistics
                    print(
                        f"Epoch({epoch + 1}/{epochs}) | Batch({batch_idx:04d}/{len(train_loader)}) | ",
                        end="",
                    )
                    print(
                        f"Train_loss({(CURRENT_TRAIN_ACC):06.4f}) | Val_loss({CURRENT_VAL_ACC:.4f}) | ",
                        end="",
                    )
                    print(
                        f"Time_per_batch({sum(TIME)/len(TIME):.2f}s) | Val_time({val_time:.2f}s)"
                    )
                    TIME = []

                if writer:
                    # add values to tensorboard
                    writer.add_scalar(
                        "Loss in train GLOBAL",
                        CURRENT_TRAIN_ACC,
                        batch_idx + epoch * (len(train_loader)),
                    )
                    writer.add_scalar(
                        "Loss in val GLOBAL",
                        CURRENT_VAL_ACC,
                        batch_idx + epoch * (len(train_loader)),
                    )

        # epoch end
        end_epoch = time.time()
        TRAIN_LOSS_GLOBAL.append(sum(TRAIN_LOSS_EPOCH) / len(TRAIN_LOSS_EPOCH))
        VAL_LOSS_GLOBAL.append(sum(VAL_LOSS_EPOCH) / len(VAL_LOSS_EPOCH))

        if scheduler:
            scheduler.step(VAL_LOSS_GLOBAL[-1])

        if writer:
            # add values to tensorboard
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION LOSS, EPOCH MEAN", VAL_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar(
                "Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch
            )

        if verbose:
            print(
                f"Time elapsed in current epoch: {(end_epoch - start_epoch):.2f} secs."
            )

        if CURRENT_VAL_ACC < BEST_VAL_ACC:
            BEST_VAL_ACC = CURRENT_VAL_ACC
            model_dict = {
                "epoch": epoch + 1,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "train_loss_per_batch": TRAIN_LOSS_EPOCH,
                "train_loss_epoch_mean": TRAIN_LOSS_GLOBAL[-1],
            }

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            PATH = "checkpoints/"
            ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
            NAME = model_name + "_epoch" + str(epoch + 1) + "_" + str(ts) + ".pt"

            torch.save(model_dict, PATH + NAME)

    return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


def train_model_2(
    model,
    criterion,
    optimizer,
    device,
    train_loader,
    epochs,
    val_loader,
    checkpoint_every=None,
    verbose=True,
    writer=None,
    scheduler=None,
    model_name=None,
    save_images=True,
):
    """This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        criterion (torch.criterion): [description]
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.

    Returns:
        TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL: Lists containing the mean error of each epoch
    """

    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    if model_name is None:
        model_name = "model"

    TRAIN_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = []

    TIME = []

    BEST_VAL_ACC = 1e5

    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)
        img_size = in_frames.size(2)

    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = []  # stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            loss = criterion(frames_pred, out_frames)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH.append(loss.detach().item())

        TRAIN_LOSS_GLOBAL.append(sum(TRAIN_LOSS_EPOCH) / len(TRAIN_LOSS_EPOCH))

        # evaluation
        model.eval()

        with torch.no_grad():
            VAL_LOSS_EPOCH = []
            mse_val_loss = 0
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)
                val_loss = mae_loss(frames_pred, out_frames)
                mse_val_loss += mse_loss(frames_pred, out_frames).detach().item()

                VAL_LOSS_EPOCH.append(val_loss.detach().item())

                if writer and (val_batch_idx == 0) and save_images and epoch > 35:
                    if img_size < 1000:
                        writer.add_images("groundtruth_batch", out_frames[:10], epoch)
                        writer.add_images("predictions_batch", frames_pred[:10], epoch)
                    else:
                        writer.add_images("groundtruth_batch", out_frames[0], epoch)
                        writer.add_images("predictions_batch", frames_pred[0], epoch)

        VAL_LOSS_GLOBAL.append(sum(VAL_LOSS_EPOCH) / len(VAL_LOSS_EPOCH))

        if scheduler:
            scheduler.step(VAL_LOSS_GLOBAL[-1])

        end_epoch = time.time()
        TIME = end_epoch - start_epoch

        if verbose:
            # print statistics
            print(f"Epoch({epoch + 1}/{epochs}) | ", end="")
            print(
                f"Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val_loss({VAL_LOSS_GLOBAL[-1]:.4f}) | ",
                end="",
            )
            print(f"Time_Epoch({TIME:.2f}s)")  # this part maybe dont print

        if writer:
            # add values to tensorboard
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION LOSS, EPOCH MEAN", VAL_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar(
                "VALIDATION MSE LOSS", mse_val_loss / len(val_loader), epoch
            )
            writer.add_scalar(
                "Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch
            )

        if VAL_LOSS_GLOBAL[-1] < BEST_VAL_ACC:
            if verbose:
                print("New Best Model")
            BEST_VAL_ACC = VAL_LOSS_GLOBAL[-1]
            model_dict = {
                "epoch": epoch + 1,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "train_loss_epoch_mean": TRAIN_LOSS_GLOBAL[-1],
                "val_loss_epoch_mean": VAL_LOSS_GLOBAL[-1],
            }
            model_not_saved = True

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if model_not_saved:
                if verbose:
                    print("Saving Checkpoint")
                PATH = "checkpoints/"
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME = model_name + "_" + str(epoch + 1) + "_" + str(ts) + ".pt"

                torch.save(model_dict, PATH + NAME)
                model_not_saved = False

    # if training finished and best model not saved
    if model_not_saved:
        if verbose:
            print("Saving Checkpoint")
        PATH = "checkpoints/"
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME = model_name + "_" + str(epoch + 1) + "_" + str(ts) + ".pt"

        torch.save(model_dict, PATH + NAME)

    return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


def train_model_full(
    model,
    train_loss: str,
    optimizer,
    device,
    train_loader,
    val_loader,
    epochs: int,
    checkpoint_every=None,
    verbose: bool = True,
    writer=None,
    scheduler=None,
    loss_for_scheduler: str = "mae",
    model_name=None,
    save_images: bool = True,
    predict_diff: bool = False,
    retrain: bool = False,
    trained_model_dict=None,
    testing_loop: bool = False,
):
    """This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        train_loss (str): Train criterion to use ('mae','mse')
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.
        loss_for_scheduler (string): choose validation error to use for scheduler steps
        model_name (string): Prefix Name for the checkpoint model to be saved
        save_images (bool): If true images are saved on the tensorboard
        predict_diff (bool): If True the model predicts the difference between las input image and output

    Returns:
        TRAIN_LOSS_GLOBAL: Mean train loss in each epoch
        VAL_MAE_LOSS_GLOBAL: Lists containing the mean MAE error of each epoch in validation
        VAL_MSE_LOSS_GLOBAL: Lists containing the mean MSE error of each epoch in validation
    """
    if retrain and not trained_model_dict:
        raise ValueError("To retrain the model dict is needed")

    PREVIOUS_CHECKPOINT_NAME = None
    PATH = "checkpoints/"
    best_model_not_saved = False

    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    if train_loss in ["mae", "MAE"]:
        train_criterion = mae_loss
    if train_loss in ["mse", "MSE"]:
        train_criterion = mse_loss
    if model_name is None:
        model_name = "model"

    TIME = []

    if retrain:
        TRAIN_LOSS_GLOBAL = trained_model_dict["train_loss_epoch_mean"]
        VAL_MAE_LOSS_GLOBAL = trained_model_dict["val_mae_loss"]
        VAL_MSE_LOSS_GLOBAL = trained_model_dict["val_mse_loss"]

        if trained_model_dict["validation_loss"] in ["mae", "MAE"]:
            BEST_VAL_ACC = np.min(VAL_MAE_LOSS_GLOBAL)
        if trained_model_dict["validation_loss"] in ["mse", "MSE"]:
            BEST_VAL_ACC = np.min(VAL_MSE_LOSS_GLOBAL)

        first_epoch = trained_model_dict["epoch"]
        print(
            f"Start from pre trained model, epoch: {first_epoch}, last train loss: {TRAIN_LOSS_GLOBAL[-1]}, best val loss: {BEST_VAL_ACC}"
        )

    else:
        TRAIN_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch
        VAL_MAE_LOSS_GLOBAL = []
        VAL_MSE_LOSS_GLOBAL = []

        BEST_VAL_ACC = 1e5

        first_epoch = 0

    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        # writer.add_graph(model, input_to_model=in_frames, verbose=False)
        img_size = in_frames.size(2)

    for epoch in range(first_epoch, epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = 0  # stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):

            if testing_loop and batch_idx == 10:
                break

            model.train()

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            if train_loss in [
                "mae",
                "MAE",
                "mse",
                "MSE",
            ]:
                if predict_diff:
                    diff = torch.subtract(out_frames[:, 0], in_frames[:, 2]).unsqueeze(
                        1
                    )
                    loss = train_criterion(frames_pred, diff)
                else:
                    loss = train_criterion(frames_pred, out_frames)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH += loss.detach().item()

        TRAIN_LOSS_GLOBAL.append(TRAIN_LOSS_EPOCH / len(train_loader))

        # evaluation
        model.eval()

        with torch.no_grad():
            mse_val_loss = 0
            mae_val_loss = 0

            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):

                if testing_loop and val_batch_idx == 10:
                    break

                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)

                if predict_diff:
                    frames_pred = torch.add(
                        frames_pred[:, 0], in_frames[:, 2]
                    ).unsqueeze(1)
                    frames_pred = torch.clamp(frames_pred, min=0, max=1)
                    mae_val_loss += mae_loss(frames_pred, out_frames).detach().item()
                    mse_val_loss += mse_loss(frames_pred, out_frames).detach().item()

                if not predict_diff:
                    mae_val_loss += mae_loss(frames_pred, out_frames).detach().item()
                    mse_val_loss += mse_loss(frames_pred, out_frames).detach().item()
                    frames_pred = torch.clamp(frames_pred, min=0, max=1)

        VAL_MAE_LOSS_GLOBAL.append(mae_val_loss / len(val_loader))
        VAL_MSE_LOSS_GLOBAL.append(mse_val_loss / len(val_loader))

        if scheduler:
            if loss_for_scheduler in ["mae", "MAE"]:
                scheduler.step(VAL_MAE_LOSS_GLOBAL[-1])
            if loss_for_scheduler in ["mse", "MSE"]:
                scheduler.step(VAL_MSE_LOSS_GLOBAL[-1])

        end_epoch = time.time()
        TIME = end_epoch - start_epoch

        if verbose:
            # print statistics
            print(f"Epoch({epoch + 1}/{epochs}) | ", end="")
            print(
                f"Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val MAE({VAL_MAE_LOSS_GLOBAL[-1]:.4f}) | Val MSE({VAL_MSE_LOSS_GLOBAL[-1]:.4f}) |",
                end="",
            )
            print(f"Time_Epoch({TIME:.2f}s)")

        if writer:
            # add values to tensorboard
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION MAE", VAL_MAE_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION MSE", VAL_MSE_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar(
                "Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch
            )

        if loss_for_scheduler in ["mae", "MAE"]:
            actual_loss = VAL_MAE_LOSS_GLOBAL[-1]
        if loss_for_scheduler in ["mse", "MSE"]:
            actual_loss = VAL_MSE_LOSS_GLOBAL[-1]

        if actual_loss < BEST_VAL_ACC:
            BEST_VAL_ACC = actual_loss

            if verbose:
                print("New Best Model")
            best_model_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": loss_for_scheduler,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "train_loss_epoch_mean": TRAIN_LOSS_GLOBAL.copy(),
                "val_mae_loss": VAL_MAE_LOSS_GLOBAL.copy(),
                "val_mse_loss": VAL_MSE_LOSS_GLOBAL.copy(),
            }
            best_model_not_saved = True
            best_model_epoch = epoch + 1

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if verbose:
                print("Saving Checkpoint")

            actual_model_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": loss_for_scheduler,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "train_loss_epoch_mean": TRAIN_LOSS_GLOBAL.copy(),
                "val_mae_loss": VAL_MAE_LOSS_GLOBAL.copy(),
                "val_mse_loss": VAL_MSE_LOSS_GLOBAL.copy(),
            }

            ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
            NAME = model_name + "_" + str(epoch + 1) + "_" + str(ts) + ".pt"
            torch.save(actual_model_dict, os.path.join(PATH, NAME))
            # delete previous checkpoint saved
            if PREVIOUS_CHECKPOINT_NAME:
                try:
                    os.remove(os.path.join(PATH, PREVIOUS_CHECKPOINT_NAME))
                except OSError as e:  ## if failed, report it back to the user ##
                    print("Error: Couldnt delete checkpoint")
            PREVIOUS_CHECKPOINT_NAME = NAME

            if best_model_not_saved and epoch > 0:
                PATH = "checkpoints/"
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME = (
                    model_name
                    + "_"
                    + str(best_model_epoch)
                    + "_"
                    + str(ts)
                    + "_BEST.pt"
                )
                torch.save(best_model_dict, os.path.join(PATH, NAME))
                best_model_not_saved = False

    # if training finished and best model not saved
    if best_model_not_saved and epoch > 0:
        if verbose:
            print("Saving Checkpoint")
        PATH = "checkpoints/"
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME = model_name + "_" + str(epoch + 1) + "_" + str(ts) + ".pt"

        torch.save(best_model_dict, os.path.join(PATH, NAME))

    return (
        TRAIN_LOSS_GLOBAL,
        VAL_MAE_LOSS_GLOBAL,
        VAL_MSE_LOSS_GLOBAL,
    )


def train_model_double_val(
    model,
    train_loss,
    optimizer,
    device,
    train_loader,
    val_loader_w_csv,
    val_loader_wo_csv,
    epochs,
    verbose=True,
    writer=None,
    testing_loop=False,
):
    """This train function evaluates on two validation datasets per epoch.

    Args:
        model (torch.model): [description]
        train_loss (str): Train criterion to use ('mae','mse')
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader_w_csv ([type]): [description]
        val_loader_wo_csv ([type]): [description]
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.

    Returns:
        TRAIN_LOSS_GLOBAL: Mean train loss in each epoch
        VAL_MAE_LOSS_GLOBAL: Lists containing the mean MAE error of each epoch in validation
        VAL_MSE_LOSS_GLOBAL: Lists containing the mean MSE error of each epoch in validation
    """
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    if train_loss in ["mae", "MAE"]:
        train_criterion = mae_loss
    if train_loss in ["mse", "MSE"]:
        train_criterion = mse_loss

    TIME = []

    TRAIN_LOSS_GLOBAL = []  # perists through epochs, stores the mean of each epoch
    VAL_MAE_LOSS_GLOBAL_W_CSV = []
    VAL_MSE_LOSS_GLOBAL_W_CSV = []

    VAL_MAE_LOSS_GLOBAL_WO_CSV = []
    VAL_MSE_LOSS_GLOBAL_WO_CSV = []

    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = 0  # stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):

            if testing_loop and batch_idx == 1:
                break

            model.train()

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            if train_loss in [
                "mae",
                "MAE",
                "mse",
                "MSE",
            ]:
                loss = train_criterion(frames_pred, out_frames)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH += loss.detach().item()

        TRAIN_LOSS_GLOBAL.append(TRAIN_LOSS_EPOCH / len(train_loader))

        # evaluation
        model.eval()

        with torch.no_grad():
            mse_val_loss_w_csv = 0
            mae_val_loss_w_csv = 0

            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader_w_csv):

                if testing_loop and val_batch_idx == 1:
                    break

                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)

                mae_val_loss_w_csv += mae_loss(frames_pred, out_frames).detach().item()
                mse_val_loss_w_csv += mse_loss(frames_pred, out_frames).detach().item()
                frames_pred = torch.clamp(frames_pred, min=0, max=1)

            mse_val_loss_wo_csv = 0
            mae_val_loss_wo_csv = 0

            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader_wo_csv):

                if testing_loop and val_batch_idx == 1:
                    break

                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)

                mae_val_loss_wo_csv += mae_loss(frames_pred, out_frames).detach().item()
                mse_val_loss_wo_csv += mse_loss(frames_pred, out_frames).detach().item()
                frames_pred = torch.clamp(frames_pred, min=0, max=1)

        VAL_MAE_LOSS_GLOBAL_W_CSV.append(mae_val_loss_w_csv / len(val_loader_w_csv))
        VAL_MSE_LOSS_GLOBAL_W_CSV.append(mse_val_loss_w_csv / len(val_loader_w_csv))

        VAL_MAE_LOSS_GLOBAL_WO_CSV.append(mae_val_loss_wo_csv / len(val_loader_wo_csv))
        VAL_MSE_LOSS_GLOBAL_WO_CSV.append(mse_val_loss_wo_csv / len(val_loader_wo_csv))

        end_epoch = time.time()
        TIME = end_epoch - start_epoch

        if verbose:
            # print statistics
            print(f"Epoch({epoch + 1}/{epochs}) | ", end="")
            print(f"Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f})")
            print(
                f"WITH CSV: Val MAE({VAL_MAE_LOSS_GLOBAL_W_CSV[-1]:.4f}) | Val MSE({VAL_MSE_LOSS_GLOBAL_W_CSV[-1]:.4f}) |"
            )
            print(
                f"WITHOUT CSV: Val MAE({VAL_MAE_LOSS_GLOBAL_WO_CSV[-1]:.4f}) | Val MSE({VAL_MSE_LOSS_GLOBAL_WO_CSV[-1]:.4f}) |"
            )
            print(f"Time_Epoch({TIME:.2f}s)")  # this part maybe dont print

        if writer:
            # add values to tensorboard
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar(
                "VALIDATION MAE W CSV", VAL_MAE_LOSS_GLOBAL_W_CSV[-1], epoch
            )
            writer.add_scalar(
                "VALIDATION MSE W CSV", VAL_MSE_LOSS_GLOBAL_W_CSV[-1], epoch
            )
            writer.add_scalar(
                "VALIDATION MAE WO CSV", VAL_MAE_LOSS_GLOBAL_WO_CSV[-1], epoch
            )
            writer.add_scalar(
                "VALIDATION MSE WO CSV", VAL_MSE_LOSS_GLOBAL_WO_CSV[-1], epoch
            )

    results_dict = {
        "TRAIN LOSS, EPOCH MEAN": TRAIN_LOSS_GLOBAL,
        "VALIDATION MAE W CSV": VAL_MAE_LOSS_GLOBAL_W_CSV,
        "VALIDATION MSE W CSV": VAL_MSE_LOSS_GLOBAL_W_CSV,
        "VALIDATION MAE WO CSV": VAL_MAE_LOSS_GLOBAL_WO_CSV,
        "VALIDATION MSE WO CSV": VAL_MSE_LOSS_GLOBAL_WO_CSV,
    }

    return results_dict
