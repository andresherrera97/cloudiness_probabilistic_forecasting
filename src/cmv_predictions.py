import os
import fire
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import logging
from models import CloudMotionVector
from data_handlers import GOES16Dataset
from data_handlers.utils import classify_array_in_integer_classes
from postprocessing.transform import quantile_2_bin
from metrics import logscore_bin_fn


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PrCMV")


def plot_predictions(predictions, num_cols=6):
    num_predictions = predictions.shape[0]
    num_rows = (num_predictions + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 3 * num_rows))
    axes = axes.flatten()

    for i in range(num_predictions):
        axes[i].imshow(predictions[i], cmap="gray")
        axes[i].set_title(
            f"Pred {i+1}, min={np.nanmin(predictions[i]):.2f}, "
            f"max={np.nanmax(predictions[i]):.2f}, "
            f"mean={np.nanmean(predictions[i]):.2f}",
            fontsize=8,
        )

    # Hide any unused subplots
    for i in range(num_predictions, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main(
    dataset: str = "salto_down",
    subset: str = "train",
    time_horizon: int = 60,
    n_quantiles: int = 9,
    noise_method: str = "not_claude",
    return_last_frame: bool = True,
    angle_noise_std: int = 15,
    magnitude_noise_std: float = 4 / (60 * 60),
    debug: bool = False,
):
    logger.info("Starting CMV model")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Subset: {subset}")
    logger.info(f"Time horizon: {time_horizon} min")
    logger.info(f"Noise method: {noise_method}")
    logger.info(f"Angle noise std: {angle_noise_std}")
    logger.info(f"Magnitude noise std: {magnitude_noise_std}")
    logger.info(f"Num Quantiles: {n_quantiles}")
    
    quantiles = np.linspace(0, 1, n_quantiles + 2)[1:-1]

    cmv = CloudMotionVector(
        n_quantiles=n_quantiles,
        angle_noise_std=angle_noise_std,
        magnitude_noise_std=magnitude_noise_std,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # initialize dataloaders
    if dataset.lower() in ["moving_mnist", "mnist", "mmnist"]:
        dataset_path = "datasets/moving_mnist_dataset/"
    elif dataset.lower() in ["goes16", "satellite", "sat", "salto", "salto_1024"]:
        dataset_path = "datasets/salto/"
    elif dataset.lower() in ["downsample", "salto_down", "salto_512"]:
        dataset_path = "datasets/salto_downsample/"
    elif dataset.lower() in ["debug", "debug_salto"]:
        dataset_path = "datasets/debug_salto/"
    else:
        raise ValueError(f"Wrong dataset! {dataset} not recognized")

    if subset.lower() == "train":
        train_dataset = GOES16Dataset(
            path=os.path.join(dataset_path, "train/"),
            num_in_images=2,
            minutes_forward=time_horizon,
            expected_time_diff=10,
            inpaint_pct_threshold=1.0,
        )
        dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    elif subset.lower() == "val":
        val_dataset = GOES16Dataset(
            path=os.path.join(dataset_path, "val/"),
            num_in_images=2,
            minutes_forward=time_horizon,
            expected_time_diff=10,
            inpaint_pct_threshold=1.0,
        )
        dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    elif subset.lower() == "test":
        test_dataset = GOES16Dataset(
            path=os.path.join(dataset_path, "test/"),
            num_in_images=2,
            minutes_forward=time_horizon,
            expected_time_diff=10,
            inpaint_pct_threshold=1.0,
        )
        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        raise ValueError(f"Wrong subset! {subset} not recognized")

    crps_per_batch = []
    logscore_per_batch = []

    logger.info(f"Calculating CRPS for {subset} set...")
    for index, (in_frames, out_frames) in enumerate(dataloader):
        in_frames = in_frames.cpu().numpy().astype(np.float32)
        out_frames = out_frames.cpu().numpy().astype(np.float32)

        sorted_predictions, nan_mask = cmv.probabilistic_prediction(
            n_quantiles=n_quantiles,
            imgi=in_frames[0, 0, :, :],
            imgf=in_frames[0, 1, :, :],
            period=10 * 60,
            time_step=10 * 60,
            time_horizon=time_horizon * 60,
            noise_method=noise_method,
            return_last_frame=return_last_frame,
        )

        crps_batch = cmv.calculate_crps(sorted_predictions, nan_mask, out_frames[0, 0])
        crps_per_batch.append(crps_batch)
        preds_bin = quantile_2_bin(
            quantiles=quantiles,
            quantiles_values=sorted_predictions,
            num_bins=n_quantiles+1,
        )

        bin_output = classify_array_in_integer_classes(
            out_frames[0, 0], num_bins=10
        )

        logscore_per_batch.append(
            logscore_bin_fn(
                predictions=torch.tensor(preds_bin).to(device),
                targets=torch.tensor(bin_output).to(device).unsqueeze(0),
                nan_mask=torch.tensor(nan_mask).to(device),
            ).detach().item()
        )
        if debug and index > 2:
            break

    dataset_crps = torch.mean(torch.tensor(crps_per_batch))
    dataset_logscore = torch.mean(torch.tensor(logscore_per_batch))
    logger.info(f"CRPS on dataset: {dataset_crps.item()}")
    logger.info(f"Logscore on dataset: {dataset_logscore.item()}")

    # # Load images
    # img_i = np.load("datasets/goes16/salto/train/2022_119/2022_119_UTC_170020.npy")
    # img_i = img_i.astype(np.float32)
    # img_f = np.load("datasets/goes16/salto/train/2022_119/2022_119_UTC_171020.npy")
    # img_f = img_f.astype(np.float32)

    # # Plot images
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].imshow(img_i, cmap="gray")
    # axes[0].set_title("Initial Image")
    # axes[1].imshow(img_f, cmap="gray")
    # axes[1].set_title("Final Image")
    # plt.show()

    # predictions = cmv.predict(
    #     imgi=img_i,
    #     imgf=img_f,
    #     period=10 * 60,
    #     time_step=10 * 60,
    #     time_horizon=60 * 60,
    # )

    # plot_predictions(predictions)

    # # noisy predictions
    # noisy_predictions = cmv.noisy_predict(
    #     imgi=img_i,
    #     imgf=img_f,
    #     period=10 * 60,
    #     time_step=10 * 60,
    #     time_horizon=60 * 60,
    # )

    # plot_predictions(noisy_predictions)


if __name__ == "__main__":
    fire.Fire(main)
