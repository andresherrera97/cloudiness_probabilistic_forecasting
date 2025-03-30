import datetime
import wandb
import torch
import os
import random
import fire
import logging
import multiprocessing
from models import DeterministicUNet, UNetConfig
import numpy as np
from typing import Optional, List


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Train Script")

os.environ["WANDB__SERVICE_WAIT"] = "300"


def set_all_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def main(
    model_name: str = "det",
    dataset: str = "salto_down",
    input_frames: int = 3,
    spatial_context: int = 0,
    output_activation: str = "sigmoid",
    time_horizon: Optional[int] = None,
    epochs: int = 5,
    optimizer: str = "SGD",
    scheduler: Optional[str] = None,
    num_train_samples: Optional[int] = None,
    print_every_n_batches: int = 500,
    num_val_samples: Optional[int] = None,
    batch_size: int = 8,
    num_filters: int = 16,
    learning_rate: float = 1e-3,
    max_lr: float = 0.1,
    checkpoint_folder: Optional[str] = "",
    train_metric: Optional[str] = None,
    val_metric: Optional[str] = None,
    save_experiment: bool = False,
    crop_or_downsample: Optional[str] = None,
    project: str = "cloud_probabilistic_forecasting",
    warmup_start: float = 0.3,
    num_workers: int = 0,
    prefetch_loader: bool = False,
    predict_background: bool = False,
    use_data_augmentation: bool = False,
):
    set_all_seeds(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    n_cores = multiprocessing.cpu_count()
    logger.info(f"Number of cores: {n_cores}")

    unet_config = UNetConfig(
        in_frames=input_frames,
        spatial_context=spatial_context,
        filters=num_filters,
        output_activation=output_activation,
        device=device,
    )

    logger.info("Selected model: Deterministic UNet")
    logger.info(f"    - input_frames: {input_frames}")
    logger.info(f"    - filters: {num_filters}")
    logger.info(f"    - Output activation: {output_activation}")
    logger.info(f"    - Crop or downsample: {crop_or_downsample}")
    unet = DeterministicUNet(config=unet_config)

    logger.info("Initializing model...")
    unet.model.to(device)
    unet.initialize_weights()
    unet.initialize_optimizer(method=optimizer, lr=learning_rate)
    # initialize dataloaders

    if dataset.lower() in ["goes16", "satellite", "sat", "salto", "salto_1024"]:
        dataset_path = "datasets/salto/"
    elif dataset.lower() in ["downsample", "salto_down", "salto_512"]:
        dataset_path = "datasets/salto_downsample/"
    elif dataset.lower() in ["debug", "debug_salto"]:
        dataset_path = "datasets/debug_salto/"
    else:
        raise ValueError(f"Wrong dataset! {dataset} not recognized")

    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Time horizon: {time_horizon}")

    unet.create_dataloaders(
        dataset=dataset,
        path=dataset_path,
        batch_size=batch_size,
        time_horizon=time_horizon,
        crop_or_downsample=crop_or_downsample,
        prefetch_loader=prefetch_loader,
        num_workers=num_workers,
    )

    if scheduler is not None:
        unet.initialize_scheduler(
            method=scheduler,
            step_size=20,
            gamma=0.3,
            patience=20,
            min_lr=1e-7,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(unet.train_loader),
            warmup_start=warmup_start,
        )

    logger.info("Initialization done.")

    # start a new wandb run to track this script
    if save_experiment:
        run_name = f'{model_name}_{time_horizon}_{crop_or_downsample}_{datetime.datetime.now().strftime("%Y-%m-%d")}'
        run = wandb.init(
            project=project,
            name=run_name,
            config={
                "time_horizon": time_horizon,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "learning_rate": learning_rate,
                "max_lr": max_lr,
                "architecture": unet.name,
                "dataset": dataset,
                "epochs": epochs,
                "input_frames": input_frames,
                "batch_size": batch_size,
                "train_metric": train_metric,
                "val_metric": val_metric,
                "output_activation": output_activation,
                "num_filters": num_filters,
                "crop_or_downsample": crop_or_downsample,
                "spatial_context": spatial_context,
                "use_data_augmentation": use_data_augmentation,
            },
        )

        wandb.watch(unet.model, log_freq=500)
    else:
        run = None

    if checkpoint_folder is None:
        checkpoint_path = None
        logger.info("Checkpoint folder is None: Not saving checkpoints.")
    else:
        checkpoint_path = os.path.join(
            f"checkpoints/{dataset}/", checkpoint_folder, model_name
        )
        logger.info(f"Checkpoint path: {checkpoint_path}")

    logger.info("Starting training...")
    train_loss, val_loss = unet.fit(
        n_epochs=epochs,
        num_train_samples=num_train_samples,
        print_train_every_n_batch=print_every_n_batches,
        num_val_samples=num_val_samples,
        device=device,
        run=run,
        verbose=True,
        model_name=unet.name,
        train_metric=train_metric,
        val_metric=val_metric,
        checkpoint_path=checkpoint_path,
        predict_background=predict_background,
        use_data_augmentation=use_data_augmentation,
    )
    logger.info("Training done.")
    logger.info(f"    - Train loss: {train_loss[-1]}")
    logger.info(f"    - Val loss: {val_loss[-1]}")

    if save_experiment:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
