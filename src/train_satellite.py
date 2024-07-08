import datetime
import wandb
import torch
import os
import random
import fire
import logging
from models import (
    MeanStdUNet,
    MedianScaleUNet,
    BinClassifierUNet,
    QuantileRegressorUNet,
    MonteCarloDropoutUNet,
)
import numpy as np
from typing import Optional, List


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Train Satellite")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def main(
    model_name: str,
    num_bins: Optional[int] = None,
    input_frames: int = 3,
    time_horizon: int = 1,
    epochs: int = 5,
    optimizer: str = "SGD",
    scheduler: Optional[str] = None,
    num_train_samples: Optional[int] = None,
    print_every_n_batches: int = 500,
    num_val_samples: Optional[int] = None,
    batch_size: int = 8,
    num_filters: int = 16,
    learning_rate: float = 1e-3,
    quantiles: Optional[List[float]] = None,
    dropout_p: Optional[float] = None,
    num_ensemble_preds: Optional[int] = None,
    checkpoint_folder: str = "",
    checkpoint_metric: str = "crps",
    save_experiment: bool = False,
):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    if model_name in ["mean_std", "meanstd"]:
        logger.info("Selected model: MeanStdUNet")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        probabilistic_unet = MeanStdUNet(in_frames=input_frames, filters=num_filters)
    elif model_name in ["median", "median_scale"]:
        logger.info("Selected model: MedianScaleUNet")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        probabilistic_unet = MedianScaleUNet(in_frames=input_frames, filters=num_filters)
    elif model_name in ["bin_classifier", "bin"]:
        logger.info("Selected model: BinClassifierUNet")
        logger.info(f"    - Bins: {num_bins}")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        probabilistic_unet = BinClassifierUNet(
            n_bins=num_bins, in_frames=input_frames, filters=num_filters, device=device
        )
    elif model_name in ["quantile_regressor", "qr"]:
        logger.info("Selected model: QuantileRegressorUNet")
        logger.info(f"    - Quantiles: {quantiles}")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        probabilistic_unet = QuantileRegressorUNet(
            quantiles=quantiles, in_frames=input_frames, filters=num_filters
        )
    elif model_name in ["monte_carlo_dropout", "mcd"]:
        logger.info("Selected model: MonteCarloDropoutUNet")
        logger.info(f"    - Dropout prob: {dropout_p}")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        logger.info(f"    - Ensemble preds: {num_ensemble_preds}")
        probabilistic_unet = MonteCarloDropoutUNet(
            dropout_p=dropout_p,
            in_frames=input_frames,
            filters=num_filters,
            n_quantiles=num_ensemble_preds,
        )
    else:
        raise ValueError(f"Wrong class type! {model_name} not recognized.")

    logger.info("Initializing model...")
    probabilistic_unet.model.to(device)
    probabilistic_unet.initialize_weights()
    probabilistic_unet.initialize_optimizer(method=optimizer, lr=learning_rate)
    if scheduler is not None:
        probabilistic_unet.initialize_scheduler(
            method=scheduler,
            step_size=20,
            gamma=0.3,
            patience=20,
            min_lr=1e-7,
        )
    probabilistic_unet.create_dataloaders(
        dataset="satellite",
        path="/clusteruy/home03/DeepCloud/deepCloud/data/uru/",
        cosangs_csv_path="datasets/uru2020_day_pct_",
        batch_size=batch_size,
        time_horizon=time_horizon,
        binarization_method="integer_classes",  # needed for BinClassifierUNet
    )
    logger.info("Initialization done.")

    # start a new wandb run to track this script
    if save_experiment:
        run_name = f'{model_name}_{datetime.datetime.now().strftime("%Y-%m-%d")}'
        run = wandb.init(
            project="cloudiness_probabilistic_forecasting",
            name=run_name,
            config={
                "time_horizon": time_horizon,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "learning_rate": learning_rate,
                "architecture": probabilistic_unet.name,
                "dataset": "satellite",
                "epochs": epochs,
                "num_bins": num_bins,
                "input_frames": input_frames,
                "batch_size": batch_size,
                "quantiles": quantiles,
                "dropout_p": dropout_p,
                "num_ensemble_preds": num_ensemble_preds,
            },
        )

        wandb.watch(probabilistic_unet.model, log_freq=100)
    else:
        run = None

    logger.info("Starting training...")
    train_loss, val_loss = probabilistic_unet.fit(
        n_epochs=epochs,
        num_train_samples=num_train_samples,
        print_train_every_n_batch=print_every_n_batches,
        num_val_samples=num_val_samples,
        device=device,
        run=run,
        verbose=True,
        model_name=probabilistic_unet.name,
        checkpoint_metric=checkpoint_metric,
        checkpoint_path=os.path.join(
            "checkpoints/uru2020/", checkpoint_folder, model_name
        ),
    )
    logger.info("Training done.")
    logger.info(f"    - Train loss: {train_loss[-1]}")
    logger.info(f"    - Val loss: {val_loss[-1]}")

    if save_experiment:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
