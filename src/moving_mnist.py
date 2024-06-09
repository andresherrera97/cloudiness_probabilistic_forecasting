import datetime
import wandb
import torch
from models import (
    MeanStdUNet,
    BinClassifierUNet,
    QuantileRegressorUNet,
    MonteCarloDropoutUNet,
)
import numpy as np
import random
import fire
from typing import Optional, List
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Train MMNIST")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def main(
    model_name: str,
    num_bins: int = 10,
    input_frames: int = 3,
    epochs: int = 5,
    num_train_samples: Optional[int] = None,
    print_every_n_batches: int = 500,
    num_val_samples: Optional[int] = None,
    batch_size: int = 32,
    num_filters: int = 16,
    learning_rate: float = 1e-3,
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    dropout_p: float = 0.5,
    num_ensemble_preds: int = 1,
    save_experiment: bool = False,
):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    if model_name == "mean_std":
        logger.info("Selected model: MeanStdUNet")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        probabilistic_unet = MeanStdUNet(
            in_frames=input_frames,
            filters=num_filters
        )
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
            quantiles=quantiles,
            in_frames=input_frames,
            filters=num_filters
        )
    elif model_name == "monte_carlo_dropout":
        logger.info("Selected model: MonteCarloDropoutUNet")
        logger.info(f"    - Dropout prob: {dropout_p}")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        probabilistic_unet = MonteCarloDropoutUNet(
            dropout_p=dropout_p, in_frames=input_frames, filters=num_filters
        )
    else:
        raise ValueError(f"Wrong class type! {model_name} not recognized.")

    logger.info("Initializing model...")
    probabilistic_unet.model.to(device)
    probabilistic_unet.initialize_weights()
    probabilistic_unet.initialize_optimizer(method="SGD", lr=learning_rate)
    probabilistic_unet.create_dataloaders(
        path="datasets/moving_mnist_dataset/",
        batch_size=batch_size,
        binarization_method="integer_classes",  # needed for BinClassifierUNet
    )
    logger.info("Initialization done.")

    # start a new wandb run to track this script
    if save_experiment:
        run = wandb.init(
            project="cloudiness_probabilistic_forecasting",
            name=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            config={
                "optimizer": "SGD",
                "learning_rate": learning_rate,
                "architecture": probabilistic_unet.name,
                "dataset": "moving_mnist",
                "epochs": epochs,
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
        checkpoint_path=f"checkpoints/mmnist/{model_name}/",
        ensemble_predictions=num_ensemble_preds,
    )
    logger.info("Training done.")
    logger.info(f"    - Train loss: {train_loss[-1]}")
    logger.info(f"    - Val loss: {val_loss[-1]}")

    if save_experiment:
        wandb.finish()

    # logger.info("Loading model from best checkpoint...")
    # probabilistic_unet.load_checkpoint(
    #     checkpoint_path="checkpoints/prueba/qr_model_mmnist_test_005.pt",
    #     device=device,
    # )
    # logger.info("Loading done.")

if __name__ == "__main__":
    fire.Fire(main)
