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
    DeterministicUNet,
    IQUNetPipeline,
    UNetConfig,
    MixtureDensityUNet,
)
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
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(
    model_name: str,
    dataset: str = "mmnist",
    num_bins: Optional[int] = None,
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
    n_components: int = 3,
    learning_rate: float = 1e-3,
    quantiles: Optional[List[float]] = None,
    predict_diff: bool = False,
    dropout_p: Optional[float] = None,
    num_ensemble_preds: Optional[int] = None,
    checkpoint_folder: Optional[str] = "",
    train_metric: Optional[str] = None,
    val_metric: Optional[str] = None,
    save_experiment: bool = False,
    binarization_method: Optional[str] = None,
    cos_dim: Optional[int] = None,
    crop_or_downsample: Optional[str] = None,
):
    set_all_seeds(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    unet_config = UNetConfig(
        in_frames=input_frames,
        spatial_context=spatial_context,
        filters=num_filters,
        output_activation=output_activation,
        device=device,
    )

    if model_name.lower() in ["mean", "mean_std", "meanstd"]:
        logger.info("Selected model: MeanStdUNet")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        logger.info(f"    - Output activation: {output_activation}")
        probabilistic_unet = MeanStdUNet(unet_config)
    elif model_name.lower() in ["median", "median_scale"]:
        logger.info("Selected model: MedianScaleUNet")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        logger.info(f"    - Output activation: {output_activation}")
        probabilistic_unet = MedianScaleUNet(unet_config)
    elif model_name.lower() in ["bin_classifier", "bin"]:
        binarization_method = "integer_classes"
        logger.info("Selected model: BinClassifierUNet")
        logger.info(f"    - Bins: {num_bins}")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        logger.info(f"    - Train metric: {train_metric}")
        logger.info(f"    - Val metric: {val_metric}")
        logger.info(f"    - Output activation: {output_activation}")
        logger.info(f"    - Binarization method: {binarization_method}")
        probabilistic_unet = BinClassifierUNet(
            config=unet_config,
            n_bins=num_bins,
        )
    elif model_name.lower() in ["quantile_regressor", "qr"]:
        if num_bins is not None and quantiles is None:
            quantiles = np.linspace(0, 1, num_bins + 1)[1:-1]
        logger.info("Selected model: QuantileRegressorUNet")
        logger.info(f"    - Quantiles: {quantiles}")
        logger.info(f"    - Num Bins: {len(quantiles)+1}")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        logger.info(f"    - Predict diff: {predict_diff}")
        logger.info(f"    - Train metric: {train_metric}")
        logger.info(f"    - Val metric: {val_metric}")
        logger.info(f"    - Output activation: {output_activation}")
        probabilistic_unet = QuantileRegressorUNet(
            config=unet_config,
            quantiles=quantiles,
            predict_diff=predict_diff,
        )
    elif model_name.lower() in ["monte_carlo_dropout", "mcd"]:
        logger.info("Selected model: MonteCarloDropoutUNet")
        logger.info(f"    - Dropout prob: {dropout_p}")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        logger.info(f"    - Ensemble preds: {num_ensemble_preds}")
        logger.info(f"    - Output activation: {output_activation}")
        probabilistic_unet = MonteCarloDropoutUNet(
            config=unet_config,
            dropout_p=dropout_p,
            n_quantiles=num_ensemble_preds,
        )
    elif model_name.lower() in ["deterministic", "det", "unet"]:
        logger.info("Selected model: Deterministic UNet")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        logger.info(f"    - Output activation: {output_activation}")
        logger.info(f"    - Crop or downsample: {crop_or_downsample}")
        probabilistic_unet = DeterministicUNet(config=unet_config)
    elif model_name.lower() in ["iqn", "iqn_unet"]:
        num_taus = num_bins - 1
        train_metric = "pinball"
        val_metric = "pinball"
        logger.info("Selected model: IQN_UNet")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        logger.info(f"    - Cosine embedding dimension: {cos_dim}")
        logger.info(f"    - Num Taus: {num_taus}")
        logger.info(f"    - Predict diff: {predict_diff}")
        logger.info(f"    - Train metric: {train_metric}")
        logger.info(f"    - Val metric: {val_metric}")
        probabilistic_unet = IQUNetPipeline(
            config=unet_config,
            cosine_embedding_dimension=cos_dim,
            num_taus=num_taus,
            predict_diff=predict_diff,
        )
    elif model_name.lower() in ["mdn"]:
        logger.info("Selected model: MixtureDensityUNet")
        logger.info(f"    - input_frames: {input_frames}")
        logger.info(f"    - filters: {num_filters}")
        logger.info(f"    - Output activation: {output_activation}")
        logger.info(f"    - Num components: {n_components}")
        probabilistic_unet = MixtureDensityUNet(
            config=unet_config,
            n_components=n_components,
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
    if dataset.lower() in ["moving_mnist", "mnist", "mmnist"]:
        dataset_path = "datasets/moving_mnist_dataset/"
    elif dataset.lower() in ["goes16", "satellite", "sat"]:
        dataset_path = "datasets/goes16/salto/"
    else:
        raise ValueError(f"Wrong dataset! {dataset} not recognized")

    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Time horizon: {time_horizon}")

    probabilistic_unet.create_dataloaders(
        dataset=dataset,
        path=dataset_path,
        batch_size=batch_size,
        time_horizon=time_horizon,
        binarization_method=binarization_method,  # needed for BinClassifierUNet
        crop_or_downsample=crop_or_downsample,
    )

    logger.info("Initialization done.")

    # start a new wandb run to track this script
    if save_experiment:
        run_name = f'{model_name}_{time_horizon}_{datetime.datetime.now().strftime("%Y-%m-%d")}'
        run = wandb.init(
            project="cloud_probabilistic_forecasting",
            name=run_name,
            config={
                "time_horizon": time_horizon,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "learning_rate": learning_rate,
                "architecture": probabilistic_unet.name,
                "dataset": dataset,
                "epochs": epochs,
                "num_bins": num_bins,
                "input_frames": input_frames,
                "batch_size": batch_size,
                "quantiles": quantiles,
                "dropout_p": dropout_p,
                "num_ensemble_preds": num_ensemble_preds,
                "predict_diff": predict_diff,
                "train_metric": train_metric,
                "val_metric": val_metric,
                "output_activation": output_activation,
                "num_filters": num_filters,
                "cos_dim": cos_dim,
                "crop_or_downsample": crop_or_downsample,
            },
        )

        wandb.watch(probabilistic_unet.model, log_freq=500)
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
    train_loss, val_loss = probabilistic_unet.fit(
        n_epochs=epochs,
        num_train_samples=num_train_samples,
        print_train_every_n_batch=print_every_n_batches,
        num_val_samples=num_val_samples,
        device=device,
        run=run,
        verbose=True,
        model_name=probabilistic_unet.name,
        train_metric=train_metric,
        val_metric=val_metric,
        checkpoint_path=checkpoint_path,
    )
    logger.info("Training done.")
    logger.info(f"    - Train loss: {train_loss[-1]}")
    logger.info(f"    - Val loss: {val_loss[-1]}")

    if save_experiment:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
