from models import PersistenceEnsemble
import visualization as viz
import logging
import fire
import torch
from typing import Optional


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Persistence Ensemble")


def main(
    dataset: str,
    time_horizon: Optional[int] = None,
    num_quantiles: int = 9,
    batch_size: int = 1,
    create_plots: bool = False,
):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    peen = PersistenceEnsemble(n_quantiles=num_quantiles, device=device)

    if dataset.lower() in ["moving_mnist", "mnist", "mmnist"]:
        dataset_path = "datasets/moving_mnist_dataset/"
    else:
        dataset_path = "/clusteruy/home03/DeepCloud/deepCloud/data/uru/"

    peen.create_dataloaders(
        dataset=dataset,
        path=dataset_path,
        batch_size=batch_size,
        time_horizon=time_horizon,
        cosangs_csv_path="datasets/uru2020_day_pct_",  # only for satellite dataset
    )

    in_frames, out_frames, predictions, crps = peen.random_example()
    logging.info(f"CRPS in random example: {crps}")

    if create_plots:
        viz.show_image_list(
            images_list=in_frames[0].tolist(),
            fig_name="figures/peen_input.jpg",
            save_fig=True,
        )

        viz.show_image_list(
            images_list=predictions[0].tolist(),
            fig_name="figures/peen_quantiles_pred.jpg",
            save_fig=True,
        )

        viz.plot_quantile_predictions(
            input_images=in_frames[0],
            pred_quantiles=predictions[0],
            quantiles=peen.quantiles,
            target_img=out_frames[0],
            pixel_coords=[32, 32],
        )

    train_crps = peen.predict_on_dataset(dataset="train")
    logging.info(f"Train CRPS: {train_crps}")

    val_crps = peen.predict_on_dataset(dataset="validation")
    logging.info(f"Validation CRPS: {val_crps}")


if __name__ == "__main__":
    fire.Fire(main)
