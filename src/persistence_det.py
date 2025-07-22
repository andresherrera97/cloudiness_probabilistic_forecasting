import os
import fire
import logging
import numpy as np
import utils.utils as utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test Persistence Deterministic")


def evaluate_for_time_horizon(
    time_horizon: int = 60,
    dataset_path: str = "datasets/salto_downsample",
    subset: str = "val",
):
    # Ensure the output directory exists
    path_to_dataset = f"{dataset_path}/{subset}/"
    logger.info(f"Dataset path: {path_to_dataset}")

    output_index = time_horizon // 10

    sequence_df = utils.sequence_df_generator_folders(
        path=path_to_dataset,
        in_channel=3,
        output_index=output_index,
        min_time_diff=5,
        max_time_diff=15,
    )

    # Keep only the first num_in_images columns and the last one
    sequence_df = sequence_df.iloc[:, list(range(3)) + [-1]]
    logger.info(f"Sequence DataFrame shape: {sequence_df.shape}")

    mean_error = []
    mse = []
    mbe = []

    for _, row in sequence_df.iterrows():
        # in_img_0_path = os.path.join(path_to_dataset, row[0])
        # in_img_1_path = os.path.join(path_to_dataset, row[1])
        in_img_2_path = os.path.join(path_to_dataset, row[2])
        target_img_path = os.path.join(path_to_dataset, row[sequence_df.columns[-1]])

        # in_img_1 = np.load(in_img_1_path, allow_pickle=True).astype(np.float32) / 255
        in_img_2 = np.load(in_img_2_path, allow_pickle=True).astype(np.float32) / 255
        target_img = (
            np.load(target_img_path, allow_pickle=True).astype(np.float32) / 255
        )

        # Calculate the optical flow using TV-L1
        prediction = in_img_2

        mean_error.append(np.nanmean(np.abs(prediction - target_img)))
        mse.append(np.nanmean((prediction - target_img) ** 2))
        mbe.append(np.nanmean(prediction - target_img))

    logger.info(f"Mean error across all predictions: {np.mean(mean_error)}")
    logger.info(f"RMSE across all predictions: {np.sqrt(np.mean(mse))}")
    logger.info(f"MBE across all predictions: {np.mean(mbe)}")


def main( 
    dataset_path: str = "datasets/salto_downsample",
    subset: str = "val",
):
    # run the evaluate_for_time_horizon function
    for time_horizon in [60, 120, 180, 240, 300]:
        print(f"=== Time Horizon: {time_horizon} ===")
        evaluate_for_time_horizon(
            time_horizon=time_horizon,
            dataset_path=dataset_path,
            subset=subset,
        )


if __name__ == "__main__":
    fire.Fire(main)
