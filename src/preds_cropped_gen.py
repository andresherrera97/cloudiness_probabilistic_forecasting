import os
import torch
import multiprocessing
import fire
import logging
import numpy as np
import matplotlib.pyplot as plt

from models import DeterministicUNet, UNetConfig
import utils.utils as utils


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Train Script")


def get_checkpoint_path(time_horizon: int) -> str:
    if time_horizon == 60:
        return "checkpoints/salto_down/salto_down_unet_det_60/det/UNet_IN3_F32_SC0_BS_8_TH60_E7_BVM0_05_D2025-06-11_13:44.pt"
    elif time_horizon == 120:
        return "checkpoints/salto_down/salto_down_unet_det_120/det/UNet_IN3_F32_SC0_BS_8_TH120_E7_BVM0_06_D2025-06-11_04:40.pt"
    elif time_horizon == 180:
        return "checkpoints/salto_down/salto_down_unet_det_180/det/UNet_IN3_F32_SC0_BS_8_TH180_E9_BVM0_07_D2025-06-11_09:05.pt"
    elif time_horizon == 240:
        return "checkpoints/salto_down/salto_down_unet_det_240/det/UNet_IN3_F32_SC0_BS_8_TH240_E9_BVM0_08_D2025-06-11_10:10.pt"
    elif time_horizon == 300:
        return "checkpoints/salto_down/salto_down_unet_det_300/det/UNet_IN3_F32_SC0_BS_8_TH300_E9_BVM0_09_D2025-06-11_07:07.pt"
    else:
        raise ValueError(f"Unsupported time horizon: {time_horizon}")


def generate_real_dataset(
    path_to_dataset: str,
    subset: str = "val",
    move_crop_center: bool = True,
    crop_size: int = 64,
    image_size: int = 512,
    output_path: str = "datasets/les/target_[subset]_crop_64x64_MR/PR/",
) -> None:
    output_path = output_path.replace("[subset]", subset)
    sequence_df = utils.sequence_df_generator_folders(
        path=path_to_dataset,
        in_channel=3,
        output_index=1,
        min_time_diff=5,
        max_time_diff=15,
    )
    # Keep only the first num_in_images columns and the last one
    sequence_df = sequence_df.iloc[:, list(range(3)) + [-1]]
    logger.info(f"Sequence DataFrame shape: {sequence_df.shape}")
    crop_start = (image_size - crop_size) // 2
    crop_end = crop_start + crop_size
    if move_crop_center:
        # Move the crop center by 5 pixels in x and -8 pixels in y
        crop_start_x = crop_start + 5
        crop_end_x = crop_end + 5
        crop_start_y = crop_start - 8
        crop_end_y = crop_end - 8
    else:
        crop_start_x = crop_start
        crop_end_x = crop_end
        crop_start_y = crop_start
        crop_end_y = crop_end
    logger.info(f"Crop size: {crop_size}")
    logger.info(f"Crop start_x: {crop_start_x}, Crop end_x: {crop_end_x}")
    logger.info(f"Crop start_y: {crop_start_y}, Crop end_y: {crop_end_y}")
    for _, row in sequence_df.iterrows():
        day_folder = row[sequence_df.columns[-1]].split("/")[0]
        os.makedirs(
            os.path.join(output_path, day_folder),
            exist_ok=True
        )
        output_filename = row[sequence_df.columns[-1]].split("/")[-1]

        target_img_path = os.path.join(path_to_dataset, row[sequence_df.columns[-1]])
        target_img = np.load(target_img_path, allow_pickle=True).astype(np.float16) / 255.0

        # plt.imshow(pred.squeeze().cpu().detach().numpy() * 255, cmap="gray")
        # plt.axis("off")
        # plt.show()

        targer_crop = (
            target_img[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        ).astype(np.float16)

        # plt.imshow(pred_crop, cmap="gray")
        # plt.axis("off")
        # plt.show()

        output_filename = os.path.join(
            output_path, day_folder, output_filename
        )
        np.save(output_filename, targer_crop)


def main(
    time_horizon: int = 60,
    dataset_path: str = "datasets/salto_downsample",
    subset: str = "val",
    crop_size: int = 64,
    image_size: int = 512,
    move_crop_center: bool = True,
    output_path: str = "predictions/les/pred_crop_64x64_MR/PR/",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    n_cores = multiprocessing.cpu_count()
    logger.info(f"Number of cores: {n_cores}")

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    path_to_dataset = f"{dataset_path}/{subset}/"
    logger.info(f"Dataset path: {path_to_dataset}")

    if time_horizon == 0:
        logger.error("Generating dataset with real images, not predictions.")
        generate_real_dataset(
            path_to_dataset=path_to_dataset,
            subset=subset,
            move_crop_center=move_crop_center,
            crop_size=64,
            image_size=512,
            output_path="predictions/les/target_crop_64x64_MR/PR/",
        )
        return

    unet_config = UNetConfig(
        in_frames=3,
        spatial_context=0,
        filters=32,
        output_activation="sigmoid",
        device=device,
    )

    unet = DeterministicUNet(config=unet_config)
    checkpoint_path = get_checkpoint_path(time_horizon)
    unet.load_checkpoint(checkpoint_path=checkpoint_path, device=device)

    logger.info(f"Loaded model from {checkpoint_path}")

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

    # Generate predictions
    crop_start = (image_size - crop_size) // 2
    crop_end = crop_start + crop_size
    if move_crop_center:
        # Move the crop center by 5 pixels in x and -8 pixels in y
        crop_start_x = crop_start + 5
        crop_end_x = crop_end + 5
        crop_start_y = crop_start - 8
        crop_end_y = crop_end - 8
    else:
        crop_start_x = crop_start
        crop_end_x = crop_end
        crop_start_y = crop_start
        crop_end_y = crop_end
    logger.info(f"Crop size: {crop_size}")
    logger.info(f"Crop start_x: {crop_start_x}, Crop end_x: {crop_end_x}")
    logger.info(f"Crop start_y: {crop_start_y}, Crop end_y: {crop_end_y}")

    for _, row in sequence_df.iterrows():
        day_folder = row[sequence_df.columns[-1]].split("/")[0]
        os.makedirs(
            os.path.join(output_path, day_folder),
            exist_ok=True
        )
        output_filename = row[sequence_df.columns[-1]].split("/")[-1]

        in_img_0_path = os.path.join(path_to_dataset, row[0])
        in_img_1_path = os.path.join(path_to_dataset, row[1])
        in_img_2_path = os.path.join(path_to_dataset, row[2])

        in_img_0 = torch.from_numpy(
            np.load(in_img_0_path, allow_pickle=True).astype(np.float16) / 255.0
        ).to(device)
        in_img_1 = torch.from_numpy(
            np.load(in_img_1_path, allow_pickle=True).astype(np.float16) / 255.0
        ).to(device)
        in_img_2 = torch.from_numpy(
            np.load(in_img_2_path, allow_pickle=True).astype(np.float16) / 255.0
        ).to(device)

        in_img_0 = in_img_0.unsqueeze(0).unsqueeze(0)
        in_img_1 = in_img_1.unsqueeze(0).unsqueeze(0)
        in_img_2 = in_img_2.unsqueeze(0).unsqueeze(0)
        input_images = torch.cat((in_img_0, in_img_1, in_img_2), dim=1)
        pred = unet.model(input_images.float())
        # plt.imshow(pred.squeeze().cpu().detach().numpy() * 255, cmap="gray")
        # plt.axis("off")
        # plt.show()

        pred_crop = (
            pred[0, 0, crop_start_y:crop_end_y, crop_start_x:crop_end_x].cpu().detach().numpy()
        ).astype(np.float16)

        # plt.imshow(pred_crop, cmap="gray")
        # plt.axis("off")
        # plt.show()

        output_filename = os.path.join(
            output_path, day_folder, output_filename
        )
        np.save(output_filename, pred_crop)


if __name__ == "__main__":
    fire.Fire(main)
