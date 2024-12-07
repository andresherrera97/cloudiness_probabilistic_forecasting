import math
import torch
import fire
import logging
from models import DeterministicUNet, UNetConfig
from typing import Optional
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContextResolutionEval")


def get_model_path(
    crop_or_downsample: Optional[str],
    time_horizon: int = 60,
):
    if time_horizon == 60:
        if crop_or_downsample is None or crop_or_downsample == "crop_1024_down_1":
            return "checkpoints/goes16/det32_60min_CROP0_DOWN0/det/UNet_IN3_F32_SC0_BS_4_TH60_E14_BVM0_05_D2024-11-27_20:08.pt"
        elif crop_or_downsample == "down_2" or crop_or_downsample == "crop_1024_down_2":
            return "checkpoints/goes16/60min_crop_1024_down_2/det/UNet_IN3_F32_SC0_BS_4_TH60_E16_BVM0_05_D2024-11-30_02:31.pt"
        elif crop_or_downsample == "down_4" or crop_or_downsample == "crop_1024_down_4":
            return "checkpoints/goes16/60min_crop_1024_down_4/det/UNet_IN3_F32_SC0_BS_4_TH60_E8_BVM0_05_D2024-11-29_12:09.pt"
        elif crop_or_downsample == "down_8" or crop_or_downsample == "crop_1024_down_8":
            return "checkpoints/goes16/60min_crop_1024_down_8/det/UNet_IN3_F32_SC0_BS_4_TH60_E16_BVM0_05_D2024-11-30_02:31.pt"
        elif (
            crop_or_downsample == "down_16" or crop_or_downsample == "crop_1024_down_16"
        ):
            return "checkpoints/goes16/60min_crop_1024_down_16/det/UNet_IN3_F32_SC0_BS_4_TH60_E15_BVM0_05_D2024-11-30_00:46.pt"
        elif (
            crop_or_downsample == "down_32" or crop_or_downsample == "crop_1024_down_32"
        ):
            return "checkpoints/goes16/60min_crop_1024_down_32/det/UNet_IN3_F32_SC0_BS_4_TH60_E15_BVM0_05_D2024-11-30_09:14.pt"
        elif (
            crop_or_downsample == "crop_512" or crop_or_downsample == "crop_512_down_1"
        ):
            return "checkpoints/goes16/60min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E15_BVM0_05_D2024-12-02_15:39.pt"
        # to update if necessary
        elif crop_or_downsample == "crop_512_down_2":
            return "checkpoints/goes16/60min_crop_512_down_2/det/UNet_IN3_F32_SC0_BS_4_TH60_E15_BVM0_05_D2024-12-03_09:15.pt"
        elif crop_or_downsample == "crop_512_down_4":
            return "checkpoints/goes16/60min_crop_512_down_4/det/UNet_IN3_F32_SC0_BS_4_TH60_E15_BVM0_05_D2024-12-03_09:15.pt"
        elif crop_or_downsample == "crop_512_down_8":
            return "checkpoints/goes16/60min_crop_512_down_8/det/UNet_IN3_F32_SC0_BS_4_TH60_E15_BVM0_05_D2024-12-03_18:25.pt"
        elif crop_or_downsample == "crop_512_down_16":
            return "checkpoints/goes16/60min_crop_512_down_16/det/UNet_IN3_F32_SC0_BS_4_TH60_E12_BVM0_05_D2024-12-03_08:11.pt"
        elif (
            crop_or_downsample == "crop_256" or crop_or_downsample == "crop_256_down_1"
        ):
            return "checkpoints/goes16/60min_crop_256_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E10_BVM0_06_D2024-12-03_00:52.pt"
        elif crop_or_downsample == "crop_256_down_2":
            raise ValueError("Not implemented")


def crop_or_downsample_image(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    crop_or_downsample: Optional[str],
):
    if crop_or_downsample is not None:
        if "crop" in crop_or_downsample and "down" in crop_or_downsample:
            # crop_X_down_Y
            if crop_or_downsample.split("_")[0] != "crop":
                raise ValueError("Invalid crop_or_downsample value, first crop")
            crop_value = int(crop_or_downsample.split("_")[1])
            down_value = int(crop_or_downsample.split("_")[3])
            border_value = (1024 - crop_value) // 2
            input_tensor = input_tensor[
                :, :, border_value:-border_value, border_value:-border_value
            ]
            input_tensor = input_tensor[:, :, ::down_value, ::down_value]
            output_tensor = output_tensor[
                :, :, border_value:-border_value, border_value:-border_value
            ]
            output_tensor = output_tensor[:, :, ::down_value, ::down_value]
        elif "crop" in crop_or_downsample and "down" not in crop_or_downsample:
            crop_value = int(crop_or_downsample.split("_")[-1])
            border_value = (1024 - crop_value) // 2
            input_tensor = input_tensor[
                :, :, border_value:-border_value, border_value:-border_value
            ]
            output_tensor = output_tensor[
                :, :, border_value:-border_value, border_value:-border_value
            ]
        elif "down" in crop_or_downsample and "crop" not in crop_or_downsample:
            down_value = int(crop_or_downsample.split("_")[-1])
            input_tensor = input_tensor[:, :, ::down_value, ::down_value]
            output_tensor = output_tensor[:, :, ::down_value, ::down_value]
        else:
            raise ValueError("Invalid crop_or_downsample value")

    return input_tensor, output_tensor


def get_upscale_factor(crop_or_downsample: Optional[str]):
    if crop_or_downsample is not None:
        if "crop" in crop_or_downsample and "down" in crop_or_downsample:
            down_value = int(crop_or_downsample.split("_")[3])
            return down_value
        elif "crop" in crop_or_downsample and "down" not in crop_or_downsample:
            return 1
        elif "down" in crop_or_downsample and "crop" not in crop_or_downsample:
            down_value = int(crop_or_downsample.split("_")[-1])
            return down_value
    else:
        return 1


def get_crop_size(crop_or_downsample: Optional[str]):
    if crop_or_downsample is not None:
        if "crop" in crop_or_downsample:
            return int(crop_or_downsample.split("_")[1])
    return 1024


def evaluate_model(
    unet: DeterministicUNet,
    device: str,
    eval_crop_size: int = 32,
    crop_or_downsample: Optional[str] = None,
    upsample_method: str = "nearest",
):
    scale_factor = get_upscale_factor(crop_or_downsample)
    upsample_function = torch.nn.Upsample(
        scale_factor=scale_factor, mode=upsample_method
    )
    expected_crop_size = get_crop_size(crop_or_downsample)

    val_loss_per_batch = []  # stores values for this validation run
    val_loss_cropped_per_batch = []  # stores values for this validation run
    unet.deterministic_metrics.start_epoch()

    with torch.no_grad():
        for val_batch_idx, (in_frames, out_frames) in enumerate(unet.val_loader):

            in_frames = in_frames.to(
                device=device, dtype=unet.torch_dtype
            )  # 1024 x 1024

            out_frames = out_frames.to(
                device=device, dtype=unet.torch_dtype
            )  # 1024 x 1024

            in_frames_processed, out_frames_processed = crop_or_downsample_image(
                input_tensor=in_frames,
                output_tensor=out_frames,
                crop_or_downsample=crop_or_downsample,
            )

            with torch.autocast(device_type="cuda", dtype=unet.torch_dtype):
                frames_pred = unet.model(in_frames_processed)
                # calculate val loss to check it matches with the loss during training
                val_loss = unet.calculate_loss(frames_pred, out_frames_processed)
                val_loss_per_batch.append(val_loss.detach().item())

                # evaluate upsampled prediction with 32x32 original crop
                crop_border = (1024 - eval_crop_size) // 2
                out_frames_cropped = out_frames[
                    :, :, crop_border:-crop_border, crop_border:-crop_border
                ]
                persistence_pred_cropped = in_frames[
                    :,
                    unet.in_frames - 1 :,
                    crop_border:-crop_border,
                    crop_border:-crop_border,
                ]

                frames_pred_upsample = upsample_function(frames_pred)
                original_crop_size = frames_pred_upsample.shape[-1]
                if original_crop_size != expected_crop_size:
                    raise ValueError(
                        "Upsampled image has different size than expected crop size"
                    )
                crop_border_pred = (original_crop_size - eval_crop_size) // 2
                if crop_border_pred > 0:
                    frames_pred_upsample_cropped = frames_pred_upsample[
                        :,
                        :,
                        crop_border_pred:-crop_border_pred,
                        crop_border_pred:-crop_border_pred,
                    ]

                if (
                    out_frames_cropped.shape[-1]
                    != frames_pred_upsample_cropped.shape[-1]
                ):
                    raise ValueError(
                        "Cropped image and prediction have different sizes"
                    )

                val_loss_cropped = unet.calculate_loss(
                    frames_pred_upsample_cropped, out_frames_cropped
                )
                val_loss_cropped_per_batch.append(val_loss_cropped.detach().item())

                unet.deterministic_metrics.run_per_batch_metrics(
                    y_true=out_frames_cropped,
                    y_pred=frames_pred_upsample_cropped,
                    y_persistence=persistence_pred_cropped,
                    pixel_wise=False,
                    eps=1e-5,
                )

    val_loss_in_epoch = sum(val_loss_per_batch) / len(val_loss_per_batch)
    val_loss_cropped_in_epoch = sum(val_loss_cropped_per_batch) / len(
        val_loss_cropped_per_batch
    )
    forecasting_metrics = unet.deterministic_metrics.end_epoch()
    return val_loss_in_epoch, val_loss_cropped_in_epoch, forecasting_metrics


def main(
    crop_or_downsample: Optional[str] = None,
    input_frames: int = 3,
    spatial_context: int = 0,
    output_activation: str = "sigmoid",
    time_horizon: int = 60,
    num_filters: int = 32,
    batch_size: int = 1,
    upsample_method: str = "nearest",
    eval_crop_size: int = 32,
    test_all_models: bool = False,
    overwrite: bool = False,
    run_id: str = "",
):
    if upsample_method not in ["nearest", "bilinear", "bicubic", "trilinear"]:
        raise ValueError("Invalid upsample method")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

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
    if not test_all_models:
        logger.info(f"    - Crop or downsample: {crop_or_downsample}")
    unet = DeterministicUNet(config=unet_config)

    unet.model.to(device)
    dataset = "goes16"
    dataset_path = "datasets/goes16/salto/"

    # The crop_downsample parameter is not used in this evaluation as this is
    # done during evaluation to keep original images
    unet.create_dataloaders(
        dataset=dataset,
        path=dataset_path,
        batch_size=batch_size,
        time_horizon=time_horizon,
        binarization_method=None,  # needed for BinClassifierUNet
        crop_or_downsample=None,
    )

    if test_all_models:

        trained_models = [
            None,
            "down_2",
            "down_4",
            "down_8",
            "down_16",
            "down_32",
            "crop_512",
            "crop_512_down_2",
            "crop_512_down_4",
            "crop_512_down_8",
            "crop_512_down_16",
            "crop_256",
        ]

        if not overwrite:
            try:
                df_previous_results = pd.read_csv(f"evaluation_results{run_id}.csv")
                models_tested = df_previous_results["model"].values.tolist()
                models_to_test = [model for model in trained_models if model not in models_tested]
            except FileNotFoundError:
                logger.warning("No previous results found")
                models_to_test = trained_models
                df_previous_results = None
        else:
            models_to_test = trained_models
            df_previous_results = None

        logger.info(f"Models to test: {models_to_test}")

        results = []

        for crop_or_downsample in models_to_test:
            logger.info(f"Testing model: {crop_or_downsample}")
            checkpoint_path = get_model_path(crop_or_downsample, time_horizon)
            unet.load_checkpoint(checkpoint_path=checkpoint_path, device=device)
            val_loss_in_epoch, val_loss_cropped_in_epoch, forecasting_metrics = (
                evaluate_model(
                    unet, device, eval_crop_size, crop_or_downsample, upsample_method
                )
            )
            logger.info(f"Model: {crop_or_downsample}")
            logger.info(f"Validation loss: {val_loss_in_epoch}")
            logger.info(f"Validation loss cropped: {val_loss_cropped_in_epoch}")
            for key, value in forecasting_metrics.items():
                logger.info(f"{key}: {value}")

            result = {
                "model": crop_or_downsample,
                "val_loss": val_loss_in_epoch,
                "val_loss_cropped": val_loss_cropped_in_epoch,
                **forecasting_metrics,
            }
            results.append(result)

        df_results = pd.DataFrame(results)
        if df_previous_results is not None:
            df_results = pd.concat([df_previous_results, df_results], ignore_index=True)
        df_results.to_csv(f"evaluation_results{run_id}.csv", index=False)
    else:
        checkpoint_path = get_model_path(crop_or_downsample, time_horizon)
        unet.load_checkpoint(checkpoint_path=checkpoint_path, device=device)

        val_loss_in_epoch, val_loss_cropped_in_epoch, forecasting_metrics = (
            evaluate_model(
                unet, device, eval_crop_size, crop_or_downsample, upsample_method
            )
        )

        logger.info(f"Validation loss: {val_loss_in_epoch}")
        logger.info(f"Validation loss cropped: {val_loss_cropped_in_epoch}")
        for key, value in forecasting_metrics.items():
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    fire.Fire(main)
