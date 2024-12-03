import torch
import fire
import logging
from models import DeterministicUNet, UNetConfig
from typing import Optional


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContextResolutionEval")


CHECKPOINT_PATH = ""
CROP_SIZE = 32


def crop_or_downsample_image(
    input_tensor: torch.Tensor, output_tensor: torch.Tensor, crop_or_downsample: Optional[str]
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


def post_process_prediction(pred: torch.Tensor, crop_or_downsample: Optional[str]):
    if crop_or_downsample is not None:
        if "crop" in crop_or_downsample and "down" in crop_or_downsample:
            # crop_X_down_Y
            if crop_or_downsample.split("_")[0] != "crop":
                raise ValueError("Invalid crop_or_downsample value, first crop")
            crop_value = int(crop_or_downsample.split("_")[1])
            down_value = int(crop_or_downsample.split("_")[3])
            border_value = (1024 - crop_value) // 2
            pred = pred[:, :, ::down_value, ::down_value]
        elif "crop" in crop_or_downsample and "down" not in crop_or_downsample:
            crop_value = int(crop_or_downsample.split("_")[-1])
            border_value = (1024 - crop_value) // 2
            pred = pred
        elif "down" in crop_or_downsample and "crop" not in crop_or_downsample:
            down_value = int(crop_or_downsample.split("_")[-1])
            pred = pred[:, :, ::down_value, ::down_value]
        else:
            raise ValueError("Invalid crop_or_downsample value")

    return pred

def evaluate_model(unet: DeterministicUNet, device: str, crop_or_downsample: Optional[str] = None):
    val_loss_per_batch = []  # stores values for this validation run
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
                crop_or_downsample=crop_or_downsample
            )

            with torch.autocast(device_type="cuda", dtype=unet.torch_dtype):
                frames_pred = unet.model(in_frames_processed)
                persistence_pred = in_frames_processed[:, unet.in_frames - 1 :, :, :]
                val_loss = unet.calculate_loss(frames_pred, out_frames_processed)

                unet.deterministic_metrics.run_per_batch_metrics(
                    y_true=out_frames_processed,
                    y_pred=frames_pred,
                    y_persistence=persistence_pred,
                    pixel_wise=False,
                    eps=1e-5,
                )

                # evaluate upsampled prediction with 32x32 original crop
                crop_border = (1024 - CROP_SIZE) // 2
                out_frames_cropped = out_frames[:, :, crop_border:-crop_border, crop_border:-crop_border]
                

            val_loss_per_batch.append(val_loss.detach().item())

    val_loss_in_epoch = sum(val_loss_per_batch) / len(val_loss_per_batch)
    forecasting_metrics = unet.deterministic_metrics.end_epoch()
    return val_loss_in_epoch, forecasting_metrics


def main(
    crop_or_downsample: str,
    input_frames: int = 3,
    spatial_context: int = 0,
    output_activation: str = "sigmoid",
    time_horizon: int = 60,
    num_filters: int = 32,
    batch_size: int = 1,
):
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

    unet.model.to(device)
    unet.load_checkpoint(checkpoint_path=CHECKPOINT_PATH, device=device)

    val_loss_in_epoch, forecasting_metrics = evaluate_model(unet, device)

    logger.info(f"Validation loss: {val_loss_in_epoch}")
    for key, value in forecasting_metrics.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    fire.Fire(main)
