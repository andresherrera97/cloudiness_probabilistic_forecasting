import fire
import json
import torch
from models import DeterministicUNet, UNetConfig
from typing import Optional


checkpoint_paths_60 = {
    "crop_128_down_2": [
        "checkpoints/goes16/60min_crop_128_down_2/det/UNet_IN3_F32_SC0_BS_4_TH60_E48_BVM0_07_D2024-12-19_09:14.pt",
        "checkpoints/goes16/60min_crop_128_down_2/det/UNet_IN3_F32_SC0_BS_4_TH60_E15_BVM0_07_D2024-12-19_06:45.pt",
        "checkpoints/goes16/60min_crop_128_down_2/det/UNet_IN3_F32_SC0_BS_4_TH60_E10_BVM0_07_D2024-12-19_06:23.pt",
        "checkpoints/goes16/60min_crop_128_down_2/det/UNet_IN3_F32_SC0_BS_4_TH60_E3_BVM0_07_D2024-12-19_05:52.pt",
        "checkpoints/goes16/60min_crop_128_down_2/det/UNet_IN3_F32_SC0_BS_4_TH60_E0_BVM0_07_D2024-12-18_23:56.pt",
    ],
    "crop_512_down_8": [
        "checkpoints/goes16/60min_crop_512_down_8/det/UNet_IN3_F32_SC0_BS_4_TH60_E15_BVM0_05_D2024-12-03_18:25.pt",
        "checkpoints/goes16/60min_crop_512_down_8/det/UNet_IN3_F32_SC0_BS_4_TH60_E12_BVM0_05_D2024-12-03_08:11.pt",
        "checkpoints/goes16/60min_crop_512_down_8/det/UNet_IN3_F32_SC0_BS_4_TH60_E7_BVM0_05_D2024-12-02_15:45.pt",
        "checkpoints/goes16/60min_crop_512_down_8/det/UNet_IN3_F32_SC0_BS_4_TH60_E6_BVM0_05_D2024-12-02_13:31.pt",
        "checkpoints/goes16/60min_crop_512_down_8/det/UNet_IN3_F32_SC0_BS_4_TH60_E2_BVM0_06_D2024-12-02_05:48.pt",
        "checkpoints/goes16/60min_crop_512_down_8/det/UNet_IN3_F32_SC0_BS_4_TH60_E0_BVM0_06_D2024-12-02_02:01.pt",
    ],
    "crop_256": [
        "checkpoints/goes16/60min_crop_256_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E10_BVM0_06_D2024-12-03_00:52.pt",
        "checkpoints/goes16/60min_crop_256_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E4_BVM0_06_D2024-12-02_09:33.pt",
        "checkpoints/goes16/60min_crop_256_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E1_BVM0_06_D2024-12-02_03:57.pt",
        "checkpoints/goes16/60min_crop_256_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E0_BVM0_07_D2024-12-02_02:01.pt",
    ],
    "crop_512": [
        "checkpoints/goes16/60min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E15_BVM0_05_D2024-12-02_15:39.pt",
        "checkpoints/goes16/60min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E7_BVM0_05_D2024-12-01_16:52.pt",
        "checkpoints/goes16/60min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E6_BVM0_06_D2024-12-01_14:19.pt",
        "checkpoints/goes16/60min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E2_BVM0_06_D2024-12-01_04:06.pt",
        "checkpoints/goes16/60min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E1_BVM0_06_D2024-12-01_01:47.pt",
        "checkpoints/goes16/60min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH60_E0_BVM0_06_D2024-11-30_23:20.pt",
    ],
}

checkpoint_paths_300 = {
    "down_2": [
        "checkpoints/goes16/300min_crop_1024_down_2/det/UNet_IN3_F32_SC0_BS_4_TH300_E11_BVM0_09_D2024-12-17_04:53.pt",
        "checkpoints/goes16/300min_crop_1024_down_2/det/UNet_IN3_F32_SC0_BS_4_TH300_E10_BVM0_09_D2024-12-17_03:12.pt",
        "checkpoints/goes16/300min_crop_1024_down_2/det/UNet_IN3_F32_SC0_BS_4_TH300_E5_BVM0_10_D2024-12-16_19:23.pt",
        "checkpoints/goes16/300min_crop_1024_down_2/det/UNet_IN3_F32_SC0_BS_4_TH300_E3_BVM0_10_D2024-12-16_16:06.pt",
        "checkpoints/goes16/300min_crop_1024_down_2/det/UNet_IN3_F32_SC0_BS_4_TH300_E0_BVM0_10_D2024-12-16_11:22.pt",
    ],
    "crop_64": [
        "checkpoints/goes16/300min_crop_64/det/UNet_IN3_F32_SC0_BS_4_TH300_E15_BVM0_11_D2024-12-16_17:34.pt",
        "checkpoints/goes16/300min_crop_64/det/UNet_IN3_F32_SC0_BS_4_TH300_E6_BVM0_11_D2024-12-16_06:06.pt",
        "checkpoints/goes16/300min_crop_64/det/UNet_IN3_F32_SC0_BS_4_TH300_E5_BVM0_11_D2024-12-16_04:49.pt",
        "checkpoints/goes16/300min_crop_64/det/UNet_IN3_F32_SC0_BS_4_TH300_E0_BVM0_11_D2024-12-15_19:49.pt",
    ],
    "crop_256_down_4": [
        "checkpoints/goes16/300min_crop_256_down_4/det/UNet_IN3_F32_SC0_BS_4_TH300_E5_BVM0_10_D2024-12-14_04:50.pt",
        "checkpoints/goes16/300min_crop_256_down_4/det/UNet_IN3_F32_SC0_BS_4_TH300_E0_BVM0_11_D2024-12-14_01:32.pt",
    ],
    "crop_512": [
        "checkpoints/goes16/512min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH300_E36_BVM0_10_D2024-12-15_08:24.pt",
        "checkpoints/goes16/512min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH300_E28_BVM0_10_D2024-12-15_00:59.pt",
        "checkpoints/goes16/512min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH300_E25_BVM0_10_D2024-12-14_22:01.pt",
        "checkpoints/goes16/512min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH300_E13_BVM0_10_D2024-12-14_10:37.pt",
        "checkpoints/goes16/512min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH300_E5_BVM0_10_D2024-12-14_04:50.pt",
        "checkpoints/goes16/512min_crop_512_down_1/det/UNet_IN3_F32_SC0_BS_4_TH300_E0_BVM0_10_D2024-12-14_01:32.pt",
    ],
}


def get_checkpoint_paths(time_horizon: int):
    if time_horizon == 60:
        return checkpoint_paths_60
    elif time_horizon == 300:
        return checkpoint_paths_300
    else:
        raise ValueError("time_horizon must be 60 or 300")


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
        if crop_or_downsample == "persistence":
            return 1
        elif "crop" in crop_or_downsample and "down" in crop_or_downsample:
            down_value = int(crop_or_downsample.split("_")[3])
            return down_value
        elif "crop" in crop_or_downsample and "down" not in crop_or_downsample:
            return 1
        elif "down" in crop_or_downsample and "crop" not in crop_or_downsample:
            down_value = int(crop_or_downsample.split("_")[-1])
            return down_value
    else:
        return 1


def get_crop_size(crop_or_downsample: Optional[str], eval_crop_size: int):
    if crop_or_downsample == "persistence":
        return eval_crop_size
    if crop_or_downsample is not None:
        if "crop" in crop_or_downsample:
            return int(crop_or_downsample.split("_")[1])
    return 1024


def evaluate_model(
    unet: DeterministicUNet,
    device: str,
    dataset: str = "val",
    eval_crop_size: int = 32,
    crop_or_downsample: Optional[str] = None,
    upsample_method: str = "nearest",
    debug: bool = False,
):
    scale_factor = get_upscale_factor(crop_or_downsample)
    upsample_function = torch.nn.Upsample(
        scale_factor=scale_factor, mode=upsample_method
    )
    expected_crop_size = get_crop_size(crop_or_downsample, eval_crop_size)

    val_loss_per_batch = []  # stores values for this validation run
    val_loss_cropped_per_batch = []  # stores values for this validation run

    if dataset == "val":
        data_loader = unet.val_loader
    elif dataset == "test":
        data_loader = unet.test_loader

    with torch.no_grad():
        for val_batch_idx, (in_frames, out_frames) in enumerate(data_loader):

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
                else:
                    frames_pred_upsample_cropped = frames_pred_upsample

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

            if debug and val_batch_idx == 5:
                break

    val_loss_in_epoch = sum(val_loss_per_batch) / len(val_loss_per_batch)
    val_loss_cropped_in_epoch = sum(val_loss_cropped_per_batch) / len(
        val_loss_cropped_per_batch
    )

    return (
        val_loss_in_epoch,
        val_loss_cropped_in_epoch,
    )


def main(
    batch_size: int = 1,
    eval_crop_size: int = 32,
    debug: bool = False,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    unet_config = UNetConfig(
        in_frames=3,
        spatial_context=0,
        filters=32,
        output_activation="sigmoid",
        device=device,
    )
    unet = DeterministicUNet(config=unet_config)
    unet.model.to(device)
    dataset = "goes16"
    dataset_path = "datasets/goes16/salto/"

    results_json = {
        60: {
            "crop_128_down_2": {},
            "crop_512_down_8": {},
            "crop_256": {},
            "crop_512": {},
        },
        300: {
            "down_2": {},
            "crop_64": {},
            "crop_256_down_4": {},
            "crop_512": {},
        },
    }

    for time_horizon in [60, 300]:
        print(f"----- time_horizon: {time_horizon} -----")
        unet.create_dataloaders(
            dataset=dataset,
            path=dataset_path,
            batch_size=batch_size,
            time_horizon=time_horizon,
            binarization_method=None,  # needed for BinClassifierUNet
            crop_or_downsample=None,
            shuffle=False,
            create_test_loader=True,
        )
        checkpoint_paths = get_checkpoint_paths(time_horizon)
        for crop_or_downsample, paths in checkpoint_paths.items():
            print(f"crops: {crop_or_downsample}")
            for path in paths:
                unet.load_checkpoint(checkpoint_path=path, device=device)
                (val_loss_in_epoch, val_loss_cropped_in_epoch) = evaluate_model(
                    unet=unet,
                    device=device,
                    dataset="val",
                    eval_crop_size=eval_crop_size,
                    crop_or_downsample=crop_or_downsample,
                    debug=debug,
                )
                (
                    test_loss_in_epoch,
                    test_loss_cropped_in_epoch,
                ) = evaluate_model(
                    unet=unet,
                    device=device,
                    dataset="test",
                    eval_crop_size=eval_crop_size,
                    crop_or_downsample=crop_or_downsample,
                    debug=debug,
                )
                print(f"    - {path}")
                print(
                    f"        val_loss: {val_loss_in_epoch:.4f} , test_loss: {test_loss_in_epoch:.4f}"
                )
                print(
                    f"        val_loss_cropped: {val_loss_cropped_in_epoch:.4f} , test_loss_cropped: {test_loss_cropped_in_epoch:.4f}"
                )
                results_json[time_horizon][crop_or_downsample][path] = {
                    "val_loss": val_loss_in_epoch,
                    "val_loss_cropped": val_loss_cropped_in_epoch,
                    "test_loss": test_loss_in_epoch,
                    "test_loss_cropped": test_loss_cropped_in_epoch,
                }

    with open("val_vs_test.json", "w") as f:
        json.dump(results_json, f)


if __name__ == "__main__":
    fire.Fire(main)
