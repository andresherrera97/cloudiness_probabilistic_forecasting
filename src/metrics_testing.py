from metrics import crps_batch
from data_handlers import MovingMnistDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":

    BATCH_SIZE = 32
    NUM_BINS = 10
    INPUT_FRAMES = 3

    val_dataset = MovingMnistDataset(
        path="datasets/moving_mnist_dataset/validation/",
        input_frames=INPUT_FRAMES,
        num_bins=NUM_BINS,
        shuffle=False,
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
        print(f"in_frames: {in_frames.shape}")
        print(f"out_frames: {out_frames.shape}")

        crps_per_image = crps_batch(out_frames, out_frames)
        # print(f"crps_per_image: {crps_per_image}")
        break
