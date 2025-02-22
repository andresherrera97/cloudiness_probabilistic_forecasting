import time
import timm
import tqdm
from pathlib import Path
import torch
import numpy as np
import re
from functools import partial
from contextlib import suppress
from torch.utils.tensorboard import SummaryWriter

# import schedulefree
# cadamw comes from https://github.com/kyleliang919/C-Optim/blob/main/c_adamw.py
from cadamw import AdamW


from datetime import datetime, timedelta


def extract_time(path):
    # Extracts HHMMSS from something like '..._UTC_102020.npy'
    m = re.search(r"_UTC_(\d{6})\.npy$", str(path))
    return datetime.strptime(m.group(1), "%H%M%S") if m else None


def find_samples(
    files, period_in_seconds=600, horizon_in_minutes=80, tolerance_in_seconds=60
):
    samples = []
    times = [extract_time(f) for f in files]
    print("extracting samples...")
    for i in tqdm.tqdm(range(len(files) - 3)):
        t1, t2, t3 = times[i : i + 3]
        # Check consecutive images: each roughly 1 minute apart (tolerance: tol seconds)
        if (
            abs((t2 - t1).total_seconds() - 600) > tolerance_in_seconds
            or abs((t3 - t2).total_seconds() - 600) > tolerance_in_seconds
        ):
            continue
        # Fourth image should be ~h minutes after the third (±tol seconds)
        expected = t3 + timedelta(minutes=horizon_in_minutes)

        j = horizon_in_minutes // 10
        visited_j = set()
        while True:
            if (j in visited_j) or (i + 2 + j >= len(files)) or (i + 2 + j < 0):
                break
            visited_j.add(j)
            t4 = times[i + 2 + j]
            time_diff = (t4 - expected).total_seconds()
            if time_diff <= tolerance_in_seconds:
                sample = files[i : i + 3] + [files[i + 2 + j]]
                samples.append(sample)
                break
            else:
                # move in the direction of the expected time
                j = j + 1 if time_diff < 0 else j - 1
    return samples


# Example usage:
# files = [Path(...), ...]  # your sorted list of Path objects
# valid_samples = find_samples(files, h=80)


class GOES16Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        split="train",
        h=60,
        crop_size=224,
        original_resolution=1024,
        split_portion=0.7,
    ):
        """
        h is time horizon, in minutes, from the last image in the input and the target image
        """
        self.dataset_path = Path(dataset_path)
        print("listing images...")
        all_images = sorted(list(self.dataset_path.glob("**/*.npy")))
        image_day = [path.parent.name for path in all_images]
        unique_days = sorted(list(set(image_day)))
        train_days = unique_days[: int(len(unique_days) * split_portion)]
        is_train = [iday in train_days for iday in image_day]
        if split == "train":
            self.images = [img for img, train in zip(all_images, is_train) if train]
        else:
            self.images = [img for img, train in zip(all_images, is_train) if not train]
        print(f"got {len(self.images)} images")
        # samples are sequences of 3 consecutive images and one target image 1 hour afterwards
        self.samples = find_samples(
            self.images, horizon_in_minutes=h, tolerance_in_seconds=59
        )
        if crop_size:
            self.low, self.high = (original_resolution - crop_size) // 2, (
                original_resolution + crop_size
            ) // 2
        else:
            self.low, self.high = 0, original_resolution

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        files_in_sample = self.samples[idx]
        data = [
            np.load(f)[self.low : self.high, self.low : self.high]
            for f in files_in_sample
        ]
        sample = np.stack(data) / 255.0
        return sample


class I2IViT(torch.nn.Module):
    def __init__(self, model_name="vit_large_patch16_224"):
        super().__init__()
        self.encoder = timm.create_model(model_name)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                self.encoder.num_features, 256, kernel_size=2, stride=2
            ),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder.forward_features(x)[:, 1:]  # skip cls token (I think, check)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, 14, 14)
        x = self.decoder(x)
        return x


class PrefetchLoader:
    def __init__(
        self,
        loader,
        device,
        img_dtype=torch.float32,
    ):
        self.loader = loader
        self.device = device
        self.img_dtype = img_dtype
        self.is_cuda = torch.cuda.is_available() and device.type == "cuda"

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress
        for next_input in self.loader:
            with stream_context():
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_input = next_input.to(self.img_dtype)
            if not first:
                yield input
            else:
                first = False
            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)
            input = next_input
        yield input

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def main(
    dataset_path="data/goes16/downloads/salto1024_all",
    h=60,
    # Ensure we use GPU if available
    device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    batch_size=32,
    num_workers=32,
    epochs=1,
    lr=1e-2,
):
    # Initialize dataset and dataloader
    dataset = GOES16Dataset(dataset_path, h=h)
    dataloader = PrefetchLoader(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        ),
        device,
    )

    # Model setup
    model = I2IViT().to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(dataloader), pct_start=0.05
    )
    criterion = torch.nn.L1Loss()  # MAE Loss

    # TensorBoard logger
    writer = SummaryWriter()
    model.train()
    # optimizer.train()
    st = time.time()
    for epoch in range(epochs):
        for step, sample in enumerate(dataloader):
            x, y = sample[:, :3], sample[:, 3:]
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(
                f"Epoch {epoch+1}, Step {step+1} / {len(dataloader)}, {(step+1)/(time.time()-st)} steps/s: Loss = {loss.item()}",
                end="\r",
            )
            writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + step)

    writer.close()
    print("Training complete!")


main()
