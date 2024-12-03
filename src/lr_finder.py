import torch
import fire
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from models import DeterministicUNet, UNetConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LR Finder")


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Save the initial model and optimizer states
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

    def range_test(
        self,
        train_loader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5,
    ):
        """
        Performs the learning rate range test.

        Args:
            train_loader: DataLoader for training data
            start_lr: starting learning rate
            end_lr: ending learning rate
            num_iter: number of iterations to run the test
            smooth_f: smoothing factor for loss values
            diverge_th: threshold for detecting divergence
        """
        # Reset model and optimizer to initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        # Initialize learning rate multiplier
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)

        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr

        # Initialize lists to store results
        learning_rates = []
        losses = []
        best_loss = None
        avg_loss = 0.0

        # Create iterator for training data
        train_iter = iter(train_loader)

        for iteration in range(num_iter):
            # Get batch of training data
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)

            # Move data to device
            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device).float()

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update running loss
            avg_loss = (
                smooth_f * loss.item() + (1 - smooth_f) * avg_loss
                if iteration > 0
                else loss.item()
            )
            smooth_loss = avg_loss / (1 - smooth_f ** (iteration + 1))

            # Store results
            learning_rates.append(start_lr * lr_mult**iteration)
            losses.append(smooth_loss)

            # Update best loss
            if best_loss is None or smooth_loss < best_loss:
                best_loss = smooth_loss

            # Check for divergence
            if smooth_loss > diverge_th * best_loss:
                print("Stopping early - loss is diverging")
                break

            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= lr_mult

        return learning_rates, losses

    def plot(self, learning_rates, losses, sufix=""):
        """Plots the learning rate range test results."""
        plt.figure(figsize=(10, 6))
        plt.semilogx(learning_rates, losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True)
        plt.savefig(f"figures/lr_finder_{sufix}.png")
        plt.close()
        print(f"Figure saved as figures/lr_finder_{sufix}.png")

    def suggest_lr(self, learning_rates, losses, skip_start=10, skip_end=5):
        """
        Suggests a learning rate based on the test results.
        Returns the learning rate with the steepest negative gradient.
        """
        losses = np.array(losses)
        learning_rates = np.array(learning_rates)

        # Skip the first few and last few points
        losses = losses[skip_start:-skip_end]
        learning_rates = learning_rates[skip_start:-skip_end]

        # Calculate gradients
        gradients = (losses[1:] - losses[:-1]) / (
            learning_rates[1:] - learning_rates[:-1]
        )

        # Find the point with steepest negative gradient
        steepest_idx = np.argmin(gradients)

        return learning_rates[steepest_idx]


def main(
    filters: int = 32,
    spatial_context: int = 0,
    batch_size: int = 4,
    time_horizon: int = 60,
    crop_or_downsample: str = None,
    start_lr: float = 1e-7,
    end_lr: float = 10,
    num_iter: int = 100,
    smooth_f: float = 0.05,
    diverge_th: float = 5,
):
    # Load model, optimizer, criterion, and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_config = UNetConfig(
        in_frames=3,
        spatial_context=spatial_context,
        filters=filters,
        output_activation="sigmoid",
        device=device,
    )
    probabilistic_unet = DeterministicUNet(config=unet_config)

    logger.info("Initializing model...")
    probabilistic_unet.model.to(device)
    probabilistic_unet.initialize_weights()
    optimizer = torch.optim.AdamW(probabilistic_unet.model.parameters())
    # initialize dataloaders

    dataset = "goes16"
    dataset_path = "datasets/goes16/salto/"

    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Time horizon: {time_horizon}")

    probabilistic_unet.create_dataloaders(
        dataset=dataset,
        path=dataset_path,
        batch_size=batch_size,
        time_horizon=time_horizon,
        binarization_method=None,  # needed for BinClassifierUNet
        crop_or_downsample=crop_or_downsample,
    )

    criterion = probabilistic_unet.loss_fn
    train_loader = probabilistic_unet.train_loader

    # Initialize LR finder
    lr_finder = LRFinder(probabilistic_unet.model, optimizer, criterion, device)

    # Run LR range test
    learning_rates, losses = lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        smooth_f=smooth_f,
        diverge_th=diverge_th,
    )

    # Plot results
    sufix = f"filters_{filters}_batch_{batch_size}_time_{time_horizon}_crop_{crop_or_downsample}"
    lr_finder.plot(learning_rates, losses, sufix)

    # Suggest learning rate
    lr = lr_finder.suggest_lr(learning_rates, losses)
    logger.info(f"Suggested LR: {lr}")


if __name__ == "__main__":
    fire.Fire(main)
