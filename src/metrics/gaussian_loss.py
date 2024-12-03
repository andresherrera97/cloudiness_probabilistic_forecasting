import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss that prevents variance collapse
    """

    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, predictions, targets):
        mu = predictions[:, 0, :, :]
        # Add small epsilon to prevent variance collapse
        sigma = F.softplus(predictions[:, 1, :, :]) + self.eps

        targets = targets.unsqueeze(1)

        # Gaussian NLL: -log(p(x|μ,σ))
        loss = 0.5 * (torch.log(2 * torch.pi * sigma) + (targets - mu) ** 2 / sigma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def test_loss_behavior():
    """
    Test function to verify loss behavior with different mean/variance predictions
    """
    # Create sample data
    batch_size = 2
    height = width = 3
    predictions = torch.zeros(batch_size, 2, height, width)
    targets = torch.ones(batch_size, height, width)

    # Test cases
    test_cases = [
        {"mu": 1.0, "log_sigma": -5.0},  # Very small variance
        {"mu": 1.0, "log_sigma": 0.0},  # Unit variance
        {"mu": 1.0, "log_sigma": 5.0},  # Large variance
        {"mu": 5.0, "log_sigma": 0.0},  # Large mean difference
    ]

    criterion = GaussianNLLLoss()

    print("Testing loss behavior:")
    for case in test_cases:
        predictions[:, 0, :, :] = case["mu"]  # Set mean
        predictions[:, 1, :, :] = case["log_sigma"]  # Set log-variance

        loss = criterion(predictions, targets)
        print(
            f"μ={case['mu']:.1f}, log(σ)={case['log_sigma']:.1f} → loss={loss.item():.4f}"
        )


# You can also use PyTorch's built-in GaussianNLLLoss
def using_pytorch_gaussian_nll():
    criterion = torch.nn.GaussianNLLLoss(eps=1e-6, reduction="mean")

    # For PyTorch's implementation, variance needs to be passed directly (not log variance)
    def compute_loss(predictions, targets):
        mu = predictions[:, 0, :, :]
        # Convert log-variance to variance
        var = F.softplus(predictions[:, 1, :, :]) + 1e-6
        return criterion(mu, targets.unsqueeze(1), var)


# Example usage:
if __name__ == "__main__":
    test_loss_behavior()
