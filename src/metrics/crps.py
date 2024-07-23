import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import properscoring as ps
from math import isclose
from typing import Any, List, Optional


def crps_bin_classification(
    predictions: torch.Tensor, ground_truth: torch.Tensor
) -> float:
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a batch of images.

    Parameters:
    - predictions: Numpy array with shape (batch_size, num_bins, height, width)
                   representing the predicted probabilities for each pixel.
    - ground_truth: Numpy array with shape (batch_size, num_bins, height, width) or (batch_size, 1, height, width)
                   representing the ground truth values for each pixel. If dim 1 has shape 1, then the heavy
                   side function is calculated.

    Returns:
    - crps: float representing the mean CRPS for the batch.
    """
    if predictions.shape != ground_truth.shape and ground_truth.shape[1] != 1:
        raise ValueError(
            "The shape of the preds and gt must be the same unless the gt has len 1 in dim 1."
        )

    if ground_truth.shape[1] == 1:
        # Create binary array with 1s in the corresponding bin positions
        ground_truth = nn.functional.one_hot(
            ground_truth.to(torch.int64), num_classes=predictions.shape[1]
        )
        ground_truth = ground_truth.squeeze(1)
        ground_truth = ground_truth.permute(0, 3, 1, 2)

    n_bins = predictions.shape[1]
    # Calculate cumulative probabilities
    pred_cum_probs = torch.cumsum(predictions, axis=1)
    gt_cum_probs = torch.cumsum(ground_truth, axis=1)

    # Calculate squared difference and integrate using trapezoidal rule
    cdf_difference = (pred_cum_probs - gt_cum_probs) ** 2
    crps = torch.sum(cdf_difference, dim=1) / n_bins
    crps = torch.mean(crps)

    return crps


def crps_gaussian(
    target: torch.Tensor,
    mu: torch.Tensor,
    sig: torch.Tensor,
    sample_weight=None,
    eps: float = 1e-12,
) -> float:
    mu = mu.cpu().numpy()
    target = target.cpu().numpy()
    sig = sig.cpu().numpy()

    sig = sig + eps  # Avoid division by zero
    sx = (target - mu) / sig
    pdf = stats.norm.pdf(sx)
    cdf = stats.norm.cdf(sx)
    per_obs_crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - 1.0 / np.sqrt(np.pi))
    return np.average(per_obs_crps, weights=sample_weight)


def crps_quantile(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    quantiles: List[float],
    device: str,
):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a batch of images.

    Parameters:
    - predictions: Numpy array with shape (batch_size, num_bins, height, width)
                   representing the predicted probabilities for each pixel.
    - ground_truth: Numpy array with shape (batch_size, num_bins, height, width) or (batch_size, 1, height, width)
                   representing the ground truth values for each pixel. If dim 1 has shape 1, then the heavy
                   side function is calculated.
    - quantiles: List of floats representing the quantiles to be predicted.

    Returns:
    - crps: float representing the mean CRPS for the batch.
    """
    if predictions.shape[1] != len(quantiles):
        raise ValueError(
            "The shape of the preds in dim 1 must be equal to quantiles predicted."
        )

    predictions = torch.cat((torch.zeros_like(ground_truth), predictions), dim=1)
    predictions = torch.cat((predictions, torch.ones_like(ground_truth)), dim=1)

    quantiles = quantiles + [1]

    crps_point_wise = torch.zeros(
        (predictions.shape[0], predictions.shape[2], predictions.shape[3])
    ).to(device=device)

    for i, q in enumerate(quantiles[:-1]):
        # bin_prob = q - sum(quantiles[:i])
        bin_prob = q  # para la probabilidad acumulada creo que iria aca
        tau_i = predictions[:, i, :, :]
        tau_f = predictions[:, i + 1, :, :]

        tau_f_under_gt_mask = tau_f <= ground_truth[:, 0, :, :]
        complete_bin = bin_prob * (tau_f - tau_i)
        crps_point_wise[tau_f_under_gt_mask] += complete_bin[
            tau_f_under_gt_mask
        ].float()

        tau_f_over_gt_mask = tau_f > ground_truth[:, 0, :, :]
        bin_tau_i_target = bin_prob * (ground_truth[:, 0, :, :] - tau_i)
        bin_target_tau_f = (1 - bin_prob) * (tau_f - ground_truth[:, 0, :, :])
        crps_point_wise[tau_f_over_gt_mask] += (
            bin_tau_i_target[tau_f_over_gt_mask].float()
            + bin_target_tau_f[tau_f_over_gt_mask].float()
        )

    crps_point_wise = crps_point_wise**2  # esta bien hacer esto aca o es antes?

    crps = torch.mean(crps_point_wise)

    return crps


def crps_loss_fixed_bins(predicted_pdf, y, bin_borders):
    """
    Compute CRPS loss for fixed bin borders and predicted PDF values.

    Args:
    predicted_pdf: tensor of shape (N, B)
    y: tensor of shape (N,)
    bin_borders: tensor of shape (B+1,)

    Returns:
    CRPS loss: tensor of shape (N,)
    """
    N, B = predicted_pdf.shape
    device = predicted_pdf.device

    # Ensure inputs are on the same device
    y = y.to(device)
    bin_borders = bin_borders.to(device)

    # Compute CDF from PDF
    cdf_values = torch.cat(
        [torch.zeros(N, 1, device=device), torch.cumsum(predicted_pdf, dim=1)], dim=1
    )

    # Compute parts
    parts = (
        (cdf_values[:, :-1] + cdf_values[:, 1:])
        / 2
        * (bin_borders[1:] - bin_borders[:-1])
    )
    sq_parts = (
        -1
        / 3
        * (
            cdf_values[:, :-1] ** 2
            + cdf_values[:, :-1] * cdf_values[:, 1:]
            + cdf_values[:, 1:] ** 2
        )
        * (bin_borders[:-1] - bin_borders[1:])
    )

    # first part of the loss
    p1 = bin_borders[-1] - y
    # second part of the loss
    p2 = torch.sum(sq_parts, dim=1)

    # Find the bin index for each y
    purek = torch.searchsorted(bin_borders, y.view(-1, 1)) - 1
    k = torch.clamp(purek, 0, B - 1).squeeze()
    # Compute CDF at y using linear interpolation
    cdf_at_y = cdf_values[torch.arange(N), k] + (y - bin_borders[k]) / (
        bin_borders[k + 1] - bin_borders[k]
    ) * (cdf_values[torch.arange(N), k + 1] - cdf_values[torch.arange(N), k])

    p3 = (
        (cdf_at_y + cdf_values[torch.arange(N), k + 1]) / 2 * (bin_borders[k + 1] - y)
    ) * ((bin_borders[0] - y < 0) * (y < bin_borders[-1])).float()
    mask = torch.arange(B, device=predicted_pdf.device)[None, :] > purek
    p4 = torch.sum(parts * mask, dim=1) * ((y - bin_borders[-1] < 0))
    crps = torch.abs(p1) + p2 - 2 * (p3 + p4)

    return crps


class CRPSLoss:
    def __init__(
        self,
        num_bins: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        device: str = "cpu",
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ):
        if num_bins is not None and quantiles is not None:
            raise ValueError("Only one of num_bins or quantiles should be provided.")
        if num_bins is None and quantiles is None:
            raise ValueError("One of num_bins or quantiles should be provided.")
        self.num_bins = num_bins
        if self.num_bins is not None:
            self.bin_borders = torch.linspace(lower_bound, upper_bound, self.num_bins).to(device)
        else:
            self.bin_borders = None
        if quantiles is not None:
            self.quantiles = quantiles = (
                [lower_bound] + quantiles + [upper_bound]
            )  # Add 0 and 1 to the quantiles
            self.pdf_qr = [
                self.quantiles[i] - self.quantiles[i - 1]
                for i in range(1, len(self.quantiles))
            ]
        else:
            self.quantiles = None
            self.pdf_qr = None

        self.pdf_array_qr = None
        self.device = device

    def _crps_loss_fixed_bins_array(
        self, predicted_pdf: torch.Tensor, y: torch.Tensor, bin_borders, device: str
    ):
        """
        Compute CRPS loss for fixed bin borders and predicted PDF values.

        Args:
        predicted_pdf: tensor of shape (N, B, H, W)
        y: tensor of shape (N, H, W)
        bin_borders: tensor of shape (B+1,)

        Returns:
        CRPS loss: tensor of shape (N,)
        """
        N, B, H, W = predicted_pdf.shape
        if y.dim() == 3:
            y = y.unsqueeze(1)
        # Ensure inputs are on the same device
        bin_borders = bin_borders.to(device)

        # Compute CDF from PDF
        cdf_values = torch.cat(
            [
                torch.zeros(N, 1, H, W, device=device),
                torch.cumsum(predicted_pdf, dim=1),
            ],
            dim=1,
        )

        # Compute parts
        parts = (
            (cdf_values[:, :-1] + cdf_values[:, 1:])
            / 2
            * (bin_borders[1:] - bin_borders[:-1]).view(1, -1, 1, 1)
        )

        sq_parts = (
            -1
            / 3
            * (
                cdf_values[:, :-1] ** 2
                + cdf_values[:, :-1] * cdf_values[:, 1:]
                + cdf_values[:, 1:] ** 2
            )
            * (bin_borders[:-1] - bin_borders[1:]).view(1, -1, 1, 1)
        )

        # first part of the loss
        p1 = bin_borders[-1] - y
        # second part of the loss
        p2 = torch.sum(sq_parts, dim=1).unsqueeze(1)
        # Find the bin index for each y
        purek = torch.searchsorted(bin_borders, y) - 1
        k = torch.clamp(purek, 0, B - 1).view(N, 1, H, W)

        # Compute CDF at y using linear interpolation
        cdf_values_arange = cdf_values[
            torch.arange(N).view(N, 1, 1, 1).expand(N, 1, H, W),
            k,
            torch.arange(H).view(1, 1, H, 1).expand(N, 1, H, W),
            torch.arange(W).view(1, 1, 1, W).expand(N, 1, H, W),
        ]

        cdf_next_values_arange = cdf_values[
            torch.arange(N).view(N, 1, 1, 1).expand(N, 1, H, W),
            k + 1,
            torch.arange(H).view(1, 1, H, 1).expand(N, 1, H, W),
            torch.arange(W).view(1, 1, 1, W).expand(N, 1, H, W),
        ]

        cdf_at_y = cdf_values_arange + (y - bin_borders[k]) / (
            bin_borders[k + 1] - bin_borders[k]
        ) * (cdf_next_values_arange - cdf_values_arange)

        p3 = ((cdf_at_y + cdf_next_values_arange) / 2 * (bin_borders[k + 1] - y)) * (
            (bin_borders[0] - y < 0) * (y < bin_borders[-1])
        ).float()

        mask = torch.arange(B, device=predicted_pdf.device).view(
            1, -1, 1, 1
        ) > purek.view(N, 1, H, W)

        p4 = torch.sum(parts * mask, dim=1).unsqueeze(1) * (
            (y - bin_borders[-1] < 0).float().sum(dim=1)
        ).unsqueeze(1)

        crps = torch.abs(p1).sum(dim=1) + p2 - 2 * (p3.sum(dim=1) + p4)

        return crps

    def _crps_loss_variable_bins_array(
        self,
        predicted_pdf: torch.Tensor,
        y: torch.Tensor,
        bin_borders: torch.Tensor,
        device: str,
    ):
        """
        Compute CRPS loss for fixed bin borders and predicted PDF values.

        Args:
        predicted_pdf: tensor of shape (N, B, H, W)
        y: tensor of shape (N, H, W)
        bin_borders: tensor of shape (N, B+1, H, W)

        Returns:
        CRPS loss: tensor of shape (N,)
        """
        N, B, H, W = predicted_pdf.shape
        if y.dim() == 3:
            y = y.unsqueeze(1)

        if bin_borders.shape[1] == B - 1:
            bin_borders = torch.cat(
                [
                    torch.zeros(N, 1, H, W, device=device),
                    bin_borders,
                    torch.ones(N, 1, H, W, device=device),
                ],
                dim=1,
            )

        # Ensure inputs are on the same device

        # Compute CDF from PDF
        cdf_values = torch.cat(
            [
                torch.zeros(N, 1, H, W, device=device),
                torch.cumsum(predicted_pdf, dim=1),
            ],
            dim=1,
        )

        # Compute parts
        parts = (
            (cdf_values[:, :-1] + cdf_values[:, 1:])
            / 2
            * (bin_borders[:, 1:, :, :] - bin_borders[:, :-1, :, :])
        )

        sq_parts = (
            -1
            / 3
            * (
                cdf_values[:, :-1] ** 2
                + cdf_values[:, :-1] * cdf_values[:, 1:]
                + cdf_values[:, 1:] ** 2
            )
            * (bin_borders[:, :-1, :, :] - bin_borders[:, 1:, :, :])
        )

        # first part of the loss
        p1 = bin_borders[:, -1:, :, :] - y
        # second part of the loss
        p2 = torch.sum(sq_parts, dim=1).unsqueeze(1)
        # Find the bin index for each y
        # Reshape tensors to run searchsorted in parallel
        bin_borders_reshaped = bin_borders.permute(
            0, 2, 3, 1
        ).contiguous()  # Shape: [3, 32, 32, 7]
        y_reshaped = y.permute(0, 2, 3, 1).contiguous()  # Shape: [3, 32, 32, 1]

        # Run searchsorted
        purek = torch.searchsorted(bin_borders_reshaped, y_reshaped) - 1
        purek = purek.permute(0, 3, 1, 2)
        k = torch.clamp(purek, 0, B - 1).view(N, 1, H, W)
        # Compute CDF at y using linear interpolation
        cdf_values_arange = cdf_values[
            torch.arange(N).view(N, 1, 1, 1).expand(N, 1, H, W),
            k,
            torch.arange(H).view(1, 1, H, 1).expand(N, 1, H, W),
            torch.arange(W).view(1, 1, 1, W).expand(N, 1, H, W),
        ]

        cdf_next_values_arange = cdf_values[
            torch.arange(N).view(N, 1, 1, 1).expand(N, 1, H, W),
            k + 1,
            torch.arange(H).view(1, 1, H, 1).expand(N, 1, H, W),
            torch.arange(W).view(1, 1, 1, W).expand(N, 1, H, W),
        ]

        bin_borders_k = bin_borders[
            torch.arange(N).view(N, 1, 1, 1).expand(N, 1, H, W),
            k,
            torch.arange(H).view(1, 1, H, 1).expand(N, 1, H, W),
            torch.arange(W).view(1, 1, 1, W).expand(N, 1, H, W),
        ]

        bin_borders_k_next = bin_borders[
            torch.arange(N).view(N, 1, 1, 1).expand(N, 1, H, W),
            k + 1,
            torch.arange(H).view(1, 1, H, 1).expand(N, 1, H, W),
            torch.arange(W).view(1, 1, 1, W).expand(N, 1, H, W),
        ]

        cdf_at_y = cdf_values_arange + (y - bin_borders_k) / (
            bin_borders_k_next - bin_borders_k
        ) * (cdf_next_values_arange - cdf_values_arange)

        p3 = ((cdf_at_y + cdf_next_values_arange) / 2 * (bin_borders_k_next - y)) * (
            (bin_borders[:, :1, :, :] - y < 0) * (y < bin_borders[:, -1:, :, :])
        ).float()

        mask = torch.arange(B, device=predicted_pdf.device).view(
            1, -1, 1, 1
        ) > purek.view(N, 1, H, W)

        p4 = torch.sum(parts * mask, dim=1).unsqueeze(1) * (
            # (y - bin_borders[-1] < 0).float().sum(dim=1)
            (y - bin_borders[:, -1:, :, :] < 0)
            .float()
            .sum(dim=1)
        ).unsqueeze(1)

        crps = (
            torch.abs(p1).sum(dim=1).unsqueeze(1)
            + p2
            - 2 * (p3.sum(dim=1).unsqueeze(1) + p4)
        )

        return crps

    def crps_loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
    ):
        if self.num_bins is not None:
            return torch.mean(
                self._crps_loss_fixed_bins_array(pred, y, self.bin_borders, self.device)
            )
        elif self.quantiles is not None:
            if self.pdf_array_qr is None:
                batch_size, height, width = pred.shape[0], pred.shape[2], pred.shape[3]
                self.pdf_array_qr = torch.ones(
                    (batch_size, len(self.quantiles) - 1, height, width)
                ).to(self.device)
                for n in range(self.pdf_array_qr.shape[1]):
                    self.pdf_array_qr[:, n, :, :] = self.pdf_qr[n]

            return torch.mean(
                self._crps_loss_variable_bins_array(
                    predicted_pdf=self.pdf_array_qr,
                    y=y,
                    bin_borders=pred,
                    device=self.device,
                )
            )


def memoize(func):
    cache = {}

    def memoized_func(*args):
        key = str(args)
        if key not in cache:
            cache[key] = func(*args)
        return cache[key]

    return memoized_func


@memoize
def cached_bin_prob(
    lower_bound, upper_bound, quantiles, batch_size, height, width, device
):
    quantiles = (
        [lower_bound] + quantiles + [upper_bound]
    )  # Add 0 and 1 to the quantiles
    bin_prob = [quantiles[i] - quantiles[i - 1] for i in range(1, len(quantiles))]
    pdf_array_qr = torch.ones((batch_size, len(quantiles) - 1, height, width)).to(
        device
    )
    for n in range(pdf_array_qr.shape[1]):
        pdf_array_qr[:, n, :, :] = bin_prob[n]
    return pdf_array_qr


@memoize
def cached_linspace(lower_bound, upper_bound, num_bins, device):
    return torch.linspace(lower_bound, upper_bound, num_bins).to(device)


if __name__ == "__main__":
    N_BINS = 10
    BATCH_SIZE = 8
    IMG_SIZE = 32

    print("=== TESTING: CRPS_BIN ===")
    bin_target = torch.rand(
        BATCH_SIZE, N_BINS, IMG_SIZE, IMG_SIZE
    )  # (batch_size, num_bins, height, width)
    if crps_bin_classification(bin_target, bin_target) == 0:
        print("CORRECT: CRPS is 0 for identical predictions and ground truth")
    else:
        print("INCORRECT: CRPS is not 0 for identical predictions and ground truth")

    step_target = torch.zeros(
        (BATCH_SIZE, N_BINS, IMG_SIZE, IMG_SIZE)
    )  # (batch_size, num_bins, height, width)
    step_prediction_lower = torch.zeros(
        (BATCH_SIZE, N_BINS, IMG_SIZE, IMG_SIZE)
    )  # (batch_size, num_bins, height, width)
    step_prediction_higher = torch.zeros(
        (BATCH_SIZE, N_BINS, IMG_SIZE, IMG_SIZE)
    )  # (batch_size, num_bins, height, width)
    step_target[:, 6, :, :] = 1
    step_prediction_lower[:, 5, :, :] = 1
    step_prediction_higher[:, 7, :, :] = 1
    integer_target = torch.ones((BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)) * 6

    if crps_bin_classification(
        step_prediction_lower, step_target
    ) == crps_bin_classification(step_prediction_higher, step_target):
        print("CORRECT: CRPS is equal for predictions with same distance from target")
    else:
        print(
            "INCORRECT: CRPS is not equal for predictions with same distance from target"
        )

    if (
        round(crps_bin_classification(step_prediction_higher, step_target).item(), 3)
        == 1 / N_BINS
    ):
        print(
            f"CORRECT: CRPS is 1/N_BINS ({1/N_BINS:.2f}) for predictions with one bin higher than target"
        )
    else:
        print(
            f"INCORRECT: CRPS is not 1/N_BINS ({1/N_BINS:.2f}) for predictions with one bin higher than target"
        )
    if crps_bin_classification(
        step_prediction_higher, step_target
    ) == crps_bin_classification(step_prediction_higher, integer_target):
        print("CORRECT: CRPS is equal for bin target and integer target")
    else:
        print("INCORRECT: CRPS is not equal for bin target and integer target")

    # print("=== TESTING: CRPS_GAUSSIAN ===")
    # # create random array with size (batch_size, height, width)
    # target = np.random.rand(BATCH_SIZE, IMG_SIZE, IMG_SIZE)
    # moved_mean = target + np.random.rand(BATCH_SIZE, IMG_SIZE, IMG_SIZE)
    # big_std_array = np.ones_like(target) * 3
    # small_std_array = np.ones_like(target) * 0.01
    # if crps_gaussian(target, target, big_std_array) > crps_gaussian(
    #     target, target, small_std_array
    # ):
    #     print("CORRECT: CRPS increases with std deviation")
    # else:
    #     print("INCORRECT: CRPS does not increase with std deviation")

    # if crps_gaussian(target, moved_mean, small_std_array) > crps_gaussian(
    #     target, target, small_std_array
    # ):
    #     print("CORRECT: CRPS increases with mu distance from target")
    # else:
    #     print("INCORRECT: CRPS does not increases with mu distance from target")

    # one_target_array = np.ones_like(target)
    # mean_target_array = np.ones_like(target) * 0.3
    # std_target_array = np.ones_like(target) * 0.5

    # if isclose(
    #     crps_gaussian(one_target_array, mean_target_array, std_target_array),
    #     ps.crps_gaussian(1, mu=0.3, sig=0.5),
    # ):
    #     print("CORRECT: CRPS is equal to properscoring package")
    # else:
    #     print("INCORRECT: CRPS is not equal to properscoring package")

    print("\n=== TESTING: CRPS LOSS ===\n")

    # 1D function

    LOWER_BOUND = 0.0
    UPPER_BOUND = 1.0
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    NUM_BINS = 5

    BIN_BORDERS = cached_linspace(LOWER_BOUND, UPPER_BOUND, NUM_BINS + 1, DEVICE)
    
    crps_loss_1 = CRPSLoss(
        num_bins=5,
        device=DEVICE,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
    )

    # predicted_pdf: tensor of shape (N, B)
    # y: tensor of shape (N,)
    predicted_pdf_1d = torch.tensor([[0.1, 0.1, 0.4, 0.3, 0.1]])
    y_1d = torch.tensor([0.5])

    crps_1d = crps_loss_fixed_bins(predicted_pdf_1d, y_1d, BIN_BORDERS)
    print(f"crps_1d: {crps_1d}\n")

    # uniform distribution
    predicted_pdf_1d = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
    y_1d = torch.tensor([0.5])

    crps_1d = crps_loss_fixed_bins(predicted_pdf_1d, y_1d, BIN_BORDERS)
    print(f"crps_1d uniform: {crps_1d}\n")


    crps_loss_1 = CRPSLoss(num_bins=NUM_BINS, device=DEVICE,)
    # predicted_pdf: tensor of shape (N, B, H, W)
    # y: tensor of shape (N, H, W)
    predicted_pdf_array = torch.ones((3, 5, 32, 32))
    predicted_pdf_array[:, 0, :, :] = 0.1
    predicted_pdf_array[:, 1, :, :] = 0.1
    predicted_pdf_array[:, 2, :, :] = 0.4
    predicted_pdf_array[:, 3, :, :] = 0.3
    predicted_pdf_array[:, 4, :, :] = 0.1

    # predicted_pdf_array[:, 0, 1, 1] = 0.4
    # predicted_pdf_array[:, 1, 1, 1] = 0.3
    # predicted_pdf_array[:, 2, 1, :] = 0.2
    # predicted_pdf_array[:, 3, 1, 1] = 0.05
    # predicted_pdf_array[:, 4, 1, 1] = 0.05

    y_array = torch.ones((3, 1, 32, 32)) * 0.5
    # y_array[0, 0, 2, 2] = 0.1
    # y_array[0, 0, 3, 3] = 0.55
    crps_array = crps_loss_1._crps_loss_fixed_bins_array(
        predicted_pdf_array, y_array, BIN_BORDERS, DEVICE
    )
    print(f" -> crps_array: {torch.mean(crps_array)}\n")

    predicted_pdf_array = torch.ones((3, 5, 32, 32))
    predicted_pdf_array[:, 0, :, :] = 0.2
    predicted_pdf_array[:, 1, :, :] = 0.2
    predicted_pdf_array[:, 2, :, :] = 0.2
    predicted_pdf_array[:, 3, :, :] = 0.2
    predicted_pdf_array[:, 4, :, :] = 0.2

    y_array = torch.ones((3, 1, 32, 32)) * 0.5
    crps_array = crps_loss_1._crps_loss_fixed_bins_array(
        predicted_pdf_array, y_array, BIN_BORDERS, DEVICE
    )
    print(f" -> crps_array uniform: {torch.mean(crps_array)}\n")

    # CRPS loss for Quantile Regressor
    
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.90]
    # quantiles = [0] + quantiles + [1]  # Add 0 and 1 to the quantiles
    # bin_prob = [quantiles[i] - quantiles[i - 1] for i in range(1, len(quantiles))]
    
    crps_loss_qr_1 = CRPSLoss(quantiles=quantiles, device=DEVICE)
    
    predicted_pdf_array_qr = torch.ones((3, len(crps_loss_qr_1.quantiles) - 1, 32, 32))
    for n in range(predicted_pdf_array_qr.shape[1]):
        predicted_pdf_array_qr[:, n, :, :] = crps_loss_qr_1.pdf_qr[n]
    
    bin_borders_qr = torch.ones((3, len(crps_loss_qr_1.quantiles), 32, 32))
    bin_borders_qr[:, 0, :, :] = 0.0
    bin_borders_qr[:, 1, :, :] = 0.12
    bin_borders_qr[:, 2, :, :] = 0.27
    bin_borders_qr[:, 3, :, :] = 0.52
    bin_borders_qr[:, 4, :, :] = 0.77
    bin_borders_qr[:, 5, :, :] = 0.92
    bin_borders_qr[:, 6, :, :] = 1.0

    y_array = torch.ones((3, 1, 32, 32)) * 0.5
    crps_array_qr = crps_loss_qr_1._crps_loss_variable_bins_array(
        predicted_pdf=predicted_pdf_array_qr,
        y=y_array,
        bin_borders=bin_borders_qr,
        device=DEVICE,
    )
    print(f" -> crps_array qr: {torch.mean(crps_array_qr)}\n")

    # uniform distribution
    quantiles = [0.2, 0.4, 0.6, 0.8]
    # quantiles = [0] + quantiles + [1]  # Add 0 and 1 to the quantiles
    # bin_prob = [quantiles[i] - quantiles[i - 1] for i in range(1, len(quantiles))]
    crps_loss_qr_2 = CRPSLoss(quantiles=quantiles, device=DEVICE)
    predicted_pdf_array_qr = torch.ones((3, len(crps_loss_qr_2.quantiles) - 1, 32, 32))
    for n in range(predicted_pdf_array_qr.shape[1]):
        predicted_pdf_array_qr[:, n, :, :] = crps_loss_qr_2.pdf_qr[n]
    bin_borders_qr = torch.ones((3, len(crps_loss_qr_2.quantiles), 32, 32))
    bin_borders_qr[:, 0, :, :] = 0.0
    bin_borders_qr[:, 1, :, :] = 0.2
    bin_borders_qr[:, 2, :, :] = 0.4
    bin_borders_qr[:, 3, :, :] = 0.6
    bin_borders_qr[:, 4, :, :] = 0.8
    bin_borders_qr[:, 5, :, :] = 1.0

    y_array = torch.ones((3, 1, 32, 32)) * 0.5
    crps_array_qr = crps_loss_qr_2._crps_loss_variable_bins_array(
        predicted_pdf=predicted_pdf_array_qr,
        y=y_array,
        bin_borders=bin_borders_qr,
        device=DEVICE,
    )
    print(f" -> crps_array qr uniform: {torch.mean(crps_array_qr)}\n")
