import torch
from typing import Dict


class DeterministicMetrics:
    def __init__(self):
        self.per_batch_metrics = None

    def start_epoch(self) -> None:
        self.per_batch_metrics = {
            "relative_rmse": [],
            "relative_mae": [],
            "forecasting_skill": [],
            "rmse": [],
        }

    def end_epoch(self) -> Dict[str, float]:
        return {
            key: torch.mean(torch.tensor(values)).item()
            for key, values in self.per_batch_metrics.items()
        }

    def return_per_batch_metrics(self) -> Dict[str, float]:
        return self.per_batch_metrics

    def rmse(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Computes the root mean squared error.

        Parameters
        ----------
        y_true : torch.tensor
            True values.
        y_pred : torch.tensor
            Predicted values.

        Returns
        -------
        float
            Root mean squared error.
        """
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    def relative_rmse(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        pixel_wise: bool = False,
        eps: float = 1e-5,
    ):
        """
        Computes the relative root mean squared error.

        Parameters
        ----------
        y_true : torch.tensor
            True values.
        y_pred : torch.tensor
            Predicted values.

        Returns
        -------
        float
            Relative root mean squared error.
        """

        if pixel_wise:
            return (
                torch.sqrt(torch.mean(((y_true - y_pred) ** 2) / (y_true + eps))) * 100
            )

        squared_diff = (y_true - y_pred) ** 2

        # Calculate the mean of the squared differences
        mean_squared_diff = torch.mean(squared_diff)

        # Calculate the root mean squared deviation (RMSD)
        rmsd = torch.sqrt(mean_squared_diff)

        # Calculate the mean of the ground truth
        mean_ground_truth = torch.mean(y_true)

        # Calculate the relative RMSD (RRMSD)
        rrmsd = rmsd / mean_ground_truth

        return rrmsd * 100

    def relative_mae(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        pixel_wise: bool = False,
        eps: float = 1e-5,
    ):
        """
        Computes the relative mean absolute error.

        Parameters
        ----------
        y_true : torch.tensor
            True values.
        y_pred : torch.tensor
            Predicted values.

        Returns
        -------
        float
            Relative mean absolute error.
        """
        if pixel_wise:
            return torch.mean(torch.abs(y_true - y_pred) / (y_true + eps)) * 100

        return torch.mean(torch.abs(y_true - y_pred)) / torch.mean(y_true) * 100

    def forecasting_skill(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, y_persistence: torch.Tensor
    ):
        """
        Computes the forecasting skill.

        Parameters
        ----------
        y_true : torch.tensor
            True values.
        y_pred : torch.tensor
            Predicted values.
        y_persistence : torch.tensor
            Persistence forecast values.

        Returns
        -------
        float
            Forecasting skill.
        """
        if y_pred.shape != y_persistence.shape:
            raise ValueError(
                "Shapes of y_pred and y_persistence must be equal. "
                f"Got {y_pred.shape} and {y_persistence.shape}."
            )
        if (
            y_true.shape[0] != y_pred.shape[0]
            or y_true.shape[0] != y_persistence.shape[0]
        ):
            raise ValueError(
                "Batch size of y_true, y_pred and y_persistence must be equal."
                f"Got {y_true.shape[0]}, {y_pred.shape[0]} and {y_persistence.shape[0]}."
            )
        if (
            y_true.shape[2] != y_pred.shape[2]
            or y_true.shape[2] != y_persistence.shape[2]
        ):
            raise ValueError(
                "Height of y_true, y_pred and y_persistence must be equal. "
                f"Got {y_true.shape[0]}, {y_pred.shape[0]} and {y_persistence.shape[0]}."
            )
        if (
            y_true.shape[3] != y_pred.shape[3]
            or y_true.shape[3] != y_persistence.shape[3]
        ):
            raise ValueError(
                "Width of y_true, y_pred and y_persistence must be equal. "
                f"Got {y_true.shape[0]}, {y_pred.shape[0]} and {y_persistence.shape[0]}."
            )
        numerator = torch.mean((y_true - y_pred) ** 2)
        denominator = torch.mean((y_true - y_persistence) ** 2)

        return 1 - numerator / denominator

    def run_per_batch_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_persistence: torch.Tensor,
        pixel_wise: bool = False,
        eps: float = 1e-5,
    ) -> None:
        """
        Run all metrics.

        Parameters
        ----------
        y_true : torch.tensor
            True values.
        y_pred : torch.tensor
            Predicted values.
        y_persistence : torch.tensor
            Persistence forecast values.

        Returns
        -------
        dict
            Dictionary containing all metrics.
        """
        self.per_batch_metrics["relative_rmse"].append(
            self.relative_rmse(y_true, y_pred, pixel_wise, eps)
        )
        self.per_batch_metrics["relative_mae"].append(
            self.relative_mae(y_true, y_pred, pixel_wise, eps)
        )
        self.per_batch_metrics["forecasting_skill"].append(
            self.forecasting_skill(y_true, y_pred, y_persistence)
        )
        self.per_batch_metrics["rmse"].append(self.rmse(y_true, y_pred))

    def run_all_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_persistence: torch.Tensor,
        pixel_wise: bool = False,
        eps: float = 1e-5,
    ) -> Dict[str, float]:
        """
        Run all metrics.

        Parameters
        ----------
        y_true : torch.tensor
            True values.
        y_pred : torch.tensor
            Predicted values.
        y_persistence : torch.tensor
            Persistence forecast values.

        Returns
        -------
        dict
            Dictionary containing all metrics.
        """
        return {
            "relative_rmse": self.relative_rmse(y_true, y_pred, pixel_wise, eps).item(),
            "relative_mae": self.relative_mae(y_true, y_pred, pixel_wise, eps).item(),
            "forecasting_skill": self.forecasting_skill(y_true, y_pred, y_persistence).item(),
            "rmse": self.rmse(y_true, y_pred).item(),
        }
