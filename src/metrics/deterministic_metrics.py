import torch


def relative_rmse(
    y_true: torch.Tensor, y_pred: torch.Tensor, pixel_wise: bool = False
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
        return torch.sqrt(torch.mean(((y_true - y_pred) ** 2) / y_true)) * 100

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
    y_true: torch.Tensor, y_pred: torch.Tensor, pixel_wise: bool = False
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
        return torch.mean(torch.abs(y_true - y_pred) / y_true) * 100

    return torch.mean(torch.abs(y_true - y_pred)) / torch.mean(y_true) * 100
