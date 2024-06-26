# USAGE:
#   Barcharts, plots, graphs, histograms, functions to view dataset, etc
#

import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from typing import Optional, List
import torch


def show_image_list(
    images_list: List[np.ndarray],
    rows: int = 1,
    fig_name: Optional[str] = None,
    show_fig: bool = False,
    save_fig: bool = False,
) -> None:
    """Shows the images passed in a grid

    Args:
        images_list (list): Each element is a numpy array
        rows (int): Number of rows in the grid
        fig_name (str): Name of the file to save the figure
        show_fig (bool): If True, shows the figure
        save_fig (bool): If True, saves the figure
    """
    len_list = len(images_list)

    if len_list % rows == 0:
        cols = len_list // rows
    else:
        cols = len_list // rows + 1

    plt.figure(figsize=(cols * 5, rows * 6))
    for num, img in enumerate(images_list):

        plt.subplot(rows, cols, num + 1)
        plt.title("img_" + str(num))
        plt.axis("off")
        plt.imshow(img)
        num += 1

    plt.subplots_adjust(wspace=1, hspace=1)
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name)
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_quantile_predictions(
    input_images: torch.Tensor,
    pred_quantiles: torch.Tensor,
    quantiles: List[float],
    target_img: torch.Tensor,
    pixel_coords: tuple[int, int],
    fig_name: Optional[str] = None,
    width: int = 3,
) -> None:
    """
    Plots the input time series and the quantile predictions.

    Parameters:
        time_series (array-like): The input time series.
        quantile_predictions (dict): A dictionary where keys are quantile values (floats between 0 and 1),
                                     and values are arrays containing predictions for each quantile.
        quantiles (list): A list of quantiles to plot.

    Returns:
        None
    """

    time_series = input_images[:, pixel_coords[0], pixel_coords[1]].cpu().tolist()
    quantile_predictions = pred_quantiles[:, pixel_coords[0], pixel_coords[1]].cpu().tolist()
    target_value = target_img[0, pixel_coords[0], pixel_coords[1]].cpu()

    # Plot the input time series
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, "-*")

    # Plot the target value
    plt.plot(len(time_series) + 1, target_value, "ro", label="Target")

    # Get length of the input time series
    n = len(time_series)

    # Plot quantile predictions after the input time series ends
    for i, quantile in enumerate(quantiles):
        quantile_values = [quantile_predictions[i]] * width

        plt.plot(
            np.arange(n, n + len(quantile_values)),
            quantile_values,
            label=f"$q_{{{quantile:.2f}}}$",
            color=f"C{i}",
        )

        # Fill the area between quantile predictions
        if i > 0:
            prev_quantile_values = [quantile_predictions[i - 1]] * width

            plt.fill_between(
                np.arange(n, n + len(quantile_values)),
                prev_quantile_values,
                quantile_values,
                color=f"C{i}",
                alpha=0.3,
            )

    plt.xlabel("Time")
    plt.ylabel("Pixel Value")
    plt.title("Time Series with Quantile Predictions")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), reverse=True)
    plt.tight_layout()

    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()


def matrix_graph(error_array):
    """
    Takes an array where the rows advance with time of the day and each column
    is the error of the prediction.

    Args:
        error_array (array): Array containing the values of the error of a prediction
    """
    fig, (ax1) = plt.subplots(figsize=(13, 3), ncols=1)
    grafica = ax1.imshow(error_array, interpolation="none")
    # grafica = ax1.imshow(error_array[70:100], interpolation='none')
    fig.colorbar(grafica, ax=ax1)
    ax1.title.set_text("One day error")
    ax1.set_xlabel("Prediction horizon ")
    ax1.set_ylabel("Day time")

    plt.show()


def barchart_compare2(
    model1_values, model1_name, model2_values, model2_name, error_metric="RMSE"
):
    """Takes the errors list of different models and plots them in a bar chart

    Args:
        model1_values (list): [description]
        model2_values (list): [description]

    Raises:
        ValueError: If lists don't have the same length
    """

    labels = []
    if len(model1_values) != len(model2_values):
        raise ValueError("Lists must have the same length.")

    for i in range(len(model1_values)):
        labels.append(str(10 * (i + 1)))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(x - width / 2, model1_values, width, label=model1_name)
    ax.bar(x + width / 2, model2_values, width, label=model2_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Error Metric: " + error_metric)
    ax.set_xlabel("Predict horizon (min)")
    ax.set_title("Error metric comparisson")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def barchart_compare3(
    model1_values, model1_name, model2_values, model2_name, model3_values, model3_name
):
    """Takes the errors list of different models and plots them in a bar chart

    Args:
        model1_values (list): [description]
        model2_values (list): [description]
        model3_values (list): [description]

    Raises:
        ValueError: If lists don't have the same length
    """

    labels = []
    if (
        len(model1_values) != len(model2_values)
        or len(model1_values) != len(model3_values)
    ) or len(model3_values) != len(model2_values):
        raise ValueError("Lists must have the same length.")

    for i in range(len(model1_values)):
        labels.append(str(10 * (i + 1)))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(x - width / 2, model1_values, width / 2, label=model1_name)
    ax.bar(x, model2_values, width / 2, label=model2_name)
    ax.bar(x + width / 2, model3_values, width / 2, label=model3_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Error metric")
    ax.set_xlabel("Predict horizon (min)")
    ax.set_title("Error metric comparisson")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_graph(model_values, model, error_metric="RMSE"):
    """Plots the errors of the predictions for a generated sequence

    Args:
        model_values (list): List containing the values of the errors
    """
    labels = []
    for i in range(len(model_values)):
        labels.append(str(10 * (i + 1)))

    plt.plot(labels, model_values, "r.")
    plt.title(model + "Model Error")
    plt.xlabel("Predict horizon (min)")
    plt.ylabel("Error Metric: " + error_metric)
    plt.show()


def plot_graph_multiple(
    models_values,
    models_names,
    error_metric="RMSE",
    save_file=None,
    labels=None,
    title=None,
    xlabel=None,
    ylabel=None,
):
    """Plots the errors of the predictions for multiple generated sequences

    Args:
        model_values (list): List containing lists with the values of the errors for each model
        models_names (list): list containing the model names
    """

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    if labels is None:
        labels = []
        for i in range(len(models_values[0])):
            labels.append(str(10 * (i + 1)))

    for i in range(len(models_values)):
        plt.plot(labels, models_values[i], colors[i % 10], label=models_names[i])

    if title:
        plt.title(title)
    else:
        plt.title("Error Metric: " + error_metric)
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel("Predict horizon (min)")
    if ylabel:
        plt.ylabel(xlabel)
    plt.legend()
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def show_seq_and_pred(
    sequence_array, time_list, prediction_t, fig_name=None, save_fig=False, grid=True
):
    """Shows the images passed in a grid
    Args:
        sequence_array (array)
    """
    nbof_frames = sequence_array.shape[0]
    if np.max(sequence_array[0]) > 1.1:
        vmax = 100
    else:
        vmax = 1

    fontsize = 22  # 22 generates the font more like the latex text

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    # plt.figure(figsize=(25, 5))
    fig, ax = plt.subplots(1, nbof_frames, figsize=(6 * nbof_frames, 5))
    plt.subplots_adjust(wspace=0.01)
    for i in range(nbof_frames):
        if i < nbof_frames - 2:
            ax[i].imshow(sequence_array[i], vmin=0, vmax=vmax)
            input_nbr = i - (nbof_frames - 3)
            ax[i].set_title(rf"{time_list[i]} ($t_{{{input_nbr}}}$)", fontsize=fontsize)
            if i == 1:
                ax[i].set_xlabel("Inputs", fontsize=fontsize)

            ax[i].grid(grid)
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])

            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
        if i == nbof_frames - 2:
            im = ax[i].imshow(sequence_array[i], vmin=0, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=fontsize)
            ax[i].set_title(
                rf"{time_list[i]} ($t_{{{prediction_t}}}$)", fontsize=fontsize
            )
            ax[i].set_xlabel("Ground Truth", fontsize=fontsize)
            ax[i].grid(grid)
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
        if i == nbof_frames - 1:
            im = ax[i].imshow(sequence_array[i], vmin=0, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=fontsize)

            ax[i].set_title(
                rf"{time_list[-1]} ($\hat{{t_{{{prediction_t}}}}}$)", fontsize=fontsize
            )
            ax[i].set_xlabel("Prediction", fontsize=fontsize)

            ax[i].grid(grid)
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
        if nbof_frames == 5:
            line = plt.Line2D(
                (0.59, 0.59),
                (0.1, 1),
                color=Colors.concrete,
                linestyle=Linestyles.dashed,
                linewidth=3,
            )
        else:
            line = plt.Line2D(
                (0.515, 0.515),
                (0.1, 1),
                color=Colors.concrete,
                linestyle=Linestyles.dashed,
                linewidth=3,
            )
        fig.add_artist(line)
    if save_fig:
        fig.tight_layout()
        fig.savefig(fig_name)
    plt.show()


def show_sample_dict(sample_dict, rows):
    images_array = sample_dict["images"]
    ts_list = sample_dict["time_stamps"]

    C = images_array.shape[0]

    if C % rows == 0:
        cols = C // rows
    else:
        cols = C // rows + 1

    plt.figure(figsize=(10, 5))
    for i in range(C):

        plt.subplot(rows, cols, i + 1)
        plt.title(ts_list[i])
        plt.axis("off")
        plt.imshow(images_array[i, :, :])

    plt.subplots_adjust(wspace=1, hspace=1)
    plt.tight_layout()
    plt.show()


def plot_day_images(dataset, sleep_secs=0, start=0):
    """Shows images from dataset on a Jupyter Notebook

    Args:
        dataset (class): Dataset of one day of images
        sleep_secs (int): Time between each image in seconds
        start (int): Where to start in the images sequence.
    """

    for n in range(start, len(dataset)):
        sample = dataset[n]
        image, time_stamp = sample["image"], sample["time_stamp"]

        fig, ax = plt.subplots()

        ax.title.set_text(f"Image: {n+1} {time_stamp}")
        fig.tight_layout()

        plt.imshow(image, cmap="gray")
        plt.show()

        time.sleep(sleep_secs)
        IPython.display.clear_output(wait=True)


def plot_histogram(values, bins, normalize=True):
    """Takes a list of values or an image and it plots the histogram, showing the mean and standard
    deviation.

    Args:
        values (list / np.array): Values to plot in histogram
        bins (int): Quanitity of bins to separate the values
        normalize (bool): if True the histogram is normalize
    """
    plt.figure(figsize=(10, 5))

    if type(values) == np.ndarray:
        M, N = values.shape
        values = np.reshape(values, (M * N))

    plt.hist(values, bins=bins, density=normalize)
    l1 = plt.axvline(np.mean(values), c="r")
    l2 = plt.axvline(np.mean(values) + np.std(values), c="g")
    l3 = plt.axvline(np.mean(values) - np.std(values), c="g")
    plt.legend((l1, l2, l3), ["mean of values", "std of values"])
    if normalize:
        plt.ylabel("p(x)")
    else:
        plt.ylabel("Quantity")

    plt.xlabel("value")
    plt.show()


def show_image_w_colorbar(
    image, title=None, fig_name=None, save_fig=False, bar_max=None, colormap="viridis"
):
    """
    Shows the image with a colorbar

    Args:
        image (array): Array containing the values of the image
    """
    fig = plt.figure()
    fig.set_size_inches(8, 4)
    ax1 = fig.add_subplot(1, 1, 1)

    if bar_max:
        image_ = ax1.imshow(
            image, cmap=colormap, interpolation="none", vmin=0, vmax=bar_max
        )
    else:
        image_ = ax1.imshow(image, cmap=colormap, interpolation="none")

    fig.colorbar(image_, ax=ax1)
    if title:
        ax1.title.set_text("Image")
    if save_fig:
        fig.tight_layout()
        fig.savefig(fig_name)
    plt.show()


def error_maps_for_5_horizons(
    error_maps_list, vmax=None, colormap="coolwarm", fig_name=None, save_fig=False
):
    """Shows the images passed in a grid
    Args:
        sequence_array (array)
    """
    if len(error_maps_list) != 5:
        raise ValueError("Must input 5 Maps")

    fontsize = 22  # 22 generates the font more like the latex text

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    fig, ax = plt.subplots(1, 5, figsize=(30, 5))
    plt.subplots_adjust(wspace=0.01)
    for i in range(len(error_maps_list)):
        if i < 4:
            if vmax:
                ax[i].imshow(error_maps_list[i], vmin=0, vmax=vmax, cmap=colormap)
            else:
                ax[i].imshow(error_maps_list[i], cmap=colormap)
            if i == 0:
                ax[i].set_title(f"1 Hour", fontsize=fontsize)
            else:
                ax[i].set_title(f"{i + 1} Hours", fontsize=fontsize)

            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])

            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)

        elif i == 4:
            if vmax:
                im = ax[i].imshow(error_maps_list[i], vmin=0, vmax=vmax, cmap=colormap)
            else:
                im = ax[i].imshow(error_maps_list[i], cmap=colormap)
            cbar = plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=fontsize)

            ax[i].set_title(f"5 Hours", fontsize=fontsize)

            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)

    if save_fig:
        fig.tight_layout()
        fig.savefig(fig_name)
    plt.show()


def show_images_diff(img1, img2):
    """
    Shows the difference between two images with a colorbar

    Args:
        img1(array): Array containing the values of the image 1
        img2(array): Array containing the values of the image 2
    """
    if img1.shape != img2.shape:
        raise ValueError(
            "Images must have the same shape, img1:", img1.shape, "img2:", img2.shape
        )
    diff = abs(img1 - img2)
    fig, (ax1) = plt.subplots(figsize=(14, 4), ncols=1)
    image_ = ax1.imshow(diff, interpolation="none")
    # grafica = ax1.imshow(error_array[70:100], interpolation='none')
    fig.colorbar(image_, ax=ax1)
    ax1.title.set_text("Image")
    plt.show()


def use_filter(in_frames, filter_):
    _, C, M, N = in_frames.shape
    Cf, Mf, Nf = filter_.shape

    r1 = (Mf - 1) // 2
    r2 = (Nf - 1) // 2

    in_frames_padded = np.zeros((C, M + r1 * 2, N + r2 * 2))
    in_frames_padded[:, 1:-1, 1:-1] = in_frames[0]
    output = np.zeros((M, N))

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            W = in_frames_padded[:, i - 1 : i + 2, j - 1 : j + 2]
            output[i - 1, j - 1] = np.sum(W * filter_)

    return output


def make_plots_from_dict(load_paths, save_folder=None):
    """
    Load results from dictionary in a pickle file (or multiple dictionaries from multiple files) and display graphs

    Args:
        load_paths(str or list of strings): path of files containing dictionaries
        save_folder(str): folder to save graphs
    """
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
    if not isinstance(load_paths, list):
        load_paths = [load_paths]

    errors_metrics_list = []
    for load_path in load_paths:
        errors_file = open(load_path, "rb")
        errors_metrics_list.append(pickle.load(errors_file))
        errors_file.close()

    for metric, errors_metric in errors_metrics_list[0].items():
        error_list = []
        models = []
        for i in range(len(errors_metrics_list) - 1):
            errors_metric.update(errors_metrics_list[i + 1][metric])

        for model_name, error in errors_metric.items():
            error_list.append(errors_metric[model_name])
            models.append(model_name)

        if save_folder:
            save_file = os.path.join(save_folder, metric)
        else:
            save_file = None
        plot_graph_multiple(
            error_list, models, error_metric=metric, save_file=save_file
        )
