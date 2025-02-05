import numpy as np
import matplotlib.pyplot as plt
from models import CloudMotionVector


def plot_predictions(predictions, num_cols=6):
    num_predictions = predictions.shape[0]
    num_rows = (num_predictions + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 3 * num_rows))
    axes = axes.flatten()

    for i in range(num_predictions):
        axes[i].imshow(predictions[i], cmap="gray")
        axes[i].set_title(f"Prediction {i+1}")

    # Hide any unused subplots
    for i in range(num_predictions, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cmv = CloudMotionVector()

    # Load images
    img_i = np.load("datasets/goes16/salto/train/2022_119/2022_119_UTC_170020.npy")
    img_i = img_i.astype(np.float32)
    img_f = np.load("datasets/goes16/salto/train/2022_119/2022_119_UTC_171020.npy")
    img_f = img_f.astype(np.float32)

    # Plot images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_i, cmap="gray")
    axes[0].set_title("Initial Image")
    axes[1].imshow(img_f, cmap="gray")
    axes[1].set_title("Final Image")
    plt.show()

    predictions = cmv.predict(
        imgi=img_i,
        imgf=img_f,
        period=10 * 60,
        time_step=10 * 60,
        time_horizon=60 * 60,
    )

    plot_predictions(predictions)

    # noisy predictions
    noisy_predictions = cmv.noisy_predict(
        imgi=img_i,
        imgf=img_f,
        period=10 * 60,
        time_step=10 * 60,
        time_horizon=60 * 60,
    )

    plot_predictions(noisy_predictions)
