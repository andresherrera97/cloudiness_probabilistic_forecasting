import os
import fire
import time
import logging
import numpy as np
from models import CloudMotionVector
import utils.utils as utils
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Generate Crop with CMV Script")


def main(
    time_horizon: int = 60,
    dataset_path: str = "datasets/salto_downsample",
    subset: str = "val",
    crop_size: int = 64,
    image_size: int = 512,
    move_crop_center: bool = True,
    output_path: str = "predictions/les/cmv_pred_crop_64x64_MR/PR/",
    save_crop_dataset: bool = True,
    cmv_method: str = "tvl1",
    start_year: Optional[int] = None,
    start_doy: Optional[int] = None,
):
    if start_year is not None and not (2019 <= start_year <= 2025):
        raise ValueError(
            f"start_year not accepted: {start_year}. Must be None or outside the range 2019-2025."
        )
    if start_doy is not None and not (0 <= start_doy <= 366):
        raise ValueError(
            f"start_doy not accepted: {start_doy}. Must be None or between 0 and 366."
        )

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    path_to_dataset = f"{dataset_path}/{subset}/"
    logger.info(f"Dataset path: {path_to_dataset}")

    output_index = time_horizon // 10

    sequence_df = utils.sequence_df_generator_folders(
        path=path_to_dataset,
        in_channel=3,
        output_index=output_index,
        min_time_diff=5,
        max_time_diff=15,
    )

    # Keep only the first num_in_images columns and the last one
    sequence_df = sequence_df.iloc[:, list(range(3)) + [-1]]
    logger.info(f"Sequence DataFrame shape: {sequence_df.shape}")

    # Generate predictions
    crop_start = (image_size - crop_size) // 2
    crop_end = crop_start + crop_size
    if move_crop_center:
        # Move the crop center by 5 pixels in x and -8 pixels in y
        crop_start_x = crop_start + 5
        crop_end_x = crop_end + 5
        crop_start_y = crop_start - 8
        crop_end_y = crop_end - 8
    else:
        crop_start_x = crop_start
        crop_end_x = crop_end
        crop_start_y = crop_start
        crop_end_y = crop_end
    logger.info(f"Crop size: {crop_size}")
    logger.info(f"Crop start_x: {crop_start_x}, Crop end_x: {crop_end_x}")
    logger.info(f"Crop start_y: {crop_start_y}, Crop end_y: {crop_end_y}")

    # TV-L1 Optical Flow Calculation
    cmv_tvl1 = CloudMotionVector(method=cmv_method)

    mean_error = []
    mse = []
    mbe = []

    persistence_mean_error = []
    persistence_mse = []
    persistence_mbe = []

    time_list = []

    for _, row in sequence_df.iterrows():
        day_folder = row[sequence_df.columns[-1]].split("/")[0]
        year = int(day_folder.split("_")[0])
        doy = int(day_folder.split("_")[1])
        if start_doy is not None and start_year is not None:
            if (year < start_year) or (year == start_year and doy < start_doy):
                continue

        os.makedirs(os.path.join(output_path, day_folder), exist_ok=True)
        output_filename = row[sequence_df.columns[-1]].split("/")[-1]

        # in_img_0_path = os.path.join(path_to_dataset, row[0])
        in_img_1_path = os.path.join(path_to_dataset, row[1])
        in_img_2_path = os.path.join(path_to_dataset, row[2])
        target_img_path = os.path.join(path_to_dataset, row[sequence_df.columns[-1]])

        in_img_1 = np.load(in_img_1_path, allow_pickle=True).astype(np.float32)
        in_img_2 = np.load(in_img_2_path, allow_pickle=True).astype(np.float32)
        target_img = (
            np.load(target_img_path, allow_pickle=True).astype(np.float32)
        )
        target_img = target_img / 255  # Normalize target image to [0, 1]

        # Calculate the optical flow using TV-L1
        start_time = time.time()

        prediction = cmv_tvl1.predict(
            imgi=in_img_1,
            imgf=in_img_2,
            period=10,
            time_step=10,
            time_horizon=time_horizon,
        )
        prediction = prediction[-1] / 255  # Normalize prediction to [0, 1]
        elapsed_time = time.time() - start_time
        time_list.append(elapsed_time)

        mean_error.append(np.nanmean(np.abs(prediction - target_img)))
        mse.append(np.nanmean((prediction - target_img) ** 2))
        mbe.append(np.nanmean(prediction - target_img))

        persistence_prediction = in_img_2 / 255  # Normalize persistence prediction to [0, 1]
        persistence_mean_error.append(
            np.nanmean(np.abs(persistence_prediction - target_img))
        )
        persistence_mse.append(np.nanmean((persistence_prediction - target_img) ** 2))
        persistence_mbe.append(np.nanmean(persistence_prediction - target_img))

        if save_crop_dataset:
            pred_crop = prediction[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            pred_crop = pred_crop.astype(np.float16)
            output_filename = os.path.join(output_path, day_folder, output_filename)
            np.save(output_filename, pred_crop)

    logger.info(f"Mean error across all predictions: {np.nanmean(mean_error)}")
    logger.info(f"RMSE across all predictions: {np.sqrt(np.nanmean(mse))}")
    logger.info(f"MBE across all predictions: {np.nanmean(mbe)}")
    logger.info(f"Total time taken for predictions: {np.sum(time_list):.2f} seconds")
    logger.info(f"Average time per prediction: {np.mean(time_list):.2f} seconds")

    logger.info(f"Persistence Mean Error: {np.mean(persistence_mean_error)}")
    logger.info(f"Persistence RMSE: {np.sqrt(np.mean(persistence_mse))}")
    logger.info(f"Persistence MBE: {np.mean(persistence_mbe)}")


if __name__ == "__main__":
    fire.Fire(main)


# Tuning the parameters of the TV-L1 algorithm is key to getting the best results for your specific satellite imagery. Different scenarios, like tracking fast-moving clouds, slow-moving glaciers, or detecting changes in water bodies, can benefit from different parameter settings.

# The TV-L1 implementation in OpenCV, which you create using cv2.optflow.createOptFlow_DualTVL1(), has several adjustable parameters. Let's go through them one by one.

# How to Set Parameters
# In the Python script, you can set these parameters when you create the TV-L1 object. Instead of calling it with no arguments:

# # Custom parameters
# tvl1 = cv2.optflow.createOptFlow_DualTVL1(
#     tau=0.2,
#     lambda_ = 0.1,
#     theta = 0.25,
#     nscales = 4,
#     warps = 4,
#     epsilon = 0.005,
#     scaleStep = 0.7,
#     gamma=0.0
# )
# Note on lambda_: The parameter is called lambda in the algorithm's theory, but because lambda is a reserved keyword in Python, it is named lambda_ in the OpenCV function.

# Adjustable Parameters Explained
# Here are the main parameters you can modify, what they mean, and guidance on how to tune them for satellite images. The default values are generally a good starting point.

# 1. tau (Time Step)
# Description: This is the time-step of the numerical scheme that solves the underlying partial differential equations. It influences the "tightness" of the gradient descent optimization.
# Default Value: 0.25
# Tuning Advice:
# tau must be less than or equal to 0.25 for the numerical scheme to be stable.
# Smaller values (e.g., 0.1 to 0.2) can lead to more accurate flow fields but will increase the computation time as more iterations might be needed to converge.
# If you observe noisy or unstable flow results, try slightly decreasing tau. For most satellite applications, the default is fine.
# 2. lambda_ (Regularization Weight)
# Description: This parameter controls the weight of the "regularization" term in the energy function. The regularization term is what makes the flow field smooth.
# Default Value: 0.15
# Tuning Advice:
# Increase lambda_ (e.g., 0.2, 0.3): This will produce a smoother, more regular flow field. This is useful if your satellite images are noisy or if you are tracking large, coherently moving objects like a single cloud mass. A higher value penalizes variations in the flow more heavily.
# Decrease lambda_ (e.g., 0.1, 0.05): This will produce a less smooth, more detailed flow field that is more faithful to the image data. This is better if you need to capture fine-grained motion or sharp boundaries between moving and stationary objects (e.g., the edge of a cloud).
# 3. theta (Tightness Parameter)
# Description: This parameter is related to the "tightness" of the data term's approximation. It affects how closely the algorithm sticks to the brightness constancy assumption (i.e., that a pixel's intensity doesn't change between frames).
# Default Value: 0.3
# Tuning Advice:
# Smaller values of theta lead to a tighter approximation and can be more accurate, but may also be more sensitive to noise and illumination changes, which can be common in satellite imagery.
# Larger values make the optimization more stable.
# For most cases, the default value works well. You might consider slightly decreasing it if you trust the image intensity values and believe the brightness constancy holds well.
# 4. nscales (Number of Scales)
# Description: The algorithm operates on a multi-scale (pyramidal) representation of the images. This parameter defines how many different resolution levels are used. Using multiple scales is crucial for detecting large displacements.
# Default Value: 5
# Tuning Advice:
# Increase nscales (e.g., 6, 7): This is one of the most important parameters for satellite imagery, especially if you have fast-moving objects like clouds or are working with high-resolution images. More scales allow the algorithm to capture larger motions by first estimating the flow at a very coarse resolution and then refining it at finer levels.
# Decrease nscales: If you are only interested in very small, subtle movements and want to speed up computation, you could reduce this number.
# 5. warps (Number of Warps)
# Description: At each scale level, the algorithm can perform warping operations. This means it uses the current flow estimate to warp the second image towards the first and then computes an update to the flow. This iterative refinement at each scale improves accuracy.
# Default Value: 5
# Tuning Advice:
# Increase warps: More warps lead to a more accurate flow estimate at each scale level, especially for large motions. This comes at the cost of increased computation time. If you increase nscales, you might also want to ensure warps is sufficiently high.
# Decrease warps: If computation time is a critical constraint, you can reduce this number.
# 6. epsilon (Stopping Criterion)
# Description: This is the tolerance for the stopping criterion of the iterative optimization process. The iterations stop when the improvement in the solution is smaller than this value.
# Default Value: 0.01
# Tuning Advice:
# Decrease epsilon (e.g., 0.005): This will make the optimization more precise, leading to potentially more accurate results, but it will take longer to compute.
# Increase epsilon: This will make the computation faster but may sacrifice some accuracy.
# 7. scaleStep (Scale Factor)
# Description: The downsampling factor used to create the image pyramid for the multi-scale approach. For example, a value of 0.8 means each subsequent scale level is 80% the size of the previous one.
# Default Value: 0.8
# Tuning Advice:
# The value must be between 0 and 1.
# A value closer to 1 (e.g., 0.9 or 0.95) creates more levels in the pyramid between the original and smallest scales. This can help capture a wider range of motion speeds but significantly increases computation time.
# A smaller value (e.g., 0.5 or 0.7) creates a coarser pyramid and is faster but might miss motions that fall "between" the scales. For fast-moving clouds, a smaller scaleStep (like 0.5 or 0.6) might be more effective than the default.


# For Fast-Moving Clouds: Your primary focus should be on capturing large movements.

# Increase nscales (e.g., to 6 or 7).
# Increase warps (e.g., to 7 or 10).
# Consider a slightly smaller scaleStep (e.g., 0.7 or 0.6) to have a wider range of coarse scales.
# You might want a smoother flow field, so try slightly increasing lambda_ (e.g., to 0.2).
