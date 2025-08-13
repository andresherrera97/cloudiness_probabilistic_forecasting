import cv2
import torch
import yaml
import datetime as datetime
import numpy as np
from metrics import CRPSLoss


class CloudMotionVector:
    def __init__(
        self,
        dcfg=None,
        method: str = "tvl1",
        device: str = "cpu",
        n_quantiles: int = 9,
        angle_noise_std: int = 15,
        magnitude_noise_std: float = 4 / (60 * 60),
    ):
        method = method.lower()
        if method not in ["tvl1", "farneback"]:
            raise ValueError(
                f"Method {method} not recognized. Use 'tvl1' or 'farneback'."
            )
        self.method = method
        if method in ["farneback"]:
            # Load configuration
            if dcfg is None:
                stream = open("src/les-prono/admin_scripts/config.yaml", "r")
                self.dcfg = yaml.load(stream, yaml.FullLoader)  # dict
            else:
                self.dcfg = dcfg

            self.cmvcfg = self.dcfg["algorithm"]["cmv"]
            self.pyr_scale = self.cmvcfg["pyr_scale"]
            self.levels = self.cmvcfg["levels"]
            self.winsize = self.cmvcfg["winsize"]
            self.iterations = self.cmvcfg["iterations"]
            self.poly_n = self.cmvcfg["poly_n"]
            self.poly_sigma = self.cmvcfg["poly_sigma"]
        elif method == "tvl1":
            # Use default parameters for TV-L1, but allow tuning for speed
            self.tvl1 = cv2.optflow.createOptFlow_DualTVL1()
            # Speed up TV-L1 by reducing the number of scales and warps
            self.tvl1.setWarpingsNumber(1)  # default is 5, lower is faster
            self.tvl1.setOuterIterations(20)  # default is 30, lower is faster
            self.tvl1.setEpsilon(0.05)  # default is 0.01, higher is faster
            # # Values Set in LES paper for optimal performance
            self.tvl1.setLambda(0.055)  # default is 0.15, lower is faster
            self.tvl1.setScalesNumber(6)

            # original time: 0,74 per pred
            # with optimal values from paper: 0,51 per pred
            # with optimal values and other optimizations: 0,04 per pred

        # Add noise parameters to configuration
        self.magnitude_noise_std = magnitude_noise_std  # 2km/h speed noise, 0,5km resolution -> 4/3600 pixels/s noise
        self.angle_noise_std = (
            np.pi * angle_noise_std
        ) / 180  # Default 15 degrees noise
        self.n_quantiles = n_quantiles
        self.quantiles = list(np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1])
        self.device = device
        self.crps_loss = CRPSLoss(quantiles=self.quantiles, device=device)

    def predict_farneback(
        self,
        imgi: np.ndarray,
        imgf: np.ndarray,
        period: int,
        time_step: int,
        time_horizon: int,
    ) -> np.ndarray:
        """Predicts next image using openCV optical Flow

        Args:
            imgi (numpy.ndarray): first image used for prediction
            imgf (numpy.ndarray): last image used for prediction
            period (int): time difference between imgi and imgf in seconds
            delta_t (int): time passed between imgf and predicted image in seconds
            predict_horizon (int): Length of the prediction horizon (Cuantity of images returned)

        Returns:
            [Numpy array]: Numpy array with predicted images
        """

        if torch.is_tensor(imgi):
            imgi = imgi.numpy()
            imgf = imgf.numpy()

        flow = cv2.calcOpticalFlowFarneback(
            imgi,
            imgf,
            None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0,
        )
        cmv = -flow / period

        i_idx, j_idx = np.meshgrid(np.arange(cmv.shape[1]), np.arange(cmv.shape[0]))

        # img(t+k) + cmv -> img(t+k+1)
        map_i = i_idx + cmv[:, :, 0] * time_step
        map_j = j_idx + cmv[:, :, 1] * time_step
        map_x, map_y = map_i.astype(np.float32), map_j.astype(np.float32)

        base_img = imgf  # base_img imagen a la que le voy a aplicar el campo
        predictions = []

        num_steps = time_horizon // time_step

        for _ in range(num_steps):

            next_img = cv2.remap(
                base_img,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan,  # fill value for moving borders
            )

            predictions.append(next_img)
            base_img = next_img

        return np.array(predictions)

    def predict_tvl1(
        self,
        imgi: np.ndarray,
        imgf: np.ndarray,
        period: int,
        time_step: int,
        time_horizon: int,
    ) -> np.ndarray:
        """
        Warps an image using the calculated optical flow to predict the next image.
        ...
        """
        flow = self.tvl1.calc(imgi, imgf, None)

        cmv_per_step = (flow / period) * time_step

        # Create a grid of coordinates corresponding to the image shape
        h, w = imgf.shape[:2]
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

        # The new map is the original coordinates + the scaled flow vectors for one step
        map_x = (x_coords + cmv_per_step[..., 0]).astype(np.float32)
        map_y = (y_coords + cmv_per_step[..., 1]).astype(np.float32)

        # Use remap to warp the image
        base_img = imgf
        predictions = []
        num_steps = time_horizon // time_step

        for _ in range(num_steps):
            next_img = cv2.remap(
                base_img,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan,
            )
            predictions.append(next_img)
            base_img = next_img

        return np.array(predictions)

    def predict(
        self,
        imgi: np.ndarray,
        imgf: np.ndarray,
        period: int,
        time_step: int,
        time_horizon: int,
    ) -> np.ndarray:
        """Predicts next image using openCV optical Flow

        Args:
            imgi (numpy.ndarray): first image used for prediction
            imgf (numpy.ndarray): last image used for prediction
            period (int): time difference between imgi and imgf in seconds
            time_step (int): time passed between imgf and predicted image in seconds
            time_horizon (int): Length of the prediction horizon (Cuantity of images returned)

        Returns:
            [Numpy array]: Numpy array with predicted images
        """
        if self.method == "farneback":
            return self.predict_farneback(imgi, imgf, period, time_step, time_horizon)
        elif self.method == "tvl1":
            return self.predict_tvl1(imgi, imgf, period, time_step, time_horizon)

    def add_noise_to_vectors_CLAUDE(self, cmv):
        """
        Add Gaussian noise to both magnitude and angle of motion vectors

        Args:
            cmv (numpy.ndarray): Cloud motion vectors of shape (H, W, 2)

        Returns:
            numpy.ndarray: Noisy motion vectors
        """
        # Convert to polar coordinates (magnitude and angle)
        magnitude = np.sqrt(cmv[:, :, 0] ** 2 + cmv[:, :, 1] ** 2)
        angle = np.arctan2(cmv[:, :, 1], cmv[:, :, 0])

        # Generate noise for magnitude (multiplicative noise)
        magnitude_noise = np.random.normal(1.0, 0.1, magnitude.shape)
        noisy_magnitude = magnitude * magnitude_noise

        # Generate noise for angle (additive noise)
        angle_noise = np.random.normal(0, self.angle_noise_std, angle.shape)
        noisy_angle = angle + angle_noise

        # Convert back to Cartesian coordinates
        noisy_cmv = np.zeros_like(cmv)
        noisy_cmv[:, :, 0] = noisy_magnitude * np.cos(noisy_angle)
        noisy_cmv[:, :, 1] = noisy_magnitude * np.sin(noisy_angle)

        return noisy_cmv

    def add_noise_to_vectors(self, cmv: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to both magnitude and angle of motion vectors

        Args:
            cmv (numpy.ndarray): Cloud motion vectors of shape (H, W, 2)

        Returns:
            numpy.ndarray: Noisy motion vectors
        """
        # Convert to polar coordinates (magnitude and angle)
        magnitude = np.sqrt(cmv[:, :, 0] ** 2 + cmv[:, :, 1] ** 2)
        angle = np.arctan2(cmv[:, :, 1], cmv[:, :, 0])

        # Generate noise for magnitude (multiplicative noise)
        # magnitude_noise_std_per_hour = self.magnitude_noise_std * time_step
        magnitude_noise_std_per_hour = self.magnitude_noise_std
        magnitude_noise = np.random.normal(
            0, magnitude_noise_std_per_hour, magnitude.shape
        )
        noisy_magnitude = magnitude + magnitude_noise

        # Generate noise for angle (additive noise)
        angle_noise = np.random.normal(0, self.angle_noise_std, angle.shape)
        noisy_angle = angle + angle_noise

        # Convert back to Cartesian coordinates
        noisy_cmv = np.zeros_like(cmv)
        noisy_cmv[:, :, 0] = noisy_magnitude * np.cos(noisy_angle)
        noisy_cmv[:, :, 1] = noisy_magnitude * np.sin(noisy_angle)

        return noisy_cmv

    def noisy_predict(
        self,
        imgi: np.ndarray,
        imgf: np.ndarray,
        period: int,
        time_step: int,
        time_horizon: int,
        noise_method: str = "claude",
        return_last_frame: bool = True,
    ) -> np.ndarray:
        """Predicts next image using openCV optical Flow with added noise"""

        if torch.is_tensor(imgi):
            imgi = imgi.numpy()
            imgf = imgf.numpy()

        flow = cv2.calcOpticalFlowFarneback(
            imgi,
            imgf,
            None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0,
        )
        cmv = -flow / period

        # Add noise to the motion vectors
        # TODO: check this function and compare with claude recommendation
        if noise_method == "claude":
            noisy_cmv = self.add_noise_to_vectors_CLAUDE(cmv)
        else:
            noisy_cmv = self.add_noise_to_vectors(cmv)

        i_idx, j_idx = np.meshgrid(
            np.arange(noisy_cmv.shape[1]), np.arange(noisy_cmv.shape[0])
        )

        # Use noisy CMV for prediction
        map_i = i_idx + noisy_cmv[:, :, 0] * time_step
        map_j = j_idx + noisy_cmv[:, :, 1] * time_step
        map_x, map_y = map_i.astype(np.float32), map_j.astype(np.float32)

        base_img = imgf
        predictions = []
        num_steps = time_horizon // time_step

        for _ in range(num_steps):
            next_img = cv2.remap(
                base_img,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan,
            )
            predictions.append(next_img)
            base_img = next_img

        if return_last_frame:
            return predictions[-1]
        return np.array(predictions)

    def sort_arrays_optimized(self, arrays):
        # Stack arrays into 3D array
        stacked = np.stack(arrays)

        # Create a mask of where we have nan values
        has_nan = np.any(np.isnan(stacked), axis=0)

        # Create a masked array where NaNs are masked
        masked_stacked = np.ma.masked_array(stacked, mask=np.isnan(stacked))

        # Sort along axis 0 (first dimension)
        # This preserves the mask and sorts only the valid values
        sorted_masked = np.ma.sort(masked_stacked, axis=0)

        # Convert back to regular array with NaNs
        result = sorted_masked.filled(np.nan)

        return result, has_nan

    def probabilistic_prediction(
        self,
        n_quantiles: int,
        imgi: np.ndarray,
        imgf: np.ndarray,
        period: int,
        time_step: int,
        time_horizon: int,
        noise_method: str = "not_claude",
        return_last_frame: bool = True,
    ):
        noisy_predictions = []

        for _ in range(n_quantiles):
            noisy_predictions.append(
                self.noisy_predict(
                    imgi,
                    imgf,
                    period,
                    time_step,
                    time_horizon,
                    noise_method,
                    return_last_frame,
                )
            )

        # Sort the arrays along the first axis
        sorted_predictions, nan_mask = self.sort_arrays_optimized(noisy_predictions)

        return sorted_predictions, nan_mask

    def calculate_crps(self, predictions, nan_mask, target):
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions, device=self.device)
            nan_mask = torch.tensor(nan_mask, device=self.device)
            target = torch.tensor(target, device=self.device)

        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(0)

        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)

        if nan_mask.dim() == 2:
            nan_mask = nan_mask.unsqueeze(0).unsqueeze(0)

        crps = self.crps_loss.crps_loss(
            pred=predictions,
            y=target,
            nan_mask=nan_mask,
        )
        return crps
