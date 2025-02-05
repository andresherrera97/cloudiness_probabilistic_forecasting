import cv2
import torch
import yaml
import datetime as datetime
import numpy as np


class CloudMotionVector:
    def __init__(self, dcfg=None):
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

        # Add noise parameters to configuration
        self.magnitude_noise_std = 4 / (60 * 60)  # 2km/h speed noise, 0,5km resolution -> 4/3600 pixels/s noise
        self.angle_noise_std = (np.pi*15)/180  # Default 15 degrees noise

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

    def add_noise_to_vectors_CLAUDE(self, cmv):
        """
        Add Gaussian noise to both magnitude and angle of motion vectors
        
        Args:
            cmv (numpy.ndarray): Cloud motion vectors of shape (H, W, 2)
            
        Returns:
            numpy.ndarray: Noisy motion vectors
        """
        # Convert to polar coordinates (magnitude and angle)
        magnitude = np.sqrt(cmv[:, :, 0]**2 + cmv[:, :, 1]**2)
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

    def add_noise_to_vectors(self, cmv: np.ndarray, time_step: int) -> np.ndarray:
        """
        Add Gaussian noise to both magnitude and angle of motion vectors

        Args:
            cmv (numpy.ndarray): Cloud motion vectors of shape (H, W, 2)

        Returns:
            numpy.ndarray: Noisy motion vectors
        """
        # Convert to polar coordinates (magnitude and angle)
        magnitude = np.sqrt(cmv[:, :, 0]**2 + cmv[:, :, 1]**2)
        angle = np.arctan2(cmv[:, :, 1], cmv[:, :, 0])

        # Generate noise for magnitude (multiplicative noise)
        # magnitude_noise_std_per_hour = self.magnitude_noise_std * time_step
        magnitude_noise_std_per_hour = self.magnitude_noise_std
        magnitude_noise = np.random.normal(0, magnitude_noise_std_per_hour, magnitude.shape)
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
        noisy_cmv = self.add_noise_to_vectors(cmv, time_step)
        noisy_cmv_claude = self.add_noise_to_vectors_CLAUDE(cmv)
        
        print(f"cmv min: {cmv.min()}, cmv max: {cmv.max()}")
        print(f"noisy_cmv min: {noisy_cmv.min()}, noisy_cmv max: {noisy_cmv.max()}")
        print(f"noisy_cmv_claude min: {noisy_cmv_claude.min()}, noisy_cmv_claude max: {noisy_cmv_claude.max()}")
        
        print(f"cmv mean: {cmv.mean()}, cmv std: {cmv.std()}")
        print(f"noisy_cmv mean: {noisy_cmv.mean()}, noisy_cmv std: {noisy_cmv.std()}")
        print(f"noisy_cmv_claude mean: {noisy_cmv_claude.mean()}, noisy_cmv_claude std: {noisy_cmv_claude.std()}")
        
        i_idx, j_idx = np.meshgrid(np.arange(noisy_cmv.shape[1]), np.arange(noisy_cmv.shape[0]))

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

        return np.array(predictions)
