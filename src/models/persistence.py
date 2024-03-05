import pandas as pd
import datetime as datetime
import numpy as np
import cv2 as cv
import torch


class Persistence:
    """
    Class that predicts the next images using naive prediction.
    """

    def __init__(self,):
        pass

    def generate_prediction(self, image, i: int = 0):
        # Default implementation for the base class
        return np.array(image[-1, :, :])

    def predict(self, image, predict_horizon, img_timestamp=None, predict_direct=False):
        """Takes an image and predicts the next images on the predict_horizon depending on class instance
            normal: identical image
            noisy: adds gaussanian noise
            blurred: blurs it with a gaussanian window

        Args:
            image (array): Image used as prediction
            img_timestamp (datetime): time stamp of image
            predict_horizon (int): Length of the prediction horizon.
            predict_direct (bool): True to return only the last prediction

        Returns:
            [Numpy array], [list]: Array containing preditions and list containing timestamps
        """
        if torch.is_tensor(image):
            image = image.numpy()

        predictions_list = []

        for input_images in image:

            if not predict_direct:
                # generate a prediction for each time step
                predictions = []
                for i in range(predict_horizon):
                    predictions.append(self.generate_prediction(input_images, i))
                predictions = np.array(predictions)

            else:
                predictions = self.generate_prediction(input_images)
                predictions = predictions[np.newaxis]

            predictions_list.append(predictions)

        return np.array(predictions_list)


class NoisyPersistence(Persistence):
    """Sub class of Persistence, adds white noise to predictions.

    Args:
        Persistence ([type]): [description]
    """
    def __init__(self, sigma: int):
        # sigma (int): standard deviation of the gauss noise
        super().__init__()
        self.sigma = sigma

    def generate_prediction(self, image, i: int = 0):
        return np.clip(image + np.random.normal(0, self.sigma, image.shape), 0, 1)


class BlurredPersistence(Persistence):
    """
    Sub class of Persistence, returns predictions after passign through a gauss filter.

    Args:
        Persistence ([type]): [description]
    """
    def __init__(self, kernel_size=(0, 0), kernel_size_list=None):
        super().__init__()
        # kernel_size (tuple): size of kernel
        self.kernel_size = kernel_size
        self.kernel_size_list = kernel_size_list

    def generate_prediction(self, image, i: int = 0):
        if self.kernel_size_list:
            kernel_size = self.kernel_size_list[i]
        else:
            kernel_size = self.kernel_size
        return cv.GaussianBlur(image, kernel_size, 0)
