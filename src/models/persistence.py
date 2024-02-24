
import pandas as pd
import datetime as datetime
import numpy as np
import cv2 as cv
import torch


class Persistence:
    """
    Class that predicts the next images using naive prediction.
    """
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

        predictions = []
        M, N = image.shape

        for i in range(predict_horizon): 
            if (isinstance(self, NoisyPersistence)):
                predictions.append(np.clip(image + np.random.normal(0, self.sigma, (M, N)), 0, 255))
            elif (isinstance(self, BlurredPersistence)):
                if self.kernel_size_list:
                    kernel_size = self.kernel_size_list[i]
                else:
                    kernel_size = self.kernel_size
                blurred_pred = cv.GaussianBlur(image, kernel_size, 0)
                predictions.append(blurred_pred)
                if not self.kernel_size_list:
                    image = blurred_pred
            else:
                predictions.append(np.array(image))

        if predict_direct:
            return np.array(predictions[-1])[np.newaxis]
        else:
            if img_timestamp is not None:
                predict_timestamp = pd.date_range(
                    start=img_timestamp,
                    periods=predict_horizon+1,
                    freq='10min'
                )
                return np.array(predictions), predict_timestamp
            else:
                return np.array(predictions)


class NoisyPersistence(Persistence):
    """Sub class of Persistence, adds white noise to predictions.

    Args:
        Persistence ([type]): [description]
    """
    def __init__(self, sigma):
        # sigma (int): standard deviation of the gauss noise
        self.sigma = sigma


class BlurredPersistence(Persistence):
    """
    Sub class of Persistence, returns predictions after passign through a gauss filter.

    Args:
        Persistence ([type]): [description]
    """
    def __init__(self, kernel_size=(0, 0), kernel_size_list=None):
        # kernel_size (tuple): size of kernel
        self.kernel_size = kernel_size
        self.kernel_size_list = kernel_size_list
