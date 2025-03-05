## utils.py

import numpy as np

class Utils:
    """
    Helper functions.
    """
    @staticmethod
    def normalize_image(image):
        """
        Normalize an image to [0, 1] range.
        :param image: 2D numpy array
        :return: Normalized image
        """
        return (image - np.min(image)) / (np.max(image) - np.min(image))