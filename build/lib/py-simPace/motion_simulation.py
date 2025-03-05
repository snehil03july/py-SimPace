import numpy as np
import cv2

class MotionSimulation:
    """
    Core functions for simulating motion artifacts in MRI data.
    """
    @staticmethod
    def apply_motion(image, rotation=0, translation=(0, 0)):
        """
        Apply rotation and translation to an MRI slice.
        :param image: 2D numpy array (single MRI slice)
        :param rotation: Rotation angle in degrees
        :param translation: Tuple (x_shift, y_shift)
        :return: Transformed image
        """
        rows, cols = image.shape
        M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
        M_trans = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        rotated = cv2.warpAffine(image, M_rot, (cols, rows))
        translated = cv2.warpAffine(rotated, M_trans, (cols, rows))
        return translated