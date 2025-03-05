## tests/test_motion_simulation.py

import numpy as np
import unittest
from py_simpace.motion_simulation import MotionSimulation

class TestMotionSimulation(unittest.TestCase):
    def test_apply_motion(self):
        image = np.ones((100, 100))
        transformed = MotionSimulation.apply_motion(image, rotation=10, translation=(5, 5))
        self.assertEqual(transformed.shape, image.shape)