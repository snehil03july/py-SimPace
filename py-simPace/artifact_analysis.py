## artifact_analysis.py

import numpy as np
import matplotlib.pyplot as plt

class ArtifactAnalysis:
    """
    Tools for measuring and visualizing motion effects.
    """
    @staticmethod
    def compare_images(original, transformed):
        """
        Display original and motion-affected images side by side.
        :param original: Original MRI slice
        :param transformed: Motion-affected MRI slice
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title("Original")
        axes[1].imshow(transformed, cmap='gray')
        axes[1].set_title("Motion-Affected")
        plt.show()