"""
PySimPace - Artifact Analysis and Visualization
-----------------------------------------------

Provides:
- Visualization functions
- SSIM, PSNR metrics
- Basic QC plots
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import logging

logger = logging.getLogger(__name__)

def plot_structural_comparison(clean_vol: np.ndarray, corrupted_vol: np.ndarray, slice_idx: int = None):
    """
    Plot structural MRI clean vs corrupted, middle slice or chosen slice.

    """
    if slice_idx is None:
        slice_idx = clean_vol.shape[2] // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(clean_vol[:, :, slice_idx], cmap='gray')
    axes[0].set_title('Clean Structural MRI')
    axes[0].axis('off')

    axes[1].imshow(corrupted_vol[:, :, slice_idx], cmap='gray')
    axes[1].set_title('Corrupted Structural MRI')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_fmri_volume_comparison(clean_fmri: np.ndarray, corrupted_fmri: np.ndarray, vol_idx: int = 0, slice_idx: int = None):
    """
    Plot one fMRI volume clean vs corrupted.

    """
    if slice_idx is None:
        slice_idx = clean_fmri.shape[2] // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(clean_fmri[:, :, slice_idx, vol_idx], cmap='gray')
    axes[0].set_title(f'Clean fMRI (vol {vol_idx})')
    axes[0].axis('off')

    axes[1].imshow(corrupted_fmri[:, :, slice_idx, vol_idx], cmap='gray')
    axes[1].set_title(f'Corrupted fMRI (vol {vol_idx})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def compute_ssim(clean_vol: np.ndarray, corrupted_vol: np.ndarray) -> float:
    """
    Compute SSIM between two 3D volumes (mean across slices).

    Returns:
    --------
    mean_ssim : float
    """
    ssim_scores = []
    for k in range(clean_vol.shape[2]):
        ssim_score = ssim(clean_vol[:, :, k], corrupted_vol[:, :, k], data_range=clean_vol.max() - clean_vol.min())
        ssim_scores.append(ssim_score)

    mean_ssim = np.mean(ssim_scores)
    logger.info(f"Mean SSIM: {mean_ssim:.4f}")
    return mean_ssim


def compute_psnr(clean_vol: np.ndarray, corrupted_vol: np.ndarray) -> float:
    """
    Compute PSNR between two 3D volumes.

    Returns:
    --------
    psnr_value : float
    """
    psnr_value = psnr(clean_vol, corrupted_vol, data_range=clean_vol.max() - clean_vol.min())
    logger.info(f"PSNR: {psnr_value:.2f} dB")
    return psnr_value


def compute_fd(translations, rotations_rad, radius=50.0):
    """
    Compute FD time series from translations and rotations.

    Returns:
    --------
    fd : np.ndarray
    """
    diff_trans = np.diff(translations, axis=0)
    diff_rot = np.diff(rotations_rad, axis=0) * radius

    fd = np.sum(np.abs(diff_trans), axis=1) + np.sum(np.abs(diff_rot), axis=1)
    return fd
