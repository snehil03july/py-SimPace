"""
PySimPace - Transform Utilities
-------------------------------

Provides helper functions to apply:
- Affine transforms to 3D volumes (image space)
- Rotations and translations in k-space
"""

import numpy as np
from scipy.ndimage import affine_transform
from scipy.fft import fftshift, ifftshift
import logging

logger = logging.getLogger(__name__)

def apply_affine_to_volume(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Apply an affine transform to a 3D volume (image space).

    Parameters:
    -----------
    volume : np.ndarray
        Input 3D volume.
    affine : np.ndarray
        4x4 affine matrix.

    Returns:
    --------
    transformed_volume : np.ndarray
    """
    # Decompose affine into rotation + translation
    rotation = affine[:3, :3]
    translation = affine[:3, 3]

    # Inverse affine for scipy affine_transform
    inv_rotation = np.linalg.inv(rotation)
    inv_translation = -inv_rotation @ translation

    transformed_volume = affine_transform(
        volume,
        inv_rotation,
        offset=inv_translation,
        order=3,  # Cubic interpolation
        mode='nearest'
    )
    return transformed_volume


def apply_kspace_translation(kspace: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Apply translation in k-space (phase modulation).

    Parameters:
    -----------
    kspace : np.ndarray
        Input k-space data.
    affine : np.ndarray
        4x4 affine matrix.

    Returns:
    --------
    shifted_kspace : np.ndarray
    """
    shape = kspace.shape
    grid = [np.fft.fftfreq(n) for n in shape]
    kx, ky, kz = np.meshgrid(grid[0], grid[1], grid[2], indexing='ij')

    # Extract translations (in voxels)
    translation = affine[:3, 3]

    phase_shift = np.exp(-2j * np.pi * (
        kx * translation[0] +
        ky * translation[1] +
        kz * translation[2]
    ))

    shifted_kspace = kspace * phase_shift
    return shifted_kspace


def apply_kspace_rotation(kspace: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Apply rotation in k-space (approximation via rotating image space).

    Parameters:
    -----------
    kspace : np.ndarray
        Input k-space data.
    affine : np.ndarray
        4x4 affine matrix.

    Returns:
    --------
    rotated_kspace : np.ndarray
    """
    # Approximate: apply rotation to image → fft again
    # Extract rotation part
    rotation_affine = np.eye(4)
    rotation_affine[:3, :3] = affine[:3, :3]

    # IFFT → rotate → FFT again
    image = np.real(ifftshift(np.fft.ifftn(ifftshift(kspace))))
    rotated_image = apply_affine_to_volume(image, rotation_affine)
    rotated_kspace = fftshift(np.fft.fftn(fftshift(rotated_image)))

    return rotated_kspace


def apply_affine_to_slice(slice2D: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Apply affine transform to 2D slice.

    Parameters:
    -----------
    slice2D : np.ndarray
    affine : np.ndarray

    Returns:
    --------
    transformed_slice : np.ndarray
    """
    from scipy.ndimage import affine_transform

    rotation = affine[:2, :2]
    translation = affine[:2, 3]

    inv_rotation = np.linalg.inv(rotation)
    inv_translation = -inv_rotation @ translation

    transformed_slice = affine_transform(
        slice2D,
        inv_rotation,
        offset=inv_translation,
        order=3,
        mode='nearest'
    )
    return transformed_slice
