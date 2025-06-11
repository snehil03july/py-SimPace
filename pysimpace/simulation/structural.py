"""
PySimPace - Structural MRI Motion Simulation Module
---------------------------------------------------

This module provides functions to simulate realistic motion artifacts in 3D structural MRI volumes (T1, T2, FLAIR, etc).

Supported techniques:
- K-space combination method (Shaw et al., 2019) for physical realism
- Image-space approximation for fast preview / augmentation
- Rigid-body motion (rotation + translation)
- Multiple motion events per scan

References:
- Shaw et al. (2019), https://doi.org/10.1016/j.neuroimage.2019.03.016
- SimPACE: https://pubmed.ncbi.nlm.nih.gov/25003176/

Author: Snehil Kumar + Contributors
Version: 2.0.0
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from pysimpace.simulation.transforms import apply_affine_to_volume, apply_kspace_translation, apply_kspace_rotation
from pysimpace.simulation.models import generate_random_affine_transforms
import logging

logger = logging.getLogger(__name__)


def simulate_structural_motion(volume_data: np.ndarray,
                               transforms: list = None,
                               num_transforms: int = 1,
                               max_rot: float = 5.0,
                               max_trans: float = 5.0,
                               image_space: bool = False,
                               use_blended: bool = False,
                               num_segments: int = 8,
                               ghosting: bool = True,
                               apply_gibbs: bool = False,                # NEW
                               gibbs_strength: float = 0.05,             # NEW
                               seed: int = None) -> np.ndarray:
    """
    Apply motion and artifact simulation to a 3D structural MRI volume.

    Parameters:
    ----------- (existing + new ones above)
    """

    if seed is not None:
        np.random.seed(seed)

    if transforms is None:
        logger.info(f"Generating {num_transforms} random transforms (max_rot={max_rot}, max_trans={max_trans})")
        transforms = generate_random_affine_transforms(num_transforms, max_rot, max_trans, volume_data.shape)

    # --- Apply motion ---
    if image_space:
        logger.info("Applying motion in image space (slice-wise approximation)")
        corrupted_volume = _simulate_image_space_motion(volume_data, transforms)
    elif use_blended:
        logger.info(f"Applying realistic k-space blended motion (num_segments={num_segments}, ghosting={ghosting})")
        corrupted_volume = _simulate_kspace_motion_blended(volume_data,
                                                           transforms,
                                                           num_segments=num_segments,
                                                           ghosting=ghosting)
    else:
        logger.info("Applying motion using k-space combination method")
        corrupted_volume = _simulate_kspace_motion(volume_data, transforms)

    # --- Apply Gibbs ringing ---
    if apply_gibbs:
        from pysimpace.simulation.noise import apply_gibbs_ringing
        logger.info(f"Applying Gibbs ringing (strength={gibbs_strength})")
        corrupted_volume = apply_gibbs_ringing(corrupted_volume, strength=gibbs_strength)

    return corrupted_volume




def _simulate_kspace_motion(volume_data: np.ndarray, transforms: list) -> np.ndarray:
    """
    Simulate motion artifacts by combining k-space segments under different transforms.
    """

    kspace_full = fftn(volume_data)
    kspace_full = fftshift(kspace_full)

    segments = np.array_split(kspace_full, len(transforms), axis=-1)  # Split along kz axis

    corrupted_segments = []
    for idx, (segment, affine) in enumerate(zip(segments, transforms)):
        logger.debug(f"K-space segment {idx+1}/{len(transforms)}: Applying rotation + translation")
        # Approximate: apply rotation and translation in k-space (phase modulation + shift)
        segment_rotated = apply_kspace_rotation(segment, affine)
        segment_shifted = apply_kspace_translation(segment_rotated, affine)
        corrupted_segments.append(segment_shifted)

    # Recombine k-space segments
    corrupted_kspace = np.concatenate(corrupted_segments, axis=-1)
    corrupted_kspace = ifftshift(corrupted_kspace)
    corrupted_volume = np.real(ifftn(corrupted_kspace))

    return corrupted_volume


def _simulate_image_space_motion(volume_data: np.ndarray, transforms: list) -> np.ndarray:
    """
    Simulate motion artifacts by applying transforms to blocks of slices (image space).
    """

    z_dim = volume_data.shape[2]
    corrupted_volume = np.zeros_like(volume_data)

    slice_blocks = np.array_split(np.arange(z_dim), len(transforms))

    for idx, (slice_indices, affine) in enumerate(zip(slice_blocks, transforms)):
        logger.debug(f"Image space block {idx+1}/{len(transforms)}: slices {slice_indices}")
        # Apply transform to whole volume, then extract relevant slices
        transformed_volume = apply_affine_to_volume(volume_data, affine)
        corrupted_volume[:, :, slice_indices] = transformed_volume[:, :, slice_indices]

    return corrupted_volume


def _simulate_kspace_motion_blended(volume_data: np.ndarray,
                                     transforms: list,
                                     num_segments: int = 8,
                                     ghosting: bool = True) -> np.ndarray:
    """
    Simulate motion artifacts by blending k-space segments from multiple transformed volumes.

    Parameters:
    -----------
    volume_data : np.ndarray
        Input 3D volume (X, Y, Z)
    transforms : list of 4x4 np.ndarray
        List of rigid-body transforms to apply. Should be len(transforms) >= num_segments.
    num_segments : int
        Number of k-space segments (along Z) to split and blend.
    ghosting : bool
        Whether to apply additional ghosting artifacts (phase encode axis).

    Returns:
    --------
    corrupted_volume : np.ndarray
        Motion-corrupted 3D volume.
    """

    from scipy.fft import fftn, ifftn, fftshift, ifftshift
    from pysimpace.simulation.transforms import apply_affine_to_volume
    import numpy as np

    logger.info(f"Simulating k-space blended motion: {num_segments} segments, ghosting={ghosting}")

    # --- Step 1: Generate Transformed Volumes ---
    transformed_volumes = []
    # Safe segmenting of transforms
    transforms_to_use = (transforms * (num_segments // len(transforms) + 1))[:num_segments]

    for i, affine in enumerate(transforms_to_use):
        logger.debug(f"Generating transformed volume {i+1}/{num_segments}")
        transformed_volume = apply_affine_to_volume(volume_data, affine)
        transformed_volumes.append(transformed_volume)

    # --- Step 2: Compute k-space for each transformed volume ---
    kspace_list = []
    for i, vol in enumerate(transformed_volumes):
        logger.debug(f"FFT of transformed volume {i+1}/{num_segments}")
        kspace = fftshift(fftn(vol))
        kspace_list.append(kspace)

    # --- Step 3: Blend k-space segments ---
    kz = kspace_list[0].shape[-1]
    segment_edges = np.linspace(0, kz, num_segments + 1, dtype=int)

    # Precompute full weight map over Z slices
    kz = kspace_list[0].shape[-1]
    blended_kspace = np.zeros_like(kspace_list[0], dtype=np.complex64)
    weight_map = np.zeros(kz)

    for i in range(num_segments):
        start_idx = segment_edges[i]
        end_idx = segment_edges[i + 1]

        logger.debug(f"Blending segment {i+1}/{num_segments}: slices {start_idx}:{end_idx}")

        # Triangular weights for this segment
        segment_weight = _triangular_weights(end_idx - start_idx)

        # Add segment's contribution to blended_kspace and accumulate weights
        for j, k in enumerate(range(start_idx, end_idx)):
            blended_kspace[:, :, k] += kspace_list[i][:, :, k] * segment_weight[j]
            weight_map[k] += segment_weight[j]

    # Normalize to avoid intensity scaling
    for k in range(kz):
        if weight_map[k] > 0:
            blended_kspace[:, :, k] /= weight_map[k]


    # --- Step 4: Optional Ghosting ---
    if ghosting:
        logger.debug("Applying ghosting...")
        blended_kspace = _apply_ghosting(blended_kspace, phase_axis=1, num_ghosts=2, ghost_strength=0.05)

    # --- Step 5: Inverse FFT to get final volume ---
    blended_kspace = ifftshift(blended_kspace)
    corrupted_volume = np.real(ifftn(blended_kspace))

    return corrupted_volume


def _triangular_weights(length):
    """
    Triangular weights for blending segments.
    """
    center = (length - 1) / 2.0
    weights = 1.0 - np.abs(np.arange(length) - center) / center
    return weights

def _apply_ghosting(kspace: np.ndarray, phase_axis: int = 1, num_ghosts: int = 2, ghost_strength: float = 0.05):
    """
    Apply ghosting artifact by injecting shifted copies of k-space along phase axis.

    Parameters:
    -----------
    kspace : np.ndarray
    phase_axis : int
    num_ghosts : int
    ghost_strength : float

    Returns:
    --------
    kspace : np.ndarray (with ghosting applied)
    """
    kspace_with_ghosts = kspace.copy()

    for g in range(1, num_ghosts + 1):
        shift = g * 5  # shift by fixed amount; can be randomized
        kspace_shifted = np.roll(kspace, shift=shift, axis=phase_axis)
        kspace_with_ghosts += ghost_strength * kspace_shifted
        logger.debug(f"Ghost {g}: shift={shift}, strength={ghost_strength}")

    return kspace_with_ghosts
