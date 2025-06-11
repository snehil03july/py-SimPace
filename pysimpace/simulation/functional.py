"""
PySimPace - Functional MRI (fMRI) Motion Simulation
--------------------------------------------------

Simulates:
- Inter-volume motion
- Intra-volume slice-wise motion
- Physiological noise
- Spin history effects
"""

import numpy as np
from pysimpace.simulation.transforms import apply_affine_to_volume, apply_affine_to_slice
from pysimpace.simulation.noise import generate_physio_noise, apply_spin_history_correction
import logging

logger = logging.getLogger(__name__)


def simulate_fmri_motion(fmri_data: np.ndarray,
                         trajectory,
                         intra: bool = True,
                         physio: bool = True,
                         parallel: bool = True,
                         seed: int = None) -> np.ndarray:
    """
    Simulate motion and artifacts on 4D fMRI data.

    Parameters:
    -----------
    fmri_data : np.ndarray
        4D fMRI data (X, Y, Z, T).
    trajectory : MotionTrajectory
        Motion parameters for volumes/slices.
    intra : bool
        Enable slice-wise intra-volume motion.
    physio : bool
        Enable physiological noise.
    parallel : bool
        Enable per-volume parallelism (speeds up large fMRI).
    seed : int
        Random seed.

    Returns:
    --------
    corrupted_fmri : np.ndarray
    """
    if seed is not None:
        np.random.seed(seed)

    n_volumes = fmri_data.shape[-1]
    n_slices = fmri_data.shape[2]

    logger.info(f"Simulating fMRI motion: n_volumes={n_volumes}, intra={intra}, physio={physio}, parallel={parallel}")

    # Generate physio noise factors
    if physio:
        physio_factors = generate_physio_noise(n_volumes)
    else:
        physio_factors = np.ones(n_volumes)

    # --- Define motion function per volume ---
    def motion_fn(t):
        clean_vol = fmri_data[..., t]

        # Inter-volume motion
        inter_affine = trajectory.volumes[t]
        moved_vol = apply_affine_to_volume(clean_vol, inter_affine)

        # Intra-volume motion (slice-wise)
        if intra and trajectory.slices is not None:
            corrupted_vol = np.zeros_like(moved_vol)
            for k in range(n_slices):
                slice_affine = trajectory.slices[t][k]
                corrupted_slice = apply_affine_to_slice(moved_vol[:, :, k], slice_affine)
                corrupted_vol[:, :, k] = corrupted_slice
        else:
            corrupted_vol = moved_vol

        # Apply spin history (simple model for now)
        # You can later add spin history correction if desired
        # corrupted_vol = apply_spin_history_correction(corrupted_vol, motion_event=True/False)

        # Apply physiological noise
        corrupted_vol *= physio_factors[t]

        return corrupted_vol

    # --- Run motion simulation ---
    corrupted_fmri = np.zeros_like(fmri_data)

    if parallel:
        from concurrent.futures import ThreadPoolExecutor

        logger.info("Running fMRI simulation in parallel mode")
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(motion_fn, range(n_volumes)))

        for t, vol in enumerate(results):
            corrupted_fmri[..., t] = vol

    else:
        logger.info("Running fMRI simulation in sequential mode")
        for t in range(n_volumes):
            corrupted_fmri[..., t] = motion_fn(t)

    return corrupted_fmri

