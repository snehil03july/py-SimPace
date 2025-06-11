"""
PySimPace - Motion Models
-------------------------

Provides:
- MotionTrajectory class
- Random affine transform generators
"""

import numpy as np

def generate_random_affine_transforms(num_transforms: int,
                                      max_rot_deg: float,
                                      max_trans_mm: float,
                                      vol_shape: tuple) -> list:
    """
    Generate random affine transforms (rotation + translation) for a 3D volume.

    Parameters:
    -----------
    num_transforms : int
        Number of transforms to generate.
    max_rot_deg : float
        Maximum rotation angle (degrees).
    max_trans_mm : float
        Maximum translation (voxels).
    vol_shape : tuple
        Shape of the input volume.

    Returns:
    --------
    transforms : list of 4x4 np.ndarray
    """
    transforms = []
    center = np.array(vol_shape) / 2

    for _ in range(num_transforms):
        angles_rad = np.deg2rad(np.random.uniform(-max_rot_deg, max_rot_deg, size=3))
        translations = np.random.uniform(-max_trans_mm, max_trans_mm, size=3)

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
                       [0, np.sin(angles_rad[0]), np.cos(angles_rad[0])]])

        Ry = np.array([[np.cos(angles_rad[1]), 0, np.sin(angles_rad[1])],
                       [0, 1, 0],
                       [-np.sin(angles_rad[1]), 0, np.cos(angles_rad[1])]])

        Rz = np.array([[np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
                       [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx

        # Build 4x4 affine
        affine = np.eye(4)
        affine[:3, :3] = R
        affine[:3, 3] = translations

        transforms.append(affine)

    return transforms


class MotionTrajectory:
    """
    MotionTrajectory for fMRI simulation.

    Stores:
    - volume-wise 4x4 affine transforms (inter-volume motion)
    - optional slice-wise transforms (intra-volume motion)
    """

    def __init__(self, n_volumes: int, n_slices: int = None):
        self.volumes = [np.eye(4) for _ in range(n_volumes)]
        self.slices = None

        if n_slices is not None:
            self.slices = []
            for _ in range(n_volumes):
                slice_transforms = [np.eye(4) for _ in range(n_slices)]
                self.slices.append(slice_transforms)

    def set_volume_transform(self, t: int, affine: np.ndarray):
        self.volumes[t] = affine

    def set_slice_transform(self, t: int, k: int, affine: np.ndarray):
        if self.slices is None:
            raise ValueError("Trajectory was initialized without slice-wise support.")
        self.slices[t][k] = affine


def generate_smooth_motion_trajectory(n_volumes: int,
                                      target_fd_mm: float,
                                      vol_shape: tuple,
                                      smoothing_sigma: float = 3.0,
                                      seed: int = None) -> list:
    """
    Generate a smooth motion trajectory with approximate target mean FD.

    Parameters:
    -----------
    n_volumes : int
    target_fd_mm : float
        Target mean FD (mm)
    vol_shape : tuple
    smoothing_sigma : float
        Gaussian smoothing kernel std-dev (controls smoothness)
    seed : int

    Returns:
    --------
    list of 4x4 np.ndarray (one per volume)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random walk in translation and rotation
    # Translations in mm
    translations = np.cumsum(np.random.randn(n_volumes, 3), axis=0)
    # Rotations in deg → convert to rad
    rotations_deg = np.cumsum(np.random.randn(n_volumes, 3), axis=0)
    rotations_rad = np.deg2rad(rotations_deg)

    # Smooth the trajectories
    from scipy.ndimage import gaussian_filter1d

    translations = gaussian_filter1d(translations, sigma=smoothing_sigma, axis=0)
    rotations_rad = gaussian_filter1d(rotations_rad, sigma=smoothing_sigma, axis=0)

    # Scale to match target FD approximately
    est_fd = _estimate_fd(translations, rotations_rad, vol_shape)
    scale = target_fd_mm / (est_fd + 1e-8)
    translations *= scale
    rotations_rad *= scale

    # Build affine matrices
    transforms = []
    for t in range(n_volumes):
        tx, ty, tz = translations[t]
        rx, ry, rz = rotations_rad[t]

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])

        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])

        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx

        affine = np.eye(4)
        affine[:3, :3] = R
        affine[:3, 3] = [tx, ty, tz]

        transforms.append(affine)

    return transforms


def _estimate_fd(translations, rotations_rad, vol_shape):
    """
    Rough estimate of mean FD for given trajectory.
    """
    # Voxel size → assume isotropic for now
    voxel_size_mm = 1.0  # safe default → user can improve later
    radius = 50.0  # typical brain radius in mm

    diff_trans = np.diff(translations, axis=0)
    diff_rot = np.diff(rotations_rad, axis=0) * radius

    fd = np.sum(np.abs(diff_trans), axis=1) + np.sum(np.abs(diff_rot), axis=1)
    mean_fd = np.mean(fd)
    return mean_fd



def extract_rotations(R):
    """
    Approximate rotation angles (rad) from rotation matrix.

    Returns:
    --------
    np.ndarray of (rx, ry, rz)
    """
    rx = np.arctan2(R[2,1], R[2,2])
    ry = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    rz = np.arctan2(R[1,0], R[0,0])
    return np.array([rx, ry, rz])

