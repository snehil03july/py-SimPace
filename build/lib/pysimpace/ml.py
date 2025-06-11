"""
PySimPace Machine Learning Integration

- Generate paired (clean, motion-corrupted) training datasets
- Provide standard PyTorch Dataset class for training

Author: Your Name
"""

import os
import glob
import csv
import random
import numpy as np
import nibabel as nib
import logging
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pysimpace.simulation.structural import simulate_structural_motion
from pysimpace.simulation.functional import simulate_fmri_motion
from pysimpace.io import load_nifti, save_nifti

logger = logging.getLogger(__name__)

# === Function: Generate Paired Training Data ===

def generate_training_pairs(clean_dir: str,
                            output_dir: str,
                            n_samples: int = 100,
                            artifact_configs: list = None,
                            structural: bool = True,
                            use_blended: bool = True,
                            save_format: str = 'nifti',
                            seed: int = 42):
    """
    Generate paired (clean, corrupted) training data.

    Parameters:
    -----------
    clean_dir : str
        Folder containing clean MRI scans (NIfTI .nii.gz).
    output_dir : str
        Where to save clean/ and corrupted/ folders + pairs.csv.
    n_samples : int
        Number of samples to generate.
    artifact_configs : list of dicts
        List of artifact configurations to choose from.
        If None, use default diverse configs.
    structural : bool
        If True → structural MRI (3D), else fMRI (4D).
    use_blended : bool
        Use blended motion simulation (recommended).
    save_format : str
        'nifti' or 'npy' (npy is faster for DL).
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    None (writes files to output_dir).
    """

    np.random.seed(seed)
    random.seed(seed)

    logger.info(f"Generating {n_samples} training pairs...")
    logger.info(f"Reading clean images from: {clean_dir}")

    clean_paths = glob.glob(os.path.join(clean_dir, "*.nii.gz"))
    if len(clean_paths) == 0:
        raise ValueError(f"No NIfTI files found in {clean_dir}!")

    os.makedirs(os.path.join(output_dir, "clean"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "corrupted"), exist_ok=True)

    pairs_csv_path = os.path.join(output_dir, "pairs.csv")

    # Default artifact configs if none provided
    if artifact_configs is None:
        artifact_configs = [
            {'apply_gibbs': True, 'ghosting': False, 'apply_spike_noise': False, 'apply_intensity_drift': False},
            {'apply_gibbs': False, 'ghosting': True, 'apply_spike_noise': False, 'apply_intensity_drift': False},
            {'apply_gibbs': True, 'ghosting': True, 'apply_spike_noise': True, 'apply_intensity_drift': True},
            {'apply_gibbs': False, 'ghosting': False, 'apply_spike_noise': False, 'apply_intensity_drift': False},  # clean baseline
        ]
        logger.info(f"Using default artifact configs (len={len(artifact_configs)})")

    with open(pairs_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['clean_path', 'corrupted_path'])

        for i in tqdm(range(n_samples), desc="Generating pairs"):
            # Pick clean image randomly
            clean_path = random.choice(clean_paths)
            clean_data, affine, header = load_nifti(clean_path)

            # Select random artifact config for this sample
            artifact_config = random.choice(artifact_configs)

            # Generate corrupted image
            if structural:
                transforms = None  # use default random transforms
                corrupted_data = simulate_structural_motion(
                    clean_data,
                    transforms=transforms,
                    use_blended=use_blended,
                    num_segments=8,
                    ghosting=artifact_config.get('ghosting', False),
                    apply_gibbs=artifact_config.get('apply_gibbs', False),
                    gibbs_strength=0.05,
                    apply_spike_noise=artifact_config.get('apply_spike_noise', False),
                    n_spikes=10,
                    spike_intensity=10.0,
                    apply_intensity_drift=artifact_config.get('apply_intensity_drift', False),
                    drift_factor=0.05,
                    seed=seed + i
                )
            else:
                from pysimpace.simulation.models import MotionTrajectory, generate_smooth_motion_trajectory

                n_vols = clean_data.shape[-1]
                n_slices = clean_data.shape[2]

                trajectory = MotionTrajectory(n_volumes=n_vols, n_slices=n_slices)
                volume_transforms = generate_smooth_motion_trajectory(
                    n_volumes=n_vols,
                    target_fd_mm=0.5,
                    vol_shape=clean_data.shape[:3],
                    smoothing_sigma=3.0,
                    seed=seed + i
                )
                for t in range(n_vols):
                    trajectory.set_volume_transform(t, volume_transforms[t])

                corrupted_data = simulate_fmri_motion(
                    clean_data,
                    trajectory,
                    intra=True,
                    physio=True,
                    parallel=True,
                    seed=seed + i
                )

            # Save clean and corrupted images
            clean_basename = f"clean_{i:04d}"
            corrupted_basename = f"corrupted_{i:04d}"

            if save_format == 'nifti':
                clean_save_path = os.path.join(output_dir, "clean", f"{clean_basename}.nii.gz")
                corrupted_save_path = os.path.join(output_dir, "corrupted", f"{corrupted_basename}.nii.gz")

                save_nifti(clean_data, affine, header, clean_save_path)
                save_nifti(corrupted_data, affine, header, corrupted_save_path)

            elif save_format == 'npy':
                clean_save_path = os.path.join(output_dir, "clean", f"{clean_basename}.npy")
                corrupted_save_path = os.path.join(output_dir, "corrupted", f"{corrupted_basename}.npy")

                np.save(clean_save_path, clean_data)
                np.save(corrupted_save_path, corrupted_data)

            else:
                raise ValueError("save_format must be 'nifti' or 'npy'.")

            # Write to CSV
            writer.writerow([clean_save_path, corrupted_save_path])

    logger.info(f"Training pairs generated → CSV saved to: {pairs_csv_path}")
    logger.info("Done.")


# === PyTorch Dataset ===

class MRIPairedDataset(Dataset):
    """
    PyTorch Dataset for paired (clean, corrupted) MRI data.
    """

    def __init__(self, csv_file, transform=None, structural=True, load_as_tensor=True):
        """
        Parameters:
        -----------
        csv_file : str
            Path to pairs.csv file.
        transform : callable, optional
            Optional transform to apply to both images.
        structural : bool
            True for structural MRI (3D), False for fMRI (4D).
        load_as_tensor : bool
            If True, convert to PyTorch tensors.
        """
        self.pairs = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.pairs.append((row['clean_path'], row['corrupted_path']))

        self.transform = transform
        self.structural = structural
        self.load_as_tensor = load_as_tensor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, corrupted_path = self.pairs[idx]

        # Load clean
        if clean_path.endswith(".npy"):
            clean_img = np.load(clean_path)
        else:
            clean_img, _, _ = load_nifti(clean_path)

        # Load corrupted
        if corrupted_path.endswith(".npy"):
            corrupted_img = np.load(corrupted_path)
        else:
            corrupted_img, _, _ = load_nifti(corrupted_path)

        # Normalize to [0,1]
        clean_img = (clean_img - clean_img.min()) / (clean_img.max() - clean_img.min() + 1e-8)
        corrupted_img = (corrupted_img - corrupted_img.min()) / (corrupted_img.max() - corrupted_img.min() + 1e-8)

        # Optional transform
        if self.transform:
            clean_img, corrupted_img = self.transform(clean_img, corrupted_img)

        # Convert to tensor
        if self.load_as_tensor:
            clean_img = torch.tensor(clean_img, dtype=torch.float32).unsqueeze(0)  # Add channel dim
            corrupted_img = torch.tensor(corrupted_img, dtype=torch.float32).unsqueeze(0)

        return clean_img, corrupted_img
