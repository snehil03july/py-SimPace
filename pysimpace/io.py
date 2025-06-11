"""
PySimPace - NIfTI I/O Utilities
-------------------------------

Load and save structural and fMRI scans
"""

import nibabel as nib
import numpy as np
import os

def load_nifti(path: str, normalize: bool = True) -> tuple:
    """
    Load a NIfTI file and return data + affine + header

    Parameters:
    -----------
    path : str
        Path to .nii or .nii.gz file
    normalize : bool
        Normalize intensities to [0, 1]

    Returns:
    --------
    data : np.ndarray
    affine : np.ndarray
    header : nib.Nifti1Header
    """
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    if normalize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    return data, affine, header


def save_nifti(data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, output_path: str):
    """
    Save NumPy data as a NIfTI file

    Parameters:
    -----------
    data : np.ndarray
    affine : np.ndarray
    header : nib.Nifti1Header
    output_path : str
        Output file path (.nii.gz)
    """
    out_img = nib.Nifti1Image(data, affine, header)
    nib.save(out_img, output_path)


def ensure_dir(path: str):
    """
    Ensure that a directory exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)
