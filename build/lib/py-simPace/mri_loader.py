## mri_loader.py

import os
import numpy as np
import nibabel as nib
import pydicom

class MRILoader:
    """
    Functions to load and preprocess MRI data.
    """
    @staticmethod
    def load_nifti(file_path):
        """
        Load an MRI scan from a NIfTI file.
        :param file_path: Path to the .nii or .nii.gz file
        :return: Numpy array of MRI data
        """
        nifti_img = nib.load(file_path)
        return nifti_img.get_fdata()


    @staticmethod
    def load_dicom(directory):
        """
        Load an MRI scan from a folder of DICOM files.
        :param directory: Path to the DICOM directory
        :return: 3D numpy array of MRI data
        """
        dicom_files = [pydicom.dcmread(os.path.join(directory, f)) for f in sorted(os.listdir(directory)) if f.endswith('.dcm')]
        return np.stack([f.pixel_array for f in dicom_files])