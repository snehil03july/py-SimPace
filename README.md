# py-simPace

## Overview
**py-simPace** (Simulated Prospective Acquisition CorrEction) is an open-source Python library designed to simulate motion artifacts in MRI data. Inspired by MATLAB's SimPACE, this project enables researchers to introduce controlled motion distortions at the slice acquisition level, study the impact of motion on MRI scans, and evaluate motion correction techniques. 

This project is part of **Snehil's PhD research** on enhancing MRI motion correction using GANs and contributes to the development of tools for understanding and mitigating motion-induced errors in functional MRI (fMRI) and structural MRI.

## Key Features
- **Motion Simulation**: Apply controlled motion artifacts (translations, rotations) at the slice level.
- **MRI Data Handling**: Load and process standard MRI file formats (NIfTI, DICOM).
- **Artifact Analysis**: Compare original vs. motion-corrupted MRI data.
- **Motion Correction Interface**: Provide a framework to integrate and evaluate correction algorithms.
- **Visualization Tools**: Generate motion heatmaps, k-space representations, and comparative analysis.
- **Extensible API**: Designed for researchers to easily incorporate motion effects into their studies.

## Installation
```bash
pip install py-simpace
```
Or install from source:
```bash
git clone https://github.com/snehil-xyz/py-simpace.git
cd py-simpace
pip install -r requirements.txt
```

## PyPI Deployment
**py-simPace** will be available on [PyPI](https://pypi.org/) for easy installation and distribution. Once deployed, users can install it directly using:
```bash
pip install py-simpace
```
Future updates will be pushed to PyPI for seamless access to the latest features.

## Dependencies
- `numpy`
- `scipy`
- `nibabel`
- `pydicom`
- `matplotlib`
- `opencv-python`

## Project Structure
```
py-simPace/
│── py_simpace/
│   ├── __init__.py
│   ├── motion_simulation.py    # Core functions for simulating motion artifacts
│   ├── mri_loader.py           # Functions to load and preprocess MRI data
│   ├── artifact_analysis.py    # Tools for measuring & visualizing motion effects
│   ├── motion_correction.py    # Interface for motion correction methods
│   ├── utils.py                # Helper functions
│
│── tests/                      # Unit tests
│── examples/                   # Jupyter Notebooks for usage examples
│── requirements.txt            # Dependencies
│── setup.py                    # Installation script
│── README.md                   # Project documentation
│── LICENSE                     # Open-source license
```

## Usage

### 1. Load MRI Data
```python
from py_simpace.mri_loader import load_mri
mri_data = load_mri("sample.nii.gz")
```

### 2. Simulate Motion Artifacts
```python
from py_simpace.motion_simulation import apply_motion
motion_data = apply_motion(mri_data, rotation=5, translation=(2, 1))
```

### 3. Analyze Motion Effects
```python
from py_simpace.artifact_analysis import compare_images
compare_images(mri_data, motion_data)
```

## Contribution
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-name`).
3. Commit changes and push to your branch.
4. Submit a pull request.

## License
MIT License

## Acknowledgments
This project is developed as part of **Snehil's PhD research** on MRI motion correction at the **University of Exeter**. The aim is to advance the study of motion artifacts in fMRI and structural MRI, aiding researchers in testing and developing correction methods.

