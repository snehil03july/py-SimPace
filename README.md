# py-simPace

## Overview
**py-simPace** (Simulated Prospective Acquisition CorrEction) is an open-source Python library designed to simulate motion artifacts in MRI data. Inspired by MATLAB's SimPACE, this package allows researchers to introduce controlled motion distortions at the slice acquisition level, study the impact of motion on MRI scans, and evaluate motion correction techniques.

This project is part of **Snehil's PhD research** on MRI motion correction at the **University of Exeter**, contributing to the development of tools for understanding and mitigating motion-induced errors in functional MRI (fMRI) and structural MRI.

## Features
- 🏃 **Motion Simulation**: Introduce controlled motion artifacts (translations, rotations) at the slice level.
- 📂 **MRI Data Handling**: Load and process standard MRI file formats (NIfTI, DICOM).
- 🔍 **Artifact Analysis**: Compare original vs. motion-corrupted MRI data.
- 🛠 **Motion Correction Interface**: Provides a framework to integrate and evaluate correction algorithms.
- 📊 **Visualization Tools**: Generate motion heatmaps, k-space representations, and comparative analysis.
- 🧩 **Extensible API**: Designed for researchers to easily incorporate motion effects into their studies.

## Installation
### Install from PyPI
You can install `py-simPace` directly from **PyPI**:
```bash
pip install py-simpace
```

### Install from Source
Alternatively, you can clone the repository and install manually:
```bash
git clone https://github.com/snehil-xyz/py-simpace.git
cd py-simpace
pip install -r requirements.txt
```

## Dependencies
`py-simPace` requires the following Python libraries:
- `numpy`
- `scipy`
- `nibabel`
- `pydicom`
- `matplotlib`
- `opencv-python`

These dependencies are automatically installed when using `pip install py-simpace`.

## Usage Guide
### 1️⃣ Load MRI Data
```python
from py_simpace.mri_loader import MRILoader
mri_data = MRILoader.load_nifti("sample.nii.gz")
```

### 2️⃣ Simulate Motion Artifacts
```python
from py_simpace.motion_simulation import MotionSimulation
motion_data = MotionSimulation.apply_motion(mri_data[:, :, 50], rotation=10, translation=(5, 5))
```

### 3️⃣ Analyze Motion Effects
```python
from py_simpace.artifact_analysis import ArtifactAnalysis
ArtifactAnalysis.compare_images(mri_data[:, :, 50], motion_data)
```

## PyPI Deployment & Updates
**py-simPace** is deployed on **PyPI**, and you can always update it using:
```bash
pip install --upgrade py-simpace
```

For package details, visit: [https://pypi.org/project/py-simpace/](https://pypi.org/project/py-simpace/)

## Contribution Guidelines
Contributions are welcome! Follow these steps:
1. **Fork the repository** on GitHub.
2. **Create a new branch** (`feature-branch`).
3. **Commit your changes** and push the branch.
4. **Submit a pull request**.

### Running Unit Tests
```bash
pytest tests/
```

## License
**MIT License** – Free to use and modify.

## Acknowledgments
This project is developed as part of **Snehil's PhD research** on MRI motion correction at the **University of Exeter**. It aims to enhance the study of motion artifacts in MRI, aiding researchers in testing and developing correction methods.

For inquiries, contact **Snehil** via GitHub: [@snehil03july](https://github.com/snehil03july/)

