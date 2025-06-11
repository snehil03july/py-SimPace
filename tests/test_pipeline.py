import numpy as np

import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
PySimPace Test Pipeline — Final Version with Your Real Paths
"""

import numpy as np
import matplotlib.pyplot as plt
from pysimpace.io import load_nifti, save_nifti
from pysimpace.simulation.structural import simulate_structural_motion
from pysimpace.simulation.functional import simulate_fmri_motion
from pysimpace.simulation.models import generate_smooth_motion_trajectory, MotionTrajectory, extract_rotations
from pysimpace.analysis import plot_structural_comparison, plot_fmri_volume_comparison, compute_ssim, compute_psnr, compute_fd

# === Your Actual Paths ===
structural_path = r"C:\Users\sk895\OneDrive - University of Exeter\Desktop\Projects\ulf_enc\training_data\POCEMR001\3T\POCEMR001_T2.nii.gz"
fmri_path = r"C:\Users\sk895\OneDrive - University of Exeter\Desktop\Projects\fmri_data\ds002_R2.0.5\sub-01\func\sub-01_task-deterministicclassification_run-01_bold.nii.gz"

# === Structural MRI Simulation ===
structural_data, s_affine, s_header = load_nifti(structural_path)
print(f"Structural MRI shape: {structural_data.shape}")

structural_transforms = generate_smooth_motion_trajectory(
    n_volumes=3,
    target_fd_mm=1.0,
    vol_shape=structural_data.shape,
    smoothing_sigma=2.0,
    seed=42
)

corrupted_structural = simulate_structural_motion(
    structural_data,
    transforms=structural_transforms,
    use_blended=True,
    ghosting=True,
    apply_gibbs=True,  # if you want
    gibbs_strength=0.05,
    seed=42
)


save_nifti(corrupted_structural, s_affine, s_header, "corrupted_structural.nii.gz")
plot_structural_comparison(structural_data, corrupted_structural)
compute_ssim(structural_data, corrupted_structural)
compute_psnr(structural_data, corrupted_structural)

# === fMRI Simulation ===
fmri_data, f_affine, f_header = load_nifti(fmri_path)
fmri_data = fmri_data[..., :10]  # For speed
print(f"fMRI shape: {fmri_data.shape}")

n_vols = fmri_data.shape[-1]
n_slices = fmri_data.shape[2]
trajectory = MotionTrajectory(n_volumes=n_vols, n_slices=n_slices)

volume_transforms = generate_smooth_motion_trajectory(
    n_volumes=n_vols,
    target_fd_mm=0.5,
    vol_shape=fmri_data.shape[:3],
    smoothing_sigma=3.0,
    seed=42
)

for t in range(n_vols):
    trajectory.set_volume_transform(t, volume_transforms[t])

translations = np.array([volume_transforms[t][:3, 3] for t in range(n_vols)])
rotations_rad = np.array([extract_rotations(volume_transforms[t][:3, :3]) for t in range(n_vols)])
fd = compute_fd(translations, rotations_rad)

plt.figure()
plt.plot(fd)
plt.title("Framewise Displacement (FD) - fMRI")
plt.xlabel("Frame")
plt.ylabel("FD (mm)")
plt.grid(True)
plt.show()

corrupted_fmri = simulate_fmri_motion(
    fmri_data,
    trajectory,
    intra=False,      # First test without intra (very fast)
    physio=True,
    parallel=True     # Enable parallelism 🚀
)

