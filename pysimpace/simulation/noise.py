"""
PySimPace - Physiological Noise and Spin History Simulation
-----------------------------------------------------------
"""

import numpy as np

def generate_physio_noise(n_volumes: int,
                          heart_rate_hz: float = 1.0,
                          resp_rate_hz: float = 0.3,
                          amp_heart: float = 0.01,
                          amp_resp: float = 0.02,
                          drift_amp: float = 0.005) -> np.ndarray:
    """
    Generate physiological noise time series.

    Returns an array of multiplicative factors per volume.

    """
    t = np.arange(n_volumes)
    fs = 1.0  # 1 volume per second → adjust if TR known

    heart_signal = amp_heart * np.sin(2 * np.pi * heart_rate_hz * t / fs + np.random.uniform(0, 2*np.pi))
    resp_signal = amp_resp * np.sin(2 * np.pi * resp_rate_hz * t / fs + np.random.uniform(0, 2*np.pi))
    drift_signal = drift_amp * np.cumsum(np.random.randn(n_volumes)) / np.sqrt(n_volumes)

    physio_noise = 1.0 + heart_signal + resp_signal + drift_signal
    return physio_noise


def apply_spin_history_correction(volume: np.ndarray,
                                  motion_event: bool,
                                  slice_order: str = 'sequential',
                                  base_factor: float = 0.8,
                                  recovery_rate: float = 0.05) -> np.ndarray:
    """
    Apply spin history effect after a motion event.

    Parameters:
    -----------
    volume : np.ndarray
    motion_event : bool
    slice_order : 'sequential' or 'interleaved'
    base_factor : initial scaling for first slice(s)
    recovery_rate : per-slice recovery rate toward 1.0

    Returns:
    --------
    corrected_volume : np.ndarray
    """
    corrected_volume = volume.copy()
    n_slices = volume.shape[2]

    factor = base_factor
    for k in range(n_slices):
        corrected_volume[:, :, k] *= factor
        factor += recovery_rate
        factor = min(factor, 1.0)

    return corrected_volume


def apply_gibbs_ringing(volume: np.ndarray, strength: float = 0.05) -> np.ndarray:
    """
    Apply Gibbs ringing artifact to a 3D volume.

    Parameters:
    -----------
    volume : np.ndarray
        Input 3D volume (X, Y, Z).
    strength : float
        Strength of ringing (0.0 = no effect, 0.05 = typical, 0.1 = strong).

    Returns:
    --------
    np.ndarray: Volume with Gibbs ringing applied.
    """
    from scipy.fft import fftn, ifftn, fftshift, ifftshift

    # Forward FFT to k-space
    kspace = fftshift(fftn(volume))

    # Create Hanning window in k-space → simulates Gibbs ringing via windowing
    shape = volume.shape
    hann_x = np.hanning(shape[0])
    hann_y = np.hanning(shape[1])
    hann_z = np.hanning(shape[2])

    hann_3d = np.outer(hann_x, hann_y).reshape(shape[0], shape[1], 1) * hann_z.reshape(1, 1, shape[2])

    # Blend between no window (1.0) and Hanning (hann_3d)
    window = (1.0 - strength) + strength * hann_3d

    # Apply window to k-space
    kspace_windowed = kspace * window

    # Inverse FFT to image space
    kspace_windowed = ifftshift(kspace_windowed)
    corrupted_volume = np.real(ifftn(kspace_windowed))

    return corrupted_volume

