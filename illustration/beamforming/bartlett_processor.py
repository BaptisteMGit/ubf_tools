#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   bartlett_processor.py
@Time    :   2024/08/30 10:50:36
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Illusatration of the classical Bartlett beamforming processor
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt


# ======================================================================================================================
# Parameters
# ======================================================================================================================

f0 = 50
SL = 140  # Source level (dB re 1 µPa at 1 m)
WNL = 50  # White Noise level (dB re 1 µPa at 1 m)

# Number of sensors
NB_SENSORS = 15
R = 7 * 1e3  # Distance between the source and the array (m)
c = 1500  # Speed of sound in water (m/s)
fs = 5000  # Sampling frequency (Hz)
p0 = 1e-6  # Reference pressure (Pa)

ALPHA = 1
# ======================================================================================================================
# Functions
# ======================================================================================================================


def build_array(m, f0, alpha=ALPHA):
    """
    Build a linear array of sensors

    Parameters
    ----------
    m : int
        Number of sensors
    f0 : float
        Central frequency of the source (Hz)

    """
    delta_sensor = c / np.min(f0) * alpha  # Distance between sensors (m)
    print(f"Sensor spacing: {delta_sensor} m")
    d_sensor = np.arange(0, m) * delta_sensor

    return d_sensor


def generate_time_series(m, sl, r, theta, f0, T, fs):
    """ "
    Generate received time series assuming a linear array of sensors and a set of monochromatic sources

    Parameters
    ----------
    m : int
        Number of sensors
    delta_sensor : float
        Distance between sensors (m)
    R : array
        Distance between the sources and the center of the array (m)
    theta : array
        Sources angles (°)
    f0 : float
        Sources central frequencies (Hz)
    T : float
        Duration of the time series (s)
    """
    n_sources = len(theta)

    # Convert theta to radians
    theta = np.deg2rad(theta)

    # Time vector
    t = np.arange(0, T, 1 / fs)

    # Sensors positions
    d_sensor = build_array(m, f0)

    received_signal = np.zeros((m, len(t)), dtype=np.complex128)

    for i in range(n_sources):

        # Source signal
        omega_0 = 2 * np.pi * f0[i]
        k = omega_0 / c
        a = p0 * 10 ** (sl[i] / 20)
        s = a * np.exp(1j * omega_0 * t)  # Assume the source is a monochromatic signal

        # Received signals (based on approximation detailed p.18 Jensen)
        r_src_sensor = r[i] - d_sensor * np.sin(theta[i])
        # r_src_sensor = np.sqrt(
        #     (d_sensor - r[i] * np.sin(theta[i])) ** 2 + (r[i] * np.cos(theta[i])) ** 2
        # )

        phase = 1 / r_src_sensor * np.exp(1j * k * r_src_sensor)
        # phase = (
        #     1 / r[i] * np.exp(1j * k * (r[i] - d_sensor * np.sin(theta[i])))
        # )

        # Cast phase to a matrix
        phase = np.tile(phase, (len(t), 1)).T
        x = s * phase

        # Add white noise
        a = p0 * 10 ** (WNL / 20)
        noise = (
            a
            / np.sqrt(2)
            * (np.random.randn(m, len(t)) + 1j * np.random.randn(m, len(t)))
        )

        x += noise

        # Superposition of signals
        received_signal += x

    # Derive noise and signal power to ensure the SNR is correct
    noise_power = np.mean(np.abs(noise) ** 2)
    signal_power = np.mean(np.abs(x) ** 2)

    tl = 20 * np.log10(R)
    snr = 10 * np.log10(signal_power / noise_power)
    print(f"Measured snr: {snr} dB")
    print(f"Theoretical snr : {SL - tl - WNL} dB")

    return t, received_signal


def bartlett_processor_das(t, x, f0):
    # Generate time series
    # t, x = generate_time_series(m, delta_sensor, sl, r, theta, f0, T, fs)

    omega_0 = 2 * np.pi * f0
    k = omega_0 / c

    m = x.shape[0]
    d_sensor = build_array(m, f0)

    ## Apply delay-and-sum beamforming

    # Define the angle vector
    alpha = np.linspace(-90, 90, 1000)
    p = np.zeros_like(alpha, dtype=np.float64)

    for i, th in enumerate(alpha):

        # Compute steering vector
        steering_vector = np.exp(-1j * k * d_sensor * np.sin(np.deg2rad(th)))

        # Apply beamforming
        p_th = np.zeros_like(t)
        for l in range(NB_SENSORS):
            p_th += np.real(steering_vector[l].conj().T * x[l])

        # Store beam output power
        p[i] = np.mean(p_th**2)

    # for k in [10, 200, 300]:
    #     print(f"Angle {alpha[k]}°: {p[k]}")

    # Normalize power
    p = p / np.max(p)

    return alpha, p


def bartlett_processor_f(x, f0):

    omega_0 = 2 * np.pi * f0
    k = omega_0 / c

    m = x.shape[0]
    d_sensor = build_array(m, f0)
    ## Apply delay-and-sum beamforming

    # Define the angle vector
    alpha = np.linspace(-90, 90, 1000)
    p = np.zeros_like(alpha, dtype=np.float64)

    # Compute CSDM at frequency f0
    # csdm = (x @ x.conj().T) / x.shape[
    #     1
    # ]  # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples

    ff, csdm = estimate_csdm(np.real(x))
    idx_f0 = np.argmin(np.abs(ff - f0))
    ff_0 = ff[idx_f0]
    print(f"Frequency: {ff_0} Hz")
    csdm = csdm[idx_f0]

    for i, th in enumerate(alpha):

        # Compute steering vector
        steering_vector = np.exp(-1j * k * d_sensor * np.sin(np.deg2rad(th)))

        # Apply beamforming
        # P_bart = steering_vector @ csdm @ steering_vector
        P_bart = steering_vector.conj().T @ csdm @ steering_vector

        # Store beam output power
        p[i] = np.abs(P_bart)

    # for k in [10, 200, 300]:
    #     print(f"Angle {alpha[k]}°: {p[k]}")

    # Normalize power
    p = p / np.max(p)

    return alpha, p


def mvdr_processor(x, f0):
    """
    Compute the minimum variance distortionless response beamformer
    """

    omega_0 = 2 * np.pi * f0
    k = omega_0 / c

    m = x.shape[0]
    d_sensor = build_array(m, f0)
    # Compute CSDM at frequency f0
    ff, csdm = estimate_csdm(np.real(x))
    idx_f0 = np.argmin(np.abs(ff - f0))
    ff_0 = ff[idx_f0]
    print(f"Frequency: {ff_0} Hz")
    # csdm = csdm[idx_f0]
    csdm = np.mean(csdm, axis=0)

    # csdm = (x @ x.conj().T) / x.shape[
    #     1
    # ]  # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples

    # Compute the pseudo inverse of the CSDM
    csdm_inv = np.linalg.pinv(csdm)

    # Define the angle vector
    alpha = np.linspace(-90, 90, 1000)
    p = np.zeros_like(alpha, dtype=np.float64)

    for i, th in enumerate(alpha):
        # Compute steering vector
        steering_vector = np.exp(-1j * k * d_sensor * np.sin(np.deg2rad(th)))

        # Apply beamforming
        P_mvd = 1 / (steering_vector.conj().T @ csdm_inv @ steering_vector)

        # Store beam output power
        p[i] = np.abs(P_mvd)

    # Normalize power
    p = p / np.max(p)

    return alpha, p


def music_processor(x, f0):
    """
    Compute the MUSIC beamformer
    """

    omega_0 = 2 * np.pi * f0
    k = omega_0 / c

    m = x.shape[0]
    d_sensor = build_array(m, f0)

    num_expected_signals = 2

    # part that doesn't change with theta_i
    csdm = (x @ x.conj().T) / x.shape[
        1
    ]  # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples

    w, v = np.linalg.eig(
        csdm
    )  # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    eig_val_order = np.argsort(np.abs(w))  # find order of magnitude of eigenvalues
    v = v[:, eig_val_order]  # sort eigenvectors using this order
    # We make a new eigenvector matrix representing the "noise subspace", it's just the rest of the eigenvalues
    V = np.zeros((NB_SENSORS, NB_SENSORS - num_expected_signals), dtype=np.complex64)
    for i in range(NB_SENSORS - num_expected_signals):
        V[:, i] = v[:, i]

    # Define the angle vector
    alpha = np.linspace(-90, 90, 1000)
    p = np.zeros_like(alpha, dtype=np.float64)

    for i, th in enumerate(alpha):
        # Steering Vector
        steering_vector = np.exp(-1j * k * d_sensor * np.sin(np.deg2rad(th)))

        output = 1 / (
            steering_vector.conj().T @ V @ V.conj().T @ steering_vector
        )  # The main MUSIC equation
        output = np.abs(output.squeeze())  # take magnitude
        p[i] = output

    p /= np.max(p)  # normalize

    return alpha, p


def estimate_csdm(x):
    n_sensors, n_samples = x.shape

    # Define STFT
    nperseg = 2**8
    w = np.sin(np.pi * np.arange(0, nperseg) / nperseg)
    alpha_overlap = 2 / 3
    hop = int((1 - alpha_overlap) * nperseg)
    SFT = sp.ShortTimeFFT(
        w, hop=nperseg, fs=fs, fft_mode="onesided", scale_to="magnitude"
    )

    # Dummy stft to get frequency and time vectors
    stft_x = SFT.stft(x[0])  # perform the STFT

    # # Plot spectrogram
    # plt.figure()
    # plt.imshow(20 * np.log10(np.abs(stft_x)), aspect="auto")
    # plt.colorbar()
    # plt.show()

    ff = SFT.f
    tt = SFT.t(n_samples) - np.min(
        SFT.t(n_samples)
    )  # For some reasons the time vector is not starting at 0 ?

    # Ugly loop version
    stft_x = np.zeros((n_sensors, len(ff), len(tt)), dtype=np.complex128)

    for i in range(n_sensors):
        # Derive stft matrix
        stft_xi = SFT.stft(x[i])  # perform the STFT

        # Store stft
        stft_x[i] = stft_xi

    # Compute the cross-spectral density matrix
    csdm = np.zeros((len(ff), n_sensors, n_sensors), dtype=np.complex128)

    for k in range(len(ff)):
        csdm_fk = np.zeros((n_sensors, n_sensors), dtype=np.complex128)
        for l in range(len(tt)):
            d_fk_tl = stft_x[
                :, k, l
            ]  # Extract the STFT at frequency fk = ff[k] and time snapshots tl = tt[l]
            csdm_fk += np.outer(d_fk_tl, d_fk_tl.conj())

        # CSDM is estimated as the average of the CSDM at each time snapshot
        csdm_fk /= len(tt)

        # Store CSDM
        csdm[k] = csdm_fk

    return ff, csdm


if __name__ == "__main__":

    T = 3
    fs = 100

    # Plot received signals in a subplot
    WNL = 50
    r = [R, R]
    f0 = [50, 50]
    theta = [45, 50]
    sl = [140, 140]
    t, x = generate_time_series(NB_SENSORS, sl, r, theta, f0, T, fs)

    # f, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    # for i in range(4):
    #     axs[i].plot(t, np.real(x[i]))
    #     axs[i].set_ylabel(f"Sensor {i+1}")
    # axs[-1].set_xlabel("Time [s]")
    # plt.tight_layout()
    # plt.show()

    # Plot beamformed signal
    # Single source at 45°
    r = [R]
    f0 = [50]
    theta = [45]
    sl = [140]

    # 3 sources at -40°, 20° and 25°
    # r = [R, R, R]
    # f0 = [30, 40, 50]
    # theta = [-40, 20, 25]
    # sl = [140, 140, 140]

    # 2 sources at 45° and 30°
    # r = [R, R]
    # f0 = [40, 50]
    # theta = [0, 35]
    # sl = [140, 140]

    src_idx_to_locate = 0
    t, x = generate_time_series(NB_SENSORS, sl, r, theta, f0, T, fs)

    # Compare CSDM estimation
    # ff, csdm = estimate_csdm(np.real(x))
    # csdm_f = csdm[np.argmin(np.abs(ff - f0[src_idx_to_locate]))]
    # plt.figure()
    # plt.imshow(np.abs(csdm_f), aspect="auto")
    # plt.colorbar()

    # csdm_f = csdm[np.argmin(np.abs(ff - f0[1]))]
    # plt.figure()
    # plt.imshow(np.abs(csdm_f), aspect="auto")
    # plt.colorbar()

    # csdm_mean = np.mean(csdm, axis=0)
    # plt.figure()
    # plt.imshow(np.abs(csdm_mean), aspect="auto")
    # plt.colorbar()

    # csdm = (x @ x.conj().T) / x.shape[1]
    # plt.figure()
    # plt.imshow(np.abs(csdm), aspect="auto")
    # plt.colorbar()

    # plt.show()

    # Delay-and-sum beamformer
    alpha, p_das = bartlett_processor_das(t, x, f0[src_idx_to_locate])

    # plt.figure()
    # plt.plot(alpha, 10 * np.log10(p_das))
    # plt.axvline(theta[src_idx_to_locate], color="r", linestyle="--")
    # plt.xlabel("Angle [°]")
    # plt.ylabel("Power [dB]")
    # plt.show()

    alpha, p_f = bartlett_processor_f(x, f0[src_idx_to_locate])

    # MVDR beamformer
    alpha, p_mvdr = mvdr_processor(x, f0[src_idx_to_locate])

    # MUSIC beamformer
    alpha, p_music = music_processor(x, f0[src_idx_to_locate])

    # Save results to .dat file
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\illustration\beamforming"
    fpath = os.path.join(root, "beamforming_1src_aliasing.dat")

    # Create pandas dataframe
    import pandas as pd

    df = pd.DataFrame(
        {
            "Angle": alpha,
            "Bartlett": 10 * np.log10(p_f),
            "MVDR": 10 * np.log10(p_mvdr),
            "MUSIC": 10 * np.log10(p_music),
        }
    )
    # Save to .dat file
    df.to_csv(fpath, sep=" ", index=False)

    plt.figure()
    plt.plot(alpha, 10 * np.log10(p_f), color="k", linestyle="-", label="Bartlett")
    # plt.plot(alpha, 10 * np.log10(p_das), color="r", linestyle="--", label="DAS")
    plt.plot(alpha, 10 * np.log10(p_mvdr), color="b", label="MVDR")
    plt.plot(alpha, 10 * np.log10(p_music), color="g", label="MUSIC")

    plt.axvline(theta[src_idx_to_locate], color="g", linestyle="--")
    plt.xlabel("Angle [°]")
    plt.ylabel("Power [dB]")
    plt.legend()
    plt.show()
