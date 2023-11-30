import numpy as np
import scipy.io as sio


def pulse(T, f, fs):
    """Generate pulse defined in Jensen et al. (2000)"""
    t = np.arange(0, T, 1 / fs)
    s = np.zeros(len(t))
    idx_tpulse = np.logical_and(0 < t, t < 4 / f)
    t_pulse = t[idx_tpulse]
    omega = 2 * np.pi * f
    s[idx_tpulse] = (
        1 / 2 * np.sin(omega * t_pulse) * (1 - np.cos(1 / 4 * omega * t_pulse))
    )
    return s, t


def sine_wave(f0, fs, T, A, phi):
    if fs < 2 * f0:
        raise ValueError(
            "Sampling frequency must be at least twice the frequency of the signal"
        )

    t = np.arange(0, T, 1 / fs)
    y = A * np.sin(2 * np.pi * f0 * t + phi)
    return y, t


def ship_noise():
    fpath = r"C:\Users\baptiste.menetrier\Desktop\ressource\ShipNoise_obs_SamuelPinson\ship_source_signal.mat"
    ship_d = sio.loadmat(fpath)
    t = ship_d["t"].squeeze()
    y = ship_d["sig_s_t"].squeeze()
    return y, t


def ship_spectrum(f):
    f = np.array(f)
    fc = 15
    Q = 2
    Aship = 1 / (1 - f**2 / fc**2 + 1j * f / fc / Q)
    return Aship
