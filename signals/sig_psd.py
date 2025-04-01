#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sig_psd.py
@Time    :   2025/03/31 23:00:48
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np


def psd_to_timeserie(psd, df):

    # Set f=0 and f=fs/2 to 0   -> psd has exactly nf=np.fft.rfftfreq points
    psd = np.concatenate(([0], psd, [0]))

    # Number of frequency components in psd
    nf = psd.shape[0]

    # Define module of the spectrum
    X_f_mod = np.sqrt(psd)  # Definition of PSD Sxx(f) = |X(f)|^2

    # Generate random phase to create the complexe spectrum
    phi_t = np.random.randn(2 * nf - 1)
    X_f_ang = np.angle(np.fft.rfft(phi_t))
    # Y_f = np.fft.rfft(phi_t)

    # Use random phase to create spectrum
    X_f = X_f_mod * np.exp(1j * X_f_ang)
    # X_f = X_f_mod * Y_f

    # Inverse fourier transform to get time signal
    x_t = np.fft.irfft(X_f)
    nt = x_t.shape[0]

    # Correct for rfft factor
    x_t *= nt * np.sqrt(df / 2)
    # x_t *= nt

    # Time vector
    fs = nt * df
    t = np.linspace(0, 1 / fs * (nt - 1), nt)

    # We can assert x_t has the required psd
    # fs = nt * df
    # ff, sxx = sp.welch(x_t, fs=fs)
    # assert np.allclose(psd, np.abs(X_f))

    return t, x_t


def colored_noise(T, fs, noise_color="white"):

    nt = T * fs
    f = np.fft.rfftfreq(nt, 1 / fs)[1:-1]
    if noise_color == "white":
        psd = np.ones(f.shape)
    elif noise_color == "pink":
        psd = 1 / f
    elif noise_color == "brown":
        psd = 1 / f**2
    elif noise_color == "blue":
        psd = f
    elif noise_color == "purple":
        psd = f**2
    else:
        raise ValueError("Unknown noise color")

    df = 1 / T
    # t, x = psd_to_timeserie(psd, df)
    t, x = psd_to_timeserie(psd, df)

    return t, x, f, psd


if __name__ == "__main__":
    import scipy.signal as sp
    import matplotlib.pyplot as plt

    fs = 5000
    T = 10
    # t, x, f_, psd_ = colored_noise(T, fs, noise_color="purple")

    f_ = np.fft.rfftfreq(int(T * fs), 1 / fs)[1:-1]
    psd_ = f_**2 + 3 * f_**3 + 1
    df = 1 / T
    t, x = psd_to_timeserie(psd_, df)

    plt.figure()
    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    # plt.show()

    # Derive and plot psd
    f, psd = sp.welch(x, fs, nperseg=1024, noverlap=512)

    plt.figure()
    plt.plot(f, 10 * np.log10(psd), label="reached", marker="o")
    plt.plot(f_, 10 * np.log10(psd_), label="target")
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.legend()
    plt.show()
