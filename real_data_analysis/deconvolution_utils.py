#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   deconvolution_utils.py
@Time    :   2024/11/13 10:43:19
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# References
# ======================================================================================================================
"""[1] Alina-Georgiana Meresescu. Inverse Problems of Deconvolution Applied in the Fields of Geosciences and 
Planetology. Paleontology. Université Paris Saclay (COmUE), 2018. English. ffNNT : 2018SACLS316ff. fftel-01982218f
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt


def crosscorr_deconvolution(x, y):
    """Cross-correlation deconvolution based on the formulation from [1]"""
    # Compute cross-correlation
    # r_xy = sp.correlate(y, x, mode="same")
    # r_xy = np.fft.irfft(np.fft.rfft(x) * np.conj(np.fft.rfft(y)))
    r_yx = np.fft.irfft(np.fft.rfft(y) * np.conj(np.fft.rfft(x)), n=len(x))

    # Estime y_rec from x and r_xy
    # y_rec = sp.convolve(x, r_xy, mode="same")
    y_rec = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(r_yx), n=len(x))
    sigma_y = np.std(y)
    sigma_y_rec = np.std(y_rec)

    # Compute the impulse response
    h = r_yx * sigma_y / sigma_y_rec

    return h


def wiener_deconvolution(x, y, rho_f=None):
    """Apply Wiener filter to estimate the impulse response"""
    # Derive ffts
    x_fft = np.fft.rfft(x)
    y_fft = np.fft.rfft(y)

    if rho_f is None:
        rho_f = np.ones_like(x_fft)

    # Derive G(f)
    g_fft = 1 / x_fft * 1 / (1 + 1 / (np.abs(x_fft) ** 2 * rho_f))

    h_fft = g_fft * y_fft

    h = np.fft.irfft(h_fft, n=len(x))

    return h


def psd_deconvolution(x, y, fs, nperseg=2**12, noverlap=2**11):
    """Derive impulse response from psd estimates"""

    # Derive psds
    f, s_xx = sp.welch(x, fs, nperseg, noverlap)
    f, s_yy = sp.welch(y, fs, nperseg, noverlap)

    # Estimate s_hh =
    pass


if __name__ == "__main__":
    x = np.random.randn(100)
    y = np.random.randn(100)

    # h = crosscorr_deconvolution(x, y)
    r_xy = sp.correlate(x, y, mode="full")

    # Estime y_rec from x and r_xy
    y_rec = sp.convolve(x, r_xy, mode="full")
    sigma_y = np.std(y)
    sigma_y_rec = np.std(y_rec)

    # Compute the impulse response
    h = r_xy * sigma_y / sigma_y_rec

    plt.figure()
    plt.plot(h)
    plt.show()
