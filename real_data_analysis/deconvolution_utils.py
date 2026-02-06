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

    nstft = max(x.size, y.size)
    Xf = np.fft.rfft(x, n=nstft)
    Yf = np.fft.rfft(y, n=nstft)
    R_YXf = Yf * np.conj(Xf)
    y_rec = np.fft.irfft(Xf * R_YXf, n=nstft)
    r_yx = np.fft.irfft(R_YXf, n=nstft)

    # r_yx = np.fft.irfft(np.fft.rfft(y) * np.conj(np.fft.rfft(x)), n=len(x))
    # Estime y_rec from x and r_xy
    # y_rec = sp.convolve(x, r_xy, mode="same")
    # y_rec = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(r_yx), n=len(x))
    sigma_y = np.std(y)
    sigma_y_rec = np.std(y_rec)

    # Compute the impulse response
    h = r_yx * sigma_y / sigma_y_rec

    return h


def wiener_deconvolution(x, y, rho_f=None):
    """Apply Wiener filter to estimate the impulse response"""
    # Derive ffts
    nstft = max(x.size, y.size)
    x_fft = np.fft.rfft(x, nstft)
    y_fft = np.fft.rfft(y, nstft)

    if rho_f is None:
        rho_f = np.ones_like(x_fft)

    # Derive G(f)
    g_fft = 1 / x_fft * 1 / (1 + 1 / (np.abs(x_fft) ** 2 * rho_f))

    h_fft = g_fft * y_fft

    h = np.fft.irfft(h_fft, n=nstft)

    return h


if __name__ == "__main__":
    import scipy

    fc = 50
    t0 = 1
    fs = 1000
    T = 10
    t = np.arange(0, T, 1 / fs)
    ns = t.size
    x = np.random.randn(ns)
    h_ricker = (1 - 2 * (np.pi * fc * (t - t0)) ** 2) * np.exp(
        -((np.pi * fc * (t - t0)) ** 2)
    )
    y = scipy.fft.irfft(scipy.fft.rfft(x) * scipy.fft.rfft(h_ricker))
    # y = np.convolve(x, h_ricker, mode="same")
    h = crosscorr_deconvolution(x, y)
    h = scipy.fft.irfft(scipy.fft.rfft(y) / scipy.fft.rfft(x))
    h = wiener_deconvolution(x, y)

    print(h_ricker / h)

    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t, x)
    axs[1].plot(t, h_ricker, label="h")
    axs[1].plot(t, h, label="h_hat")
    # axs[1].plot(t, h_ricker / h)
    axs[1].legend()
    axs[2].plot(t, y)

    plt.show()

    # r_xy = sp.correlate(x, y, mode="full")

    # # Estime y_rec from x and r_xy
    # y_rec = sp.convolve(x, r_xy, mode="full")
    # sigma_y = np.std(y)
    # sigma_y_rec = np.std(y_rec)

    # # Compute the impulse response
    # h = r_xy * sigma_y / sigma_y_rec

    # plt.figure()
    # plt.plot(h)
    # plt.show()
