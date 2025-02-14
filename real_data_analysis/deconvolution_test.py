#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   deconvolution_test.py
@Time    :   2024/11/13 10:42:42
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
from real_data_analysis.deconvolution_utils import *
from publication.PublicationFigure import PubFigure

PubFigure()


def apply_bandpass_filter(x, ts):
    """Apply a bandpass filter to the input signal x"""
    x_fft = np.fft.rfft(x)
    fft_freq = np.fft.rfftfreq(len(x), ts)

    # Define the bandpass filter
    a = 1
    q = 10
    fc = 1e4
    fr = fft_freq / fc
    bandpass_filter = a / (1 - fr**2 + 1 / q * 1j * fr)
    # bandpass_filter = a / (1 + 1j * q * (fr - 1 / fr))
    # bandpass_filter[0] = bandpass_filter[1]

    bandpass_filter_info = {
        "a": a,
        "q": q,
        "fc": fc,
        "f": fft_freq,
        "amp": np.abs(bandpass_filter),
        "phase": np.angle(bandpass_filter),
        "ri": np.fft.irfft(bandpass_filter, n=len(x)),
    }

    # Apply the bandpass filter
    y_fft = x_fft * bandpass_filter
    y = np.fft.irfft(y_fft, n=len(x))

    return y, bandpass_filter_info


def generate_chirp(f0=0, f1=50 * 1e3, t1=100 * 1e-3):
    fs = 2 * 1e6
    ts = 1 / fs
    t = np.arange(0, t1 + ts, ts)
    n = len(t)
    # x = np.random.randn(n)
    x = sp.chirp(t, f0=f0, f1=f1, t1=t1, method="linear")
    # Apply window to avoid edge effects
    w = sp.windows.tukey(n, alpha=0.05)
    x *= w
    # Pad with few zeros to avoid any edge effects
    npad = int(n * 0.05)
    pad = (npad, npad)
    x = np.pad(x, pad)
    w = np.pad(w, pad)
    # Update time vector
    t = np.arange(0, len(x)) * ts
    n = len(t)

    return x, w, t, ts, n


def generate_white_noise(t1=100 * 1e-3):
    fs = 2 * 1e6
    ts = 1 / fs
    t = np.arange(0, t1 + ts, ts)
    n = len(t)
    x = np.random.randn(n)
    # Apply window to avoid edge effects
    w = sp.windows.tukey(n, alpha=0.05)
    x *= w
    # Pad with few zeros to avoid any edge effects
    npad = int(n * 0.05)
    pad = (npad, npad)
    x = np.pad(x, pad)
    w = np.pad(w, pad)
    # Update time vector
    t = np.arange(0, len(x)) * ts
    n = len(t)

    return x, w, t, ts, n


def plot_signals(t, x, y, h, w=None):
    # Plot input and output signals
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t, x)
    if w is not None:
        axs[0].plot(t, w)
    axs[0].set_ylabel(r"$x$")
    axs[1].plot(t, y)
    axs[1].set_ylabel(r"$y$")
    axs[2].plot(t, h)
    axs[2].set_ylabel(r"$h$")
    fig.supxlabel(r"$t \, \textrm{[s]}$")


def plot_spectrum(x, y, ts):
    # Plot input and output frequency spectrum
    x_fft = np.fft.rfft(x)
    y_fft = np.fft.rfft(y)
    fft_freq = np.fft.rfftfreq(len(x), ts)
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(fft_freq, np.abs(x_fft))
    axs[0].set_ylabel(r"$|X(f)|$")
    axs[1].plot(fft_freq, np.abs(y_fft))
    axs[1].set_ylabel(r"$|Y(f)|$")
    fig.supxlabel(r"$f \, \textrm{[Hz]}$")


def plot_filter_estimate(filter, h_hat, t, t_h, fmax=None):

    if fmax is None:
        fmax = 4 * filter["fc"]
    # Plot the bandpass filter
    ts = t[1] - t[0]

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(filter["f"], filter["amp"], label=r"$|H(f)|$")
    axs[0].plot(filter["f"], np.abs(np.fft.rfft(h_hat)), label=r"$|\hat{H}(f)|$")
    axs[0].set_ylabel(r"$|H(f)|$")
    axs[0].legend()

    axs[1].plot(filter["f"], filter["phase"], label=r"$\phi(f)$")
    axs[1].plot(
        filter["f"],
        np.angle(np.fft.rfft(h_hat)),
        label=r"$\hat{\phi}(f)$",
    )
    axs[1].set_ylabel(r"$\phi(f)$")
    axs[1].legend()
    fig.supxlabel(r"$f \, \textrm{[Hz]}$")
    plt.xlim([0, fmax])

    plt.figure(figsize=(10, 8))
    plt.plot(t_h, h_hat, label=r"$\hat{h}$")
    plt.plot(t, filter["ri"], label=r"$h$")
    plt.xlim([0, 0.005])
    plt.legend()


def test_crosscorr_deconvolution(snr_dB=0, sig="chirp"):

    if sig == "chirp":
        # Generate a chirp signal
        x, w, t, ts, n = generate_chirp(f0=0, f1=60 * 1e3, t1=100 * 1e-3)
    else:
        # Generate a random signal
        x, w, t, ts, n = generate_white_noise(t1=100 * 1e-3)

    w /= np.std(x)
    x /= np.std(x)

    y, bandpass_filter_info = apply_bandpass_filter(x, ts)
    h = bandpass_filter_info["ri"]

    # Add white noise
    sigma_noise = np.sqrt(10 ** (-snr_dB / 10))
    noise = np.random.normal(loc=0, scale=sigma_noise, size=x.size) * np.std(y)
    y += noise
    y /= np.std(y)

    h_hat = crosscorr_deconvolution(x, y)
    # h_hat *= np.std(h) / np.std(h_hat)

    print(f"Bandwidth : {bandpass_filter_info['fc'] / bandpass_filter_info['q']} Hz")

    # Plot input and output signals
    plot_signals(t, x, y, h, w)
    # Plot input and output frequency spectrum
    plot_spectrum(x, y, ts)
    # Plot the bandpass filter
    # t_h = np.arange(-len(h_hat) // 2, len(h_hat) // 2) * ts
    plot_filter_estimate(bandpass_filter_info, h_hat, t, t)

    # Evaluate deconvolution accuracy by comparing estimate output signal to true output signal
    y_hat_fft = np.fft.rfft(x) * np.fft.rfft(h_hat)
    y_hat = np.fft.irfft(y_hat_fft, n=len(x))
    # y_hat = sp.convolve(x, h_hat, mode="same")
    # t_hat = np.arange(0, len(y_hat)) * ts
    plt.figure()
    plt.plot(t, y, label=r"$y$")
    plt.plot(t, y_hat, label=r"$\hat{y}$")
    plt.legend()

    # Derive correlation between h and h_hat
    h = bandpass_filter_info["ri"]
    r_h_hhat = sp.correlate(h, h_hat, mode="same") / np.sqrt(
        np.sum(h**2) * np.sum(h_hat**2)
    )
    print(f"Cross correlation r_h_hhat = {np.max(r_h_hhat)}")

    r_y_yhat = sp.correlate(y, y_hat, mode="full") / np.sqrt(
        np.sum(y**2) * np.sum(y_hat**2)
    )
    print(f"Cross correlation r_y_yhat = {np.max(r_y_yhat)}")


def test_wiener_deconvolution(snr_dB=0, sig="chirp"):

    if sig == "chirp":
        # Generate a chirp signal
        f1 = 100 * 1e3
        x, w, t, ts, n = generate_chirp(f0=0, f1=f1, t1=100 * 1e-3)
    else:
        # Generate a random signal
        x, w, t, ts, n = generate_white_noise(t1=100 * 1e-3)
        f1 = np.fft.rfftfreq(len(x), ts)[-1]

    w /= np.std(x)
    x /= np.std(x)

    y, bandpass_filter_info = apply_bandpass_filter(x, ts)
    h = bandpass_filter_info["ri"]

    ff, s_yy = sp.welch(y, fs=1 / ts, nperseg=2**14, noverlap=2**13)
    ff, s_hh = sp.welch(h, fs=1 / ts, nperseg=2**14, noverlap=2**13)

    # Add white noise
    sigma_noise = np.sqrt(10 ** (-snr_dB / 10))
    noise = np.random.normal(loc=0, scale=sigma_noise, size=x.size) * np.std(y)
    _, s_nn = sp.welch(noise, fs=1 / ts, nperseg=2**14, noverlap=2**13)
    y += noise
    y /= np.std(y)

    # rho_f = s_hh / s_nn**3
    # rho_f = s_yy / s_nn
    # # Interp rho_f to match fft_freq
    fft_freq = np.fft.rfftfreq(len(x), ts)
    transmitted_band_idx = fft_freq > f1
    # rho_f = np.interp(fft_freq, ff, rho_f)
    # rho_f[fft_freq < 500] = rho_f[fft_freq >= 500][0]

    # Dicus 1981
    sigma_n2 = np.abs(np.fft.rfft(noise)) ** 2
    sigma_h2 = np.mean(np.abs(np.fft.rfft(y)) ** 2) - sigma_n2 / np.mean(
        np.abs(np.fft.rfft(x)) ** 2
    )
    rho_f = sigma_h2 / sigma_n2

    plt.figure()
    plt.plot(fft_freq, rho_f, label=r"$\rho_f$")
    # plt.figure()

    # plt.plot(ff, s_hh, label=r"$S_{hh}$")
    # plt.plot(ff, s_nn, label=r"$S_{nn}$")
    # plt.legend()

    # Estimate h
    h_hat = wiener_deconvolution(x, y, rho_f=rho_f)

    H_hat = np.fft.rfft(h_hat)
    H_hat[fft_freq > f1] = 0
    h_hat = np.fft.irfft(H_hat, n=len(x))
    # h_hat *= np.std(h) / np.std(h_hat)

    print(f"Bandwidth : {bandpass_filter_info['fc'] / bandpass_filter_info['q']} Hz")

    # Plot input and output signals
    plot_signals(t, x, y, h, w)
    # Plot input and output frequency spectrum
    plot_spectrum(x, y, ts)
    # Plot the bandpass filter
    plot_filter_estimate(bandpass_filter_info, h_hat, t, t_h=t, fmax=f1)

    # Evaluate deconvolution accuracy by comparing estimate output signal to true output signal
    y_hat_fft = np.fft.rfft(x) * np.fft.rfft(h_hat)
    y_hat = np.fft.irfft(y_hat_fft, n=len(x))
    # y_hat /= np.std(y_hat)
    # y_hat = sp.convolve(x, h_hat, mode="same")
    # t_hat = np.arange(0, len(y_hat)) * ts
    plt.figure()
    plt.plot(t, y, label=r"$y$")
    plt.plot(t, y_hat, label=r"$\hat{y}$")
    plt.legend()

    # Derive correlation between h and h_hat
    h = bandpass_filter_info["ri"]
    r_h_hhat = sp.correlate(h, h_hat, mode="same") / np.sqrt(
        np.sum(h**2) * np.sum(h_hat**2)
    )
    print(f"Wiener filter r_h_hhat = {np.max(r_h_hhat)}")

    r_y_yhat = sp.correlate(y, y_hat, mode="full") / np.sqrt(
        np.sum(y**2) * np.sum(y_hat**2)
    )
    print(f"Wiener filter r_y_yhat = {np.max(r_y_yhat)}")


if __name__ == "__main__":
    # test_crosscorr_deconvolution(snr_dB=20, sig="chirp")
    # test_crosscorr_deconvolution(snr_dB=20, sig="white_noise")
    test_wiener_deconvolution(snr_dB=20, sig="chirp")
    test_wiener_deconvolution(snr_dB=20, sig="white_noise")
    plt.show()
