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
"""
[1] 
Alina-Georgiana Meresescu. Inverse Problems of Deconvolution Applied in the Fields of Geosciences and 
Planetology. Paleontology. Université Paris Saclay (COmUE), 2018. English. ffNNT : 2018SACLS316ff. fftel-01982218f

[2] 
Bonnel, J., Thode, A., Wright, D., & Chapman, R. (2020). Nonlinear time-warping made simple: A step-by-step tutorial
on underwater acoustic modal separation with a single hydrophone. The Journal of the Acoustical Society of America, 
147(3), 1897-1926. https://doi.org/10.1121/10.0000937

"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal import resample_poly

from misc import BandFilter
from publication.publication_figure import color


# ======================================================================================================================
# Déconvolution functions
# ======================================================================================================================
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
    x_fft = np.fft.rfft(x, n=nstft)
    y_fft = np.fft.rfft(y, n=nstft)

    if rho_f is None:
        rho_f = np.ones_like(x_fft)

    # Derive G(f)
    g_fft = 1 / x_fft * 1 / (1 + 1 / (np.abs(x_fft) ** 2 * rho_f))

    h_fft = g_fft * y_fft

    h = np.fft.irfft(h_fft, n=nstft)

    return h


def regularized_spectral_deconvolution(x, y, eps_ratio=1e-2):
    """
    Regularized spectral deconvolution taken from [2] Tutorial MATLAB material.

    Parameters
    ----------
    x : ndarray
        Source signal.
    y : ndarray
        Received signal.
    eps_ratio : float
        Regularization level relative to max(|X(f)|²).

    Returns
    -------
    h : ndarray
        Estimated impulse response.
    """
    nfft = max(len(x), len(y))

    Xf = np.fft.rfft(x, n=nfft)
    Yf = np.fft.rfft(y, n=nfft)
    eps = np.max(np.abs(Xf) ** 2) * eps_ratio

    Hf = (Yf * np.conj(Xf)) / np.maximum(np.abs(Xf) ** 2, eps)

    h = np.fft.irfft(Hf, n=nfft)

    return h


# ======================================================================================================================
# Test functions : synthetic data
# ======================================================================================================================


def generate_test_case(
    fs=1000,
    duration=10.0,
    fc=50.0,
    sigma_noise=0.1,
):
    """
    Generate a synthetic source, impulse response and received signal.
    """

    t = np.arange(0, duration, 1 / fs)

    # Impulsive source
    # x = np.zeros_like(t)
    # x[len(t) // 4] = 1.0

    # White noise source
    x = np.random.normal(
        scale=1.0,
        size=t.shape,
    )

    # Ricker wavelet impulse response
    t0 = 1.0
    h = (1 - 2 * (np.pi * fc * (t - t0)) ** 2) * np.exp(-((np.pi * fc * (t - t0)) ** 2))
    h /= np.max(np.abs(h))

    # Convolve in frequency domain
    y_fft = np.fft.rfft(x) * np.fft.rfft(h)
    y = np.fft.irfft(y_fft, n=len(t))

    noise = np.random.normal(
        scale=sigma_noise,
        size=y.shape,
    )

    y += noise

    return t, x, h, y


def plot_test_case(
    t,
    x,
    h,
    y,
    normalize=False,
):
    """
    Plot the synthetic deconvolution test case.

    Parameters
    ----------
    t : ndarray
        Time vector.
    x : ndarray
        Source signal.
    h : ndarray
        Impulse response.
    y : ndarray
        Received signal.
    normalize : bool, optional
        Normalize signals before plotting.
    """

    x_plot = x.copy()
    h_plot = h.copy()
    y_plot = y.copy()

    if normalize:

        def _norm(sig):
            vmax = np.max(np.abs(sig))
            return sig / vmax if vmax > 0 else sig

        x_plot = _norm(x_plot)
        h_plot = _norm(h_plot)
        y_plot = _norm(y_plot)

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True,
    )

    # Source signal
    axs[0].plot(t, x_plot)
    axs[0].set_title("Source signal $x(t)$")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    # Impulse response
    axs[1].plot(t[: len(h_plot)], h_plot)
    axs[1].set_title("Impulse response $h(t)$")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)

    # Received signal
    axs[2].plot(t[: len(y_plot)], y_plot)
    axs[2].set_title("Received signal $y(t) = x(t) * h(t)$")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Amplitude")
    axs[2].grid(True)

    plt.show()


def test_deconvolution_methods(sigma_noise=0.1):
    """
    Compare impulse response reconstruction errors.
    """

    t, x, h_true, y = generate_test_case(sigma_noise=sigma_noise)
    # plot_test_case(
    #     t,
    #     x,
    #     h_true,
    #     y,
    #     normalize=True,
    # )

    h_cross = crosscorr_deconvolution(x, y)
    h_wiener = wiener_deconvolution(x, y, rho_f=1 / sigma_noise**2)
    h_reg = regularized_spectral_deconvolution(x, y)

    n = min(
        len(h_true),
        len(h_cross),
        len(h_wiener),
        len(h_reg),
    )

    h_true = h_true[:n]

    methods = {
        "Cross-correlation": h_cross[:n],
        "Wiener": h_wiener[:n],
        "Regularized spectral": h_reg[:n],
    }

    print("\nImpulse response reconstruction errors")
    print("-" * 50)

    for name, h_hat in methods.items():

        rmse = np.sqrt(np.mean((h_hat - h_true) ** 2))

        corr = np.corrcoef(
            h_hat,
            h_true,
        )[0, 1]

        print(f"{name:25s}" f" RMSE={rmse:.4e}" f"  Corr={corr:.4f}")


def plot_deconvolution_comparison(sigma_noise=0.1):
    """
    Compare estimated impulse responses.
    """

    t, x, h_true, y = generate_test_case(sigma_noise=sigma_noise)
    plot_test_case(
        t,
        x,
        h_true,
        y,
        normalize=True,
    )

    h_cross = crosscorr_deconvolution(x, y)
    h_wiener = wiener_deconvolution(x, y)
    h_reg = regularized_spectral_deconvolution(x, y)

    n = len(h_true)

    fig, axs = plt.subplots(
        4,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    axs[0].plot(t, h_true, "k", lw=2)
    axs[0].set_title("Reference impulse response")

    axs[1].plot(t, h_cross[:n], "r")
    axs[1].set_title("Cross-correlation")

    axs[2].plot(t, h_wiener[:n], "b")
    axs[2].set_title("Wiener")

    axs[3].plot(t, h_reg[:n], "g")
    axs[3].set_title("Regularized spectral")

    axs[-1].set_xlabel("Time (s)")

    # Reconstructed signals
    reconstruction_errors = {}

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(12, 10),
        sharex=True,
    )

    ax.plot(t, y, "k", lw=2)
    ax.set_title("Reference received signal")

    # y_hat_crosscorr = sp.convolve(x, h_cross, mode="same")
    y_hat_crosscorr = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(h_cross), n=len(t))
    reconstruction_errors["cross_corr"] = np.sqrt(np.mean((y - y_hat_crosscorr) ** 2))

    ax.plot(t, y_hat_crosscorr, "r", linestyle="--", alpha=0.8)
    ax.set_title("Reconstructed received signal (Cross-correlation)")

    # y_hat_wiener = sp.convolve(x, h_wiener, mode="same")
    y_hat_wiener = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(h_wiener), n=len(t))
    reconstruction_errors["wiener"] = np.sqrt(np.mean((y - y_hat_wiener) ** 2))
    ax.plot(t, y_hat_wiener, "b", linestyle="--", alpha=0.8)
    ax.set_title("Reconstructed received signal (Wiener)")

    # y_hat_reg = sp.convolve(x, h_reg, mode="same")
    y_hat_reg = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(h_reg), n=len(t))
    reconstruction_errors["regularized"] = np.sqrt(np.mean((y - y_hat_reg) ** 2))
    ax.plot(t, y_hat_reg, "g", linestyle="--", alpha=0.8)
    ax.set_title("Reconstructed received signal (Regularized spectral)")

    ax.set_xlabel("Time (s)")

    print("\nReconstruction errors")
    print("-" * 50)

    for name, rmse in sorted(
        reconstruction_errors.items(),
        key=lambda x: x[1],
    ):
        print(f"{name:<20s} RMSE = {rmse:.6e}")

    plt.tight_layout()
    plt.show()


# ======================================================================================================================
# Test functions : real data
# ======================================================================================================================


def test_css_deconvolution(source_signal_fpath, received_signal_fpath):
    """
    Compare deconvolution methods on real CSS data.
    """

    # Load data
    fs_source, source_signal = wavfile.read(source_signal_fpath)
    fs_received, received_signal = wavfile.read(received_signal_fpath)

    # Check sampling frequencies
    if fs_source != fs_received:
        print(
            f"Warning: source and received signals have different sampling frequencies "
            f"({fs_source} Hz vs {fs_received} Hz). Resampling to common frequency.",
        )
        if fs_source < fs_received:
            print(
                f"Resampling (decimation) received from {fs_received} Hz to {fs_source} Hz"
            )
            fs_src = fs_received
            fs_target = fs_source

            # Resample to common frequency
            received_signal_rs, fs = match_sampling_frequency(
                received_signal,
                fs_src=fs_src,
                fs_target=fs_target,
            )

            plot_resampling_check(
                signal=received_signal,
                signal_rs=received_signal_rs,
                fs_src=fs_src,
                fs_target=fs_target,
            )
            received_signal = received_signal_rs

        else:
            print(
                f"Resampling (decimation) source signal from {fs_source} Hz to {fs_received} Hz"
            )
            fs_src = fs_source
            fs_target = fs_received

            # Resample to common frequency
            source_signal_rs, fs = match_sampling_frequency(
                source_signal,
                fs_src=fs_src,
                fs_target=fs_target,
            )
            plot_resampling_check(
                signal=source_signal,
                signal_rs=source_signal_rs,
                fs_src=fs_src,
                fs_target=fs_target,
            )
            source_signal = source_signal_rs

        fs = fs_target
    else:
        fs = fs_received

    # Plot signals
    plot_source_received(
        source_signal,
        received_signal,
        fs,
    )

    h_cross = crosscorr_deconvolution(
        source_signal,
        received_signal,
    )

    # Define frequency-dependent regularization parameter for Wiener deconvolution considering the last 10 % or recording assuming a noise floor in the last part of the recording. This is a common practice to avoid over-amplification of noise at frequencies where the source signal is weak.
    # rcv_signal_noise = received_signal[int(0.9 * len(received_signal)) :]
    # rcv_signal_sig = received_signal[: int(0.9 * len(received_signal))]
    # nfft = max(len(received_signal), len(source_signal))
    # rho_f = (
    #     np.abs(np.fft.rfft(rcv_signal_noise, n=nfft)) ** 2
    #     / np.abs(np.fft.rfft(rcv_signal_sig, n=nfft)) ** 2
    # ) * 1e6

    h_wiener = wiener_deconvolution(
        source_signal,
        received_signal,
        # rho_f=rho_f,
    )

    h_reg = regularized_spectral_deconvolution(
        source_signal,
        received_signal,
    )

    # # Apply bandpass filtering to the estimated impulse responses
    # bf = BandFilter(
    #     order=4,
    #     lowcut=1,
    #     highcut=5000,
    # )
    # h_cross = bf.apply_filter(h_cross, fs=fs)
    # h_wiener = bf.apply_filter(h_wiener, fs=fs)
    # h_reg = bf.apply_filter(h_reg, fs=fs)

    return {
        "fs": fs,
        "source_signal": source_signal,
        "received_signal": received_signal,
        "crosscorr": h_cross,
        "wiener": h_wiener,
        "regularized": h_reg,
    }


def plot_source_received(
    s_source,
    s_received,
    fs,
    normalize=True,
):
    """
    Plot source and received signals before deconvolution.

    Parameters
    ----------
    s_source : ndarray
        Source signal (resampled if needed).
    s_received : ndarray
        Received signal.
    fs : float
        Common sampling frequency.
    normalize : bool
        If True, normalize both signals by their max amplitude.
    """

    x = s_source.astype(float)
    y = s_received.astype(float)

    Xf = np.fft.rfft(x)
    Yf = np.fft.rfft(y)
    f_x = np.fft.rfftfreq(len(x), d=1 / fs)
    f_y = np.fft.rfftfreq(len(y), d=1 / fs)

    if normalize:
        x = x / np.max(np.abs(x))
        y = y / np.max(np.abs(y))

    t_x = np.arange(len(x)) / fs
    t_y = np.arange(len(y)) / fs

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 16), sharex="col")

    axs[0, 0].plot(t_x, x, label="Source signal", color=color(0))
    axs[1, 0].plot(t_y, y, label="Received signal", color=color(1))
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Amplitude (normalized)" if normalize else "Amplitude")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Amplitude (normalized)" if normalize else "Amplitude")
    # axs[1, 0].set_title("Source vs received signal (before deconvolution)")

    axs[0, 1].plot(f_x, np.abs(Xf), label="Source signal spectrum", color=color(0))
    axs[1, 1].plot(
        f_y,
        np.abs(Yf),
        label="Received signal spectrum",
        color=color(1),
    )
    axs[0, 1].set_xlabel("Frequency [Hz]")
    axs[0, 1].set_ylabel("Spectrum magnitude")
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_ylabel("Spectrum magnitude")
    # axs[1, 0].set_title("Source vs received signal (before deconvolution)")

    for ax in axs.flatten():
        ax.legend()
        ax.grid(True)

    # plt.tight_layout()
    # plt.show()


def match_sampling_frequency(
    signal,
    fs_src,
    fs_target,
):
    """
    Resample signal to match target sampling frequency.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    fs_src : float
        Source sampling frequency.
    fs_target : float
        Target sampling frequency (received signal).

    Returns
    -------
    signal_resampled : ndarray
        Resampled signal.
    fs_out : float
        Updated sampling frequency (== fs_target).
    """

    # Rational approximation of resampling ratio
    up = int(fs_target)
    down = int(fs_src)

    signal_rs = resample_poly(
        signal,
        up,
        down,
    )

    return signal_rs, fs_target


def plot_css_deconvolution_results(deconv_results, reference_deconv_fpath=None):
    """
    Visualize and compare CSS deconvolution results.

    Parameters
    ----------
    deconv_results : dict
        Output of test_css_deconvolution().

    reference_deconv_fpath : str or None
        Optional path to reference deconvolution result for comparison.
    """

    fs = deconv_results["fs"]
    source_signal = deconv_results["source_signal"]
    received_signal = deconv_results["received_signal"]

    # Remove signals from deconv_results to keep only impulse responses
    deconv_results = {
        name: h
        for name, h in deconv_results.items()
        if name not in ["fs", "source_signal", "received_signal"]
    }

    t_y = np.arange(len(received_signal)) / fs

    fig, axs = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=False,
        constrained_layout=True,
    )

    # =====================================================================
    # Source signal
    # =====================================================================

    t_x = np.arange(len(source_signal)) / fs
    # Normalize for better visualization
    source_signal_norm = source_signal / np.max(np.abs(source_signal))

    axs[0].plot(t_x, source_signal_norm)
    axs[0].set_title("Source signal")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    # =====================================================================
    # Estimated impulse responses
    # =====================================================================

    ax = axs[1]

    for name, h in deconv_results.items():

        t_h = np.arange(len(h)) / fs

        ax.plot(
            t_h,
            h / np.max(np.abs(h)),
            label=name,
            linewidth=1.5,
        )

    # Add reference deconv solution
    if reference_deconv_fpath is not None:
        fs_ref, h_ref = wavfile.read(reference_deconv_fpath)
        t_h_ref = np.arange(len(h_ref)) / fs_ref

        ax.plot(
            t_h_ref,
            h_ref / np.max(np.abs(h_ref)),
            label="Reference",
            color="k",
            linestyle="--",
            linewidth=2,
        )

    ax.set_title("Estimated impulse responses")
    ax.set_ylabel("Normalized amplitude")
    ax.legend()
    ax.grid(True)

    # =====================================================================
    # Signal reconstruction
    # =====================================================================

    ax = axs[2]

    ax.plot(
        t_y,
        received_signal,
        color="k",
        linewidth=2,
        label="received",
    )

    reconstruction_errors = {}

    for name, h in deconv_results.items():

        n = len(h)
        Hf = np.fft.rfft(h, n=n)
        Xf = np.fft.rfft(source_signal, n=n)
        y_hat = np.fft.irfft(Hf * Xf)
        t_y_hat = np.arange(len(y_hat)) / fs

        # rcv_sig_rebuild_f = fft(sig_deconv_t, NFFT_deconv) .* fft(s_source_ok, NFFT_deconv);
        # rcv_sig_rebuild_t = ifft(rcv_sig_rebuild_f);

        # y_hat = sp.convolve(
        #     source_signal,
        #     h,
        #     mode="same",
        # )

        # n = min(len(y_hat), len(received_signal))

        # y_hat = y_hat[:n]
        # y_ref = received_signal[:n]
        y_ref = received_signal

        reconstruction_errors[name] = np.sqrt(np.mean((y_ref - y_hat) ** 2))

        ax.plot(
            t_y_hat,
            y_hat,
            label=f"{name}",
            alpha=0.8,
        )

    ax.set_title("Received signal reconstruction")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

    # =====================================================================
    # Reconstruction error
    # =====================================================================

    ax = axs[3]

    methods = []
    rmses = []

    for name, h in deconv_results.items():

        y_hat = sp.convolve(
            source_signal,
            h,
            mode="same",
        )

        n = min(len(y_hat), len(received_signal))

        err = received_signal[:n] - y_hat[:n]

        methods.append(name)
        rmses.append(np.sqrt(np.mean(err**2)))

    ax.bar(methods, rmses)

    ax.set_title("Reconstruction RMSE")
    ax.set_ylabel("RMSE")
    ax.grid(True)

    plt.show()

    # =====================================================================
    # Numerical summary
    # =====================================================================

    print("\nReconstruction errors")
    print("-" * 50)

    for name, rmse in sorted(
        reconstruction_errors.items(),
        key=lambda x: x[1],
    ):
        print(f"{name:<20s} RMSE = {rmse:.6e}")


def plot_css_deconvolution_spectrograms(
    received_signal,
    deconv_results,
    fs,
    nfft=1024,
    nperseg=303,
    max_time=None,
    dynamic_range_db=60,
):
    """
    Compare spectrograms of deconvolution results.

    Parameters
    ----------
    received_signal : ndarray
        Received signal.

    deconv_results : dict
        Output from test_css_deconvolution().

    fs : float
        Sampling frequency [Hz].

    nfft : int
        FFT size.

    nperseg : int
        Window length.

    max_time : float or None
        Maximum displayed time (s).

    dynamic_range_db : float
        Display dynamic range below peak.
    """

    signals = {"received": received_signal}
    signals.update(deconv_results)

    spectrograms = {}

    global_max_db = -np.inf

    # Compute all spectrograms
    for name, sig in signals.items():

        f, t, S = spectrogram(
            sig,
            fs=fs,
            window="hamming",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=nfft,
            mode="magnitude",
        )

        S_db = 20 * np.log10(S + 1e-12)

        global_max_db = max(global_max_db, np.max(S_db))

        spectrograms[name] = (f, t, S_db)

    vmin = global_max_db - dynamic_range_db
    vmax = global_max_db

    nplots = len(signals)

    fig, axs = plt.subplots(
        nplots,
        1,
        figsize=(12, 3 * nplots),
        sharex=True,
        constrained_layout=True,
    )

    if nplots == 1:
        axs = [axs]

    for ax, (name, (f, t, S_db)) in zip(
        axs,
        spectrograms.items(),
    ):

        pcm = ax.pcolormesh(
            t,
            f,
            S_db,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(name)

        ax.set_ylim(0, fs / 2)

        if max_time is not None:
            ax.set_xlim(0, max_time)

    axs[-1].set_xlabel("Time [s]")

    cbar = fig.colorbar(
        pcm,
        ax=axs,
        shrink=0.95,
        label="Magnitude [dB]",
    )

    plt.show()


def plot_resampling_check(signal, signal_rs, fs_src, fs_target):
    """
    Visual check of resampling effect.
    """

    t1 = np.arange(len(signal)) / fs_src
    t2 = np.arange(len(signal_rs)) / fs_target

    plt.figure(figsize=(10, 4))

    plt.plot(t1, signal, label="Original signal", color=color(0))
    plt.plot(t2, signal_rs, "o", markersize=2, label="Resampled signal", color=color(1))

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":

    # # ====================
    # # Synthetic data
    # # ====================
    # test_deconvolution_methods()
    # plot_deconvolution_comparison()

    # ====================
    # Real data: CSS deconvolution from Bonnel et al. 2020 [2]
    # ====================

    root_wav = r"C:\Users\baptiste.menetrier\Desktop\ressource\Bonnel_supp_publi_2020_warping\matlab_code\experimental_data\c_css_deconv"
    received_file = os.path.join(root_wav, "css_received_signal.wav")
    source_file = os.path.join(root_wav, "css_source_signal.wav")

    results = test_css_deconvolution(
        source_signal_fpath=source_file,
        received_signal_fpath=received_file,
    )

    reference_deconv_file = os.path.join(root_wav, "css_ready_to_warp.wav")

    plot_css_deconvolution_results(
        deconv_results=results, reference_deconv_fpath=reference_deconv_file
    )

    # Plot spectrogram of estimated impulse response
    fs = results["fs"]
    h_reg = results["regularized"]
    nperseg = 1024
    nw = 50
    noverlap = nperseg - nw

    (
        ff,
        tt,
        Sxx,
    ) = sp.stft(h_reg, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hamming")

    plt.figure()
    plt.pcolormesh(tt, ff, 20 * np.log10(np.abs(Sxx)))

    plt.show()

    # # ====================
    # # Real data: Groix
    # # ====================
    # root_wav = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data\extracted_signal_for_warping"
    # received_file = os.path.join(root_wav, "OBS1_seq_144_pulse_25.wav")
    # source_file = os.path.join(root_wav, "source_synthetic_chirp_144.wav")

    # results = test_css_deconvolution(
    #     source_signal_fpath=source_file,
    #     received_signal_fpath=received_file,
    # )

    # plot_css_deconvolution_results(deconv_results=results, reference_deconv_fpath=None)
