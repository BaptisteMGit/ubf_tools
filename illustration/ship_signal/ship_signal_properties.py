#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ship_signal_properties.py
@Time    :   2024/09/19 11:40:52
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import scipy.signal as sp
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

from scipy.special import erf
from get_data.ais.ais_tools import *
from matplotlib.dates import DateFormatter
from get_data.wav.signal_processing_utils import *
from signals.signals import ship_spectrum, generate_ship_signal

from publication.PublicationFigure import PubFigure

PubFigure(legend_fontsize=8, label_fontsize=8, ticks_fontsize=8, title_fontsize=8)

IMG_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\ship_signal"


# ======================================================================================================================
# Functions
# ======================================================================================================================
def plot_ship_spectrum(fs):

    f = np.linspace(0, fs / 2, int(1e3))
    Aship = ship_spectrum(f)

    # Module
    plt.figure()
    plt.plot(f, np.abs(Aship))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$| \tilde{e}_{ship} (f) |$")
    plt.title("Ship spectrum module")
    plt.grid()
    # plt.show()

    # Phase
    plt.figure()
    plt.plot(f, np.angle(Aship))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$arg( \tilde{e}_{ship} )$" + " [rad]")
    plt.title("Ship spectrum phase")
    plt.grid()
    # plt.show()
    plt.close("all")

    # Inverse Fourier transform of Aship -> "minimum phase pulse"
    tau = 10
    nfftinv = int(2**12)
    e = np.fft.ifft(Aship * np.exp(-1j * 2 * np.pi * f * tau), n=nfftinv).real
    df = f[1] - f[0]
    fs = nfftinv * df
    t = np.arange(0, len(e)) * 1 / fs
    plt.figure()
    plt.plot(t, e)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$e_{ship}(t)$")
    plt.title("Ship signal")
    plt.grid()

    fpath = os.path.join(
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ship_sig",
        "pulse_model.dat",
    )
    df = pd.DataFrame({"t": t, "e": e / np.max(np.abs(e))})
    df.to_csv(fpath, sep=" ", index=False)

    plt.show()


def plot_instant_freq_perturbation_spectrum(fs, tau_corr_fi, std_fi):
    Ttot = 10
    t = np.arange(0, Ttot, 1 / fs)
    f = np.arange(0, fs, 1 / Ttot)
    Nt = len(t)

    Gaussian = np.zeros_like(f)

    Gaussian[0 : int(np.floor(Nt / 2))] = (
        np.sqrt(2 * np.pi)
        * tau_corr_fi
        * std_fi**2
        * np.exp(-2 * (np.pi * f[0 : int(np.floor(Nt / 2))] * tau_corr_fi) ** 2)
    )
    Gaussian[-1 : int(np.floor(Nt / 2)) : -1] = np.conj(
        Gaussian[1 : int(np.ceil(Nt / 2))]
    )
    delta_fi = np.random.randn(Nt)
    delta_fi = np.fft.ifft(
        np.fft.fft(delta_fi) * np.sqrt(Gaussian * fs)
    )  # 19/09/2024 for clarity

    plt.figure()
    plt.plot(f, Gaussian)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Gaussian")
    plt.title("Gaussian spectrum")
    plt.grid()
    # plt.show

    plt.figure()
    plt.plot(t, delta_fi)
    plt.xlabel("Time [s]")
    plt.ylabel("Instant frequency perturbation")
    plt.title("Instant frequency perturbation")
    plt.grid()
    # plt.show()

    delta_ph = np.cumsum(delta_fi) / fs  # random instant phase perturbation

    plt.figure()
    plt.plot(t, delta_ph)
    plt.xlabel("Time [s]")
    plt.ylabel("Instant phase perturbation")
    plt.title("Instant phase perturbation")
    plt.grid()
    plt.show()


def samuel_test_cases():

    # Common parameters
    Ttot = 10**4
    n = 1
    f0 = 10
    T0 = 1 / f0
    A = 0.1
    fs = 100
    std_fi = A * f0

    # Test case 1
    tau_corr_fi_1 = 10 * T0
    s_1, t = generate_ship_signal(
        Ttot, f0, std_fi=std_fi, tau_corr_fi=tau_corr_fi_1, fs=100, Nh=n
    )

    # Test case 2
    tau_corr_fi_2 = 1 / 10 * T0
    s_2, t = generate_ship_signal(
        Ttot, f0, std_fi=std_fi, tau_corr_fi=tau_corr_fi_2, fs=100, Nh=n
    )

    # Test case 3
    tau_corr_fi_3 = 1 * T0
    s_3, t = generate_ship_signal(
        Ttot, f0, std_fi=std_fi, tau_corr_fi=tau_corr_fi_3, fs=100, Nh=n
    )

    # Derive auto-correlation
    lag = np.arange(-len(t) + 1, len(t)) * 1 / fs
    R_1 = sp.correlate(s_1, s_1, mode="full")
    R_1 = R_1 / np.max(R_1)
    R_2 = sp.correlate(s_2, s_2, mode="full")
    R_2 = R_2 / np.max(R_2)
    R_3 = sp.correlate(s_3, s_3, mode="full")
    R_3 = R_3 / np.max(R_3)

    # Derive psd
    f, Pxx_1 = sp.welch(s_1, fs, nperseg=2**10, noverlap=2**9)
    f, Pxx_2 = sp.welch(s_2, fs, nperseg=2**10, noverlap=2**9)
    f, Pxx_3 = sp.welch(s_3, fs, nperseg=2**10, noverlap=2**9)

    # Plot
    # Time domain
    plt.figure()
    plt.plot(t, s_1, label="tau_corr_fi = 10 * T0")
    plt.plot(t, s_2, label="tau_corr_fi = 1/10 * T0")
    plt.plot(t, s_3, label="tau_corr_fi = 1 * T0")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Ship signal")
    plt.legend()
    plt.grid()

    # Auto-correlation
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    # First row : autocorr
    ax[0, 0].plot(lag, R_1, label="tau_corr_fi = 10 * T0")
    ax[0, 0].set_xlim(-1, 1)
    ax[0, 1].plot(lag, R_2, label="tau_corr_fi = 1/10 * T0")
    ax[0, 1].set_xlim(-10, 10)
    ax[0, 2].plot(lag, R_3, label="tau_corr_fi = 1 * T0")
    ax[0, 2].set_xlim(-1, 1)
    for i in range(3):
        ax[0, i].set_xlabel(r"$\tau $" + " [s]")
        ax[0, i].set_title(f"Test case {i+1}")
        ax[0, i].grid()
    ax[0, 0].set_ylabel(r"$R_{ss} (\tau)$")

    # Second row : PSD
    ax[1, 0].plot(f - f0, 10 * np.log10(Pxx_1), label="tau_corr_fi = 10 * T0")
    ax[1, 0].set_xlim(-2.5, 2.5)
    ax[1, 0].set_ylim(-50, 0)
    ax[1, 1].plot(f - f0, 10 * np.log10(Pxx_2), label="tau_corr_fi = 1/10 * T0")
    ax[1, 1].set_xlim(-1, 1)
    ax[1, 1].set_ylim(-30, 10)
    ax[1, 2].plot(f - f0, 10 * np.log10(Pxx_3), label="tau_corr_fi = 1 * T0")
    ax[1, 2].set_xlim(-2.5, 2.5)
    ax[1, 2].set_ylim(-40, 0)

    for i in range(3):
        ax[1, i].set_xlabel(r"$f - f_0$" + " [Hz]")
        ax[1, i].grid()
    ax[1, 0].set_ylabel(r"$S_{ss} (f)$")

    # plt.title("PSD of the ship signal")
    fpath = os.path.join(IMG_ROOT, "ship_testcases_Rss_Sss.png")
    plt.savefig(fpath)


def stdfi_taucorrfi_influence():

    tau_corr_fi_per_T0 = [0.01, 0.1, 1, 5, 10]
    std_fi_per_f0 = [0.01, 0.1, 1, 5, 10]
    Ttot = 10**4
    f0 = 5
    fs = 100
    # n = int(np.floor(fs / 2 / f0) - 1)
    n = 1

    Rxx = []
    Sxx = []
    signals = []
    for tau_corr_fi in tau_corr_fi_per_T0:
        tau_corr_fi = tau_corr_fi * 1 / f0
        for std_fi in std_fi_per_f0:
            std_fi = std_fi * f0

            s, t = generate_ship_signal(
                Ttot, f0, std_fi=std_fi, tau_corr_fi=tau_corr_fi, fs=fs, Nh=n
            )
            signals.append(s)

            # Derive auto-correlation
            lag = np.arange(-len(t) + 1, len(t)) * 1 / fs
            R = sp.correlate(s, s, mode="full")
            R = R / np.max(R)
            Rxx.append(R)

            # Derive psd
            f, Pxx = sp.welch(s, fs, nperseg=2**10, noverlap=2**9)
            Sxx.append(Pxx)

    # Parameters string
    param_str = f"(Ttot={Ttot} s, f0={f0} Hz, fs={fs} Hz, n={n})"
    # Plot Rxx
    fig, ax = plt.subplots(
        len(tau_corr_fi_per_T0), len(std_fi_per_f0), figsize=(16, 10)
    )
    fig.suptitle(r"$R_{xx}(\tau)$" + "\n" + param_str)
    for i in range(len(tau_corr_fi_per_T0)):
        ax[i, 0].set_ylabel(
            r"$\tau_{corr_fi} = $" + str(tau_corr_fi_per_T0[i]) + r"$T_0$"
        )
        for j in range(len(std_fi_per_f0)):
            ax[0, j].set_title(r"$\sigma_{fi} = $" + str(std_fi_per_f0[j]) + r"$f_0$")
            ax[-1, j].set_xlabel(r"$\tau $" + " [s]")

            ax[i, j].plot(lag, Rxx[i * len(std_fi_per_f0) + j])
            ax[i, j].set_xlim(-1, 1)
            ax[i, j].grid()

    # Save
    IMG_ROOT = (
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\ship_signal"
    )
    fpath = os.path.join(
        IMG_ROOT, f"ship_stdfi_taucorrfi_influence_Rss_{n}harmonics.png"
    )
    plt.savefig(fpath)

    # Plot Sxx
    fig, ax = plt.subplots(
        len(tau_corr_fi_per_T0), len(std_fi_per_f0), figsize=(16, 10)
    )
    fig.suptitle(r"$S_{xx}(f)$" + "\n" + param_str)
    for i in range(len(tau_corr_fi_per_T0)):
        ax[i, 0].set_ylabel(
            r"$\tau_{corr_fi} = $" + str(tau_corr_fi_per_T0[i]) + r"$T_0$"
        )
        for j in range(len(std_fi_per_f0)):
            ax[0, j].set_title(r"$\sigma_{fi} = $" + str(std_fi_per_f0[j]) + r"$f_0$")
            ax[-1, j].set_xlabel(r"$f - f_0$" + " [Hz]")

            ax[i, j].plot(f - f0, 10 * np.log10(Sxx[i * len(std_fi_per_f0) + j]))
            ax[i, j].set_xlim(-5, 5)
            ax[i, j].set_ylim(-50, 10)
            ax[i, j].grid()

    # Save
    fpath = os.path.join(
        IMG_ROOT, f"ship_stdfi_taucorrfi_influence_Sss_{n}harmonics.png"
    )
    plt.savefig(fpath)


def fi_build_process(fs, tau_corr_fi, std_fi):
    Ttot = 10**4
    t = np.arange(0, Ttot, 1 / fs)
    f = np.arange(0, fs, 1 / Ttot)
    Nt = len(t)

    freq = np.fft.fftfreq(Nt, 1 / fs)
    G = np.zeros_like(freq)
    G = (
        np.sqrt(2 * np.pi)
        * tau_corr_fi
        * std_fi**2
        * np.exp(-2 * (np.pi * freq * tau_corr_fi) ** 2)
    )

    delta_fi = np.random.randn(Nt)
    delta_fi = np.fft.ifft(
        np.fft.fft(delta_fi) * np.sqrt(G * fs)
    )  # 19/09/2024 for clarity
    delta_ph = np.cumsum(delta_fi) / fs  # random instant phase perturbation

    shifted_freq = np.fft.fftshift(freq)
    delta_fi_fft = np.abs(np.fft.fft(delta_fi))

    delta_fi_th_np = np.random.normal(
        0, 1 / (tau_corr_fi * np.pi * 2), len(shifted_freq)
    )
    delta_fi_th_np_fft = np.abs(np.fft.fft(delta_fi_th_np))

    # Plot psd
    f, delta_fi_psd = sp.welch(delta_fi, fs, nperseg=2**10, noverlap=2**9)
    plt.figure()
    plt.plot(f, 10 * np.log10(delta_fi_psd))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.title("PSD of delta_fi")
    plt.grid()

    plt.figure()
    plt.plot(shifted_freq, np.fft.fftshift(delta_fi_fft))
    # plt.plot(shifted_freq, np.fft.fftshift(G))
    plt.plot(shifted_freq, np.fft.fftshift(delta_fi_th_np_fft))
    # plt.hist(delta_fi_th_np, bins=100)

    # plt.show()

    plt.figure()
    # plt.plot(shifted_freq, np.fft.fftshift(delta_fi_fft))
    delta_fi_th = (
        1
        / (std_fi * np.sqrt(2 * np.pi))
        * np.exp(-1 / 2 * (shifted_freq / std_fi) ** 2)
    )

    plt.plot(shifted_freq, delta_fi_th)
    plt.plot(shifted_freq, delta_fi_th_np)

    # fi = np.random.randn(Nt)
    # mod_fi_fft = np.abs(np.fft.fft(fi))
    # plt.figure()
    # plt.plot(shifted_freq, np.fft.fftshift(mod_fi_fft))
    # # plt.show()

    # delta_ph_fft = np.abs(np.fft.fft(delta_ph))
    # plt.figure()
    # plt.plot(shifted_freq, np.fft.fftshift(delta_ph_fft))
    plt.show()


def load_wav_data(
    fmin=2, fmax=48, filter_type="bandpass", filter_corners=4, save_wav=False
):

    date = "2013-05-09 13:30:00"
    ch = ["BDH"]
    rcv_id = "RR47"
    duration_s = 60 * 60 * 1.6
    # Select frequency properties
    nperseg = 2**12
    noverlap = int(nperseg * 3 / 4)
    data = {}
    raw_sig, filt_sig, corr_sig = get_rhumrum_data(
        station_id=rcv_id,
        date=date,
        duration_sec=duration_s,
        channels=ch,
        plot=False,
        fmin=fmin,
        fmax=fmax,
        filter_type=filter_type,
        filter_corners=filter_corners,
    )

    sig = corr_sig["BDH"]
    data["data"] = sig.data
    data["sig"] = sig

    # Derive stft
    f, tt, stft = sp.stft(
        data["data"],
        fs=sig.meta.sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    data["f"] = f
    data["tt"] = tt
    data["stft"] = stft

    if save_wav:
        # Convert date to datetime
        date = date.replace(" ", "_").replace(":", "-")
        fpath = os.path.join(
            r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ship_sig",
            f"{date}.wav",
        )
        wavfile.write(fpath, sig.meta.sampling_rate, data["data"])

    return data


def filter_real_data(dsp_args):

    # Load wav data
    data = load_wav_data()

    img_root_filter = os.path.join(IMG_ROOT, "ray_selection")
    if not os.path.exists(img_root_filter):
        os.makedirs(img_root_filter)

    PubFigure(
        legend_fontsize=12, label_fontsize=18, ticks_fontsize=14, title_fontsize=20
    )

    # Create a time vector composed of datetime objects for the x-axis
    t0 = data["sig"].meta.starttime.datetime
    t1 = data["sig"].meta.endtime.datetime
    dtt = data["tt"][1] - data["tt"][0]
    tt_datetime = pd.date_range(start=t0, end=t1, periods=len(data["tt"]))

    # Plot stft
    vmin = -80
    vmax = -20
    plt.figure(figsize=(14, 6))
    plt.pcolormesh(
        # data["tt"],
        tt_datetime,
        data["f"],
        20 * np.log10(np.abs(data["stft"])),
        vmin=vmin,
        vmax=vmax,
        cmap="jet",
        shading="gouraud",
    )
    plt.ylabel("$f$" + " [Hz]")
    plt.xlabel("$t$" + " [s]")
    cbar = plt.colorbar()
    cbar.set_label("Amplitude [dB]")

    # Format x-axis with time and show only the date once
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.gca().xaxis.set_major_locator(
        mdates.MinuteLocator(interval=30)
    )  # Two tick per hour
    plt.gca().xaxis.set_minor_locator(
        mdates.MinuteLocator(interval=15)
    )  # Minor ticks every 15 minutes

    # Add the date as a single label above the axis
    plt.gcf().autofmt_xdate()
    date_str = t0.strftime("%Y-%m-%d")
    plt.annotate(
        date_str, xy=(0.05, -0.1), xycoords="axes fraction", ha="center", fontsize=14
    )

    fpath = os.path.join(img_root_filter, f"stft.png")
    plt.savefig(fpath)

    # Derive dsp
    dsp_param_label = f"nperseg{int(dsp_args['nperseg'])}_overlap{dsp_args['overlap_coef']}_window{dsp_args['window']}"
    fmin_band = 3
    fmax_band = 18
    # fmin_band = 3
    # fmax_band = 45
    ff, Pxx = sp.welch(
        data["data"],
        fs=data["sig"].meta.sampling_rate,
        nperseg=dsp_args["nperseg"],
        noverlap=int(dsp_args["nperseg"] * dsp_args["overlap_coef"]),
        window=dsp_args["window"],
    )
    Pxx_in_band = Pxx[(ff >= fmin_band) & (ff <= fmax_band)]
    ff_in_band = ff[(ff >= fmin_band) & (ff <= fmax_band)]

    # Find peaks in coherence to locate first harmonic
    distance_samples = np.floor(1.5 / (ff_in_band[1] - ff_in_band[0]))
    idx_f_peaks, _ = sp.find_peaks(
        10 * np.log10(Pxx_in_band), distance=distance_samples
    )
    f_peaks = ff_in_band[idx_f_peaks]
    k = np.arange(1, len(f_peaks) + 1)
    harmonic_labels = []
    j = 0
    for i in range(0, len(f_peaks)):
        if i % 2 == 0:
            harmonic_labels.append(f"${k[j]}f_0$")
            j += 1
        else:
            harmonic_labels.append(r"$\frac{" + f"{2+i}" + r"}{2}f_0$")

    f0 = f_peaks[0]

    # Plot dsp
    plt.figure()
    plt.plot(ff_in_band, 10 * np.log10(Pxx_in_band))
    # Add detected peaks
    plt.scatter(f_peaks, 10 * np.log10(Pxx_in_band[idx_f_peaks]), color="r")
    # Add labels
    for i, txt in enumerate(harmonic_labels):
        plt.annotate(
            txt,
            (f_peaks[i], 10 * np.log10(Pxx_in_band[idx_f_peaks[i]])),
            xytext=(f_peaks[i] + 0.1, 10 * np.log10(Pxx_in_band[idx_f_peaks[i]]) + 0.5),
            ha="left",
        )
    plt.xlabel("$f$" + " [Hz]")
    plt.ylabel("$S_{xx}(f)$" + " [dB $(Pa^2Hz^{-1})$]")
    plt.grid()
    fpath = os.path.join(img_root_filter, f"psd_{dsp_param_label}.png")
    plt.savefig(fpath)

    # Plot dsp with th harmonics and f0
    f_rays_th_f0 = np.arange(1, 20) * f0
    f__rays_th = [(2 + i) / 2 * f0 for i in range(1, 20) if i % 2 != 0]
    f_rays_th = np.concatenate((f_rays_th_f0, f__rays_th))
    idx_in_band = f_rays_th <= fmax_band
    f_rays_th = f_rays_th[idx_in_band]
    f_rays_th_idx = [
        np.argmin(np.abs(ff_in_band - f_rays_th[i])) for i in range(len(f_rays_th))
    ]
    f_ray_th_ff = ff_in_band[f_rays_th_idx]
    Pxx_ray_th = 10 * np.log10(Pxx_in_band[f_rays_th_idx])

    harmonic_labels_f0 = [f"${i}f_0$" for i in range(1, len(f_rays_th_f0) + 1)]
    harmonic_labels_ = [
        f"$\\frac{{{2+i}}}{{2}}f_0$" for i in range(1, 20) if i % 2 != 0
    ]
    harmonic_labels = np.concatenate((harmonic_labels_f0, harmonic_labels_))[
        idx_in_band
    ]

    plt.figure()
    # Add vertical lines at the th peaks
    for i in range(len(f_ray_th_ff)):
        plt.axvline(x=f_ray_th_ff[i], color="g", linestyle="--", alpha=0.5)
        # Annotate the peaks
        plt.annotate(
            harmonic_labels[i],
            (f_ray_th_ff[i], np.min(10 * np.log10(Pxx_in_band))),
            xytext=(f_ray_th_ff[i] + 0.1, np.min(10 * np.log10(Pxx_in_band))),
            ha="left",
        )
    plt.plot(ff_in_band, 10 * np.log10(Pxx_in_band))
    # Add detected peaks
    plt.scatter(f_peaks, 10 * np.log10(Pxx_in_band[idx_f_peaks]), color="r")
    # for i, txt in enumerate(harmonic_labels):
    #     plt.annotate(
    #         txt,
    #         (f_ray_th_ff[i], Pxx_ray_th),
    #         xytext=(f_ray_th_ff[i] + 0.1, Pxx_ray_th + 0.5),
    #         ha="left",
    #     )
    plt.xlabel("$f$" + " [Hz]")
    plt.ylabel("$S_{xx}(f)$" + " [dB $(Pa^2Hz^{-1})$]")
    plt.grid()
    fpath = os.path.join(img_root_filter, f"psd_rays_th_{dsp_param_label}.png")
    plt.savefig(fpath)

    # Band pass around the one ray
    idx_ray = 4
    f_ray = f_peaks[idx_ray]
    fs = data["sig"].meta.sampling_rate
    f_ray_width = 0.25
    fmin = f_ray - f_ray_width
    fmax = f_ray + f_ray_width
    filter_type = "bandpass"
    filter_corners = 20

    print(f"fmin = {fmin}, fmax = {fmax}")
    data_filtered = load_wav_data(
        fmin=fmin, fmax=fmax, filter_type=filter_type, filter_corners=filter_corners
    )
    sig_filtered = data_filtered["data"]

    # Plot filtered signal
    # plt.figure()
    # plt.plot(data["sig"].times(), sig_filtered)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.title("Filtered signal")
    # plt.grid()

    # # Derive stft
    # f, tt, stft = sp.stft(
    #     sig_filtered,
    #     fs=fs,
    #     window="hann",
    #     nperseg=2**12,
    #     noverlap=int(2**11 * 1 / 2),
    # )

    # # Plot stft
    # plt.figure()
    # plt.pcolormesh(tt, f, np.abs(stft))
    # plt.ylabel("Frequency [Hz]")
    # plt.xlabel("Time [s]")
    # plt.ylim([fmin - 1, fmax + 1])
    # plt.colorbar()

    # # Derive dsp
    ff, Pxx = sp.welch(
        sig_filtered,
        fs=fs,
        nperseg=2**14,
        noverlap=2**13,
    )

    # Plot dsp
    plt.figure()
    plt.plot(ff, (Pxx))
    plt.xlabel("$f$" + " [Hz]")
    plt.ylabel("$S_{xx}(f)$" + " [dB $(Pa^2Hz^{-1})$]")
    plt.grid()
    plt.xlim([fmin - 0.5, fmax + 0.5])
    fpath = os.path.join(img_root_filter, f"psd_filtered_ray_f{f_ray}Hz.png")
    plt.savefig(fpath)

    # # Plot fft of the unmodified signal and the filtered signal
    # fft_data = np.fft.rfft(data["data"])
    # fft_filtered = np.fft.rfft(sig_filtered)
    # freq = np.fft.rfftfreq(len(data["data"]), 1 / fs)
    # plt.figure()
    # plt.plot(freq, np.abs(fft_data))
    # plt.plot(freq, np.abs(fft_filtered))
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Amplitude")
    # plt.grid()

    # Derive auto-correlation
    rxx_ifft = np.fft.fftshift(np.fft.irfft(Pxx))
    lag = np.arange(-len(rxx_ifft) / 2, len(rxx_ifft) / 2) * 1 / fs

    plt.figure()
    plt.plot(lag, rxx_ifft)
    plt.xlabel(r"$\tau$" + " [s]")
    plt.ylabel(r"$R_{xx}( \tau )$")
    fpath = os.path.join(img_root_filter, f"rxx_filtered_ray_f{f_ray}Hz.png")
    plt.savefig(fpath)

    data_filtered["Pxx"] = Pxx
    data_filtered["ff"] = ff
    data_filtered["rxx"] = rxx_ifft
    data_filtered["lag"] = lag
    data_filtered["f_ray"] = f_ray
    data_filtered["fmin"] = fmin
    data_filtered["fmax"] = fmax

    # plt.show()
    plt.close("all")

    return data_filtered


def show_delta_fi_rxx():
    # Theoretical delta_fi rxx
    fs = 100
    f0 = 10

    tau_corr_fi_c = 0.5
    std_fi_c = 1e-2
    tau_corr_fi = tau_corr_fi_c * 1 / f0
    std_fi = std_fi_c * f0

    print(f"tau_corr_fi = {tau_corr_fi}, std_fi = {std_fi}")
    corr_lag = np.linspace(-tau_corr_fi * 5, tau_corr_fi * 5, int(1e3))
    rxx = std_fi**2 * np.exp(-1 / 2 * (corr_lag / tau_corr_fi) ** 2)

    lab = r"$\sigma_{fi}^2 e^{-\frac{1}{2} (\tau/\tau_{corrfi})^2 }$"

    title = (
        r"$\tau_{corrfi}$"
        + f" = {tau_corr_fi_c} T0, "
        + r"$\sigma_{fi}$"
        + f" {std_fi_c} f0"
    )
    plt.figure(figsize=(16, 8))
    plt.plot(
        corr_lag,
        rxx,
        label=lab,
    )
    plt.xlabel(r"$\tau$" + " [s]")
    plt.ylabel(r"$R_{\delta fi \delta fi}(\tau)$")
    plt.grid()
    plt.legend()
    plt.title(title)

    # Save
    fpath = os.path.join(IMG_ROOT, "delta_fi_rxx.png")
    plt.savefig(fpath)

    plt.close()

    # Theoretical delta_fi sxx
    # freq = np.fft.fftfreq(len(corr_lag), 1 / fs)
    freq = np.linspace(-1000, 1000, int(1e3))
    sxx = (
        std_fi**2
        * tau_corr_fi
        * np.sqrt(2 * np.pi)
        * np.exp(-2 * (np.pi * freq * tau_corr_fi) ** 2)
    )

    # Derive sxx from rxx
    rxx_Ts = corr_lag[1] - corr_lag[0]
    rxx_fs = 1 / rxx_Ts
    print(f"rxx_fs = {rxx_fs}")
    sxx_from_rxx = np.fft.fftshift(np.fft.fft(rxx))
    sxx_from_rxx_freq = np.fft.fftshift(np.fft.fftfreq(len(rxx), rxx_Ts))

    lab_sxx_th = r"$\sigma_{fi}^2 \tau_{corrfi} \sqrt{2 \pi} e^{-2 (\pi f \tau_{corrfi})^2} \times f_s$"
    lab_sxx_fft = r"$FFT \{ R_{\delta fi \delta fi}(\tau) \}$"
    plt.figure()
    plt.plot(freq, sxx * rxx_fs, label=lab_sxx_th)
    plt.plot(sxx_from_rxx_freq, np.abs(sxx_from_rxx), label=lab_sxx_fft)
    plt.xlim([-1000, 1000])
    plt.legend()
    plt.grid()
    plt.xlabel(r"$f$" + " [Hz]")
    plt.ylabel(r"$S_{\delta fi \delta fi}(f)$")
    plt.title(title)

    # Save
    fpath = os.path.join(IMG_ROOT, "delta_fi_sxx.png")
    plt.savefig(fpath)

    plt.close()

    # Store data into a dictionary
    data = {}
    data["rxx"] = rxx
    data["corr_lag"] = corr_lag
    # Convert into a pandas dataframe
    df = pd.DataFrame(data)
    # Save data
    fpath = os.path.join(IMG_ROOT, "delta_fi_rxx_th_data.dat")
    df.to_csv(fpath, index=False, sep=" ")

    f_to_save = (freq > -1000) & (freq < 1000)
    data = {}
    data["sxx"] = sxx[f_to_save] * rxx_fs
    data["freq"] = freq[f_to_save]
    # Convert into a pandas dataframe
    df = pd.DataFrame(data)
    # Save data
    fpath = os.path.join(IMG_ROOT, "delta_fi_sxx_th_data.dat")
    df.to_csv(fpath, index=False, sep=" ")

    f_to_save = (sxx_from_rxx_freq > -1000) & (sxx_from_rxx_freq < 1000)
    data = {}
    data["sxx_from_rxx"] = np.abs(sxx_from_rxx)[f_to_save]
    data["sxx_from_rxx_freq"] = sxx_from_rxx_freq[f_to_save]
    # Convert into a pandas dataframe
    df = pd.DataFrame(data)
    # Save data
    fpath = os.path.join(IMG_ROOT, "delta_fi_sxx_fft_data.dat")
    df.to_csv(fpath, index=False, sep=" ")

    # plt.show()


def fit_real_data():

    fs = 100

    data = filter_real_data()
    rxx = data["rxx"]
    Ttot = len(data["data"]) / fs

    # Generate signal with the properties of interest
    n = 1
    f0 = data["f_ray"]

    # Define the harmonic amplitude as Amp = |e(nw0)| = sqrt(Rss(0))
    df = data["ff"][1] - data["ff"][0]
    Amp = np.sqrt(rxx[len(rxx) // 2])
    Amp = np.sum(data["Pxx"]) * (df)
    # A_harmonics = [A]

    # tau_corr_fi_l = np.arange(1, 2, 1) * 1 / f0
    # std_fi_l = np.arange(1, 2, 1) * f0
    tau_corr_fi_coef = np.array([0.01, 0.05, 0.1, 0.5, 1])
    std_fi_coef = np.array([0.01, 0.05, 0.1, 0.5, 1])

    t = np.arange(0, Ttot, 1 / fs)
    f = np.arange(0, fs, 1 / Ttot)
    freq = np.fft.fftfreq(len(t), 1 / fs)
    Nt = len(t)

    max_pxx = np.max(data["Pxx"])

    plt.figure()
    plt.plot(data["ff"], data["Pxx"], label="Real data")
    # plt.plot(data["lag"], data["rxx"], label="Real data")

    for tau_corr_fi_c in tau_corr_fi_coef:
        tau_corr_fi = tau_corr_fi_c * 1 / f0
        for std_fi_c in std_fi_coef:
            std_fi = std_fi_c * f0

            ### Create the signal and derive associated PSD ###
            # random instant frequency perturbation delta_fi with Gaussian power spectrum
            # G = np.zeros_like(freq)
            # G = (
            #     np.sqrt(2 * np.pi)
            #     * tau_corr_fi
            #     * std_fi**2
            #     * np.exp(-2 * (np.pi * freq * tau_corr_fi) ** 2)
            # )

            # delta_fi = np.random.randn(len(t))
            # delta_fi = np.fft.ifft(
            #     np.fft.fft(delta_fi) * np.sqrt(G * fs)
            # )  # 19/09/2024 for clarity
            # delta_ph = np.cumsum(delta_fi) / fs  # random instant phase perturbation

            # # Derive ship signal at the single harmonic considered here
            # s = Amp * np.exp(1j * 2 * np.pi * 1 * (f0 * t + delta_ph))

            # # Derive psd
            # ff, Pxx = sp.welch(
            #     s,
            #     fs=fs,
            #     nperseg=2**14,
            #     noverlap=2**13,
            # )

            ### Derive the theoretical PSD ###
            # sigma_phi from equation 9
            n_psd = len(data["Pxx"])
            tau_corr = np.arange(0, n_psd) * 1 / fs
            part1 = (
                tau_corr
                / tau_corr_fi
                * np.sqrt(np.pi / 2)
                * erf(1 / np.sqrt(2) * tau_corr / tau_corr_fi)
            )
            part2 = -1 + np.exp(-1 / 2 * (tau_corr / tau_corr_fi) ** 2)
            sigma_phi = 2 * std_fi**2 * tau_corr_fi**2 * (part1 + part2)

            # Rxx from equation 6
            Rxx = Amp**2 * np.exp(
                1j * 2 * np.pi * f0 * n * tau_corr
                - 1 / 2 * n**2 * (2 * np.pi * f0) ** 2 * sigma_phi**2
            )

            # Derive Pxx
            #  / fs * 10
            Pxx = np.abs(np.fft.fftshift(np.fft.fft(Rxx)))
            freq_pxx = np.fft.fftshift(np.fft.fftfreq(len(Rxx), 1 / fs))
            plt.plot(
                freq_pxx,
                Pxx,
                label=f"tau_corr_fi = {tau_corr_fi_c} T0, std_fi = {std_fi_c} f0",
            )

            # # Plot rxx
            # plt.plot(
            #     tau_corr,
            #     np.real(Rxx),
            #     label=f"tau_corr_fi = {tau_corr_fi_c} T0, std_fi = {std_fi_c} f0",
            # )
            # plt.xlabel(r"$\tau$" + " [s]")
            # plt.ylabel(r"$R_{xx}(\tau)$")
            # plt.grid()
            # plt.legend()

            # Plot dsp
            # plt.plot(
            #     ff,
            #     Pxx,
            #     label=r"$\tau_{corrfi}$"
            #     + f" = {tau_corr_fi_c} T0, "
            #     + r"$\sigma_{fi}$"
            #     + f" {std_fi_c} f0",
            # )
            plt.xlabel("$f$" + " [Hz]")
            plt.ylabel("$S_{xx}(f)$")
            plt.grid()

            max_pxx = max(np.max(Pxx), max_pxx)

    plt.ylim([0, max_pxx * 1.15])
    plt.xlim([data["fmin"] - 0.25, data["fmax"] + 0.25])
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    fs = 100
    f0 = 1
    tau_corr_fi = 1 / 5
    std_fi = f0 * 1 / 100

    # plot_ship_spectrum(fs)
    # plot_instant_freq_perturbation_spectrum(fs, tau_corr_fi, std_fi)

    # samuel_test_cases()
    # stdfi_taucorrfi_influence()

    std_fi = 0.1 * f0
    # fi_build_process(fs, tau_corr_fi, std_fi)

    data = load_wav_data(save_wav=True)

    # dsp_args = {}

    # for nperseg in [2**10, 2**11, 2**12, 2**13, 2**14]:
    #     for overlap_coef in [1 / 2, 3 / 4, 9 / 10, 1 / 10]:
    #         for window in ["hann", "hamming", "blackman"]:
    #             dsp_args["nperseg"] = nperseg
    #             dsp_args["overlap_coef"] = overlap_coef
    #             dsp_args["window"] = window

    #             data = filter_real_data(dsp_args=dsp_args)

    # for nperseg in [2**13]:
    #     for overlap_coef in [1 / 4, 1 / 2, 3 / 4, 9 / 10, 1 / 10]:
    #         for window in [
    #             "hann",
    #             "hamming",
    #             "blackman",
    #             "bartlett",
    #             "boxcar",
    #             "triang",
    #             "cosine",
    #         ]:
    #             dsp_args["nperseg"] = nperseg
    #             dsp_args["overlap_coef"] = overlap_coef
    #             dsp_args["window"] = window

    #             data = filter_real_data(dsp_args=dsp_args)

    # fit_real_data()

    # show_delta_fi_rxx()
