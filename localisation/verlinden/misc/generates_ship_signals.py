#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   generates_ship_signals.py
@Time    :   2024/09/18 09:02:13
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
import xarray as xr
import pandas as pd
import scipy.signal as sp
import scipy.fft as fft
import matplotlib.pyplot as plt
from signals.signals import generate_ship_signal, ship_spectrum

# ======================================================================================================================
# Parameters
# ======================================================================================================================
fs = 100
dt = 10.24
sig_type = "ship"
f0_min = 2.5
f0_max = 7.5

# ======================================================================================================================
# Functions
# ======================================================================================================================


def generate_random_ship_signals(dt, fs, nsig, rbr=20):
    """
    Generate random ship signals.
    """

    f0_min = 4
    f0_max = 5
    std_fi_min = 1e-3
    std_fi_max = 1e-1
    tau_corr_fi_min = 1e-3
    tau_corr_fi_max = 0.5
    sig = {}
    meta_data = {}

    for i in range(nsig):
        f0 = np.random.uniform(f0_min, f0_max)
        std_fi_c = np.random.uniform(std_fi_min, std_fi_max)
        std_fi = std_fi_c * f0
        tau_corr_fi_c = np.random.uniform(tau_corr_fi_min, tau_corr_fi_max)
        tau_corr_fi = tau_corr_fi_c * 1 / f0

        event_sig_info = {
            "sig_type": sig_type,
            "f0": f0,
            "std_fi": std_fi,
            "tau_corr_fi": tau_corr_fi,
            "fs": fs,
        }

        src_sig, t_src_sig = generate_ship_signal(
            Ttot=dt,
            f0=event_sig_info["f0"],
            std_fi=event_sig_info["std_fi"],
            tau_corr_fi=event_sig_info["tau_corr_fi"],
            fs=event_sig_info["fs"],
        )

        src_sig *= np.hanning(len(src_sig))

        src_sig = src_sig / np.sqrt(np.var(src_sig))

        # Add random noise
        # Find peaks in the PSD
        f, psd = sp.welch(src_sig, fs, nperseg=2**10, noverlap=2**9)
        log_psd = 10 * np.log10(psd)
        f_peaks, _ = sp.find_peaks(log_psd, height=np.max(log_psd) - 10)
        # Define the PSD of the rays as the mean PSD of rays peaks
        psd_ray = np.median(log_psd[f_peaks])
        # Define the PSD of the noise according to the desired Rays to Background Ratio (RBR)
        psd_wn = psd_ray - rbr
        # Derive the noise standard deviation assuming white noise (ie psd = cst) : PSD = sigma^2 / fs (linear scale)
        wn_power_dB = 10 * np.log10(fs) + psd_wn
        sigma_noise = np.sqrt(10 ** (wn_power_dB / 10))
        # print(f"PSD rays : {psd_ray} dB")
        # print(f"PSD noise : {psd_wn} dB")
        # Compute noise vector
        src_sig += sigma_noise * np.random.normal(0, 1, len(src_sig))

        sig["t"] = t_src_sig
        sig[f"s{i}"] = src_sig
        meta_data[f"s{i}"] = [f0, std_fi_c, tau_corr_fi_c]

    df = pd.DataFrame(sig)
    df_meta = pd.DataFrame(meta_data)

    return df, df_meta


def derive_stft(df, fs, nperseg=2**8, noverlap=2**7):
    """
    Derive the Short Time Fourier Transform (STFT) of the signals.
    """

    stft_array = []
    for col in df.columns[1:]:
        sig = df[col].values
        f, t, stft = sp.stft(sig, fs, nperseg=nperseg, noverlap=noverlap)
        stft_array.append(stft)

    stft_array = np.array(stft_array)

    # Convert to xarray
    xr_stft = xr.Dataset(
        {
            "stft": (["sig", "ff", "tt"], stft_array),
            "stft_mod": (["sig", "ff", "tt"], np.abs(stft_array)),
            "stft_arg": (["sig", "ff", "tt"], np.angle(stft_array)),
        },
        coords={
            "ff": f,
            "tt": t,
            "sig": df.columns[1:],
        },
        attrs={
            "fs": fs,
            "nperseg": nperseg,
            "noverlap": noverlap,
        },
    )

    xr_stft.ff.attrs["units"] = "Hz"
    xr_stft.ff.attrs["long_name"] = "Frequency"
    xr_stft.tt.attrs["units"] = "s"
    xr_stft.tt.attrs["long_name"] = "Time"

    return xr_stft


def derive_psd(df, fs, nperseg=2**8, noverlap=2**7):
    """
    Derive the Power Spectral Density (PSD) of the signals.
    """

    psd_array = []
    for col in df.columns[1:]:
        sig = df[col].values
        f, psd = sp.welch(sig, fs, nperseg=nperseg, noverlap=noverlap)
        psd_array.append(psd)

    psd_array = np.array(psd_array)

    # Convert to xarray
    xr_psd = xr.Dataset(
        {
            "psd": (["sig", "ff"], psd_array),
        },
        coords={
            "ff": f,
            "sig": df.columns[1:],
        },
        attrs={
            "fs": fs,
            "nperseg": nperseg,
            "noverlap": noverlap,
        },
    )

    xr_psd.ff.attrs["units"] = "Hz"
    xr_psd.ff.attrs["long_name"] = "Frequency"

    return xr_psd


def derive_tf(df, fs, nfft=513):

    # Compute the Fourier Transform of the signals
    tf_array = []
    for col in df.columns[1:]:
        sig = df[col].values
        tf = fft.rfft(sig, n=nfft)
        tf_array.append(tf)

    tf_array = np.array(tf_array)
    f = fft.rfftfreq(nfft, 1 / fs)

    # Convert to xarray
    xr_tf = xr.Dataset(
        {
            "tf_mod": (["sig", "freq"], np.abs(tf_array)),
            "tf_arg": (["sig", "freq"], np.angle(tf_array)),
        },
        coords={
            "freq": f,
            "sig": df.columns[1:],
        },
        attrs={
            "fs": fs,
            "nfft": len(f),
            "df": f[1] - f[0],
        },
    )

    xr_tf.freq.attrs["units"] = "Hz"
    xr_tf.freq.attrs["long_name"] = "Frequency"

    # Save the data
    root = (
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ship_sig\ship_sig_database"
    )
    fpath = os.path.join(root, "ship_sig_database.nc")
    xr_tf.to_netcdf(fpath)

    return xr_tf


def plot_signal_bank(df_sig, xr_stft, xr_psd, xr_tf, df_meta):
    root_bank = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\ship_signal\library_signals"
    root_bank_sig = os.path.join(root_bank, "signals")
    root_bank_stft = os.path.join(root_bank, "stft")
    root_bank_psd = os.path.join(root_bank, "psd")
    root_bank_tf = os.path.join(root_bank, "tf")

    if not os.path.exists(root_bank_sig):
        os.makedirs(root_bank_sig)
    if not os.path.exists(root_bank_stft):
        os.makedirs(root_bank_stft)
    if not os.path.exists(root_bank_psd):
        os.makedirs(root_bank_psd)
    if not os.path.exists(root_bank_tf):
        os.makedirs(root_bank_tf)

    vmin = -50
    vmax = 0
    nsig = len(df_sig.columns) - 1
    for i in range(nsig):

        # Meta data
        f0, std_fi, tau_corr_fi = df_meta[f"s{i}"].values
        param_label = (
            f"$f_0={f0:.3f}$ Hz, $\sigma_f={std_fi:.3f} f_0$, " + r"$\tau_c = $"
            f"${tau_corr_fi:.3f} T_0$"
        )

        # Plot time signal
        t_mid = df_sig["t"].values[len(df_sig) // 2]
        plt.figure()
        plt.plot(df_sig["t"], df_sig[f"s{i}"])
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"Signal {i}\n" + param_label)
        plt.grid()
        plt.xlim(-1 / f0 * 3 + t_mid, 1 / f0 * 3 + t_mid)
        plt.ylim(-5, 5)

        # Save
        fpath = os.path.join(root_bank_sig, f"signal_{i}.png")
        plt.savefig(fpath)
        plt.close()

        # Plot STFT
        stft_log = 10 * np.log10(xr_stft.stft_mod.sel(sig=f"s{i}"))
        plt.figure()
        stft_log.plot(x="tt", y="ff", vmin=vmin, vmax=vmax)
        plt.title(f"STFT of signal {i}\n" + param_label)

        # Save
        fpath = os.path.join(root_bank_stft, f"stft_{i}.png")
        plt.savefig(fpath)
        plt.close()

        # Plot PSD
        psd_log = 10 * np.log10(xr_psd.psd.sel(sig=f"s{i}"))
        plt.figure()
        psd_log.plot(x="ff")
        plt.title(f"PSD of signal {i}\n" + param_label)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [dB]")
        plt.grid()
        plt.ylim(-65, 5)

        # Save
        fpath = os.path.join(root_bank_psd, f"psd_{i}.png")
        plt.savefig(fpath)
        plt.close()

        # Plot tf
        tf_log = 10 * np.log10(xr_tf.tf_mod.sel(sig=f"s{i}"))
        plt.figure()
        tf_log.plot(x="freq")
        plt.title(f"TF of signal {i}\n" + param_label)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [dB]")
        plt.grid()

        # Save
        fpath = os.path.join(root_bank_tf, f"tf_{i}.png")
        plt.savefig(fpath)


if __name__ == "__main__":

    nsig = 20
    dt = 1000
    fs = 100
    rbr = 1e3
    nperseg = 2**11
    noverlap = 2**10
    nfft = 1024
    df, df_meta = generate_random_ship_signals(dt, fs, nsig, rbr=rbr)
    xr_tf = derive_tf(df, fs, nfft)
    xr_stft = derive_stft(df, fs, nperseg=nperseg, noverlap=noverlap)
    xr_psd = derive_psd(df, fs, nperseg=nperseg, noverlap=noverlap)

    plot_signal_bank(df, xr_stft, xr_psd, xr_tf, df_meta)
    # f = xr_stft.ff.values
    # root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ship_sig"
    # fpath_ship_spectrum = os.path.join(root, "ship_spectrum.dat")
    # df_ship = pd.DataFrame(
    #     {"f": f, "Aship": np.abs(ship_spectrum(f)), "Pship": np.angle(ship_spectrum(f))}
    # )
    # df_ship.to_csv(fpath_ship_spectrum, index=False)

    # plt.figure()
    # plt.plot(f, np.abs(ship_spectrum(f)))
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Amplitude")
    # plt.title("Ship spectrum")
    # plt.show()

    # plt.figure()
    # for i in range(nsig - 1):
    #     sig_name = f"s{i}"
    #     plt.plot(df["t"], df[sig_name], label=f"sig_{i}")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # for i in range(nsig - 1):
    #     sig_name = f"s{i}"
    #     plt.figure()
    #     xr_stft.stft_mod.sel(sig=sig_name).plot(x="tt", y="ff")
    #     plt.title(f"STFT of signal {sig_name}")
    #     plt.show()

    # plt.figure()
    # for i in range(nsig - 1):
    #     sig_name = f"s{i}"
    #     psd_log = 10 * np.log10(xr_psd.psd.sel(sig=sig_name))
    #     psd_log.plot(x="ff")

    #     # Add the mean PSD
    #     median_psd = np.median(psd_log.values)
    #     # plt.axhline(
    #     #     median_psd,
    #     #     linestyle="--",
    #     #     label="Median PSD : {:.2f} dB".format(median_psd),
    #     # )

    #     rbr = np.max(psd_log.values) - median_psd
    #     print("RBR : {:.2f} dB".format(rbr))

    # plt.title(f"PSD of signal {sig_name}")
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("PSD [dB]")
    # plt.legend()

    # plt.show()

    # # noise DSP
    # nl = -20
    # sigma_noise = np.sqrt(10 ** (nl / 10))
    # print(sigma_noise)
    # # nl = 10 * np.log10(sigma_noise**2)
    # # print("NL : {:.2f} dB".format(nl))
    # noise = sigma_noise * np.random.normal(0, 1, 10**6)
    # print(10 * np.log10(np.var(noise)))

    # f, psd = sp.welch(noise, fs, nperseg=2**10, noverlap=2**9)
    # psd_log = 10 * np.log10(psd)
    # mean_psd = np.mean(psd_log)
    # plt.figure()
    # plt.plot(f, psd_log)
    # plt.axhline(mean_psd, linestyle="--", label="Mean PSD : {:.2f} dB".format(mean_psd))
    # plt.legend()
    # plt.show()
