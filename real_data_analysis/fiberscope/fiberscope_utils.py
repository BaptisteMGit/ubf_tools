#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   fiberscope.py
@Time    :   2024/11/12 17:27:58
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
import scipy.signal as sp
import matplotlib.pyplot as plt

from publication.PublicationFigure import PubFigure
from propa.rtf.rtf_utils import D_hermitian_angle_fast
from real_data_analysis.fiberscope.read_tdms import load_fiberscope_data
from real_data_analysis.deconvolution_utils import crosscorr_deconvolution
from propa.rtf.rtf_estimation.rtf_estimation_utils import (
    rtf_cs,
    rtf_cw,
    get_stft_list,
    get_csdm_from_signal,
)

# Set figure properties
PubFigure()

data_root = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\Fiberscope_campagne_oct_2024"
)
img_root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_estimation\fiberscope"
# date = "09-10-2024"
# data_path = os.path.join(data_root, f"Campagne_{date}")

processed_data_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\data_processed"


# ======================================================================================================================
# Functions
# ======================================================================================================================
def process_recording(recording_name, recording_props, processing_props):

    # Load data
    # date = "09-10-2024"
    date = recording_name.split("T")[0]
    data_path = os.path.join(data_root, f"Campagne_{date}")
    file_name = f"{recording_name}.tdms"
    file_path = os.path.join(data_path, file_name)

    img_path = os.path.join(img_root, recording_name)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    data = load_fiberscope_data(file_path)

    # Process data
    process_single_rec(data, recording_name, recording_props, processing_props)


def derive_rtf(recording_name, recording_props, processing_props, verbose=False):

    if verbose:
        print(f"Processing recording {recording_name} - RTF estimation")

    # Load data if already processed
    fpath = os.path.join(processed_data_path, f"{recording_name}.nc")
    if os.path.exists(fpath):
        data = xr.open_dataset(fpath)
    else:
        process_recording(recording_name, recording_props, processing_props)
        data = xr.open_dataset(fpath)

    # Unpack usefull properties
    # By default the reference hydrophone is the fifth one located in the water column
    ref_hydro = processing_props.get("ref_hydro", 5)  # Reference hydrophone

    # Derive rtf from recordings
    data = derive_rtf_from_recordings(data, recording_props, processing_props)

    # Derive rtf from tf estimated by deconvolution
    data = derive_rtf_from_tf(data, ref_hydro)

    # Derive GCC for comparison
    data = derive_gcc(data, recording_props, processing_props)

    # Save results
    data.to_netcdf(os.path.join(processed_data_path, f"{recording_name}_rtf.nc"))
    data.close()


def derive_gcc(data, recording_props, processing_props):

    # Unpack usefull properties
    gcc_method = processing_props.get("gcc_method", "scot")  # Method to derive the GCC
    ts = data.ts

    # Inputs : array of shape (nt, n_rcv)
    y = data.signal.T.values
    yref = data.signal.isel(
        h_index=0
    ).T.values  # Arrays have been rolled to have the reference hydrophone at the first position previously

    # Cast to the same shape as y to derive cross power density spectrums
    yref = np.expand_dims(yref, axis=1)
    tile_shape = tuple([y.shape[i] - yref.shape[i] + 1 for i in range(y.ndim)])
    yref = np.tile(yref, tile_shape)

    # Derive cross power density spectrums
    nperseg = 2**12
    noverlap = nperseg // 2
    ff, Rxy = sp.csd(yref, y, fs=1 / ts, nperseg=nperseg, noverlap=noverlap, axis=0)
    _, Rxx = sp.csd(yref, yref, fs=1 / ts, nperseg=nperseg, noverlap=noverlap, axis=0)
    _, Ryy = sp.csd(y, y, fs=1 / ts, nperseg=nperseg, noverlap=noverlap, axis=0)

    # Apply gcc weights

    if gcc_method == "scot":
        # Compute the GCC-SCOT
        w = 1 / np.abs(np.sqrt(Rxx * Ryy))
    elif gcc_method == "phat":
        # Compute the GCC-PHAT
        w = 1 / np.abs(Rxy)
    elif gcc_method == "ml":
        # Compute the GCC-ML
        gamma_xy = sp.coherence(
            yref, y, fs=1 / ts, nperseg=nperseg, noverlap=noverlap, axis=0
        )  # magnitude squared coherence estimate
        w = 1 / Rxy * gamma_xy / (1 - gamma_xy)

    gcc_f = w * Rxy

    # Add results to the dataset
    data["f_gcc"] = ff
    data["gcc_amp"] = (
        ["h_index", "f_gcc"],
        np.abs(gcc_f).T,
    )
    data["gcc_phase"] = (
        ["h_index", "f_gcc"],
        np.angle(gcc_f).T,
    )

    return data


def split_signal_noise(data, recording_props, processing_props):

    # Unpack usefull properties
    alpha_th = processing_props.get(
        "alpha_th", 0.01 * 1e-2
    )  # Threshold for signal / noise split
    split_method = processing_props.get(
        "split_method", "band_energy"
    )  # Method to split signal and noise "band_energy" or "rolling_power"
    t_interp_pulse = recording_props.get("t_interp_pulse", 1)
    f0 = recording_props.get("f0", 8e3)
    f1 = recording_props.get("f1", 15e3)
    n_em = recording_props.get("n_em", 10)  # Number of signal emissions
    n_hydro = data.sizes["h_index"]
    ts = data.ts

    # Derive the split time for the first emission received by the hydrophone n°4 (the one which is likely to received the highest number of echoes)
    # The goal of choosing a common split time is that the simultaneity of received signals is preserved (essential to derive csdm)
    hydro_for_split_time_calibration = 4
    y = data.signal.sel(time=slice(0, t_interp_pulse)).sel(
        h_index=hydro_for_split_time_calibration
    )

    if split_method == "rolling_power":
        rolling_power = y.rolling(time=1000, center=True).var().dropna("time")
        power_threshold = rolling_power.max() * alpha_th
        # Detect last instant where the power is above the threshold
        time_with_power_above_threshold = rolling_power.time.values[
            rolling_power.values > power_threshold.values
        ]
        split_time_0 = time_with_power_above_threshold[0]
        split_time_1 = time_with_power_above_threshold[-1]

    elif split_method == "band_energy":
        # Other approach : use the energy level in the frequency band of interest to split signal and noise
        # 1) Derive the stft of the original signal (signal of interest + noise)
        nperseg = 2**12
        noverlap = nperseg // 2
        ff, tt, stft = sp.stft(
            y.values,
            fs=1 / ts,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",
        )
        # 2) Compute the energy in the frequency band of interest
        f0_idx = np.argmin(np.abs(ff - f0))
        f1_idx = np.argmin(np.abs(ff - f1))
        energy_band = np.sum(np.abs(stft[f0_idx:f1_idx, :]) ** 2, axis=0) * (
            ff[1] - ff[0]
        )  # Integrate over the frequency band -> energy in V^2

        # 3) Split signal and noise based on the energy level in the frequency band of interest
        # energy_threshold = energy_band.max() * alpha_th
        energy_threshold = np.max(energy_band[tt > tt.max() * 2 / 3]) * 4
        # print(f"energy_threshold = {energy_threshold}")
        time_with_energy_above_threshold = tt[energy_band > energy_threshold]
        # split_time = time_with_energy_above_threshold[-1]
        split_time_0 = time_with_energy_above_threshold[0]
        split_time_1 = time_with_energy_above_threshold[-1]

    signal_plus_noise = []
    only_noise = []
    # Loop over each hydrophone to extract signal and noise
    for i_hydro in range(n_hydro):
        sig_array = []
        noise_array = []
        # Process each emission
        for i_em in range(n_em):
            # Extract the emission
            y = data.signal.sel(
                time=slice(i_em * t_interp_pulse - ts, (i_em + 1) * t_interp_pulse)
            ).isel(h_index=i_hydro)

            # # Update split time for the current emission
            # split_time_i_em = split_time + y.time.min().values
            # # Split signal and noise
            # signal = y.sel(time=slice(0, split_time_i_em))
            # noise = y.sel(time=slice(split_time_i_em, y.time.max()))

            # Update split time for the current emission
            split_time_i_em_0 = split_time_0 + y.time.min().values
            split_time_i_em_1 = split_time_1 + y.time.min().values

            signal = y.sel(time=slice(split_time_i_em_0, split_time_i_em_1))
            noise = xr.concat(
                [
                    y.sel(time=slice(0, split_time_i_em_0)),
                    y.sel(time=slice(split_time_i_em_1, y.time.max())),
                ],
                dim="time",
            )

            # Apply window at the very edge to avoid high frequency effects when combining with other emissions
            alpha_tukey = 0.01
            signal = signal * sp.windows.tukey(len(signal), alpha_tukey)
            noise = noise * sp.windows.tukey(len(noise), alpha_tukey)

            # Add signal and noise to arrays
            sig_array.append(signal.values)
            noise_array.append(noise.values)

        # Combine all emissions into a single signal and noise array
        sig_array = np.concatenate(sig_array)
        noise_array = np.concatenate(noise_array)

        # Store the signal and noise in the dedicated arrays
        signal_plus_noise.append(sig_array)
        only_noise.append(noise_array)

    # Pad with zeros to ensure all arrays have the same length
    max_len_sig = max([sig.size for sig in signal_plus_noise])
    max_len_noise = max([noise.size for noise in only_noise])
    for i_hydro in range(n_hydro):
        signal_plus_noise[i_hydro] = np.pad(
            signal_plus_noise[i_hydro],
            (0, max_len_sig - signal_plus_noise[i_hydro].size),
        )
        only_noise[i_hydro] = np.pad(
            only_noise[i_hydro], (0, max_len_noise - only_noise[i_hydro].size)
        )

    # Convert to arrays
    signal_plus_noise = np.array(signal_plus_noise)
    only_noise = np.array(only_noise)

    # Create new time vectors
    signal_plus_noise_time = np.linspace(
        0, signal_plus_noise.shape[1] * ts, signal_plus_noise.shape[1]
    )
    only_noise_time = np.linspace(0, only_noise.shape[1] * ts, only_noise.shape[1])

    # Add coordinate to the dataset
    data["signal_plus_noise_time"] = signal_plus_noise_time
    data["only_noise_time"] = only_noise_time

    # Add signal and noise to the dataset
    data["signal_plus_noise"] = (
        ["h_index", "signal_plus_noise_time"],
        signal_plus_noise,
    )
    data["only_noise"] = (
        ["h_index", "only_noise_time"],
        only_noise,
    )

    # Derive SNR in the frequency band of interest
    noise_fft = np.fft.rfft(only_noise, axis=1)
    signal_fft = np.fft.rfft(signal_plus_noise, axis=1)
    f_noise = np.fft.rfftfreq(only_noise.shape[1], d=ts)
    f_signal = np.fft.rfftfreq(signal_plus_noise.shape[1], d=ts)
    f_in_band_noise = np.logical_and(f_noise >= f0, f_noise <= f1)
    f_in_band_sig = np.logical_and(f_signal >= f0, f_signal <= f1)
    snr = np.sum(np.abs(signal_fft[:, f_in_band_sig]) ** 2, axis=1) / np.sum(
        np.abs(noise_fft[:, f_in_band_noise]) ** 2, axis=1
    )
    snr = 10 * np.log10(snr)

    # Add snr to the dataset
    data["snr"] = (
        ["h_index"],
        snr,
    )

    plt.figure()
    data.signal.plot(x="time", hue="h_index")
    plt.title("Original signal")
    plt.savefig(os.path.join(data.attrs["img_path"], "original_signal.png"))

    plt.figure()
    data.signal_plus_noise.plot(x="signal_plus_noise_time", hue="h_index")
    plt.title("Signal")
    plt.savefig(os.path.join(data.attrs["img_path"], "signal_plus_noise.png"))

    plt.figure()
    data.only_noise.plot(x="only_noise_time", hue="h_index")
    plt.title("Noise")
    plt.savefig(os.path.join(data.attrs["img_path"], "only_noise.png"))

    return data


def derive_rtf_from_recordings(data, recording_props, processing_props):

    # Unpack usefull properties
    ref_hydro = processing_props.get("ref_hydro", 5)  # Reference hydrophone
    method = processing_props.get("method", "cs")  # Method to derive the RTF
    n_hydro = data.sizes["h_index"]
    ts = data.ts

    # Check if the signal and noise have already been split
    if "signal_plus_noise" not in data:
        # illustrate_signal_noise_split_process(data, recording_props, processing_props)
        data = split_signal_noise(data, recording_props, processing_props)

    # By default rtf estimation method assumed the first receiver as the reference -> need to roll along the receiver axis
    idx_pos_ref_hydro = np.argmin(np.abs(data.h_index.values - ref_hydro))
    npos_to_roll = data.sizes["h_index"] - idx_pos_ref_hydro
    data = data.roll(
        h_index=npos_to_roll,
        roll_coords=True,
    )

    # Input to cs and cw functions must be array of shape (nt, n_rcv)
    x = data.signal_plus_noise.T.values
    tx = data.signal_plus_noise_time.values
    v = data.only_noise.T.values
    tv = data.only_noise_time.values
    # Covariance substraction
    tau_ir = 0.5
    nperseg = int(tau_ir / ts)
    # print(f"nperseg : {nperseg}")
    # nperseg = 2**12
    noverlap = nperseg // 2
    ff, Rx = get_csdm_from_signal(tx, x, nperseg, noverlap)
    ff, Rv = get_csdm_from_signal(tv, v, nperseg, noverlap)

    # Add Rx and R_v to the dataset
    data["f_csdm"] = ff
    data["Rx"] = (
        ["f_csdm", "h_index", "h_index"],
        np.abs(Rx),
    )
    data["Rv"] = (
        ["f_csdm", "h_index", "h_index"],
        np.abs(Rv),
    )

    if method in ["cs", "both"]:
        f, rtf = rtf_cs(ff, n_hydro, Rx, Rv)
        # Add frequency and estimated rtf to the dataset
        data["f_rtf_cs"] = f
        data["rtf_amp_cs"] = (
            ["h_index", "f_rtf_cs"],
            np.abs(rtf).T,
        )
        data["rtf_phase_cs"] = (
            ["h_index", "f_rtf_cs"],
            np.angle(rtf).T,
        )
    if method in ["cw", "both"]:
        # nperseg /= 2
        # noverlap /= 2
        x = data.signal.T.values
        ff, _, stft_list_x = get_stft_list(x, 1 / ts, nperseg, noverlap)
        stft_x = np.array(stft_list_x)

        f, rtf = rtf_cw(ff, n_hydro, stft_x, Rv)
        # Add frequency and estimated rtf to the dataset
        data["f_rtf_cw"] = f
        data["rtf_amp_cw"] = (
            ["h_index", "f_rtf_cw"],
            np.abs(rtf).T,
        )
        data["rtf_phase_cw"] = (
            ["h_index", "f_rtf_cw"],
            np.angle(rtf).T,
        )

    return data


def derive_rtf_from_tf(data, ref_hydro):

    # Unpack usefull properties
    tf_ref = data.tf_hat_amp.sel(h_index=ref_hydro) * np.exp(
        1j * data.tf_hat_phase.sel(h_index=ref_hydro)
    )

    rtf = np.zeros((data.sizes["h_index"], data.sizes["f_ir"]), dtype=complex)
    for i_hydro in range(data.sizes["h_index"]):
        tf = data.tf_hat_amp.isel(h_index=i_hydro) * np.exp(
            1j * data.tf_hat_phase.isel(h_index=i_hydro)
        )
        rtf[i_hydro, :] = tf / tf_ref

    data["rtf_amp"] = (
        ["h_index", "f_ir"],
        np.abs(rtf),
    )
    data["rtf_phase"] = (
        ["h_index", "f_ir"],
        np.angle(rtf),
    )

    return data


def illustrate_signal_noise_split_process(data, recording_props, processing_props):

    ts = data.ts
    t_interp_pulse = recording_props.get("t_interp_pulse", 1)

    # Extract the emission
    # y = data.signal.sel(
    #     time=slice(i_em * t_interp_pulse - ts, (i_em + 1) * t_interp_pulse)
    # ).isel(h_index=i_hydro)
    hydro_for_split_time_calibration = 4
    y = data.signal.sel(time=slice(0, t_interp_pulse)).sel(
        h_index=hydro_for_split_time_calibration
    )

    split_method = processing_props.get(
        "split_method", "band_energy"
    )  # Method to split signal and noise "band_energy" or "rolling_power"

    if split_method == "rolling_power":
        rolling_power = y.rolling(time=1000, center=True).var().dropna("time")

        alpha_th = 0.01 * 1e-2  # 0.01% of the maximum power
        power_threshold = rolling_power.max() * alpha_th
        # Detect last instant where the power is above the threshold
        time_with_power_above_threshold = rolling_power.time.values[
            rolling_power.values > power_threshold.values
        ]
        split_time = time_with_power_above_threshold[-1]

        # Plot signal and rolling power on two subfigures to illustrate the signal / noise split process
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        y.plot(ax=axs[0], color="b")
        axs[0].set_ylabel(r"$p \, \textrm{[V]}$")
        axs[0].set_title("Signal")
        axs[0].set_xlabel("")
        rolling_power.plot(ax=axs[1], color="r")
        axs[1].set_ylabel(r"$\sigma^2 \, \textrm{[V}^2 \textrm{]}$")
        axs[1].set_title("Signal rolling variance")
        axs[1].set_xlabel("")
        fig.supxlabel(r"$t \, \textrm{[s]}$")
        # Add vertical line to illustrate the last instant where the power is above the threshold
        annotation = (
            f"$t_c = {np.round(split_time, 2)}"
            + r"\, \textrm{s}$ \,"
            + r"$(\sigma^2 >"
            + f"{alpha_th * 1e2} \% \,"
            + r"\sigma^2_{\textrm{max}})$"
        )
        axs[0].axvline(split_time, color="k", linestyle="--")
        axs[1].axvline(split_time, color="k", linestyle="--")
        # Annotate the vertical line
        axs[0].annotate(
            annotation,
            xy=(split_time, 0),
            xytext=(split_time * 1.05, y.max() * 0.8),
            # arrowprops=dict(facecolor="black", arrowstyle="->"),
        )
        axs[1].annotate(
            annotation,
            xy=(split_time, 0),
            xytext=(split_time * 1.05, rolling_power.max() * 0.8),
            # arrowprops=dict(facecolor="black", arrowstyle="->"),
        )

        fpath = os.path.join(
            img_root, "method_illustration", "signal_noise_split_process_var_sig.png"
        )
        plt.savefig(fpath)

    if split_method == "band_energy":
        # Other approach : use the energy level in the frequency band of interest to split signal and noise
        alpha_th = 0.01 * 1e-2  # 0.001% of the maximum energy
        # 1) Derive the stft of the original signal (signal of interest + noise)
        nperseg = 2**12
        noverlap = nperseg // 2
        ff, tt, stft = sp.stft(
            y.values,
            fs=1 / ts,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",
        )
        # 2) Compute the energy in the frequency band of interest
        f0 = recording_props.get("f0", 8e3)
        f1 = recording_props.get("f1", 15e3)
        f0_idx = np.argmin(np.abs(ff - f0))
        f1_idx = np.argmin(np.abs(ff - f1))
        energy_band = np.sum(np.abs(stft[f0_idx:f1_idx, :]) ** 2, axis=0) * (
            ff[1] - ff[0]
        )  # Integrate over the frequency band -> energy in V^2

        # 3) Split signal and noise based on the energy level in the frequency band of interest
        # energy_threshold = energy_band.max() * alpha_th
        energy_threshold = np.max(energy_band[tt > tt.max() * 2 / 3]) * 5
        time_with_energy_above_threshold = tt[energy_band > energy_threshold]
        split_time_0 = time_with_energy_above_threshold[0]
        split_time_1 = time_with_energy_above_threshold[-1]

        # 4) Plot stft on the first subfigure and energy in the frequency band of interest on the second subfigure
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axs[0].pcolormesh(tt, ff, 10 * np.log10(np.abs(stft)), shading="gouraud")
        axs[0].set_ylabel(r"$f \, \textrm{[Hz]}$")
        axs[0].set_ylim([0, f1 * 1.5])

        axs[1].plot(tt, energy_band)
        axs[1].set_ylabel(r"$\mathcal{E} \, \textrm{[V}^2 \textrm{]}$")
        axs[1].set_xlabel(r"$t \, \textrm{[s]}$")

        # Annotate with the threshold
        annotation_1 = (
            r"$t_{c1}"
            + f" = {np.round(split_time_1, 2)}"
            + r"\, \textrm{s}$ \,"
            # + r"$(\mathcal{E} >"
            # + f"{alpha_th * 1e2} \% \,"
            # + r"\mathcal{E}_{\textrm{max}})$"
        )
        annotation_0 = (
            r"$t_{c0}"
            + f" = {np.round(split_time_0, 2)}"
            + r"\, \textrm{s}$ \,"
            # + r"$(\mathcal{E} >"
            # + f"{alpha_th * 1e2} \% \,"
            # + r"\mathcal{E}_{\textrm{max}})$"
        )
        axs[0].axvline(split_time_0, color="k", linestyle="--")
        axs[0].axvline(split_time_1, color="k", linestyle="--")
        axs[1].axvline(split_time_0, color="k", linestyle="--")
        axs[1].axvline(split_time_1, color="k", linestyle="--")
        axs[1].axhline(energy_threshold, color="r", linestyle="--")
        axs[0].annotate(
            annotation_0,
            xy=(split_time_0, 0),
            xytext=(split_time_0 * 1.05, f1 * 1.5 * 0.8),
        )
        axs[0].annotate(
            annotation_1,
            xy=(split_time_1, 0),
            xytext=(split_time_1 * 1.05, f1 * 1.5 * 0.8),
        )

        axs[1].annotate(
            annotation_0,
            xy=(split_time_0, 0),
            xytext=(split_time_0 * 1.05, energy_band.max() * 0.8),
        )
        axs[1].annotate(
            annotation_1,
            xy=(split_time_1, 0),
            xytext=(split_time_1 * 1.05, energy_band.max() * 0.8),
        )

        plt.savefig(
            os.path.join(
                img_root,
                "method_illustration",
                "signal_noise_split_process_energy_band.png",
            )
        )

    # Split signal and noise
    signal = y.sel(time=slice(split_time_0, split_time_1))
    # noise = y.sel(time=slice(split_time_1, y.time.max()))
    noise = xr.concat(
        [
            y.sel(time=slice(0, split_time_0)),
            y.sel(time=slice(split_time_1, y.time.max())),
        ],
        dim="time",
    )
    plt.figure()
    signal.plot(color="b")
    noise.plot(color="r")
    plt.xlabel(r"$t \, \textrm{[s]}$")
    plt.ylabel(r"$p \, \textrm{[V]}$")
    plt.title("")
    plt.legend([r"$s(t)$", r"$n(t)$"])
    plt.savefig(
        os.path.join(
            img_root,
            "method_illustration",
            "signal_noise_split_process_signal_noise.png",
        )
    )

    # Study noise to understand the noise electrical component
    nperseg = 2**12
    noverlap = nperseg // 2
    ff, tt, noise_stft = sp.stft(
        noise.values,
        fs=1 / ts,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    plt.figure()
    plt.pcolormesh(tt, ff, 10 * np.log10(np.abs(noise_stft)), shading="gouraud")
    plt.xlabel(r"$t \, \textrm{[s]}$")
    plt.ylabel(r"$f \, \textrm{[Hz]}$")
    plt.title("Noise STFT amplitude")
    plt.colorbar()
    plt.ylim([0, 20e3])
    plt.savefig(
        os.path.join(
            img_root,
            "method_illustration",
            "signal_noise_split_process_noise_stft.png",
        )
    )

    # Plot signal stft for comparison
    ff, tt, signal_stft = sp.stft(
        signal.values,
        fs=1 / ts,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    plt.figure()
    plt.pcolormesh(tt, ff, 10 * np.log10(np.abs(signal_stft)), shading="gouraud")
    plt.xlabel(r"$t \, \textrm{[s]}$")
    plt.ylabel(r"$f \, \textrm{[Hz]}$")
    plt.title("Signal STFT amplitude")
    plt.colorbar()
    plt.ylim([0, 20e3])
    plt.savefig(
        os.path.join(
            img_root,
            "method_illustration",
            "signal_noise_split_process_signal_stft.png",
        )
    )
    # signal = y.sel(time=slice(0, split_time))
    # noise = y.sel(time=slice(split_time, y.time.max()))

    # plt.figure()
    # signal.plot(color="b")
    # noise.plot(color="r")
    # plt.xlabel(r"$t \, \textrm{[s]}$")
    # plt.ylabel(r"$p \, \textrm{[V]}$")
    # plt.title("")
    # plt.legend([r"$s(t)$", r"$n(t)$"])
    # plt.savefig(
    #     os.path.join(
    #         img_root,
    #         "method_illustration",
    #         "signal_noise_split_process_signal_noise_energy_band.png",
    #     )
    # )


def analyse_rtf_estimation_results(recording_name, processing_props, verbose=False):

    if verbose:
        print(f"Analyzing recording {recording_name} - Plotting RTF estimation results")

    data_rtf = xr.open_dataset(
        os.path.join(processed_data_path, f"{recording_name}_rtf.nc")
    )

    method = processing_props.get("method", "cs")  # Method to derive the RTF

    for i_hydro in data_rtf.h_index.values:
        rtf_hydro = data_rtf.sel(h_index=i_hydro)
        plt.figure()
        rtf_hydro.rtf_amp.plot(
            x="f_ir",
            color="k",
            label=r"$\Pi_{"
            + str(i_hydro)
            + r"} \textrm{(deconvolution)}$"
            + f" ({recording_name})",
        )

        if method in ["cs", "both"]:
            rtf_hydro.rtf_amp_cs.plot(
                x="f_rtf_cs",
                label=r"$\Pi_{" + str(i_hydro) + r"}^{(CS)}$" + f" ({recording_name})",
                linestyle="-",
                color="b",
                marker="o",
                linewidth=0.2,
                markersize=2,
            )

        if method in ["cw", "both"]:
            rtf_hydro.rtf_amp_cw.plot(
                x="f_rtf_cw",
                label=r"$\Pi_{" + str(i_hydro) + r"}^{(CW)}$" + f" ({recording_name})",
                linestyle="-",
                color="r",
                marker="o",
                linewidth=0.2,
                markersize=2,
            )

        plt.title("")
        plt.xlabel(r"$f \, \textrm{[Hz]}$")
        plt.ylabel(r"$\left| \Pi \right|$")
        plt.xlim(10 * 1e3, 13 * 1e3)
        # plt.ylim([1e-1, 1e1])
        plt.yscale("log")
        plt.legend()
        plt.savefig(
            os.path.join(data_rtf.attrs["img_path"], f"rtf_amp_hydro{i_hydro}.png")
        )
        plt.close("all")

        plt.figure()
        rtf_phase = rtf_hydro.rtf_phase
        # rtf_phase = rtf_hydro.rtf_phase.sel(
        #     f_ir=rtf_hydro.f_rtf_cs.values, method="nearest"
        # )
        # rtf_phase.values = np.unwrap(rtf_phase.values)
        rtf_phase.plot(
            x="f_ir",
            color="k",
            label=r"$\Pi_{"
            + str(i_hydro)
            + r"} \textrm{(deconvolution)}$"
            + f" ({recording_name})",
            # marker="o",
            # linewidth=0.2,
            # markersize=2,
        )

        if method in ["cs", "both"]:
            # rtf_hydro.rtf_phase_cs.values = np.unwrap(rtf_hydro.rtf_phase_cs.values)
            rtf_hydro.rtf_phase_cs.plot(
                x="f_rtf_cs",
                label=r"$\Pi_{" + str(i_hydro) + r"}^{(CS)}$" + f" ({recording_name})",
                linestyle="-",
                color="b",
                marker="o",
                linewidth=0.2,
                markersize=2,
            )

        if method in ["cw", "both"]:
            # rtf_hydro.rtf_phase_cw.values = np.unwrap(rtf_hydro.rtf_phase_cw.values)
            rtf_hydro.rtf_phase_cw.plot(
                x="f_rtf_cw",
                label=r"$\Pi_{" + str(i_hydro) + r"}^{(CW)}$" + f" ({recording_name})",
                linestyle="-",
                color="r",
                marker="o",
                linewidth=0.2,
                markersize=2,
            )

        plt.title("")
        plt.xlabel(r"$f \, \textrm{[Hz]}$")
        plt.ylabel(r"$\phi \, \textrm{[rad]}$")
        # plt.xlim(fmin, fmax)
        plt.xlim(10 * 1e3, 13 * 1e3)
        plt.legend()
        plt.savefig(
            os.path.join(data_rtf.attrs["img_path"], f"rtf_phase_hydro{i_hydro}.png")
        )
        plt.close("all")

    # Plot csdm
    rx_vmax = data_rtf.Rx.mean(axis=0).values.max()
    rv_vmax = data_rtf.Rv.mean(axis=0).values.max()
    vmax = max(rx_vmax, rv_vmax)
    plt.figure()
    im = plt.pcolormesh(
        data_rtf.h_index.values,
        data_rtf.h_index.values,
        data_rtf.Rv.mean(axis=0).values,
        vmax=vmax,
    )
    plt.colorbar(im)
    plt.xlabel(r"$\textrm{Receiver id}$")
    plt.ylabel(r"$\textrm{Receiver id}$")
    plt.title(r"$\mathbf{\hat{R}_v}$")
    plt.savefig(os.path.join(data_rtf.attrs["img_path"], "noise_csdm.png"))

    plt.figure()
    im = plt.pcolormesh(
        data_rtf.h_index.values,
        data_rtf.h_index.values,
        data_rtf.Rx.mean(axis=0).values,
        vmax=vmax,
    )
    plt.colorbar(im)
    plt.xlabel(r"$\textrm{Receiver id}$")
    plt.ylabel(r"$\textrm{Receiver id}$")
    plt.title(r"$\mathbf{\hat{R}_x}$")
    plt.savefig(os.path.join(data_rtf.attrs["img_path"], "signal_csdm.png"))


def run_analysis(
    recording_name,
    recording_props,
    processing_props,
    plot_rtf_estimation=False,
    verbose=False,
):
    derive_rtf(recording_name, recording_props, processing_props, verbose=verbose)
    if plot_rtf_estimation:
        analyse_rtf_estimation_results(
            recording_name, processing_props, verbose=verbose
        )


def split_dynamic_recording(data, recording_props, processing_props):
    # Unpack usefull props
    src_speed = recording_props.get("src_speed", 0.1)  # Source speed in m/s
    src_start_pos = recording_props.get(
        "src_start_pos", "P1"
    )  # Source start position id
    dynamic_recording_name = recording_props.get("dynamic_recording_name", "")
    src_end_pos = recording_props.get("src_end_pos", "P4")  # Source end position id

    time_step = processing_props.get(
        "time_step", 10
    )  # Time step to use to devide the recording into

    recording_names = []
    start_of_current_period = 0
    end_of_current_period = time_step
    while end_of_current_period < data.time.max().values:
        # Extract the data corresponding to the current period
        data_period = data.sel(
            time=slice(start_of_current_period, end_of_current_period)
        )

        # Update time vector to start at 0
        data_period["time"] = data_period.time - start_of_current_period

        displacement_from_start_pos = (
            end_of_current_period - time_step / 2
        ) * src_speed  # Position of the source at the center of the period

        start_of_current_period += time_step
        end_of_current_period += time_step

        # Set recording name corresponding to the current period position
        recording_name = f"{dynamic_recording_name}_{src_start_pos}_r{np.round(displacement_from_start_pos, 2)}m_{src_end_pos}"
        process_single_rec(
            data_period, recording_name, recording_props, processing_props
        )
        recording_names.append(recording_name)


def process_single_rec(data, recording_name, recording_props, processing_props):
    # Unpack recording properties
    t_interp_pulse = recording_props.get(
        "t_interp_pulse", 1
    )  # Period between two signal emissions
    t_ir = recording_props.get("t_ir", 0.5)  # Approximated impulse response duration
    n_em = recording_props.get("n_em", 10)  # Number of signal emissions
    f0 = recording_props.get("f0", 2e3)  # Start frequency of the signal
    f1 = recording_props.get("f1", 20e3)  # End frequency of the signal

    # Unpack processing properties
    hydro_to_process = processing_props.get(
        "hydro_to_process", None
    )  # Hydrophone to process

    # Add attrs
    img_path = os.path.join(img_root, recording_name)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    data.attrs["recording_name"] = recording_name
    data.attrs["img_path"] = img_path

    # Restrict to desired hydrophone if specified
    if hydro_to_process is not None:
        data = data.sel(
            h_index=slice(
                hydro_to_process,
            )
        )  # Select hydrophone while keeping the dimension

    n_hydro = data.sizes["h_index"]

    # Create the source pulse signal
    ts = data.ts
    t = data.signal.sel(time=slice(0, t_ir)).time.values
    x = sp.chirp(t, f0=f0, f1=f1, t1=t.max(), method="linear")

    # sweep_hat = np.zeros((n_em, len(t)))
    ri_hat = np.zeros((n_hydro, n_em, len(t)))

    # Loop over each hydrophone to process
    for i_hydro in range(n_hydro):
        # Process each emission
        for i_em in range(n_em):
            # Extract the emission
            hydro_idx = data.h_index.isel(h_index=i_hydro)
            y = data.signal.sel(
                time=slice(i_em * t_interp_pulse - ts, i_em * t_interp_pulse + t_ir),
                h_index=hydro_idx,
            )

            # Estimate the impulse response
            h_hat = crosscorr_deconvolution(x=x, y=y.values)
            ri_hat[i_hydro, i_em, :] = h_hat

    # Take the mean impulse response over all sweeps analysed
    ri_hat_mean = np.mean(ri_hat, axis=1)

    data["t_ir"] = t
    data["ri_hat"] = (
        ["h_index", "t_ir"],
        ri_hat_mean,
    )

    # Derive the corresponding frequency response
    tf_hat_mean = np.fft.rfft(ri_hat_mean, axis=1)
    f_ir = np.fft.rfftfreq(len(t), d=ts)
    data["f_ir"] = f_ir

    # Store amplitude and phase in two separate variables to avoid issues with complex in netcdf
    data["tf_hat_amp"] = (
        ["h_index", "f_ir"],
        np.abs(tf_hat_mean),
    )
    data["tf_hat_phase"] = (
        ["h_index", "f_ir"],
        np.angle(tf_hat_mean),
    )

    # Save results
    data.to_netcdf(os.path.join(processed_data_path, f"{recording_name}.nc"))
    data.close()


def localise(
    recording_names,
    recording_name_to_loc,
    recording_props,
    processing_props,
    pos_ids=[],
    th_pos=None,
):
    # Unpack usefull props
    f0 = recording_props.get("f0", 8e3)
    f1 = recording_props.get("f1", 15e3)
    rtf_method = processing_props.get("method", "cs")

    # Load all datasets
    recording_names_all = recording_names + [recording_name_to_loc]
    datasets = {}
    for recording_name in recording_names_all:
        ds_i = xr.open_dataset(
            os.path.join(processed_data_path, f"{recording_name}_rtf.nc")
        )
        datasets[recording_name] = ds_i.sel({f"f_rtf_{rtf_method}": slice(f0, f1)})

        # datasets[recording_name] = ds_i.sel(f_rtf=slice(f0, f1))
    # print(f"SNR {datasets[recording_name_to_loc].snr.values} dB")
    # Derive distance between rtf at the position of interrest and other positions
    ds_true_pos = datasets[recording_name_to_loc]
    rtf_ref = ds_true_pos.rtf_amp_cs * np.exp(1j * ds_true_pos.rtf_phase_cs)
    dist = []
    for recording_name in recording_names:
        ds_pos = datasets[recording_name]
        rtf_pos = ds_pos.rtf_amp_cs * np.exp(1j * ds_pos.rtf_phase_cs)

        # Derive distance using hermitian angle
        dist_kwargs = {"ax_rcv": 0, "apply_mean": True}
        d = D_hermitian_angle_fast(rtf_ref.values, rtf_pos.values, **dist_kwargs)
        dist.append(d)

    # Extract position ids from names
    dyn_loc = False
    if len(pos_ids) == 0:
        if "PR" in recording_names[0]:  # Source is dynamic
            pos_ids = [float(n.split("_")[-2][1:-1]) for n in recording_names]
            dyn_loc = True
        else:
            pos_ids = [n.split("_")[-4] for n in recording_names]

    plt.figure()
    plt.plot(pos_ids, dist, color="k", marker="o", linestyle="-", markersize=5)

    if dyn_loc:
        xlabel = r"$\textrm{Range from P1}  \, \textrm{[m]}$"
        if th_pos is not None:
            plt.axvline(
                th_pos, color="r", linestyle="--", label=r"$\textrm{True position}$"
            )
    else:
        xlabel = r"$\textrm{Position}$"
        if th_pos is not None:
            # idx_th_pos = pos_ids.index(th_pos)
            plt.axvline(
                # pos_ids[th_pos],
                th_pos,
                color="r",
                linestyle="--",
                label=r"$\textrm{True position}$",
            )

    plt.xlabel(xlabel)
    plt.ylabel(r"$\theta \textrm{[°]}$")
    plt.title(r"$\textrm{" + f"{recording_name_to_loc}" + r"}$")
    plt.legend()
    plt.savefig(
        os.path.join(
            ds_true_pos.attrs["img_path"], f"{recording_name_to_loc}_localisation.png"
        )
    )
    # plt.show()
    # print(f"SNR {ds_pos.snr.values} dB")
    # print(dist)

    snrs = datasets[recording_name_to_loc].snr.values

    return pos_ids, dist, snrs


def re_order_recordings(recording_names):
    """
    Re-order the recordings so that distance from P1 are ordered in ascending order.
    """

    ordered_pos = [
        "P1",  # Reference position
        "P6",  # 5m from P1
        "P2",  # 10m from P1
        "P5",  # 15m from P1
        "P3",  # 20m from P1
        "P4",  # 25m from P1
    ]
    ordered_recording_names = []
    for pos in ordered_pos:
        for rec in recording_names:
            if pos in rec:
                ordered_recording_names.append(rec)
                break

    return ordered_pos, ordered_recording_names


# ======================================================================================================================
# Analyse results from both test
# ======================================================================================================================
# img_analysis_path = os.path.join(img_root, "analysis")
# if not os.path.exists(img_analysis_path):
#     os.makedirs(img_analysis_path)

# # Load data
# data_1 = xr.open_dataset(os.path.join(processed_data_path, f"{recording_name_1}.nc"))
# data_2 = xr.open_dataset(os.path.join(processed_data_path, f"{recording_name_2}.nc"))

# # Restrict to the common frequency range
# fmin = max(recording_props_1["f0"], recording_props_2["f0"])
# fmax = min(recording_props_1["f1"], recording_props_2["f1"])
# data_1 = data_1.sel(f_ir=slice(fmin, fmax))
# data_2 = data_2.sel(f_ir=slice(fmin, fmax))

# # Compare rtf
# data_rtf_1 = xr.open_dataset(
#     os.path.join(processed_data_path, f"{recording_name_1}_rtf.nc")
# )
# data_rtf_2 = xr.open_dataset(
#     os.path.join(processed_data_path, f"{recording_name_2}_rtf.nc")
# )

# for i_hydro in data_rtf_1.h_index.values:
#     rtf_hydro_1 = data_rtf_1.sel(h_index=i_hydro)
#     rtf_hydro_2 = data_rtf_2.sel(h_index=i_hydro)
#     plt.figure()
#     rtf_hydro_1.rtf_amp.plot(
#         x="f_ir",
#         color="k",
#         label=r"$\Pi_{"
#         + str(i_hydro)
#         + r"} \textrm{(deconvolution)}$"
#         + f" ({recording_name_1})",
#     )
#     rtf_hydro_2.rtf_amp.plot(
#         x="f_ir",
#         color="g",
#         linestyle="--",
#         label=r"$\Pi_{"
#         + str(i_hydro)
#         + r"} \textrm{(deconvolution)}$"
#         + f" ({recording_name_2})",
#     )

#     rtf_hydro_1.rtf_amp_cs.plot(
#         x="f_rtf",
#         label=r"$\Pi_{" + str(i_hydro) + r"}^{(CS)}$" + f" ({recording_name_1})",
#         linestyle="-",
#         color="b",
#         marker="o",
#         linewidth=0.2,
#         markersize=2,
#     )
#     rtf_hydro_2.rtf_amp_cs.plot(
#         x="f_rtf",
#         label=r"$\Pi_{" + str(i_hydro) + r"}^{(CS)}$" + f" ({recording_name_2})",
#         linestyle="-",
#         color="r",
#         marker="o",
#         linewidth=0.2,
#         markersize=2,
#     )

#     plt.title("")
#     plt.xlabel(r"$f \, \textrm{[Hz]}$")
#     plt.ylabel(r"$\left| \Pi \right|$")
#     plt.xlim(5 * 1e3, 1e4)
#     # plt.ylim([1e-1, 1e1])
#     plt.yscale("log")
#     plt.legend()
#     plt.savefig(
#         os.path.join(data_rtf_1.attrs["img_path"], f"rtf_amp_hydro{i_hydro}.png")
#     )

#     plt.figure()
#     rtf_hydro_1.rtf_phase.plot(
#         x="f_ir",
#         color="k",
#         label=r"$\Pi_{"
#         + str(i_hydro)
#         + r"} \textrm{(deconvolution)}$"
#         + f" ({recording_name_1})",
#     )
#     rtf_hydro_2.rtf_phase.plot(
#         x="f_ir",
#         color="g",
#         linestyle="--",
#         label=r"$\Pi_{"
#         + str(i_hydro)
#         + r"} \textrm{(deconvolution)}$"
#         + f" ({recording_name_2})",
#     )
#     rtf_hydro_1.rtf_phase_cs.plot(
#         x="f_rtf",
#         label=r"$\Pi_{" + str(i_hydro) + r"}^{(CS)}$" + f" ({recording_name_1})",
#         linestyle="-",
#         color="b",
#         marker="o",
#         linewidth=0.2,
#         markersize=2,
#     )
#     rtf_hydro_2.rtf_phase_cs.plot(
#         x="f_rtf",
#         label=r"$\Pi_{" + str(i_hydro) + r"}^{(CS)}$" + f" ({recording_name_2})",
#         linestyle="-",
#         color="r",
#         marker="o",
#         linewidth=0.2,
#         markersize=2,
#     )

#     plt.title("")
#     plt.xlabel(r"$f \, \textrm{[Hz]}$")
#     plt.ylabel(r"$\phi \, \textrm{[rad]}$")
#     # plt.xlim(fmin, fmax)
#     plt.xlim(5 * 1e3, 1e4)
#     plt.legend()
#     plt.savefig(
#         os.path.join(data_rtf_1.attrs["img_path"], f"rtf_phase_hydro{i_hydro}.png")
#     )
#     plt.close("all")


# ======================================================================================================================
# Left overs
# ======================================================================================================================

# for i_hydro in data_1.h_index.values:

#     # Plot amplitude and phase of the transfer function
#     plt.figure()
#     data_2.sel(h_index=i_hydro).tf_hat_amp.plot(x="f_ir")
#     data_1.sel(h_index=i_hydro).tf_hat_amp.plot(x="f_ir")
#     plt.title("Transfer function amplitude")
#     plt.xlim(fmin, fmax)
#     plt.savefig(
#         os.path.join(img_analysis_path, f"transfer_function_amp_hydro_{i_hydro}.png")
#     )

#     plt.figure()
#     data_2.sel(h_index=i_hydro).tf_hat_phase.plot(x="f_ir")
#     data_1.sel(h_index=i_hydro).tf_hat_phase.plot(x="f_ir")
#     plt.title("Transfer function phase")
#     plt.xlim(fmin, fmax)
#     plt.savefig(
#         os.path.join(img_analysis_path, f"transfer_function_phase_hydro_{i_hydro}.png")
#     )

#     # Derive the hermitian angle between the two transfer functions (from data 1 to data 2)
#     tf_1 = data_1.sel(h_index=i_hydro).tf_hat_amp * np.exp(
#         1j * data_1.sel(h_index=i_hydro).tf_hat_phase
#     )
#     tf_2 = data_2.sel(h_index=i_hydro).tf_hat_amp * np.exp(
#         1j * data_2.sel(h_index=i_hydro).tf_hat_phase
#     )

#     # normalise amplitude for comparison
#     tf_1 = tf_1 / data_1.sel(h_index=i_hydro).tf_hat_amp.max()
#     tf_2 = tf_2 / data_2.sel(h_index=i_hydro).tf_hat_amp.max()

#     plt.figure()
#     np.abs(tf_2).plot()
#     np.abs(tf_1).plot()
#     plt.title("Transfer function amplitude")
#     plt.xlim(fmin, fmax)
#     plt.savefig(
#         os.path.join(
#             img_analysis_path, f"transfer_function_amp_normalized_hydro{i_hydro}.png"
#         )
#     )

#     inner_prod = np.abs(
#         np.sum(
#             tf_1.conj() * tf_2,
#         )
#     )
#     norm_1 = np.linalg.norm(tf_1)
#     norm_2 = np.linalg.norm(tf_2)

#     # Cosine of Hermitian angle, clipped for stability
#     cos_angle = np.clip(inner_prod / (norm_1 * norm_2), -1.0, 1.0)
#     dist = np.arccos(cos_angle)
#     dist = np.rad2deg(dist)

#     print(f"Hydrophone {i_hydro} - Angle between transfer functions : {dist.values}")


# plt.show()


# Plot 10 first points on the C plane
# n_pt = 5
# pt_start = tf_1.size // 2
# plt.figure()
# plt.plot(
#     tf_1.real.values[pt_start : pt_start + n_pt],
#     tf_1.imag.values[pt_start : pt_start + n_pt],
#     "o",
#     label="TF 1",
# )
# plt.plot(
#     tf_2.real.values[pt_start : pt_start + n_pt],
#     tf_2.imag.values[pt_start : pt_start + n_pt],
#     "o",
#     label="TF 2",
# )
# plt.legend()
# plt.title("Transfer function in the complex plane")
# plt.savefig(os.path.join(img_analysis_path, "transfer_function_complex_plane.png"))

# plt.show()
# Plot left overs
# # Plot signal
# plt.figure()
# data.signal.plot(x="time", hue="h_index")
# plt.title(test_name)
# plt.savefig(os.path.join(img_path, f"{test_name}_signal.png"))

# # Zoom each pulse and save
# for i_em in range(n_sweep):
#     plt.figure()
#     data.signal.sel(
#         time=slice(i_em * t_interp_pulse, (i_em + 1) * t_interp_pulse)
#     ).plot(x="time", hue="h_index")
#     plt.title(test_name)
#     plt.savefig(os.path.join(img_path, f"{test_name}_signal_sweep_{i_em}.png"))
#     plt.close()

# plt.figure()
# plt.plot(t[0:5000], x[0:5000])
# plt.title(f"{test_name} - Source pulse")
# plt.savefig(os.path.join(img_path, f"{test_name}_source_pulse.png"))


# Plot impulse response
# plt.figure()
# plt.plot(t, h_hat)
# plt.title(f"{test_name} - Sweep {i_em}")
# plt.savefig(
#     os.path.join(img_path, f"{test_name}_impulse_response_sweep_{i_em}.png")
# )
# plt.close()

# plt.figure()
# plt.plot(t, y, label=r"$y$")
# plt.plot(t, y_hat, label=r"$\hat{y}$")
# plt.legend()
# plt.title(f"{test_name} - Sweep {i_em}")
# plt.savefig(
#     os.path.join(img_path, f"{test_name}_output_signal_sweep_{i_em}.png")
# )
# # plt.show()
# plt.close()

# Derive the frequency response


# Compare all estimated impulse responses
# plt.figure()
# data.ri_hat.plot(x="sweep_time", hue="sweep_idx")
# plt.title(test_name)
# plt.savefig(os.path.join(img_path, f"{test_name}_impulse_response_all_sweeps.png"))

# # Compare all transfer functions
# plt.figure()
# data.tf_hat_amp.plot(x="sweep_freq", hue="sweep_idx")
# data.tf_hat_mean_amp.plot(x="sweep_freq", color="k", linestyle="--")
# plt.xlim([f0, f1])
# plt.title(test_name)
# plt.savefig(os.path.join(img_path, f"{test_name}_transfer_function_amp_all_sweeps.png"))

# # Same thing with the phase
# plt.figure()
# data.tf_hat_phase.plot(x="sweep_freq", hue="sweep_idx")
# data.tf_hat_mean_phase.plot(x="sweep_freq", color="k", linestyle="--")
# plt.xlim([f0, f1])
# plt.title(test_name)
# plt.savefig(
#     os.path.join(img_path, f"{test_name}_transfer_function_phase_all_sweeps.png")
# )

# Plot the estimated impulse response


# plt.show()

# if __name__ == "__main__":
#     pass
