#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   signal_processing_utils.py
@Time    :   2024/09/13 14:01:31
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
from real_data_analysis.real_data_utils import plot_dsp, plot_stft, get_dsp
from get_data.wav.get_data_from_rhumrum import get_rhumrum_data


def compute_dsps(wav_data, rcv_info, nperseg=2**10, noverlap=None):
    rcv_0 = list(wav_data.keys())[0]
    available_mmsi = list(wav_data[rcv_0].keys())

    if noverlap is None:
        noverlap = nperseg // 2

    coh = {}
    sxx = {}
    syy = {}
    sxy = {}

    for i, rcv_id1 in enumerate(rcv_info["id"]):
        for j, rcv_id2 in enumerate(rcv_info["id"]):
            if i < j:
                rcv_couple_id = f"{rcv_id1}_{rcv_id2}"
                coh[rcv_couple_id] = {}
                sxx[rcv_couple_id] = {}
                syy[rcv_couple_id] = {}
                sxy[rcv_couple_id] = {}

                for mmsi in available_mmsi:

                    s1 = wav_data[rcv_id1][mmsi]
                    s2 = wav_data[rcv_id2][mmsi]

                    fs = s1["sig"].meta.sampling_rate

                    # Derive Sxx, Syy and Sxy
                    f, Sxx = sp.welch(
                        s1["data"],
                        fs=fs,
                        window="hann",
                        nperseg=nperseg,
                        noverlap=noverlap,
                    )
                    f, Syy = sp.welch(
                        s2["data"],
                        fs=fs,
                        window="hann",
                        nperseg=nperseg,
                        noverlap=noverlap,
                    )
                    f, Sxy = sp.csd(
                        s1["data"],
                        s2["data"],
                        fs=fs,
                        window="hann",
                        nperseg=nperseg,
                        noverlap=noverlap,
                    )

                    syy[rcv_couple_id][mmsi] = {}
                    sxy[rcv_couple_id][mmsi] = {}
                    sxx[rcv_couple_id][mmsi] = {}
                    coh[rcv_couple_id][mmsi] = {}
                    sxx[rcv_couple_id][mmsi]["f"] = f
                    syy[rcv_couple_id][mmsi]["f"] = f
                    sxy[rcv_couple_id][mmsi]["f"] = f
                    coh[rcv_couple_id][mmsi]["f"] = f
                    sxx[rcv_couple_id][mmsi]["Sxx"] = Sxx
                    syy[rcv_couple_id][mmsi]["Syy"] = Syy
                    sxy[rcv_couple_id][mmsi]["Sxy"] = Sxy
                    coh[rcv_couple_id][mmsi]["coh_xy"] = np.abs(Sxy) ** 2 / (Sxx * Syy)

                    # Equivalent to sp.coherence()
                    # coh[mmsi][rcv_couple_id]["coh_xy_sp"] = sp.coherence(
                    #     s1["data"],
                    #     s2["data"],
                    #     fs=fs,
                    #     window="hann",
                    #     nperseg=2**10,
                    #     noverlap=2**10 // 2,
                    # )[1]

    return coh, sxx, syy, sxy


def load_data(
    stations,
    mmsi_list,
    start_times,
    ch=["BDH"],
    fmin=8,
    fmax=48,
    filter_type="bandpass",
    filter_corners=4,
    duration_seconds=3600,
):

    nperseg = 2**10
    noverlap = int(nperseg // 2)

    loaded_data = {}
    for s_id in stations:
        data_station = {}
        for i, mmsi in enumerate(mmsi_list):
            date = start_times[mmsi].strftime("%Y-%m-%dT%H:%M:%S")

            raw_sig, filt_sig, corr_sig = get_rhumrum_data(
                station_id=s_id,
                date=date,
                duration_sec=duration_seconds,
                channels=ch,
                plot=False,
                fmin=fmin,
                fmax=fmax,
                filter_type=filter_type,
                filter_corners=filter_corners,
            )

            sig = corr_sig["BDH"]
            data = sig.data
            data_station[mmsi] = {}
            data_station[mmsi]["data"] = data
            data_station[mmsi]["sig"] = sig

            # Derive stft
            f, tt, stft = sp.stft(
                data,
                fs=sig.meta.sampling_rate,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
            )
            data_station[mmsi]["f"] = f
            data_station[mmsi]["tt"] = tt
            data_station[mmsi]["stft"] = stft

        loaded_data[s_id] = data_station

    return loaded_data


def plot_time_series(wav_data, ais_data, intersection_time):
    rcv_0 = list(wav_data.keys())[0]
    available_mmsi = list(wav_data[rcv_0].keys())
    available_stations = list(wav_data.keys())

    fig, ax = plt.subplots(len(available_mmsi), len(available_stations), sharex=True)
    for i, rcv_id in enumerate(available_stations):
        for j, mmsi in enumerate(available_mmsi):
            data = wav_data[rcv_id][mmsi]["data"]
            sig = wav_data[rcv_id][mmsi]["sig"]
            # ax[j, i].plot(sig.times("utcdatetime"), data)
            ship_name = ais_data[ais_data["mmsi"] == mmsi]["shipName"].values[0]
            t_hour = sig.times() / 3600
            # ax[j, i].plot(sig.times(), data, label=f"{ship_name}", color="k")
            ax[j, i].plot(t_hour, data, label=f"{ship_name}", color="k")

            # Plot a vertical dotted line at the intersection time
            ax[j, i].axvline(
                x=t_hour[-1] / 2,
                color="r",
                linestyle="--",
                label=f"Intersection at {r'$t_{ship' + str(j) + r'}$'} = {intersection_time[mmsi].strftime('%Y-%m-%d %H:%M')}",
            )
            # ax[j, 0].set_ylabel("Amplitude")
            ax[j, i].legend(loc="upper right", fontsize=7)

            # Set common y limits
            ax[j, i].set_ylim(-5, 5)

        ax[0, i].set_title(f"Station {rcv_id}")
        # ax[-1, i].set_xlabel("Time [s]")
        # ax[-1, i].set_xlabel("Time [h]")

    fig.supxlabel("Time [h]")
    fig.supylabel("Amplitude")
    plt.tight_layout()


def plot_spectrograms(
    wav_data,
    ais_data,
    intersection_time=None,
    fmin=4,
    fmax=18,
    delta_f=1,
    vmin=-100,
    vmax=0,
):
    rcv_0 = list(wav_data.keys())[0]
    available_mmsi = list(wav_data[rcv_0].keys())
    available_stations = list(wav_data.keys())

    fig, ax = plt.subplots(len(available_mmsi), len(available_stations), sharex=True)

    # Reshape if only one ship
    if len(available_mmsi) == 1:
        ax = np.array([ax])

    for i, rcv_id in enumerate(available_stations):
        for j, mmsi in enumerate(available_mmsi):
            stft = wav_data[rcv_id][mmsi]["stft"]
            f = wav_data[rcv_id][mmsi]["f"]
            tt = wav_data[rcv_id][mmsi]["tt"]
            ship_name = ais_data[ais_data["mmsi"] == mmsi]["shipName"].values[0]

            tt_hour = tt / 3600
            # ax[j, i].pcolormesh(tt, f, 20*np.log10(np.abs(stft)), shading="gouraud", vmin=vmin, vmax=vmax)
            ax[j, i].pcolormesh(
                tt_hour,
                f,
                20 * np.log10(np.abs(stft)),
                shading="gouraud",
                vmin=vmin,
                vmax=vmax,
            )

            # Plot a vertical dotted line at the intersection time
            if intersection_time is not None:
                ax[j, i].axvline(
                    x=tt_hour[-1] / 2,
                    color="r",
                    linestyle="--",
                    label=f"Intersection at {r'$t_{ship' + str(j) + r'}$'} = {intersection_time[mmsi].strftime('%Y-%m-%d %H:%M')}",
                )

            # ax[j, 0].set_ylabel("Frequency [Hz]")
            ax[j, i].text(
                0.02,
                0.9,
                ship_name,
                transform=ax[j, i].transAxes,
                ha="left",
                fontsize=12,
                bbox=dict(facecolor="lightyellow", edgecolor="black", alpha=0.9),
            )
            # ax[j, i].set_title(f"{ship_name}")
            ax[j, i].legend(loc="upper right", fontsize=7)
            ax[j, i].set_ylim([fmin - delta_f, fmax + delta_f])

        # ax[-1, i].set_xlabel("Time [s]")
        ax[0, i].set_title(f"Station {rcv_id}")
        # ax[-1, i].set_xlabel("Time [h]")

    fig.supxlabel("Time [h]")
    fig.supylabel("Frequency [Hz]")

    return fig, ax


def compute_coherence_spectrogram(
    signal1, signal2, fs, nperseg, noverlap, nperseg_sub, noverlap_sub
):
    # Segment the signals
    segment_length = nperseg
    step_size = nperseg - noverlap
    n_segments = (len(signal1) - noverlap) // step_size

    tt = []
    coherence_matrix = []

    for i in range(n_segments):
        # Get the time window
        start = i * step_size
        end = start + segment_length
        segment1 = signal1[start:end]
        segment2 = signal2[start:end]

        # Derive time vector (middle of the segment)
        tt.append((start + segment_length // 2) / fs)

        # Compute the coherence over the segment using Welch's method
        f, coh = sp.coherence(
            segment1, segment2, fs=fs, nperseg=nperseg_sub, noverlap=noverlap_sub
        )
        coherence_matrix.append(coh)

    tt = np.array(tt)
    coherence_matrix = np.array(coherence_matrix).T  # Transpose for plotting

    return f, tt, coherence_matrix


def process_plot(data, fmin, fmax, nperseg, save):
    # Define dsp parameters
    dsp_args = {
        "nperseg": nperseg,
        "overlap_coef": 3 / 4,
        "window": "hann",
        "fs": data["sig"].meta.sampling_rate,
        "fmin": fmin,
        "fmax": fmax,
    }

    # Derive dsp
    get_dsp(data, dsp_args=dsp_args)

    # Plot stft
    plot_stft(data, save=save)

    # Plot dsp
    fmin_plot = 4
    fmax_plot = 18
    plot_dsp(data, fmin=fmin_plot, fmax=fmax_plot, save=save)

    fmin_plot = 18
    fmax_plot = 42
    plot_dsp(data, fmin=fmin_plot, fmax=fmax_plot, save=save)

    # Select one ray
    idx_ray = np.argmax(data["Pxx_in_band"][data["idx_f_peaks"]])  # Most energetic ray
    # idx_ray = 5
    ray_bandwith = 2
    fmin_plot = data["f_peaks"][idx_ray] - ray_bandwith / 2
    fmax_plot = data["f_peaks"][idx_ray] + ray_bandwith / 2

    plot_dsp(data, fmin=fmin_plot, fmax=fmax_plot, save=save)

    plt.close("all")


def get_bifrequency_spectrum(
    s,
    fs,
    fmin=None,
    fmax=None,
    nperseg=2**12,
    noverlap=2**11,
    plot=False,
    root_img=None,
):

    # nperseg = 2**17
    # noverlap = 2**16
    ff, tt, stft = sp.stft(
        s,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = fs / 2

    ff_in_band = np.logical_and((ff >= fmin), (ff <= fmax))
    stft = stft[ff_in_band, :]
    ff = ff[ff_in_band]

    # Faster implementation with matrix multiplication
    L = stft.shape[1]
    S_f1f2 = stft @ np.conj(stft).T
    S_f1f2 = 1 / L * S_f1f2

    if plot:
        plot_bi_frequency_spectrum(S_f1f2, ff, root_img)

    return ff, S_f1f2


def plot_bi_frequency_spectrum(S_f1f2, ff, root_img=None):

    # Normalize
    S_f1f2 /= np.max(np.abs(S_f1f2))

    S_f1f2 = xr.DataArray(
        S_f1f2,
        dims=["ff1", "ff2"],
        # coords={"ff1": stft.ff.values, "ff2": stft.ff.values},
        coords={"ff1": ff, "ff2": ff},
    )

    S_f1f2_log_magnitude = 20 * np.log10(np.abs(S_f1f2))
    # Plot
    fig, ax = plt.subplots()
    S_f1f2_log_magnitude.plot(
        ax=ax,
        x="ff1",
        y="ff2",
        vmin=-30,
        vmax=0,
        cmap="viridis",
        add_colorbar=True,
        cbar_kwargs={"label": "Magnitude [dB]"},
    )

    ax.set_xlabel("$f_1$" + " [Hz]")
    ax.set_ylabel("$f_2$" + " [Hz]")
    ax.set_title("Bi-frequency spectrum")
    img_filepath = os.path.join(root_img, "bi_frequency_spectrum.png")
    plt.savefig(img_filepath)


if __name__ == "__main__":

    from signals.signals import lfm_chirp

    ### LFM Chirp ###
    f0 = 8 * 1e3
    f1 = 15 * 1e3
    fs = 10 * f1
    T = 0.1
    phi = 0

    # Signal
    s, t = lfm_chirp(f0, f1, fs, T, phi)

    # Plot lfm chirp
    # fig, ax = plt.subplots()
    # ax.plot(t, s)
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Amplitude")
    # ax.set_title("LFM Chirp")
    # # plt.show()

    # Plot spectrogram
    nperseg = 2**10
    noverlap = 2**9

    ff, tt, stft = sp.stft(
        s,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    plt.figure()
    plt.pcolormesh(tt, ff, 20 * np.log10(np.abs(stft)), shading="gouraud")
    plt.ylim([0, f1 + 2 * 1e3])
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram")
    plt.colorbar(label="Magnitude [dB]")
    # plt.show()

    # Bispectrum from spectrum
    X = np.fft.rfft(s)
    freq = np.fft.rfftfreq(len(s), 1 / fs)

    # Plot spectrum
    plt.figure()
    plt.plot(freq, np.abs(X))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Spectrum")

    X = np.atleast_2d(X)
    S_f1f2 = np.conj(X).T @ X
    # plt.show

    # Plot bispectrum
    # plot_bi_frequency_spectrum(S_f1f2, freq, root_img=".")

    plt.figure()
    plt.pcolormesh(freq, freq, np.abs(S_f1f2) / np.max(np.abs(S_f1f2)))
    plt.xlabel("$f_1$ [Hz]")
    plt.ylabel("$f_2$ [Hz]")
    plt.title("Bispectrum")
    plt.colorbar(label="Magnitude")
    plt.xlim(f0 * 0.9, f1 * 1.1)
    plt.ylim(f0 * 0.9, f1 * 1.1)
    plt.show()

    fmin = f0
    fmax = f1
    plot = True
    root_img = "."
    get_bifrequency_spectrum(
        s,
        fs,
        fmin=fmin,
        fmax=fmax,
        nperseg=nperseg,
        noverlap=noverlap,
        plot=plot,
        root_img=root_img,
    )
