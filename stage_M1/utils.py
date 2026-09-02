#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   data_selection.py
@Time    :   2026/04/07 15:50:00
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
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sp
import gc

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream, Inventory

# from obspy import read as obspy_read
# from obspy import read_inventory as obspy_read_inventory

from publication.publication_figure import PubFigure

PubFigure(disable_backend=True)

p0 = 1e-6

# ============================================================
# 📡 DATA LOADING
# ============================================================


def get_data(data_info, save=True, root_folder=None):
    """
    Download waveform data and metadata using ObsPy.

    Parameters
    ----------
    data_info : dict
        Dictionary containing client, network, stations, channel,
        start_time, end_time
    save : bool
        Save data locally
    root_folder : str
        Path to save data

    Returns
    -------
    st : Stream
    inv : Inventory
    """

    client = Client(data_info["client"])

    st = Stream()
    inv = Inventory()

    for sta_id in data_info["stations"]:
        try:
            st += client.get_waveforms(
                network=data_info["network"],
                station=sta_id,
                channel=data_info["channel"],
                location="*",
                starttime=UTCDateTime(data_info["start_time"]),
                endtime=UTCDateTime(data_info["end_time"]),
            )

            inv += client.get_stations(
                network=data_info["network"],
                station=sta_id,
                channel=data_info["channel"],
                level="response",
                starttime=UTCDateTime(data_info["start_time"]),
                endtime=UTCDateTime(data_info["end_time"]),
            )

        except Exception as e:
            print(f"Error with station {sta_id}: {e}")

    # Save
    if save and root_folder is not None:
        fname = (
            f"{data_info['network']}_{data_info['stations'][0][:3]}_"
            f"{data_info['start_time'].strftime('%Y%m%d_%H%M')}"
        )
        fpath = os.path.join(root_folder, fname)

        st.write(fpath + ".mseed", format="MSEED")
        inv.write(fpath + ".xml", format="STATIONXML")

    return st, inv


# ============================================================
# ⚙️ PROCESSING
# ============================================================


def apply_response_correction(
    st, inv, pre_filt=(0.05, 0.1, 100, 120), response_output="DEF"
):
    """
    Remove instrument response from stream.
    """
    st = st.copy()
    st.remove_response(
        inventory=inv,
        output=response_output,
        pre_filt=pre_filt,
        water_level=60,
    )
    return st


# ============================================================
# 📊 METRICS
# ============================================================


def compute_spl(trace, p0=p0):
    """
    Compute Sound Pressure Level (SPL)
    """
    data = trace.data
    p_rms = np.sqrt(np.mean(data**2))
    spl = 20 * np.log10(p_rms / p0)
    return spl


# ============================================================
# 📈 PLOTTING
# ============================================================


def save_figure(fig, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    fpath = os.path.join(save_path, filename)
    fig.savefig(fpath, bbox_inches="tight")
    # print(f"Saved: {fpath}")


def plot_signal(trace, save=False, save_path=None, show=False):
    fig = trace.plot(show=show)
    fig.suptitle(trace.id)

    if save and save_path:
        save_path = os.path.join(save_path, "signals")
        os.makedirs(save_path, exist_ok=True)

        fname = f"{trace.id}_signal.png"
        save_figure(fig, save_path, fname)

    return fig


def plot_spectrogram(
    trace, save=False, save_path=None, nperseg=2048, overlap=0.75, p0=p0
):
    noverlap = int(nperseg * overlap)

    f, t, Sxx = sp.stft(
        trace.data,
        fs=trace.stats.sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="psd",  # to get Pa**2 / Hz (assuming input is given in Pa)
    )

    Sxx_db = 10 * np.log10(
        np.abs(Sxx)
    )  # dB re 1 Pa**2 / Hz (assuming input is given in Pa)
    Sxx_db += 10 * np.log10(1 / p0)  # dB re 1uPa**2 / Hz
    clabel = r"dB re 1$\mu$Pa$^2$ / Hz"

    # # Associated datetime vector
    from datetime import timedelta
    import matplotlib.dates as mdates

    tt_datetime = pd.date_range(
        trace.stats.starttime.datetime,
        trace.stats.starttime.datetime + timedelta(seconds=t[-1]),
        freq=f"{t[1]-t[0]:.3f}s",
        inclusive="both",
    )

    # if t.max() >= 60 and t.max() < 3600:
    #     t = t / 60
    #     time_label = "Time [min]"
    # elif t.max() >= 3600:
    #     t = t / 3600
    #     time_label = "Time [hr]"

    fig = plt.figure()
    plt.pcolormesh(
        tt_datetime,
        f,
        Sxx_db,
        cmap="magma",
        vmin=np.percentile(Sxx_db, 20),
        vmax=np.percentile(Sxx_db, 95),
    )

    formatter = mdates.DateFormatter("%H:%M")
    plt.gca().xaxis.set_major_formatter(formatter)
    formatter = mdates.DateFormatter("%H:%M")
    plt.gca().xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    plt.gca().xaxis.set_major_locator(locator)
    plt.setp(plt.gca().get_xticklabels(), rotation=15, ha="right")

    plt.colorbar(label=clabel)
    plt.xlabel("Time")
    plt.ylabel("Frequency [Hz]")
    plt.title(f"{trace.id}")

    if save and save_path:
        save_path = os.path.join(save_path, "spectrogram")
        os.makedirs(save_path, exist_ok=True)

        fname = f"{trace.id}_spectrogram.png"
        save_figure(fig, save_path, fname)

    return fig


def plot_psd(trace, save=False, save_path=None, nperseg=2048, overlap=0.75):
    noverlap = int(nperseg * overlap)

    f, Pxx = sp.welch(
        trace.data,
        fs=trace.stats.sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )

    fig = plt.figure()
    Pxx_dB = 10 * np.log10(
        Pxx
    )  # dB re 1 Pa**2 / Hz       (assuming input is given in Pa)
    Pxx_dB += 10 * np.log10(1 / p0)  # dB re 1uPa**2 / Hz
    plt.plot(f, Pxx_dB)
    # plt.semilogy(f, Pxx)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"PSD [dB re 1$\mu$Pa$^2$ / Hz]")
    plt.title(trace.id)

    if save and save_path:
        save_path = os.path.join(save_path, "psd")
        os.makedirs(save_path, exist_ok=True)

        fname = f"{trace.id}_psd.png"
        save_figure(fig, save_path, fname)

    return fig


def plot_station_map(inv, save=False, save_path=None):
    """
    Plot station geographic map using ObsPy inventory.
    """

    fig = inv.plot(projection="local", show=False)
    fig.suptitle("Receiver network")

    if save and save_path is not None:
        fname = os.path.join(save_path, "station_map.png")
        fig.savefig(fname, dpi=300)
        print(f"Saved: {fname}")

    return fig


# ============================================================
# 🚀 MAIN PIPELINE
# ============================================================


def process_and_plot(
    data_info,
    pre_filt=(0.05, 0.1, 100, 120),
    response_output="DEF",
    save_figures=False,
    save_path="figures",
    plot_map=True,
    plot_sig=True,
    plot_spectro=True,
    plot_power_sprectral_density=True,
    show=False,
):
    """
    Full pipeline with optional saving and map plotting.
    """
    try:
        print("Loading data...")
        st, inv = get_data(data_info, save=False)
    except:
        return None, None

    # 🗺️ Plot map
    if plot_map:
        print("Plotting station map...")
        plot_station_map(inv, save=save_figures, save_path=save_path)

    # Remove response
    st = apply_response_correction(st, inv, pre_filt, response_output=response_output)

    print("Processing traces...")
    for tr in st:
        print(f"\n--- {tr.id} ---")

        spl = compute_spl(tr)
        print(f"SPL: {spl:.1f} dB re 1 µPa")

        try:
            if plot_sig:
                plot_signal(tr, save=save_figures, save_path=save_path, show=show)
            if plot_spectro:
                plot_spectrogram(tr, save=save_figures, save_path=save_path)
            if plot_power_sprectral_density:
                plot_psd(tr, save=save_figures, save_path=save_path)
        except:
            continue

        if show:
            plt.show()
        else:
            plt.close("all")

        gc.collect()

    plt.close("all")

    return st, inv


if __name__ == "__main__":
    pass
