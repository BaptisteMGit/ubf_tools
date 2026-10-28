#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_plot_portion.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Figures of the portion: receiver positions, shot positions, pressure signals and spectrograms.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sp

from publication.publication_figure import color, set_subfigures_abc_labels


### Positions ###
def plot_obs_pos(ds_pos, ax=None, add_legend=True, add_title=True):
    """Plot the OBS positions in geographic coordinates (longitude / latitude).

    Parameters
    ----------
    ds_pos : xarray.Dataset
        Dataset with ``rcv_lon`` and ``rcv_lat`` (dimension ``rcv_id``).
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. A new figure is created if None.
    add_legend : bool, optional
        If True, add a legend (at most 4 columns).
    add_title : bool, optional
        If True, add the title "OBS positions".

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for obs_id in ds_pos.rcv_id.values:
        rcv = ds_pos.sel(rcv_id=obs_id)
        ax.scatter(
            rcv.rcv_lon, rcv.rcv_lat, marker="d", label=f"{obs_id.upper()}", s=200
        )

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")

    if add_title:
        ax.set_title("OBS positions")
    if add_legend:
        _, labels = ax.get_legend_handles_labels()
        ncols = min(len(labels), 4)
        ax.legend(ncols=ncols)

    return ax


def plot_portion_tir_pos(portion_tree, ax=None):
    """Plot the shot positions of every section, and the OBS, in lon / lat.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree with a ``position`` group.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. A new figure is created if None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the axis.
    ax : matplotlib.axes.Axes
        Axis used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    ds_pos_section = None
    for _, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        ax.scatter(
            ds_pos_section.tir_src_lon,
            ds_pos_section.tir_src_lat,
            color=color(ds_pos_section.attrs["section_id"]),
            label=f"{ds_pos_section.attrs['section_orientation']} {ds_pos_section.attrs['section_id']}",
        )

    if ds_pos_section is not None:
        # OBS positions are shared by all sections
        plot_obs_pos(ds_pos_section, ax=ax)

    return fig, ax


def plot_pressure_section(ds_pressure_section, single_fig=False):
    """Plot the pressure signal of every OBS of a section.

    Parameters
    ----------
    ds_pressure_section : xarray.Dataset
        Pressure dataset of a section (variables ``pressure_<obs_id>``).
    single_fig : bool, optional
        If True, all OBS are drawn on a single axis. Otherwise one subplot per
        OBS is created, sharing the x and y axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot.
    axs : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
        The single axis if ``single_fig`` is True, otherwise a flat array with
        one axis per OBS.
    """
    if single_fig:
        plt.figure()
        for i, obs_id in enumerate(ds_pressure_section.obs_id.values):
            ds_pressure_section[f"pressure_{obs_id}"].plot(
                label=obs_id.upper(), color=color(i)
            )
        plt.legend(loc="upper right")
        return plt.gcf(), plt.gca()

    fig, axs = plt.subplots(
        nrows=ds_pressure_section.obs_id.size,
        ncols=1,
        figsize=(16, 14),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    axs = axs.flatten()
    for i, obs_id in enumerate(ds_pressure_section.obs_id.values):
        ds_pressure_section[f"pressure_{obs_id}"].plot(
            label=obs_id.upper(), color=color(i), ax=axs[i]
        )
        axs[i].legend(loc="upper right")

    xlab = axs[0].get_xlabel()
    ylab = axs[0].get_ylabel()

    for ax in axs:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
    fig.supxlabel(xlab)
    fig.supylabel(ylab)

    return fig, axs


def plot_spectrogram_section(
    ds_pressure_section, nperseg=2**8, noverlap=2**7, cmap="magma"
):
    """Plot the spectrogram of every OBS of a section (one subplot per OBS).

    The STFT is computed with a Hann window and the power spectral density is
    displayed in dB re 1 V^2/Hz. All subplots share the same color scale (10th
    to 99th percentile of all spectrograms) and the time axis of the first OBS.

    Parameters
    ----------
    ds_pressure_section : xarray.Dataset
        Pressure dataset of a section, in volts.
    nperseg : int, optional
        Number of samples per STFT segment.
    noverlap : int, optional
        Number of overlapping samples between segments.
    cmap : str, optional
        Matplotlib colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Flat array with one axis per OBS.
    """
    sxx_plot = []
    tt_datetime = None
    ff = None

    for i, obs_id in enumerate(ds_pressure_section.obs_id.values):
        pressure = ds_pressure_section[f"pressure_{obs_id}"]
        # Derive stft
        ff, tt, stft = sp.stft(
            pressure.values,
            fs=ds_pressure_section.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",  # V^2 / Hz
        )
        sxx = 10 * np.log10(np.abs(stft) + 1e-6)  # dB re V**2 / Hz
        sxx_plot.append(sxx)

        # Datetime vector of the first OBS is used for the shared time axis
        if i == 0:
            t0 = pd.to_datetime(
                ds_pressure_section[f"time_{obs_id}"].values[0]
            ).to_pydatetime()
            tt_datetime = pd.to_datetime(t0) + pd.to_timedelta(tt, unit="s")

    # Plot
    sxx_plot = np.array(sxx_plot)
    vmin = np.percentile(sxx_plot, 10)
    vmax = np.percentile(sxx_plot, 99)

    fig, axs = plt.subplots(
        nrows=ds_pressure_section.obs_id.size,
        ncols=1,
        figsize=(16, 14),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    axs = axs.flatten()
    for i, _ in enumerate(ds_pressure_section.obs_id.values):
        im = axs[i].pcolormesh(
            tt_datetime, ff, sxx_plot[i, ...], cmap=cmap, vmin=vmin, vmax=vmax
        )
    clabel = r"dB re 1V$^2$ / Hz"
    fig.colorbar(
        im,
        ax=axs.ravel().tolist(),
        label=clabel,
        orientation="vertical",
        fraction=1.0,
        pad=0.03,
    )

    for ax in axs:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
    fig.supxlabel("Time")
    fig.supylabel("Frequency [Hz]")

    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axs[-1].xaxis.set_major_locator(locator)

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    return fig, axs
