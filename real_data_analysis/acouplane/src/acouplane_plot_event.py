#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_plot_event.py
@Time    :   2026/10/30 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.1
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Figures of the comparison between a library and events: positions, features.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from real_data_analysis.acouplane.src.acouplane_plot_portion import (
    plot_portion_tir_pos,
)


def add_events_to_map(ax, portion_event, legend_ncols=2, legend_fontsize=16):
    """Add the selection box and the trajectories of the events to a map in lon / lat.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map (longitude on x, latitude on y).
    portion_event : xarray.DataTree
        Event tree with a ``position`` group (``lon`` / ``lat`` of each event)
        and, in its attributes, the selection box ``box_lon_min``,
        ``box_lon_max``, ``box_lat_min`` and ``box_lat_max``.
    legend_ncols : int, optional
        Number of columns of the legend.
    legend_fontsize : int, optional
        Font size of the legend.

    Returns
    -------
    None
    """
    # Selection box of the events
    attrs = portion_event.attrs
    box_keys = ("box_lon_min", "box_lon_max", "box_lat_min", "box_lat_max")
    if all(key in attrs for key in box_keys):
        ax.add_patch(
            Rectangle(
                (attrs["box_lon_min"], attrs["box_lat_min"]),
                attrs["box_lon_max"] - attrs["box_lon_min"],
                attrs["box_lat_max"] - attrs["box_lat_min"],
                fc="none",
                ec="k",
                lw=5,
                label="Selected area",
            )
        )

    # Trajectory of each event
    for event_name, node in portion_event.position.children.items():
        ds_position = node.to_dataset()
        event_id = ds_position.attrs.get("event_id", event_name)
        ax.plot(
            ds_position.lon,
            ds_position.lat,
            "-x",
            markersize=2,
            label=f"Event {event_id}",
        )

    # Legend on the upper right corner, without the duplicated entries
    ax.legend()
    legend = ax.get_legend()
    if legend is not None:
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()
        ax.legend(
            handles,
            labels,
            loc="upper right",
            fontsize=legend_fontsize,
            ncols=legend_ncols,
        )


def plot_library_and_event_map(portion, portion_event):
    """Plot the sections of the portion (library grid) and the trajectories of the events.

    Parameters
    ----------
    portion : xarray.DataTree
        Portion tree with the shot positions of every section (library grid).
    portion_event : xarray.DataTree
        Event tree, see ``add_events_to_map``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure (not closed).
    ax : matplotlib.axes.Axes
        The map.
    """
    fig, ax = plot_portion_tir_pos(portion_tree=portion)
    add_events_to_map(ax, portion_event)

    return fig, ax


def plot_event_features(ds_feature_event, rtf_amplitude_max=2, dcf_max=1.2):
    """Plot the RTF amplitude and the DCF of an event, for each receiver pair.

    Parameters
    ----------
    ds_feature_event : xarray.Dataset
        Feature dataset of an event, with the variables ``pi_ij`` and
        ``sij_scot`` of dimensions ``(rcv_id_feature, event_datetime, freq)``.
    rtf_amplitude_max : float, optional
        Upper bound of the color scale of the RTF amplitude.
    dcf_max : float, optional
        Bound of the color scale of the real and imaginary parts of the DCF.

    Returns
    -------
    fig_rtf : matplotlib.figure.Figure
        RTF amplitude, one panel per receiver pair (not closed).
    fig_dcf : matplotlib.figure.Figure
        Real (left) and imaginary (right) parts of the DCF, one row per
        receiver pair (not closed).
    """
    rcv_ids = ds_feature_event.rcv_id_feature.values
    n_pairs = len(rcv_ids)

    # RTF: one panel per receiver pair
    fig_rtf, axs_rtf = plt.subplots(
        nrows=n_pairs, ncols=1, figsize=(16, 16), sharex=True, sharey=True, squeeze=False
    )
    axs_rtf = axs_rtf.flatten()

    # DCF: real and imaginary parts
    fig_dcf, axs_dcf = plt.subplots(
        nrows=n_pairs, ncols=2, figsize=(16, 16), sharex=True, sharey=True, squeeze=False
    )

    for i_pair, rcv_id in enumerate(rcv_ids):
        ds_feature_rcv = ds_feature_event.sel(rcv_id_feature=rcv_id)

        # Amplitude of the RTF
        np.abs(ds_feature_rcv.pi_ij).plot(
            x="freq", ax=axs_rtf[i_pair], cmap="magma", vmin=0, vmax=rtf_amplitude_max
        )
        axs_rtf[i_pair].set_xlabel("")
        axs_rtf[i_pair].set_ylabel("")
        axs_rtf[i_pair].set_title(str(rcv_id).upper(), fontsize=12)

        # Real and imaginary parts of the DCF
        np.real(ds_feature_rcv.sij_scot).plot(
            x="freq", ax=axs_dcf[i_pair, 0], cmap="magma", vmin=-dcf_max, vmax=+dcf_max
        )
        np.imag(ds_feature_rcv.sij_scot).plot(
            x="freq", ax=axs_dcf[i_pair, 1], cmap="magma", vmin=-dcf_max, vmax=+dcf_max
        )
        for i_col in range(2):
            axs_dcf[i_pair, i_col].set_xlabel("")
            axs_dcf[i_pair, i_col].set_ylabel("")
            axs_dcf[i_pair, i_col].set_title(str(rcv_id).upper(), fontsize=12)

    fig_rtf.supxlabel("Frequency [Hz]")
    fig_dcf.supxlabel("Frequency [Hz]")
    axs_dcf[0, 0].set_title("Real part")
    axs_dcf[0, 1].set_title("Imaginary part")

    return fig_rtf, fig_dcf
