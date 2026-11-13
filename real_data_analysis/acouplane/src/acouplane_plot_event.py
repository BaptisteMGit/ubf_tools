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


def _section_boundaries(library_sections):
    """Boundaries and centers of the consecutive sections of a library block.

    Parameters
    ----------
    library_sections : numpy.ndarray
        Section label of each sample of the block.

    Returns
    -------
    names : list of str
        Section names, in the order of the block.
    centers : list of float
        Index at the center of each section.
    boundaries : list of float
        Index of the limit between two consecutive sections (``n_sections - 1`` values).
    """
    labels = np.asarray(library_sections).astype(str)
    names, centers, boundaries = [], [], []
    start = 0
    for index in range(1, labels.size + 1):
        if index == labels.size or labels[index] != labels[start]:
            names.append(labels[start])
            centers.append((start + index - 1) / 2)
            if index < labels.size:
                boundaries.append(index - 0.5)
            start = index
    return names, centers, boundaries


def plot_library_block_ambiguity(
    ds_surfaces,
    records,
    metrics,
    cmap="Grays_r",
    max_quantile=0.95,
    d_real_vmax=500,
    event_label="Test section",
    title=None,
):
    """Plot the ambiguity surfaces of the whole library (single block) against an event.

    The figure is a 2x2 grid: the real distance (a) is always the first panel,
    followed by the chosen metrics. The x axis is the library block (the
    sections one after the other, separated by dashed lines and named at the
    top), the y axis the samples of the event. The minima found by
    ``library_block_minima`` are marked: green circles for the minima of a
    feature surface that coincide with a minimum of the real distance
    (consistent), red crosses for those that do not (spurious), and red
    squares for the minima of the real distance on every panel.

    Parameters
    ----------
    ds_surfaces : xarray.Dataset
        Surfaces, see ``compute_block_ambiguity``.
    records : list of dict
        Minima, see ``library_block_minima``.
    metrics : sequence of str
        1 to 3 distance types displayed in the panels (b), (c) and (d).
    cmap : str, optional
        Matplotlib colormap name.
    max_quantile : float, optional
        Quantile of each feature distance used as upper bound of its color scale.
    d_real_vmax : float, optional
        Upper bound of the color scale of the real distance, in meters.
    event_label : str, optional
        Name of the event, used in the y axis label.
    title : str or None, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).

    Raises
    ------
    ValueError
        If the number of metrics is not between 1 and 3, or a metric is missing.
    """
    from real_data_analysis.acouplane.src.acouplane_plot_mfp import _set_figure_title
    from publication.publication_figure import set_subfigures_abc_labels

    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    if not 1 <= len(metrics) <= 3:
        raise ValueError(f"Between 1 and 3 metrics can be displayed (got {metrics}).")
    missing = [key for key in metrics if f"d_{key}" not in ds_surfaces]
    if missing:
        raise ValueError(f"Metric(s) {missing} not in the surfaces.")

    n_library, n_event = ds_surfaces["d_real"].values.shape
    section_names, section_centers, section_boundaries = _section_boundaries(
        ds_surfaces["library_section"].values
    )

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 16), sharex=True, sharey=True)
    panel_axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    panel_vars = ["d_real"] + [f"d_{key}" for key in metrics]
    panel_kinds = ["real"] + list(metrics)
    for ax in panel_axes[len(panel_vars):]:
        ax.set_visible(False)

    # Sample indices on both axes (the datetimes of different sections can overlap)
    x_edges = np.arange(n_library + 1) - 0.5
    y_edges = np.arange(n_event + 1) - 0.5

    for ax, var, kind in zip(panel_axes, panel_vars, panel_kinds):
        values = np.ma.masked_invalid(np.asarray(ds_surfaces[var].values, dtype=float).T)
        if kind == "real":
            vmin, vmax = 0, d_real_vmax
        else:
            finite = values.compressed()
            vmin = float(finite.min()) if finite.size else 0
            vmax = float(np.quantile(finite, max_quantile)) if finite.size else 1
        mesh = ax.pcolormesh(x_edges, y_edges, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")

        # Colorbar label from the attributes of the variable
        attrs = ds_surfaces[var].attrs
        label = attrs.get("long_name", var)
        if "units" in attrs:
            label += f" [{attrs['units']}]"
        fig.colorbar(mesh, ax=ax, label=label, shrink=0.8)

        # Limits between the library sections
        for boundary in section_boundaries:
            ax.axvline(boundary, color="tab:blue", linestyle="--", linewidth=1)

        # Minima of the real distance on every panel
        real_minima = [r for r in records if r["kind"] == "real"]
        ax.scatter(
            [r["i_library"] for r in real_minima],
            [r["j_event"] for r in real_minima],
            marker="s", s=400, facecolors="none", edgecolors="r", linewidth=3, label="Real distance minimum",
        )

        # Minima of the feature surface
        if kind != "real":
            for r in (r for r in records if r["kind"] == kind):
                if r["consistent"]:
                    ax.scatter(r["i_library"], r["j_event"], marker="o", s=200, facecolors="none",
                               edgecolors="lime", linewidth=3)
                else:
                    ax.scatter(r["i_library"], r["j_event"], marker="x", s=200, color="red", linewidth=3)

    # Section names above the first row
    top_ax = axs[0, 0]
    secax = top_ax.secondary_xaxis("top")
    secax.set_xticks(section_centers)
    secax.set_xticklabels(section_names, rotation=45, ha="left")
    secax.tick_params(length=0)

    for ax in axs.flatten():
        ax.set_xlim(-0.5, n_library - 0.5)
        ax.set_ylim(-0.5, n_event - 0.5)

    set_subfigures_abc_labels(axs, x_pos=0.5, y_pos=1.12, fontsize=20, ha="center", va="bottom")
    fig.supxlabel("Library sample (sections one after the other)")
    fig.supylabel(f"{event_label} sample")
    if title:
        _set_figure_title(fig, title)

    return fig
