#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_plot_analysis.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Figures of the analysis of the results: position error, receiver pairs, frequency, contrast, combinations.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np

from publication.publication_figure import color
from real_data_analysis.acouplane.src.acouplane_localization import (
    get_distance_type_label,
    pivot_error_summary,
)


def _finalize_figure(fig, title=None):
    """Set the title of a figure (if any) and adjust its layout.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finalize.
    title : str or None, optional
        Title of the figure.
    """
    if title:
        fig.suptitle(title)

    # A layout engine already set (e.g. constrained layout enabled in the
    # matplotlib rcParams) is kept: switching engine fails when a colorbar exists.
    if fig.get_layout_engine() is not None:
        return

    if title:
        # Leave room for the title
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()


def _make_panel_grid(n_panels, max_cols=3, panel_size=(5.5, 4.5), **subplots_kwargs):
    """Create a grid of subplots with at least ``n_panels`` axes; the extra ones are hidden.

    Parameters
    ----------
    n_panels : int
        Number of panels needed.
    max_cols : int, optional
        Maximum number of columns.
    panel_size : tuple of float, optional
        Size of one panel ``(width, height)`` in inches.
    **subplots_kwargs
        Passed to ``matplotlib.pyplot.subplots`` (e.g. ``sharey=True``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : numpy.ndarray of matplotlib.axes.Axes
        Flat array of ``n_panels`` visible axes (followed by hidden ones).
    """
    # Grid with at least n_panels axes
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
        **subplots_kwargs,
    )
    axs = axs.flatten()

    # Hide the unused axes
    for ax in axs[n_panels:]:
        ax.set_visible(False)

    return fig, axs


def plot_error_boxplots_per_rcv_pair(df_errors, distance_types, title=None):
    """Boxplots of the position error versus the distance type, one panel per receiver pair.

    Each box summarizes the errors of all the analysed intersections. Invalid
    errors (NaN) are ignored.

    Parameters
    ----------
    df_errors : pandas.DataFrame
        Records with the columns ``distance_type``, ``rcv_pair`` and ``error_m``.
    distance_types : list of str
        Distance types to display, in the order of the x axis.
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    rcv_pairs = list(df_errors["rcv_pair"].drop_duplicates())

    # One panel per receiver pair
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(rcv_pairs),
        figsize=(16, 8),
        sharey=True,
        squeeze=False,
    )
    axs = axs[0]

    for i_rcv, rcv_pair in enumerate(rcv_pairs):
        ax = axs[i_rcv]
        df_pair = df_errors[df_errors["rcv_pair"] == rcv_pair]

        for i_dist, distance_type in enumerate(distance_types, start=1):
            # Errors of all the intersections for this pair and distance type (NaN removed)
            values = (
                df_pair.loc[df_pair["distance_type"] == distance_type, "error_m"]
                .dropna()
                .to_numpy()
            )
            if values.size == 0:
                continue

            # Box of the distance type
            bp = ax.boxplot(
                [values],
                positions=[i_dist],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="k"),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color(i_rcv))
                patch.set_alpha(0.6)

        # X axis: one tick per distance type
        ax.set_xlim(0.5, len(distance_types) + 0.5)
        ax.set_xticks(range(1, len(distance_types) + 1))
        ax.set_xticklabels(
            [get_distance_type_label(key) for key in distance_types],
            rotation=45,
            ha="right",
        )
        ax.set_title(rcv_pair, fontsize=12)

    fig.supxlabel("Distance type", fontsize=16)
    fig.supylabel("Position error [m]", fontsize=16)
    _finalize_figure(fig, title)

    return fig


def plot_rcv_pair_performance_heatmaps(summary, success_threshold_m, title=None):
    """Compare the receiver pairs: median error and success rate for each distance type.

    Two heatmaps (receiver pairs in rows, distance types in columns), annotated
    with the values. The best pair(s) of each distance type are framed in red
    (no frame if all the pairs have the same value).

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of ``summarize_localization_errors``.
    success_threshold_m : float
        Threshold used to compute the success rate, only used in the title.
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    n_pairs = summary["rcv_pair"].nunique()

    # Two heatmaps side by side
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, max(5, 0.7 * n_pairs + 3.5)),
        squeeze=False,
    )

    # (statistic, panel title, colormap, annotation format, lower is better)
    panels = [
        ("median_error_m", "Median position error [m]", "viridis_r", "{:.0f}", True),
        (
            "success_rate",
            rf"Success rate (error $\leq$ {success_threshold_m:g} m)",
            "viridis",
            "{:.0%}",
            False,
        ),
    ]

    for ax, (value, panel_title, cmap, fmt, minimize) in zip(axs[0], panels):
        table = pivot_error_summary(summary, value)
        data = table.to_numpy(dtype=float)

        im = ax.imshow(data, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax)

        # Annotate each cell with its value (text color chosen for readability)
        for (i, j), val in np.ndenumerate(data):
            if not np.isfinite(val):
                text, text_color = "-", "k"
            else:
                r, g, b, _ = im.cmap(im.norm(val))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text, text_color = fmt.format(val), ("k" if luminance > 0.5 else "w")
            ax.text(j, i, text, ha="center", va="center", color=text_color)

        # Frame the best pair(s) of each distance type; nothing if all the pairs are tied
        for j in range(data.shape[1]):
            col = data[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                continue

            best_value = np.nanmin(col) if minimize else np.nanmax(col)
            is_best = finite & (col == best_value)
            if is_best.sum() == finite.sum():
                continue

            for i_best in np.flatnonzero(is_best):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i_best - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="r",
                        linewidth=3,
                    )
                )

        # Axes: distance types in columns, receiver pairs in rows
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(
            [get_distance_type_label(key) for key in table.columns],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(list(table.index))
        ax.set_title(panel_title, fontsize=14)

    _finalize_figure(fig, title)

    return fig


def plot_error_cdf(df_errors, distance_types, success_threshold_m, title=None):
    """Cumulative distribution of the position error, one panel per distance type.

    One curve per receiver pair. The fraction is computed over all the analysed
    intersections, so a curve that does not reach 1 means that some errors are
    larger than the x axis or invalid (NaN). The success threshold is drawn as a
    dashed vertical line.

    Parameters
    ----------
    df_errors : pandas.DataFrame
        Records with the columns ``distance_type``, ``rcv_pair`` and ``error_m``.
    distance_types : list of str
        Distance types to display (one panel each).
    success_threshold_m : float
        Success threshold, in meters.
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    rcv_pairs = list(df_errors["rcv_pair"].drop_duplicates())

    # Common x axis: up to the largest valid error
    all_errors = df_errors["error_m"].to_numpy(dtype=float)
    all_errors = all_errors[np.isfinite(all_errors)]
    x_max = 1.05 * all_errors.max() if all_errors.size else 1.0

    fig, axs = _make_panel_grid(len(distance_types), sharex=True, sharey=True)

    for ax, distance_type in zip(axs, distance_types):
        df_dist = df_errors[df_errors["distance_type"] == distance_type]

        for i_rcv, rcv_pair in enumerate(rcv_pairs):
            errors = df_dist.loc[df_dist["rcv_pair"] == rcv_pair, "error_m"].to_numpy(
                dtype=float
            )

            # Sorted valid errors; the fraction is relative to all the intersections (NaN included)
            valid = np.sort(errors[np.isfinite(errors)])
            if valid.size == 0:
                continue
            fraction = np.arange(1, valid.size + 1) / errors.size

            ax.step(
                np.r_[0.0, valid, x_max],
                np.r_[0.0, fraction, fraction[-1]],
                where="post",
                color=color(i_rcv),
                label=rcv_pair,
            )

        # Success threshold
        ax.axvline(success_threshold_m, color="k", linestyle="--", linewidth=1)
        ax.set_title(get_distance_type_label(distance_type), fontsize=12)
        ax.set_ylim(0, 1.02)

    axs[0].legend(loc="lower right", fontsize=9)
    fig.supxlabel("Position error [m]", fontsize=16)
    fig.supylabel("Fraction of intersections", fontsize=16)
    _finalize_figure(fig, title)

    return fig


def plot_error_vs_frequency_band(df_band_errors, distance_types, title=None):
    """Position error versus frequency band, one panel per distance type.

    One curve per receiver pair: mean of the position error over the analysed
    intersections, with the standard deviation as error bar. The bands are
    labelled by their limits.

    Parameters
    ----------
    df_band_errors : pandas.DataFrame
        Records with the columns ``distance_type``, ``rcv_pair``, ``error_m``,
        ``band_index``, ``band_label`` and ``band_center``.
    distance_types : list of str
        Distance types to display (one panel each).
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    # Bands in increasing order (x axis)
    bands = (
        df_band_errors[["band_index", "band_label", "band_center"]]
        .drop_duplicates()
        .sort_values("band_index")
    )

    # Mean and standard deviation over the intersections
    stats = (
        df_band_errors.groupby(["distance_type", "rcv_pair", "band_index"], sort=False)
        .agg(
            band_center=("band_center", "first"),
            mean_error=("error_m", "mean"),
            std_error=("error_m", lambda x: x.std(ddof=0)),
        )
        .reset_index()
    )
    rcv_pairs = list(df_band_errors["rcv_pair"].drop_duplicates())

    fig, axs = _make_panel_grid(len(distance_types), sharey=True)

    for ax, distance_type in zip(axs, distance_types):
        # One curve per receiver pair
        for i_rcv, rcv_pair in enumerate(rcv_pairs):
            sub = stats[
                (stats["distance_type"] == distance_type)
                & (stats["rcv_pair"] == rcv_pair)
            ].sort_values("band_index")
            if sub.empty:
                continue

            ax.errorbar(
                x=sub["band_center"],
                y=sub["mean_error"],
                yerr=sub["std_error"],
                color=color(i_rcv),
                marker="o",
                capsize=3,
                label=rcv_pair,
            )

        # Ticks labelled with the band limits
        ax.set_xticks(bands["band_center"])
        ax.set_xticklabels(bands["band_label"], rotation=45, ha="right")
        ax.set_title(get_distance_type_label(distance_type), fontsize=12)

    axs[0].legend(loc="best", fontsize=9)
    fig.supxlabel("Frequency band [Hz]", fontsize=16)
    fig.supylabel("Position error [m]", fontsize=16)
    _finalize_figure(fig, title)

    return fig


def plot_error_vs_bandwidth(df_bandwidth_errors, distance_types, title=None):
    """Position error versus the width of the frequency range, one panel per distance type.

    The ranges all start at the same lower limit and their upper limit
    increases. One curve per receiver pair: mean of the position error over the
    analysed intersections, with the standard deviation as error bar (NaN
    errors ignored).

    Parameters
    ----------
    df_bandwidth_errors : pandas.DataFrame
        Records with the columns ``distance_type``, ``rcv_pair``, ``error_m``,
        ``f_low`` and ``f_high``.
    distance_types : list of str
        Distance types to display (one panel each).
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    # Mean and standard deviation over the intersections
    stats = (
        df_bandwidth_errors.groupby(["distance_type", "rcv_pair", "f_high"], sort=False)
        .agg(
            mean_error=("error_m", "mean"),
            std_error=("error_m", lambda x: x.std(ddof=0)),
        )
        .reset_index()
    )
    rcv_pairs = list(df_bandwidth_errors["rcv_pair"].drop_duplicates())
    f_low = df_bandwidth_errors["f_low"].iloc[0]

    fig, axs = _make_panel_grid(len(distance_types), sharey=True)

    for ax, distance_type in zip(axs, distance_types):
        # One curve per receiver pair
        for i_rcv, rcv_pair in enumerate(rcv_pairs):
            sub = stats[
                (stats["distance_type"] == distance_type)
                & (stats["rcv_pair"] == rcv_pair)
            ].sort_values("f_high")
            if sub.empty:
                continue

            ax.errorbar(
                x=sub["f_high"],
                y=sub["mean_error"],
                yerr=sub["std_error"].fillna(0.0),
                color=color(i_rcv),
                marker="o",
                markersize=4,
                capsize=3,
                label=rcv_pair,
            )

        ax.set_title(get_distance_type_label(distance_type), fontsize=12)

    axs[0].legend(loc="best", fontsize=9)
    fig.supxlabel(
        f"Upper limit of the frequency range [{f_low:g}, f] [Hz]", fontsize=16
    )
    fig.supylabel("Position error [m]", fontsize=16)
    _finalize_figure(fig, title)

    return fig


def plot_contrast_vs_validity(
    df_records, distance_types, percentile=50.0, only_valid=False, title=None
):
    """Contrast of the ambiguity surface for each receiver pair, split by validity of the localization.

    One panel per distance type, receiver pairs on the x axis. For each pair,
    the contrast of the valid localizations (error not greater than the
    success threshold) and of the invalid ones are shown side by side, which
    tells whether the contrast is a usable quality indicator. If ``only_valid``
    is True, only the valid localizations are shown. NaN values are ignored.

    Parameters
    ----------
    df_records : pandas.DataFrame
        Records with the columns ``distance_type``, ``rcv_pair``, ``contrast``
        and ``valid`` (see ``add_validity_column``).
    distance_types : list of str
        Distance types to display (one panel each).
    percentile : float, optional
        Percentile used in the contrast (only for the axis label).
    only_valid : bool, optional
        If True, restrict the figure to the valid localizations.
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    rcv_pairs = list(df_records["rcv_pair"].drop_duplicates())

    # Groups drawn for each receiver pair: (validity, legend name, color index)
    if only_valid:
        groups = [(True, "valid", 0)]
        box_width, group_offset = 0.6, 0.0
    else:
        groups = [(True, "valid", 0), (False, "invalid", 1)]
        box_width, group_offset = 0.32, 0.2

    fig, axs = _make_panel_grid(len(distance_types), sharey=True)

    for ax, distance_type in zip(axs, distance_types):
        df_type = df_records[df_records["distance_type"] == distance_type]

        for i_pair, rcv_pair in enumerate(rcv_pairs):
            df_pair = df_type[df_type["rcv_pair"] == rcv_pair]

            for valid, _, i_group in groups:
                # Contrast of the valid (or invalid) localizations, NaN removed
                values = (
                    df_pair.loc[df_pair["valid"] == valid, "contrast"]
                    .dropna()
                    .to_numpy()
                )
                if values.size == 0:
                    continue

                # Boxes of the two groups are placed side by side around the pair position
                position = i_pair + 1
                if not only_valid:
                    position += (i_group - 0.5) * 2 * group_offset

                bp = ax.boxplot(
                    [values],
                    positions=[position],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="k"),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color(i_group))
                    patch.set_alpha(0.6)

        # X axis: one tick per receiver pair
        ax.set_xlim(0.5, len(rcv_pairs) + 0.5)
        ax.set_xticks(range(1, len(rcv_pairs) + 1))
        ax.set_xticklabels(rcv_pairs, rotation=45, ha="right")
        ax.set_title(get_distance_type_label(distance_type), fontsize=12)

    # Legend: color of each group
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color(i_group), alpha=0.6)
        for _, _, i_group in groups
    ]
    axs[0].legend(handles, [name for _, name, _ in groups], loc="best", fontsize=9)

    fig.supylabel(f"Contrast  (P{percentile:g} - min) / P{percentile:g}", fontsize=16)
    fig.supxlabel("Receiver pair", fontsize=16)
    _finalize_figure(fig, title)

    return fig


def plot_metric_vs_n_rcv(
    df_records, distance_types, value_col, ylabel, split_valid=False, title=None
):
    """Boxplots of a metric versus the number of combined receivers, one panel per distance type.

    Each box gathers the values of all the combinations having the same number
    of receivers, for all the analysed intersections. NaN values are ignored.

    Parameters
    ----------
    df_records : pandas.DataFrame
        Records with the columns ``distance_type``, ``n_rcv`` and ``value_col``
        (and ``valid`` if ``split_valid`` is True, see ``add_validity_column``).
    distance_types : list of str
        Distance types to display (one panel each).
    value_col : str
        Column to display (e.g. ``"error_m"`` or ``"contrast"``).
    ylabel : str
        Label of the y axis.
    split_valid : bool, optional
        If True, the valid and invalid localizations are drawn side by side.
    title : str or None, optional
        Title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    n_values = sorted(df_records["n_rcv"].unique())

    # Groups drawn for each number of receivers: (validity or None, legend name, color index)
    if split_valid:
        groups = [(True, "valid", 0), (False, "invalid", 1)]
        box_width, group_offset = 0.32, 0.2
    else:
        groups = [(None, "all", 0)]
        box_width, group_offset = 0.6, 0.0

    fig, axs = _make_panel_grid(len(distance_types), sharey=True)

    for ax, distance_type in zip(axs, distance_types):
        df_type = df_records[df_records["distance_type"] == distance_type]

        for i_n, n_rcv in enumerate(n_values, start=1):
            df_n = df_type[df_type["n_rcv"] == n_rcv]

            for valid, _, i_group in groups:
                # Values of the group (NaN removed)
                df_group = df_n if valid is None else df_n[df_n["valid"] == valid]
                values = df_group[value_col].dropna().to_numpy()
                if values.size == 0:
                    continue

                # Boxes of the groups are placed side by side around the position of n_rcv
                position = (
                    i_n + (i_group - 0.5) * 2 * group_offset if split_valid else i_n
                )
                bp = ax.boxplot(
                    [values],
                    positions=[position],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="k"),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color(i_group))
                    patch.set_alpha(0.6)

        # X axis: one tick per number of receivers
        ax.set_xlim(0.5, len(n_values) + 0.5)
        ax.set_xticks(range(1, len(n_values) + 1))
        ax.set_xticklabels([str(n_rcv) for n_rcv in n_values])
        ax.set_title(get_distance_type_label(distance_type), fontsize=12)

    # Legend: color of each group
    if split_valid:
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color(i_group), alpha=0.6)
            for _, _, i_group in groups
        ]
        axs[0].legend(handles, [name for _, name, _ in groups], loc="best", fontsize=9)

    fig.supxlabel("Number of combined receivers", fontsize=16)
    fig.supylabel(ylabel, fontsize=16)
    _finalize_figure(fig, title)

    return fig
