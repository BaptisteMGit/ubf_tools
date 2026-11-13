#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_plot_mfp.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Figures of the MFP ambiguity surfaces of an intersection.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import matplotlib.pyplot as plt

from publication.publication_figure import set_subfigures_abc_labels
from real_data_analysis.acouplane.src.acouplane_conventions import (
    get_mfp_axis_names,
)
from real_data_analysis.acouplane.src.acouplane_combination import (
    build_combined_mfp_dataset,
    format_rcv_combination,
)
from real_data_analysis.acouplane.src.acouplane_localization import (
    DISTANCE_TYPES,
    _argmin_2d,
)


# Metrics displayed by default next to the real distance, in the order of the panels (b), (c), (d)
DEFAULT_AMBIGUITY_METRICS = ("theta_gamma", "theta", "dcf")


def _check_ambiguity_metrics(metrics, ds_mfp_intersection):
    """Check the metrics displayed next to the real distance in an ambiguity figure.

    Parameters
    ----------
    metrics : sequence of str
        Distance types to display in the panels (b), (c) and (d), among
        ``DISTANCE_TYPES`` (at most 3, the real distance always occupies the
        panel (a)).
    ds_mfp_intersection : xarray.Dataset
        MFP dataset that must contain the corresponding variables ``d_<type>``.

    Returns
    -------
    list of str
        The metrics, as a list.

    Raises
    ------
    ValueError
        If there are no or more than 3 metrics, if a metric is unknown, or if
        it is not stored in the dataset (e.g. the centered distances need
        ``compare_centered_gamma = True``).
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)

    if not 1 <= len(metrics) <= 3:
        raise ValueError(
            f"Between 1 and 3 metrics can be displayed next to the real distance (got {metrics})."
        )

    unknown = [key for key in metrics if key not in DISTANCE_TYPES]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {unknown}. Valid metrics: {list(DISTANCE_TYPES)}."
        )

    missing = [key for key in metrics if f"d_{key}" not in ds_mfp_intersection]
    if missing:
        raise ValueError(
            f"The metric(s) {missing} are not stored in the intersection dataset "
            "(the centered distances need compare_centered_gamma = True when the MFP strategy is applied)."
        )

    return metrics


def ambiguity_figure_filename(metrics, suffix):
    """File name of an ambiguity figure, depending on the metrics displayed.

    Parameters
    ----------
    metrics : sequence of str
        Metrics displayed next to the real distance.
    suffix : str
        Part identifying the figure (receiver pair or combination of receivers).

    Returns
    -------
    str
        ``"mfp_ambiguity_4_metrics_<suffix>.png"`` for the default metrics,
        ``"mfp_ambiguity_<metric 1>_<metric 2>_..._<suffix>.png"`` otherwise.
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    if tuple(metrics) == DEFAULT_AMBIGUITY_METRICS:
        return f"mfp_ambiguity_4_metrics_{suffix}.png"

    return f"mfp_ambiguity_{'_'.join(metrics)}_{suffix}.png"


def _set_figure_title(fig, title):
    """Set the title of a figure above the panels, without covering them.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to title.
    title : str
        Title.

    Returns
    -------
    None
    """
    if fig.get_layout_engine() is not None:
        # The layout engine reserves the space of the title
        fig.suptitle(title)
    else:
        # Space is left above the panels and their (a), (b)... labels
        fig.subplots_adjust(top=0.88)
        fig.suptitle(title, y=0.99)


def plot_intersection_ambiguity(
    ds_mfp_intersection_rcv,
    target_dt,
    guess_dt,
    section1_name,
    section2_name,
    cmap="Grays_r",
    max_quantile=0.95,
    d_real_vmax=500,
    metrics=DEFAULT_AMBIGUITY_METRICS,
):
    """Plot the ambiguity surfaces of one intersection for one receiver pair.

    The figure is a 2x2 grid: the real distance (a) is always the first panel,
    followed by the chosen metrics (b), (c) and (d) (by default
    ``theta_gamma``, ``theta`` and DCF). The true intersection is marked by a
    red square on every panel and the estimated one by a magenta circle on the
    metric panels.

    Parameters
    ----------
    ds_mfp_intersection_rcv : xarray.Dataset
        MFP dataset of one intersection, selected for a single receiver pair.
    target_dt : tuple of numpy.datetime64
        True intersection ``(dt1, dt2)``, see ``get_intersection_target_position``.
    guess_dt : dict
        Estimated intersections for this receiver pair, one key per displayed
        metric, each a tuple ``(dt1, dt2)``.
    section1_name, section2_name : str
        Display names of the two sections, used in the axis labels.
    cmap : str, optional
        Matplotlib colormap name.
    max_quantile : float, optional
        Quantile of each feature distance used as upper bound of its color scale.
    d_real_vmax : float, optional
        Upper bound of the color scale of the real distance, in meters.
    metrics : sequence of str, optional
        Distance types displayed in the panels (b), (c) and (d), among
        ``"theta"``, ``"theta_gamma"``, ``"L2_gamma"``, ``"dcf"``,
        ``"theta_gamma_centered"`` and ``"L2_gamma_centered"`` (the centered
        ones need ``compare_centered_gamma = True``). If fewer than 3 are given,
        the remaining panels are hidden.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).

    Raises
    ------
    ValueError
        If a metric is invalid or not stored in the dataset.
    """
    metrics = _check_ambiguity_metrics(metrics, ds_mfp_intersection_rcv)

    fig, axs = plt.subplots(
        nrows=2, ncols=2, figsize=(16, 16), sharex=True, sharey=True
    )

    # Dimension displayed along the x axis (side 1 of the intersection)
    x_dim = get_mfp_axis_names(ds_mfp_intersection_rcv)["dim1"]

    # Plot ambiguity surfaces
    ds_mfp_intersection_rcv.d_real.plot(
        x=x_dim, cmap=cmap, ax=axs[0, 0], vmin=0, vmax=d_real_vmax
    )
    metric_axes = [axs[0, 1], axs[1, 0], axs[1, 1]]
    panels = [(f"d_{key}", ax, key) for key, ax in zip(metrics, metric_axes)]

    # Unused panels are hidden
    for ax in metric_axes[len(metrics):]:
        ax.set_visible(False)

    for var, ax, guess_key in panels:
        da = ds_mfp_intersection_rcv[var]
        da.plot(
            x=x_dim,
            cmap=cmap,
            ax=ax,
            vmin=da.min(),
            vmax=da.quantile(q=max_quantile),
        )

    # Add target position
    for ax in [axs[0, 0]] + [panel_ax for _, panel_ax, _ in panels]:
        ax.scatter(
            target_dt[0],
            target_dt[1],
            marker="s",
            s=400,
            facecolors="none",
            edgecolors="r",
            label="Target",
            linewidth=3,
        )

    # Add guess position
    for _, ax, guess_key in panels:
        # No estimate (e.g. surface without any valid value): nothing to mark
        if guess_dt.get(guess_key) is None:
            continue

        ax.scatter(
            guess_dt[guess_key][0],
            guess_dt[guess_key][1],
            marker="o",
            s=200,
            facecolors="none",
            edgecolors="m",
            label="Guess",
            linewidth=3,
        )

    for ax in axs.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    fig.supxlabel(f"Time along {section1_name} trajectory")
    fig.supylabel(f"Time along {section2_name} trajectory")

    return fig


def plot_intersection_gamma_comparison(
    ds_mfp_intersection_rcv,
    target_dt,
    guess_dt,
    section1_name,
    section2_name,
    cmap="Grays_r",
    max_quantile=0.95,
):
    """Compare the gamma distances with and without centering of gamma.

    The figure is a 2x2 grid: ``theta_gamma`` (a) and ``theta_gamma`` with
    centered gamma (b) on the first row, ``L2_gamma`` (c) and ``L2_gamma`` with
    centered gamma (d) on the second row. The true intersection is marked by a
    red square and the estimated one by a magenta circle. For each row, both
    panels share the same color scale (minimum of the two, upper quantile of the
    two) so that they can be compared directly.

    Parameters
    ----------
    ds_mfp_intersection_rcv : xarray.Dataset
        MFP dataset of one intersection, selected for a single receiver pair,
        containing the centered distances (``compare_centered_gamma = True``).
    target_dt : tuple of numpy.datetime64
        True intersection ``(dt1, dt2)``, see ``get_intersection_target_position``.
    guess_dt : dict
        Estimated intersections for this receiver pair, with the keys
        ``theta_gamma``, ``theta_gamma_centered``, ``L2_gamma`` and
        ``L2_gamma_centered``, each a tuple ``(dt1, dt2)``.
    section1_name, section2_name : str
        Display names of the two sections, used in the axis labels.
    cmap : str, optional
        Matplotlib colormap name.
    max_quantile : float, optional
        Quantile used as upper bound of the color scale.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    fig, axs = plt.subplots(
        nrows=2, ncols=2, figsize=(16, 16), sharex=True, sharey=True
    )

    # Dimension displayed along the x axis (side 1 of the intersection)
    x_dim = get_mfp_axis_names(ds_mfp_intersection_rcv)["dim1"]

    rows = [
        (
            axs[0],
            [
                ("d_theta_gamma", "theta_gamma"),
                ("d_theta_gamma_centered", "theta_gamma_centered"),
            ],
        ),
        (
            axs[1],
            [("d_L2_gamma", "L2_gamma"), ("d_L2_gamma_centered", "L2_gamma_centered")],
        ),
    ]
    for row_axs, panels in rows:
        das = [ds_mfp_intersection_rcv[var] for var, _ in panels]
        vmin = min(float(da.min()) for da in das)
        vmax = max(float(da.quantile(q=max_quantile)) for da in das)
        for ax, da, (_, guess_key) in zip(row_axs, das, panels):
            da.plot(x=x_dim, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax)
            ax.scatter(
                target_dt[0],
                target_dt[1],
                marker="s",
                s=400,
                facecolors="none",
                edgecolors="r",
                label="Target",
                linewidth=3,
            )

            # No estimate (e.g. surface without any valid value): nothing to mark
            if guess_dt.get(guess_key) is None:
                continue

            ax.scatter(
                guess_dt[guess_key][0],
                guess_dt[guess_key][1],
                marker="o",
                s=200,
                facecolors="none",
                edgecolors="m",
                label="Guess",
                linewidth=3,
            )

    for ax in axs.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    fig.supxlabel(f"Time along {section1_name} trajectory")
    fig.supylabel(f"Time along {section2_name} trajectory")

    return fig


def _combined_surfaces_and_positions(ds_mfp_intersection, rcv_combination, method):
    """Combined ambiguity surfaces of a combination of receivers, with the true and estimated intersections.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection (see ``build_mfp_intersection_dataset``).
    rcv_combination : sequence of str
        Receivers to combine.
    method : str
        Combination approach, see ``combine_distances_over_receivers``.

    Returns
    -------
    ds_combined : xarray.Dataset
        Combined surfaces, see ``build_combined_mfp_dataset``.
    target_dt : tuple of numpy.datetime64
        True intersection ``(dt1, dt2)`` (minimum of the real-space distance).
    guess_dt : dict
        Estimated intersection ``(dt1, dt2)`` of each distance type present
        (minimum of the combined surface); None if the surface has no valid value.
    """
    ds_combined = build_combined_mfp_dataset(
        ds_mfp_intersection, rcv_combination, method=method
    )
    axis_names = get_mfp_axis_names(ds_combined)
    dt1 = ds_combined[axis_names["dim1"]].values
    dt2 = ds_combined[axis_names["dim2"]].values

    # True intersection: pair of shots closest in real space
    target = _argmin_2d(ds_combined["d_real"].values)
    if target is None:
        raise ValueError(
            "The real-space distance has no valid value: the true intersection "
            "cannot be determined."
        )
    target_dt = (dt1[target[0]], dt2[target[1]])

    # Estimated intersection: pair of shots minimizing the combined surface
    guess_dt = {}
    for key in DISTANCE_TYPES:
        if f"d_{key}" not in ds_combined:
            continue
        guess = _argmin_2d(ds_combined[f"d_{key}"].values)
        guess_dt[key] = None if guess is None else (dt1[guess[0]], dt2[guess[1]])

    return ds_combined, target_dt, guess_dt


def plot_combined_intersection_ambiguity(
    ds_mfp_intersection,
    rcv_combination,
    section1_name,
    section2_name,
    method="mean_surfaces",
    cmap="Grays_r",
    max_quantile=0.95,
    d_real_vmax=500,
    metrics=DEFAULT_AMBIGUITY_METRICS,
):
    """Plot the 4 ambiguity surfaces of one intersection for a combination of receivers.

    Same figure as ``plot_intersection_ambiguity`` (real distance and the chosen
    metrics), computed on the surfaces combined over the receivers of
    the combination. The title gives the combination.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection, with all the receivers to combine.
    rcv_combination : sequence of str
        Receivers to combine, e.g. ``("obs2", "obs4")``.
    section1_name, section2_name : str
        Display names of the two sections, used in the axis labels.
    method : {"mean_surfaces"}, optional
        Combination approach, see ``combine_distances_over_receivers``.
    cmap : str, optional
        Matplotlib colormap name.
    max_quantile : float, optional
        Quantile of each feature distance used as upper bound of its color scale.
    d_real_vmax : float, optional
        Upper bound of the color scale of the real distance, in meters.
    metrics : sequence of str, optional
        Distance types displayed next to the real distance, see
        ``plot_intersection_ambiguity``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    ds_combined, target_dt, guess_dt = _combined_surfaces_and_positions(
        ds_mfp_intersection, rcv_combination, method
    )

    fig = plot_intersection_ambiguity(
        ds_mfp_intersection_rcv=ds_combined,
        target_dt=target_dt,
        guess_dt=guess_dt,
        section1_name=section1_name,
        section2_name=section2_name,
        cmap=cmap,
        max_quantile=max_quantile,
        d_real_vmax=d_real_vmax,
        metrics=metrics,
    )
    _set_figure_title(
        fig, f"Combination {format_rcv_combination(rcv_combination)} ({method})"
    )

    return fig


def plot_combined_intersection_gamma_comparison(
    ds_mfp_intersection,
    rcv_combination,
    section1_name,
    section2_name,
    method="mean_surfaces",
    cmap="Grays_r",
    max_quantile=0.95,
):
    """Compare the gamma distances with and without centering, for a combination of receivers.

    Same figure as ``plot_intersection_gamma_comparison``, computed on the
    surfaces combined over the receivers of the combination.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection, with all the receivers to combine and
        the centered distances (``compare_centered_gamma = True``).
    rcv_combination : sequence of str
        Receivers to combine, e.g. ``("obs2", "obs4")``.
    section1_name, section2_name : str
        Display names of the two sections, used in the axis labels.
    method : {"mean_surfaces"}, optional
        Combination approach, see ``combine_distances_over_receivers``.
    cmap : str, optional
        Matplotlib colormap name.
    max_quantile : float, optional
        Quantile used as upper bound of the color scale.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).

    Raises
    ------
    ValueError
        If the centered distances are not stored in ``ds_mfp_intersection``.
    """
    if "d_theta_gamma_centered" not in ds_mfp_intersection:
        raise ValueError(
            "The centered gamma distances are not stored in the intersection: set "
            "compare_centered_gamma = True and apply the MFP strategy again."
        )

    ds_combined, target_dt, guess_dt = _combined_surfaces_and_positions(
        ds_mfp_intersection, rcv_combination, method
    )

    fig = plot_intersection_gamma_comparison(
        ds_mfp_intersection_rcv=ds_combined,
        target_dt=target_dt,
        guess_dt=guess_dt,
        section1_name=section1_name,
        section2_name=section2_name,
        cmap=cmap,
        max_quantile=max_quantile,
    )
    _set_figure_title(
        fig, f"Combination {format_rcv_combination(rcv_combination)} ({method})"
    )

    return fig
