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
from real_data_analysis.acouplane.src.acouplane_combination import (
    build_combined_mfp_dataset,
    format_rcv_combination,
)
from real_data_analysis.acouplane.src.acouplane_localization import (
    DISTANCE_TYPES,
    _argmin_2d,
)


def plot_intersection_ambiguity(
    ds_mfp_intersection_rcv,
    target_dt,
    guess_dt,
    section1_name,
    section2_name,
    cmap="Grays_r",
    max_quantile=0.95,
):
    """Plot the 4 ambiguity surfaces of one intersection for one receiver pair.

    The figure is a 2x2 grid: real distance (a), ``theta_gamma`` (b), ``theta``
    (c) and DCF (d). The true intersection is marked by a red square on every
    panel and the estimated one by a magenta circle on the feature panels. The
    L2 distance on gamma is computed but not displayed.

    Parameters
    ----------
    ds_mfp_intersection_rcv : xarray.Dataset
        MFP dataset of one intersection, selected for a single receiver pair.
    target_dt : tuple of numpy.datetime64
        True intersection ``(dt1, dt2)``, see ``get_intersection_target_position``.
    guess_dt : dict
        Estimated intersections for this receiver pair, with the keys ``theta``,
        ``theta_gamma`` and ``dcf``, each a tuple ``(dt1, dt2)``.
    section1_name, section2_name : str
        Display names of the two sections, used in the axis labels.
    cmap : str, optional
        Matplotlib colormap name.
    max_quantile : float, optional
        Quantile of each feature distance used as upper bound of its color scale.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (not closed).
    """
    fig, axs = plt.subplots(
        nrows=2, ncols=2, figsize=(16, 16), sharex=True, sharey=True
    )

    # Plot ambiguity surfaces
    ds_mfp_intersection_rcv.d_real.plot(
        x="src_datetime_section1", cmap=cmap, ax=axs[0, 0], vmin=0, vmax=500
    )
    panels = [
        ("d_theta_gamma", axs[0, 1], "theta_gamma"),
        ("d_theta", axs[1, 0], "theta"),
        ("d_dcf", axs[1, 1], "dcf"),
    ]
    for var, ax, guess_key in panels:
        da = ds_mfp_intersection_rcv[var]
        da.plot(
            x="src_datetime_section1",
            cmap=cmap,
            ax=ax,
            vmin=da.min(),
            vmax=da.quantile(q=max_quantile),
        )

    # Add target position
    for ax in axs.flatten():
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
            da.plot(x="src_datetime_section1", cmap=cmap, ax=ax, vmin=vmin, vmax=vmax)
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
    dt1 = ds_combined["src_datetime_section1"].values
    dt2 = ds_combined["src_datetime_section2"].values

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
):
    """Plot the 4 ambiguity surfaces of one intersection for a combination of receivers.

    Same figure as ``plot_intersection_ambiguity`` (real distance, ``theta_gamma``,
    ``theta`` and DCF), computed on the surfaces combined over the receivers of
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
    )
    fig.suptitle(f"Combination {format_rcv_combination(rcv_combination)} ({method})", y=0.94)

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
    fig.suptitle(f"Combination {format_rcv_combination(rcv_combination)} ({method})", y=0.94)

    return fig
