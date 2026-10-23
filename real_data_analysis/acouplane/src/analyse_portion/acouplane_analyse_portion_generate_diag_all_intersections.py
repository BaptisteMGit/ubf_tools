#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_analyse_portion_generate_diag_all_intersections.py
@Time    :   2026/09/29 08:57:41
@Author  :   Menetrier Baptiste
@Version :   1.1
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Analyse of an ACOUPLANE portion: signals, features (RTF, DCF) and MFP
             ambiguity surfaces for every intersection of meridional / zonal sections.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

from publication.publication_figure import set_subfigures_abc_labels, color
from real_data_analysis.acouplane.src.acouplane_cst import (
    ACOUPLANE_PORTION_NC_ROOT_FPATH,
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR,
)
from real_data_analysis.acouplane.src.acouplane_utils import (
    open_portion,
    save_portion_inplace,
    apply_rcv_position_correction_to_portion,
    apply_mfp_strategy_to_portion_intersections,
    derive_portion_cpsd,
    derive_portion_propagation_time,
    plot_portion_tir_pos,
    plot_pressure_section,
    plot_spectrogram_section,
)

# ======================================================================================================================
# Constants
# ======================================================================================================================
portion_id = 5
ROOT_IMG_PORTION = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR, f"portion{portion_id}"
)
ROOT_IMG_PORTION_INTERSECTION = os.path.join(ROOT_IMG_PORTION, "intersections")

ROOT_IMG_PORTION_FEATURES = os.path.join(ROOT_IMG_PORTION, "features")
ROOT_IMG_PORTION_FEATURES_RTF = os.path.join(ROOT_IMG_PORTION_FEATURES, "rtf")
ROOT_IMG_PORTION_FEATURES_DCF = os.path.join(ROOT_IMG_PORTION_FEATURES, "dcf")

ROOT_IMG_PORTION_INPUT_SIGNAL = os.path.join(ROOT_IMG_PORTION, "signal")
ROOT_IMG_PORTION_INPUT_SIGNAL_PRESSURE = os.path.join(
    ROOT_IMG_PORTION_INPUT_SIGNAL, "pressure"
)
ROOT_IMG_PORTION_INPUT_SIGNAL_SPECTROGRAM = os.path.join(
    ROOT_IMG_PORTION_INPUT_SIGNAL, "spectrogram"
)

roots = [
    ROOT_IMG_PORTION,
    ROOT_IMG_PORTION_INTERSECTION,
    ROOT_IMG_PORTION_FEATURES,
    ROOT_IMG_PORTION_FEATURES_RTF,
    ROOT_IMG_PORTION_FEATURES_DCF,
    ROOT_IMG_PORTION_INPUT_SIGNAL,
    ROOT_IMG_PORTION_INPUT_SIGNAL_PRESSURE,
    ROOT_IMG_PORTION_INPUT_SIGNAL_SPECTROGRAM,
]
for root in roots:
    os.makedirs(root, exist_ok=True)

c0 = 1508  # Equivalent celerity derived from real profile

nfft = 2**15
offset_ms = 100
signal_duration_ms = 3000

reference_rcv = "obs1"
obs_ids = [
    "obs1",
    "obs2",
    "obs4",
    "obs5",
    "obs6",
    "obs7",
]
# Frequency band
fmin = 0
fmax = 200

# ======================================================================================================================
# Booleans to control behavior
# ======================================================================================================================
save_modification_to_portion = False
apply_obs_pos_correction = False
compute_cpsds = False
apply_mfp_strategy_intersections = True

plot_portion_position = True
plot_signal = False
plot_spectrogram = False
plot_signal_single_emission = False
plot_features = False
plot_intersections = True
verbose = True


# ======================================================================================================================
# Helper functions
# ======================================================================================================================
def log(message, level=0):
    """Print a progress message when ``verbose`` is True.

    Parameters
    ----------
    message : str
        Message to print.
    level : int, optional
        Indentation level: 0 for a main step (prefixed by ``[Portion <id>]``),
        1 or more for a detail of the current step.

    Returns
    -------
    None
    """
    if not verbose:
        return
    if level == 0:
        print(f"[Portion {portion_id}] {message}")
    else:
        print("    " * level + f"- {message}")


def section_name_of(node):
    """Build the display name of a section, e.g. ``"South/North 0"``.

    Parameters
    ----------
    node : xarray.DataTree
        Node of a section (in the ``position``, ``pressure`` or ``feature`` group).

    Returns
    -------
    str
        ``"<orientation> <id>"``.
    """
    return f"{node.attrs['section_orientation']} {node.attrs['section_id']}"


def get_intersection_target_position(ds_mfp_intersection):
    """Find the true intersection of two sections from the real-space distance.

    The "target" is the pair of shots (one on each section) which are the
    closest to each other in real space, i.e. the position where the two
    trajectories actually cross.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection (see ``build_mfp_intersection_dataset``),
        with the variable ``d_real`` of dimensions
        ``(src_datetime_section1, src_datetime_section2)``.

    Returns
    -------
    intersection_dt1 : numpy.datetime64
        Emission datetime, along section 1, of the closest shot pair.
    intersection_dt2 : numpy.datetime64
        Emission datetime, along section 2, of the closest shot pair.
    """
    idx_target_pos = ds_mfp_intersection.d_real.argmin(
        dim=["src_datetime_section1", "src_datetime_section2"]
    )
    intersection_dt1, intersection_dt2 = (
        ds_mfp_intersection.isel(idx_target_pos).src_datetime_section1.values,
        ds_mfp_intersection.isel(idx_target_pos).src_datetime_section2.values,
    )

    return intersection_dt1, intersection_dt2


def get_intersection_guess_positions(ds_mfp_intersection):
    """Find the estimated intersection for each feature-space distance.

    For each distance (``d_theta``, ``d_theta_gamma``, ``d_L2_gamma`` and
    ``d_dcf``) and each receiver pair, the pair of shots minimizing the distance
    is the estimated (guessed) intersection.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection, with the distances of dimensions
        ``(rcv_id_feature, src_datetime_section1, src_datetime_section2)``.

    Returns
    -------
    dict
        One entry per distance: ``"theta"``, ``"theta_gamma"``, ``"L2_gamma"``
        and ``"dcf"``. Each entry is a dict with the keys ``dt1`` and ``dt2``
        (xarray.DataArray of dimension ``rcv_id_feature``): emission datetimes of
        the estimated intersection along section 1 and section 2.
    """
    dims = ["src_datetime_section1", "src_datetime_section2"]
    metrics = dict(
        theta="d_theta",
        theta_gamma="d_theta_gamma",
        L2_gamma="d_L2_gamma",
        dcf="d_dcf",
    )

    intersections = {}
    for key, var in metrics.items():
        idx_guess_pos = ds_mfp_intersection[var].argmin(dim=dims)
        ds_guess = ds_mfp_intersection.isel(idx_guess_pos)
        intersections[key] = dict(
            dt1=ds_guess.src_datetime_section1,
            dt2=ds_guess.src_datetime_section2,
        )

    return intersections


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


# ======================================================================================================================
# Load data (lazy: nothing is read in memory until values are actually used)
# ======================================================================================================================
log(f"Step 1 - Opening file for portion {portion_id} (lazy, not loaded in memory)")
portion_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}.nc"
portion = open_portion(portion_fpath)
if verbose:
    print(portion)

obs_ids = np.asarray(obs_ids)

# NOTE : use corrected position (LMS estimation)
if apply_obs_pos_correction:
    log("Correcting OBS positions with the LMS estimation")
    apply_rcv_position_correction_to_portion(portion_tree=portion)

# ======================================================================================================================
# Plot portion positions
# ======================================================================================================================
if plot_portion_position:
    log("Plotting portion positions")
    fig, ax = plot_portion_tir_pos(portion_tree=portion)
    fpath = os.path.join(ROOT_IMG_PORTION, f"portion_{portion_id}_position.png")
    fig.savefig(fpath)
    plt.close(fig)
    log(f"Saved {fpath}", level=1)

# ======================================================================================================================
# Plot signals
# ======================================================================================================================
single_fig = False
if plot_signal:
    log("Plotting pressure signals of each section")
    for name, node in portion.pressure.children.items():
        log(f"Section '{name}'", level=1)
        ds_pressure_section = node.to_dataset()
        fig, axs = plot_pressure_section(
            ds_pressure_section=ds_pressure_section, single_fig=single_fig
        )
        fig.suptitle(section_name_of(node))
        fpath = os.path.join(
            ROOT_IMG_PORTION_INPUT_SIGNAL_PRESSURE,
            f"pressure_signal_{name}.png",
        )
        fig.savefig(fpath)
        plt.close(fig)

# ======================================================================================================================
# Plot spectrograms
# ======================================================================================================================
if plot_spectrogram:
    log("Plotting spectrograms of each section")
    for name, node in portion.pressure.children.items():
        log(f"Section '{name}'", level=1)
        ds_pressure_section = node.to_dataset()
        fig, axs = plot_spectrogram_section(
            ds_pressure_section, nperseg=2**8, noverlap=2**7, cmap="magma"
        )
        fig.suptitle(section_name_of(node))
        fpath = os.path.join(
            ROOT_IMG_PORTION_INPUT_SIGNAL_SPECTROGRAM,
            f"pressure_spectrogram_{name}.png",
        )
        fig.savefig(fpath)
        plt.close(fig)

# ======================================================================================================================
# Derive propagation time
# ======================================================================================================================
log(f"Step 2 - Estimating theoretical propagation times (c0 = {c0} m/s)")
derive_portion_propagation_time(portion_tree=portion, c0=c0)

# ======================================================================================================================
# Plot single emission and associated arrival time
# ======================================================================================================================
if plot_signal_single_emission:
    log("Plotting the first emission of each section with its theoretical arrival time")

    for name, node in portion.pressure.children.items():
        log(f"Section '{name}'", level=1)
        ds_pressure_section = node.to_dataset()
        # Associated ds_pos
        ds_pos_section = portion.position[name].to_dataset()
        tir_dt = ds_pos_section["tir_reception_datetime"].isel(src_datetime=slice(0, 1))

        fig, axs = plt.subplots(
            nrows=obs_ids.size, ncols=1, figsize=(16, 14), sharex=False, squeeze=False
        )
        axs = axs.flatten()
        for i, rcv_id in enumerate(obs_ids):
            tir_dt_rcv = tir_dt.sel(rcv_id=rcv_id)
            # Select associated data window
            press_rcv = ds_pressure_section[f"pressure_{rcv_id}"].sel(
                {
                    f"time_{rcv_id}": slice(
                        tir_dt_rcv.values[0] - np.timedelta64(offset_ms, "ms"),
                        tir_dt_rcv.values[0]
                        + np.timedelta64(signal_duration_ms - offset_ms, "ms"),
                    )
                }
            )

            # Normalize
            press_rcv = press_rcv / press_rcv.max()
            press_rcv.plot(label=rcv_id.upper(), color=color(i), ax=axs[i])
            # Plot arrivals time
            axs[i].vlines(x=tir_dt_rcv, ymin=-1, ymax=1, color="k", linestyle="--")
            # Add legend
            axs[i].legend(loc="upper right")

        xlab = axs[0].get_xlabel()
        ylab = axs[0].get_ylabel()

        for ax in axs:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")

        fig.suptitle(section_name_of(node))
        fig.supxlabel(xlab)
        fig.supylabel(ylab)

        fpath = os.path.join(
            ROOT_IMG_PORTION_INPUT_SIGNAL_PRESSURE,
            f"pressure_signal_single_emission_{name}.png",
        )
        fig.savefig(fpath)
        plt.close(fig)

# ======================================================================================================================
# CPSD
# ======================================================================================================================
if compute_cpsds:
    log(
        f"Step 3 - Computing CPSD (band {fmin}-{fmax} Hz, nfft = {nfft}, "
        f"window = {signal_duration_ms} ms, offset = {offset_ms} ms)"
    )
    derive_portion_cpsd(
        portion_tree=portion,
        offset_ms=offset_ms,
        signal_duration_s=signal_duration_ms / 1000,
        nfft=nfft,
        obs_ids=None,
        fmin=fmin,
        fmax=fmax,
    )
else:
    log("Step 3 - Computing CPSD: skipped (compute_cpsds = False), using stored CPSD")
    # Check that the CPSD has already been computed
    if "feature" not in portion.children:
        raise RuntimeError(
            f"No 'feature' group in '{portion_fpath}': the CPSD has never been computed "
            "for this portion. Set compute_cpsds = True (and save_modification_to_portion = True "
            "to keep the result)."
        )

# ======================================================================================================================
# Derive features and apply MFP to each intersection
# ======================================================================================================================
if apply_mfp_strategy_intersections:
    log("Step 4 - Applying the MFP strategy to each intersection")
    apply_mfp_strategy_to_portion_intersections(
        portion, rcv_ref=reference_rcv, fmin=fmin, fmax=fmax, verbose=verbose
    )
else:
    log("Step 4 - MFP strategy: skipped (apply_mfp_strategy_intersections = False)")

# ======================================================================================================================
# Plot features (RTF, DCF)
# ======================================================================================================================
if plot_features:
    log("Step 5 - Plotting features (RTF, DCF)")

    alpha_max = 0.5

    for name, node in portion.feature.children.items():
        log(f"Section '{name}'", level=1)
        ds_feature_section = node.to_dataset()

        # Check if frequency band match required one (in case of previous computation)
        if not (
            (fmin <= ds_feature_section.freq.min())
            and (fmax >= ds_feature_section.freq.max())
        ):
            print(
                f"    WARNING: stored frequency band [{ds_feature_section.freq.min().values:.2f}, "
                f"{ds_feature_section.freq.max().values:.2f}] Hz is not included in the requested "
                f"band [{fmin}, {fmax}] Hz: the stored band is used."
            )
        # Slice frequency band
        ds_feature_section = ds_feature_section.sel(freq=slice(fmin, fmax))

        # Check reference receiver
        stored_ref = ds_feature_section.attrs.get("reference_receiver")
        if stored_ref != reference_rcv:
            print(
                f"    WARNING: requested reference receiver '{reference_rcv}' differs from the "
                f"one stored in the portion ('{stored_ref}'): the stored one is used."
            )

        n_pairs = ds_feature_section.sizes["rcv_id_feature"]
        # RTF
        fig_gamma, axs_gamma = plt.subplots(
            nrows=n_pairs,
            ncols=1,
            figsize=(16, 16),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axs_gamma = axs_gamma.flatten()
        # DCF
        fig_dcf, axs_dcf = plt.subplots(
            nrows=n_pairs,
            ncols=2,
            figsize=(16, 16),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for iplot, rcv_id_i in enumerate(ds_feature_section.rcv_id_feature.values):

            ds_feat_rcv = ds_feature_section.sel(rcv_id_feature=rcv_id_i)

            # RTF Gamma
            gamma_max = np.nanmax(ds_feat_rcv.gamma_ij.values)
            ds_feat_rcv.gamma_ij.plot(
                x="freq",
                ax=axs_gamma[iplot],
                cmap="magma",
                vmin=-alpha_max * gamma_max,
                vmax=+alpha_max * gamma_max,
            )
            axs_gamma[iplot].set_xlabel("")
            axs_gamma[iplot].set_ylabel("")
            axs_gamma[iplot].set_title(rcv_id_i.upper(), fontsize=12)

            # DCF (sxy SCOT)
            re_sij = np.real(ds_feat_rcv.sij_scot)
            im_sij = np.imag(ds_feat_rcv.sij_scot)

            re_sij.plot(
                x="freq", ax=axs_dcf[iplot, 0], cmap="magma", vmin=-1.2, vmax=+1.2
            )
            im_sij.plot(
                x="freq", ax=axs_dcf[iplot, 1], cmap="magma", vmin=-1.2, vmax=+1.2
            )

            for icol in range(2):
                axs_dcf[iplot, icol].set_xlabel("")
                axs_dcf[iplot, icol].set_ylabel("")
                axs_dcf[iplot, icol].set_title(rcv_id_i.upper(), fontsize=12)

        fig_gamma.supxlabel("Frequency [Hz]")
        fig_gamma.suptitle(section_name_of(node))

        fpath = os.path.join(
            ROOT_IMG_PORTION_FEATURES_RTF,
            f"rtf_gamma_map_{name}.png",
        )
        fig_gamma.savefig(fpath)
        plt.close(fig_gamma)

        fig_dcf.supxlabel("Frequency [Hz]")
        fig_dcf.suptitle(section_name_of(node))
        axs_dcf[0, 0].set_title("Real part")
        axs_dcf[0, 1].set_title("Imaginary part")

        fpath = os.path.join(
            ROOT_IMG_PORTION_FEATURES_DCF,
            f"dcf_map_{name}.png",
        )
        fig_dcf.savefig(fpath)
        plt.close(fig_dcf)

# ======================================================================================================================
# Plot MFP results for each intersection
# ======================================================================================================================
if plot_intersections:
    if "mfp_intersection" not in portion.children:
        raise RuntimeError(
            "No 'mfp_intersection' group in the portion: nothing to plot. "
            "Set apply_mfp_strategy_intersections = True to compute the intersections."
        )

    log("Step 6 - Plotting the MFP ambiguity surfaces of each intersection")

    cmap = "Grays_r"
    max_quantile = 0.95

    # Iterate over meridional sections
    for section1_label, meridional_node in portion.mfp_intersection.children.items():
        # Iterate over zonal sections
        for section2_label, node in meridional_node.children.items():
            log(f"Intersection '{section1_label}' x '{section2_label}'", level=1)

            # Get intersection dataset
            ds_mfp_intersection = node.to_dataset()

            # Build sections names
            section1_name = section_name_of(portion.feature[section1_label])
            section2_name = section_name_of(portion.feature[section2_label])

            # Define root to save img for current intersections
            root_img_intersection = os.path.join(
                ROOT_IMG_PORTION_INTERSECTION, f"{section1_label}_{section2_label}"
            )
            os.makedirs(root_img_intersection, exist_ok=True)

            # Extract intersection target position (true intersection)
            target_dt = get_intersection_target_position(
                ds_mfp_intersection=ds_mfp_intersection
            )

            # Extract intersection estimated pos
            intersection_guess = get_intersection_guess_positions(
                ds_mfp_intersection=ds_mfp_intersection
            )

            for rcv_id in ds_mfp_intersection.rcv_id_feature.values:
                log(f"Receiver pair ({rcv_id}, {reference_rcv})", level=2)

                # Result for current rcv couple (rcv_i, rcv_ref)
                ds_mfp_intersection_rcv = ds_mfp_intersection.sel(rcv_id_feature=rcv_id)
                guess_dt = {
                    key: (
                        val["dt1"].sel(rcv_id_feature=rcv_id).values,
                        val["dt2"].sel(rcv_id_feature=rcv_id).values,
                    )
                    for key, val in intersection_guess.items()
                }

                fig = plot_intersection_ambiguity(
                    ds_mfp_intersection_rcv=ds_mfp_intersection_rcv,
                    target_dt=target_dt,
                    guess_dt=guess_dt,
                    section1_name=section1_name,
                    section2_name=section2_name,
                    cmap=cmap,
                    max_quantile=max_quantile,
                )

                fpath = os.path.join(
                    root_img_intersection,
                    f"mfp_ambiguity_4_metrics_{rcv_id}.png",
                )
                fig.savefig(fpath)
                plt.close(fig)

# ======================================================================================================================
# Save results (last step: the portion is closed by the save)
# ======================================================================================================================
if save_modification_to_portion:
    log(f"Step 7 - Saving results to '{portion_fpath}'")
    save_portion_inplace(portion, portion_fpath)
else:
    portion.close()
    log("Done (no modification saved to the portion file)")
