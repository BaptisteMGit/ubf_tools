#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_analyse_portion_generate_diag_all_intersections.py
@Time    :   2026/09/29 08:57:41
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Analyse of an ACOUPLANE portion: signals, features (RTF, DCF) and MFP
             ambiguity surfaces for every intersection of meridional / zonal sections.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from publication.publication_figure import color
from real_data_analysis.acouplane.src.acouplane_cst import (
    ACOUPLANE_PORTION_NC_ROOT_FPATH,
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR,
)
from real_data_analysis.acouplane.src.acouplane_console import (
    make_figure_saver,
    make_logger,
)
from real_data_analysis.acouplane.src.acouplane_frequency_bands import (
    build_cumulative_frequency_bands,
    build_frequency_bands,
    collect_errors_over_frequency_ranges,
    frequency_band_label,
)
from real_data_analysis.acouplane.src.acouplane_localization import (
    POSITION_ERROR_MODES,
    add_validity_column,
    ambiguity_contrast_from_mfp_dataset,
    best_rcv_pair_per_distance,
    errors_to_records,
    localization_errors_from_mfp_dataset,
    pivot_error_summary,
    select_and_order_distance_types,
    summarize_localization_errors,
)
from real_data_analysis.acouplane.src.acouplane_mfp import (
    apply_mfp_strategy_to_portion_intersections,
    get_intersection_guess_positions,
    get_intersection_target_position,
)
from real_data_analysis.acouplane.src.acouplane_plot_analysis import (
    plot_contrast_vs_validity,
    plot_error_boxplots_per_rcv_pair,
    plot_error_cdf,
    plot_error_vs_bandwidth,
    plot_error_vs_frequency_band,
    plot_rcv_pair_performance_heatmaps,
)
from real_data_analysis.acouplane.src.acouplane_plot_mfp import (
    ambiguity_figure_filename,
    plot_intersection_ambiguity,
    plot_intersection_gamma_comparison,
)
from real_data_analysis.acouplane.src.acouplane_plot_portion import (
    plot_portion_tir_pos,
    plot_pressure_section,
    plot_spectrogram_section,
)
from real_data_analysis.acouplane.src.acouplane_portion_io import (
    list_stored_intersections,
    open_portion,
    restrict_dataset_to_receivers,
    save_portion_inplace,
    section_name_of,
    select_intersections,
)
from real_data_analysis.acouplane.src.acouplane_preprocessing import (
    apply_rcv_position_correction_to_portion,
    derive_portion_cpsd,
    derive_portion_propagation_time,
)

# ======================================================================================================================
# Constants
# ======================================================================================================================
portion_id = 5
ROOT_IMG_PORTION = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR, f"portion{portion_id}"
)
ROOT_IMG_PORTION_INTERSECTION = os.path.join(ROOT_IMG_PORTION, "intersections")
ROOT_IMG_PORTION_RESULTS_SYNTHESIS = os.path.join(ROOT_IMG_PORTION, "results_synthesis")

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
    ROOT_IMG_PORTION_RESULTS_SYNTHESIS,
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

nfft = 2**13
offset_ms = 100
signal_duration_ms = 3000

reference_rcv = "obs1"
# Receivers used by the CPSD computation, the signal figures and the analysis of the results (the analysis only keeps
# the pairs (rcv_i, reference_rcv) with rcv_i in this list, even if more receivers are stored in the portion)
obs_ids = [
    "obs1",
    "obs2",
    "obs4",
    "obs5",
    "obs6",
    "obs7",
]
# Frequency band
fmin = 10
fmax = 800

# Intersections to process (MFP computation and ambiguity figures).
# None -> all intersections. Otherwise a list of (meridional, zonal) section labels, e.g.
intersections_to_process = [("sn2", "oe6")]
# , ("sn2", "oe7"), ("sn2", "oe8")
# intersections_to_process = None

# Analysis of the results: position error of the estimated intersection with respect to the true one.
# error_mode : shots used to measure the position error
#   "section1" -> shots of the meridional section (error mostly along North-South)
#   "section2" -> shots of the zonal section (error mostly along East-West)
#   "both"     -> sqrt(error_section1**2 + error_section2**2)
error_mode = "section1"
success_threshold_m = 100  # an intersection is successfully localized if its error <= threshold [m]
freq_bandwidth = 400  # width of the frequency bands of the band study [Hz]
# Distance types to analyse: None -> all those available, otherwise a list among "theta", "theta_gamma",
# "L2_gamma", "dcf", "theta_gamma_centered", "L2_gamma_centered" (the centered ones need compare_centered_gamma = True)

# Metrics displayed next to the real distance (always in the first panel) in the ambiguity figures: 3 among
# "theta", "theta_gamma", "L2_gamma", "dcf", "theta_gamma_centered", "L2_gamma_centered"
# (the centered ones need compare_centered_gamma = True), e.g. ["theta_gamma_centered", "L2_gamma_centered", "dcf"]
ambiguity_metrics = ["theta_gamma", "theta", "dcf"]
# It applies to all the analysis figures (position error, receiver pair comparison, frequency study, contrast).
distance_types_to_analyse = None
# Error versus width of the frequency range: ranges [fmin, f] with f = bandwidth_step, 2 * bandwidth_step, ... up to
# bandwidth_study_fmax (None -> fmax). The distances are recomputed for each range: the cost grows with the number of ranges.
bandwidth_step = 50  # [Hz]
bandwidth_study_fmax = None  # [Hz]
# Contrast of the ambiguity surface = (P - min) / P, P being the percentile `contrast_percentile` of the surface.
contrast_percentile = 50
# True -> the contrast figure only shows the valid localizations (error <= success_threshold_m); False -> valid and invalid
# localizations are shown side by side (recommended: it tells whether the contrast predicts the success of the localization)
contrast_only_valid = False

# ======================================================================================================================
# Booleans to control behavior
# ======================================================================================================================
save_modification_to_portion = False
apply_obs_pos_correction = False
compute_cpsds = False
apply_mfp_strategy_intersections = False

plot_portion_position = False
plot_signal = False
plot_spectrogram = False
plot_signal_single_emission = False
plot_features = False
plot_intersections = False
# Analysis of the results (position error of the estimated intersections)
analyse_error_vs_distance_type = True
analyse_error_vs_frequency_band = True
analyse_error_vs_bandwidth = True
analyse_ambiguity_contrast = True
analyse_rcv_pair_performance = True
# Also compute (and plot) the gamma distances with gamma centered along frequency
compare_centered_gamma = True
verbose = True

# ======================================================================================================================
# Console messages and figure saving
# ======================================================================================================================
log = make_logger(portion_id, verbose)
save_synthesis_figure = make_figure_saver(ROOT_IMG_PORTION_RESULTS_SYNTHESIS, log)

# Receivers listed in obs_ids that are missing from the stored results (warned only once)
warned_missing_receivers = set()


def restrict_to_selected_receivers(ds, reference):
    """Keep, in a dataset of features or MFP results, the receivers listed in ``obs_ids``.

    The portion can store more receivers than those selected by ``obs_ids`` (e.g. features computed
    earlier with other receivers), and the reference receiver has no pair with itself.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with the dimension ``rcv_id_feature``.
    reference : str
        Reference receiver of the pairs ``(rcv_id_feature, reference)``, never kept.

    Returns
    -------
    xarray.Dataset
        Dataset restricted to the selected receivers.

    Raises
    ------
    RuntimeError
        If none of the selected receivers is available in the dataset.
    """
    stored = [str(rcv_id) for rcv_id in ds["rcv_id_feature"].values]
    selected = [str(rcv_id) for rcv_id in obs_ids if str(rcv_id) != str(reference)]

    # Warn once about the receivers of obs_ids that are not stored
    missing = [rcv_id for rcv_id in selected if rcv_id not in stored]
    if missing and tuple(missing) not in warned_missing_receivers:
        warned_missing_receivers.add(tuple(missing))
        print(
            f"    WARNING: receivers {missing} are listed in obs_ids but are not stored in the "
            f"results (stored receivers: {stored}): they are not analysed."
        )

    # Keep the stored receivers selected by obs_ids
    try:
        ds_selected, _ = restrict_dataset_to_receivers(ds, obs_ids, reference=reference)
    except ValueError:
        raise RuntimeError(
            f"None of the receivers of obs_ids {selected} is stored in the results "
            f"(stored receivers: {stored}): nothing to analyse."
        )

    return ds_selected


# ======================================================================================================================
# Load data (lazy: nothing is read in memory until values are actually used)
# ======================================================================================================================
log(f"Step 1 - Opening file for portion {portion_id} (lazy, not loaded in memory)")
portion_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}.nc"
portion = open_portion(portion_fpath)
# if verbose:
#     print(portion)

obs_ids = np.asarray(obs_ids)

# Check the requested intersections right away (before any heavy computation)
selected_intersections = select_intersections(
    list(portion.position.children), intersections_to_process
)
log(
    f"{len(selected_intersections)} intersection(s) selected"
    + (" (all)" if intersections_to_process is None else f": {selected_intersections}"),
    level=1,
)

# Check the analysis settings right away (before any heavy computation)
if error_mode not in POSITION_ERROR_MODES:
    raise ValueError(
        f"Unknown error_mode '{error_mode}'. Valid modes: {list(POSITION_ERROR_MODES)}."
    )
if analyse_error_vs_frequency_band:
    freq_bands = build_frequency_bands(fmin, fmax, freq_bandwidth)
if analyse_error_vs_bandwidth:
    if bandwidth_study_fmax is not None and bandwidth_study_fmax > fmax:
        raise ValueError(
            f"bandwidth_study_fmax ({bandwidth_study_fmax} Hz) is greater than fmax ({fmax} Hz): "
            "there is no stored frequency bin above fmax."
        )
    bandwidth_ranges = build_cumulative_frequency_bands(
        fmin, fmax if bandwidth_study_fmax is None else bandwidth_study_fmax, bandwidth_step
    )
if analyse_ambiguity_contrast and not 0 < contrast_percentile < 100:
    raise ValueError(
        f"contrast_percentile must be in ]0, 100[ (got {contrast_percentile})."
    )

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
        portion,
        rcv_ref=reference_rcv,
        fmin=fmin,
        fmax=fmax,
        verbose=verbose,
        compare_centered_gamma=compare_centered_gamma,
        intersections=selected_intersections,
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
        ds_feature_section = node.to_dataset()
        if "gamma_ij" not in ds_feature_section:
            log(
                f"Section '{name}' skipped: features not derived "
                "(section not involved in the processed intersections)",
                level=1,
            )
            continue
        log(f"Section '{name}'", level=1)

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

    # Selected intersections that are not in the portion (not computed yet)
    stored_intersections = list_stored_intersections(portion)
    missing = [
        pair for pair in selected_intersections if pair not in stored_intersections
    ]
    if missing:
        print(
            f"    WARNING: intersections {missing} are not stored in the portion and "
            "will not be plotted: set apply_mfp_strategy_intersections = True to compute them."
        )

    # Iterate over meridional sections
    for section1_label, meridional_node in portion.mfp_intersection.children.items():
        # Iterate over zonal sections
        for section2_label, node in meridional_node.children.items():
            if (section1_label, section2_label) not in selected_intersections:
                continue
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
                    metrics=ambiguity_metrics,
                )

                fpath = os.path.join(
                    root_img_intersection,
                    ambiguity_figure_filename(ambiguity_metrics, rcv_id),
                )
                fig.savefig(fpath)
                plt.close(fig)

                # Comparison of gamma distances with / without centering of gamma
                if "d_theta_gamma_centered" in ds_mfp_intersection:
                    fig = plot_intersection_gamma_comparison(
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
                        f"mfp_ambiguity_gamma_centered_comparison_{rcv_id}.png",
                    )
                    fig.savefig(fpath)
                    plt.close(fig)
                elif compare_centered_gamma:
                    print(
                        "    WARNING: compare_centered_gamma = True but the intersections stored in "
                        "the portion do not contain the centered distances: set "
                        "apply_mfp_strategy_intersections = True to compute them."
                    )


# ======================================================================================================================
# Analysis of the results : position error of the estimated intersections
# ======================================================================================================================
run_analysis = (
    analyse_error_vs_distance_type
    or analyse_error_vs_frequency_band
    or analyse_rcv_pair_performance
    or analyse_error_vs_bandwidth
    or analyse_ambiguity_contrast
)
if run_analysis:
    log("Step 7 - Analysing the results (position error of the estimated intersections)")

    # Intersections of the selection that are stored in the portion
    stored_intersections = list_stored_intersections(portion)
    analysed_intersections = [
        pair for pair in selected_intersections if pair in stored_intersections
    ]
    missing = [
        pair for pair in selected_intersections if pair not in stored_intersections
    ]
    if missing:
        print(
            f"    WARNING: intersections {missing} are not stored in the portion and "
            "will not be analysed: set apply_mfp_strategy_intersections = True to compute them."
        )

    # Nothing to analyse if no selected intersection is stored
    if not analysed_intersections:
        raise RuntimeError(
            "None of the selected intersections is stored in the portion: nothing to analyse. "
            "Set apply_mfp_strategy_intersections = True to compute them."
        )

    # Statistics need several intersections to be meaningful
    if len(analysed_intersections) < 3:
        print(
            f"    WARNING: only {len(analysed_intersections)} intersection(s) analysed: "
            "the statistics (mean, std, boxplots, CDF) are not significant."
        )
    log(
        f"{len(analysed_intersections)} intersection(s) analysed, "
        f"position error computed on '{error_mode}'",
        level=1,
    )
    figure_title = f"Position error ({error_mode}), {len(analysed_intersections)} intersection(s)"

    # ==================================================================================================================
    # Position errors from the stored ambiguity surfaces (frequency band used for the MFP computation)
    # ==================================================================================================================
    if (
        analyse_error_vs_distance_type
        or analyse_rcv_pair_performance
        or analyse_ambiguity_contrast
    ):
        log("Position errors from the stored ambiguity surfaces", level=1)

        # Iterate over the stored intersections (one record per distance type and receiver pair)
        records = []
        for section1_label, section2_label in analysed_intersections:
            log(f"Intersection '{section1_label}' x '{section2_label}'", level=2)

            # Lazy access to the stored ambiguity surfaces of the intersection
            ds_mfp_intersection = portion[
                f"/mfp_intersection/{section1_label}/{section2_label}"
            ].to_dataset()

            # Keep the receivers selected by obs_ids
            ds_mfp_intersection = restrict_to_selected_receivers(
                ds_mfp_intersection,
                ds_mfp_intersection.attrs.get("reference_receiver", reference_rcv),
            )

            # Position error of the estimated intersection, for each distance type and receiver pair
            errors, rcv_ids = localization_errors_from_mfp_dataset(
                ds_mfp_intersection, error_mode=error_mode
            )

            # Contrast of the ambiguity surfaces (only if requested)
            contrasts = (
                ambiguity_contrast_from_mfp_dataset(
                    ds_mfp_intersection, percentile=contrast_percentile
                )
                if analyse_ambiguity_contrast
                else None
            )

            # Store
            records += errors_to_records(
                errors,
                rcv_ids,
                ds_mfp_intersection.attrs.get("reference_receiver", reference_rcv),
                contrasts=contrasts,
                intersection=f"{section1_label} x {section2_label}",
            )


        # Keep the requested distance types and flag the valid localizations (error <= threshold)
        df_errors, distance_types = select_and_order_distance_types(
            pd.DataFrame(records), distance_types_to_analyse
        )
        df_errors = add_validity_column(df_errors, success_threshold_m)

        # Save the records
        fpath = os.path.join(ROOT_IMG_PORTION_RESULTS_SYNTHESIS, "mfp_loc_error_records.csv")
        df_errors.to_csv(fpath, index=False)
        log(f"Saved {fpath}", level=2)

        # --------------------------------------------------------------------------------------------------------------
        # Position error versus distance type, for each receiver pair
        # --------------------------------------------------------------------------------------------------------------
        if analyse_error_vs_distance_type:
            log("Position error versus distance type", level=1)

            fig = plot_error_boxplots_per_rcv_pair(
                df_errors, distance_types, title=figure_title
            )
            save_synthesis_figure(fig, "mfp_loc_error_vs_distance_type.png")

        # --------------------------------------------------------------------------------------------------------------
        # Comparison of the receiver pairs
        # --------------------------------------------------------------------------------------------------------------
        if analyse_rcv_pair_performance:
            log(
                f"Comparison of the receiver pairs (success: error <= {success_threshold_m:g} m)",
                level=1,
            )

            # Statistics per distance type and receiver pair
            summary = summarize_localization_errors(df_errors, success_threshold_m)

            # Save
            fpath = os.path.join(
                ROOT_IMG_PORTION_RESULTS_SYNTHESIS, "mfp_rcv_pair_performance.csv"
            )
            summary.to_csv(fpath, index=False)
            log(f"Saved {fpath}", level=2)


            # Console summary: median error table and best receiver pair of each distance type
            if verbose:
                print(
                    "    Median position error [m] (rows: receiver pairs, columns: distance types)"
                )
                table = pivot_error_summary(summary, "median_error_m").round(0)
                print("\n".join("    " + line for line in table.to_string().splitlines()))
                for distance_type, (rcv_pair, value) in best_rcv_pair_per_distance(
                    summary
                ).items():
                    if rcv_pair is not None:
                        print(
                            f"    Lowest median error for '{distance_type}': "
                            f"{rcv_pair} ({value:.0f} m)"
                        )


            # Heatmaps of the median error and of the success rate
            fig = plot_rcv_pair_performance_heatmaps(
                summary, success_threshold_m, title=figure_title
            )
            save_synthesis_figure(fig, "mfp_rcv_pair_performance_heatmaps.png")


            # Cumulative distribution of the position error
            fig = plot_error_cdf(
                df_errors, distance_types, success_threshold_m, title=figure_title
            )
            save_synthesis_figure(fig, "mfp_rcv_pair_error_cdf.png")

        # --------------------------------------------------------------------------------------------------------------
        # Contrast of the ambiguity surfaces
        # --------------------------------------------------------------------------------------------------------------
        if analyse_ambiguity_contrast:
            log(
                f"Contrast of the ambiguity surfaces (P{contrast_percentile:g}; valid: "
                f"error <= {success_threshold_m:g} m)",
                level=1,
            )

            n_valid = int(df_errors["valid"].sum())
            log(f"{n_valid} valid localization(s) out of {len(df_errors)}", level=2)

            # Console summary: median contrast of the valid and invalid localizations
            if verbose:
                table = (
                    df_errors.groupby(["distance_type", "valid"])["contrast"]
                    .median()
                    .unstack("valid")
                    .reindex(distance_types)
                    .rename(columns={True: "valid", False: "invalid"})
                    .rename_axis(columns=None)
                    .round(3)
                )
                print("    Median contrast (rows: distance types)")
                print("\n".join("    " + line for line in table.to_string().splitlines()))

            # Figure (skipped if only the valid localizations are wanted and there is none)
            if contrast_only_valid and n_valid == 0:
                print(
                    "    WARNING: contrast_only_valid = True but no localization is valid: "
                    "contrast figure skipped."
                )
            else:
                fig = plot_contrast_vs_validity(
                    df_errors,
                    distance_types,
                    percentile=contrast_percentile,
                    only_valid=contrast_only_valid,
                    title=(
                        f"Ambiguity contrast (P{contrast_percentile:g}, error mode '{error_mode}'), "
                        f"{len(analysed_intersections)} intersection(s)"
                    ),
                )
                save_synthesis_figure(fig, "mfp_ambiguity_contrast.png")

    # ==================================================================================================================
    # Position error versus frequency band (distances are recomputed on each band)
    # ==================================================================================================================
    if analyse_error_vs_frequency_band:
        log(
            f"Position error versus frequency band (bandwidth {freq_bandwidth} Hz): "
            f"{[frequency_band_label(lo, hi) for lo, hi in freq_bands]} Hz",
            level=1,
        )

        # Distances recomputed on each frequency band (contiguous bands)
        band_records = collect_errors_over_frequency_ranges(
            portion,
            freq_bands,
            analysed_intersections,
            close_all_ranges=False,
            reference_rcv=reference_rcv,
            error_mode=error_mode,
            compare_centered_gamma=compare_centered_gamma,
            rcv_filter=restrict_to_selected_receivers,
            log=log,
        )
        if not band_records:
            raise RuntimeError(
                "No frequency band contains stored frequency bins: check fmin, fmax and "
                "freq_bandwidth."
            )

        # Keep the requested distance types and save the records
        df_band_errors, band_distance_types = select_and_order_distance_types(
            pd.DataFrame(band_records), distance_types_to_analyse
        )
        fpath = os.path.join(
            ROOT_IMG_PORTION_RESULTS_SYNTHESIS,
            "mfp_loc_error_vs_frequency_band_records.csv",
        )
        df_band_errors.to_csv(fpath, index=False)
        log(f"Saved {fpath}", level=2)


        # Mean and standard deviation of the error versus frequency band
        fig = plot_error_vs_frequency_band(
            df_band_errors, band_distance_types, title=figure_title
        )
        save_synthesis_figure(fig, "mfp_loc_error_vs_frequency_band.png")

    # ==================================================================================================================
    # Position error versus width of the frequency range ([fmin, f] with increasing f)
    # ==================================================================================================================
    if analyse_error_vs_bandwidth:
        log(
            f"Position error versus width of the frequency range: {len(bandwidth_ranges)} ranges from "
            f"{frequency_band_label(*bandwidth_ranges[0])} to {frequency_band_label(*bandwidth_ranges[-1])} Hz "
            f"(step {bandwidth_step} Hz)",
            level=1,
        )
        # Distances recomputed on each range [fmin, f] (nested ranges)
        bandwidth_records = collect_errors_over_frequency_ranges(
            portion,
            bandwidth_ranges,
            analysed_intersections,
            close_all_ranges=True,
            reference_rcv=reference_rcv,
            error_mode=error_mode,
            compare_centered_gamma=compare_centered_gamma,
            rcv_filter=restrict_to_selected_receivers,
            log=log,
        )
        if not bandwidth_records:
            raise RuntimeError(
                "No frequency range contains stored frequency bins: check fmin, fmax, "
                "bandwidth_step and bandwidth_study_fmax."
            )

        # Keep the requested distance types and save the records
        df_bandwidth_errors, bandwidth_distance_types = select_and_order_distance_types(
            pd.DataFrame(bandwidth_records), distance_types_to_analyse
        )
        fpath = os.path.join(
            ROOT_IMG_PORTION_RESULTS_SYNTHESIS,
            "mfp_loc_error_vs_bandwidth_records.csv",
        )
        df_bandwidth_errors.to_csv(fpath, index=False)
        log(f"Saved {fpath}", level=2)


        # Mean and standard deviation of the error versus width of the frequency range
        fig = plot_error_vs_bandwidth(
            df_bandwidth_errors, bandwidth_distance_types, title=figure_title
        )
        save_synthesis_figure(fig, "mfp_loc_error_vs_bandwidth.png")

# ======================================================================================================================
# Save results (last step: the portion is closed by the save)
# ======================================================================================================================
if save_modification_to_portion:
    log(f"Step 8 - Saving results to '{portion_fpath}'")
    save_portion_inplace(portion, portion_fpath)
else:
    portion.close()
    log("Done (no modification saved to the portion file)")