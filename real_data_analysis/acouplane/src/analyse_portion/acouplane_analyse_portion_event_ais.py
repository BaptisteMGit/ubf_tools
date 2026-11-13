#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_analyse_portion_event_ais.py
@Time    :   2026/10/30 10:00:00
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Comparison of a passive event (vessel passage with an AIS trajectory) to a library of
             reference features built from the sections of an ACOUPLANE portion: ambiguity surfaces,
             estimated intersection versus theoretical intersection (AIS trajectory x library section),
             and evaluation of the combinations of receivers.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import pandas as pd

from real_data_analysis.acouplane.src.acouplane_cst import (
    ACOUPLANE_PORTION_NC_ROOT_FPATH,
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR,
)
from real_data_analysis.acouplane.src.acouplane_combination import (
    RCV_COMBINATION_METHODS,
    combination_intersection_positions,
    enumerate_rcv_combinations,
    format_rcv_combination,
    rcv_combination_records,
)
from real_data_analysis.acouplane.src.acouplane_console import (
    make_figure_saver,
    make_logger,
)
from real_data_analysis.acouplane.src.acouplane_conventions import (
    library_event_error_mode,
)
from real_data_analysis.acouplane.src.acouplane_event import (
    apply_mfp_strategy_to_portions,
    build_portion_library,
    derive_portion_cpsd_welch,
    derive_portion_features,
    get_crossing_distance_m,
)
from real_data_analysis.acouplane.src.acouplane_localization import (
    add_validity_column,
    best_rcv_pair_per_distance,
    pivot_error_summary,
    select_and_order_distance_types,
    summarize_localization_errors,
)
from real_data_analysis.acouplane.src.acouplane_plot_analysis import (
    plot_error_cdf,
    plot_metric_vs_n_rcv,
    plot_rcv_pair_performance_heatmaps,
)
from real_data_analysis.acouplane.src.acouplane_plot_event import (
    plot_event_features,
    plot_library_and_event_map,
)
from real_data_analysis.acouplane.src.acouplane_plot_mfp import (
    ambiguity_figure_filename,
    plot_combined_intersection_ambiguity,
    plot_combined_intersection_gamma_comparison,
)
from real_data_analysis.acouplane.src.acouplane_portion_io import (
    list_stored_intersections,
    open_portion,
    section_name_of,
)

# ======================================================================================================================
# Constants
# ======================================================================================================================
portion_id = 5
ROOT_IMG_PORTION = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR, f"portion{portion_id}"
)
ROOT_IMG_PORTION_RESULTS_EVENT = os.path.join(ROOT_IMG_PORTION, "results_event_ais")
os.makedirs(ROOT_IMG_PORTION_RESULTS_EVENT, exist_ok=True)

# Files: the library is built from the portion, the events are stored in a separate file
portion_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}.nc"
portion_event_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}_event_ais.nc"

# Reference receiver of the pairs (rcv_i, reference_rcv): the same for the library and for the events
reference_rcv = "obs1"

# Receivers that can be used (the reference receiver is ignored). None -> all the receivers common to the
# library and to the event
obs_ids = [
    "obs1",
    "obs2",
    "obs4",
    "obs5",
    "obs6",
    "obs7",
]

# Frequency band of the features
fmin = 10
fmax = 500

# Estimation of the RTF (pi_ij), chosen separately for the library and for the event
#   "spectral_ratio" -> S_ij / S_jj (active case)
#   "cs_evd"         -> first eigenvector of the spatial covariance matrix (preferable for passive signals)
# The two can differ (e.g. library: "spectral_ratio", passive event: "cs_evd")
library_rtf_method = "spectral_ratio"
event_rtf_method = "cs_evd"

# Library: sections of the portion used as reference (None -> all the sections of the portion)
library_sections = None


# library_sections = ["sn0", "sn2", "oe7", "oe8", "oe9"]
# Event 7
# library_sections = ["sn0", "sn1", "sn2", "sn3", "oe4", "oe5", "oe6"]
# library_sections = ["oe4"]


# Events compared to the library (None -> all the events of the event file)
# event_sections = None
event_sections = ["event_3"]

# CPSD of the events (Welch method, see derive_portion_cpsd_welch)
analysis_segment_duration_s = 10
analysis_segment_alpha_overlap = 0.75
nfft = 2**13
nperseg = 2**13
noverlap = 2**12

# Position error of the estimated intersection with respect to the theoretical one
#   "library" -> error between the library samples (position along the library section)
#   "event"   -> error between the event samples (position along the event trajectory)
#   "both"    -> sqrt(error_library**2 + error_event**2)
error_mode = "both"
success_threshold_m = (
    100  # an intersection is successfully localized if its error <= threshold [m]
)

# The theoretical intersection only exists if the trajectory of the event crosses the library section: the
# comparisons whose closest samples are further apart than this distance are not analysed
max_crossing_distance_m = 200

# Approach used to combine the receivers and combinations to study
#   "mean_surfaces" -> mean of the ambiguity surfaces of the combined receiver pairs
combination_method = "mean_surfaces"
#   rcv_combinations = None  -> all the combinations of the sizes `combination_sizes` among the receivers
#   rcv_combinations = [["obs2"], ["obs2", "obs4"], ...] -> explicit combinations
rcv_combinations = None
combination_sizes = (
    None  # None -> all sizes, from 1 receiver to all of them; e.g. [1, 2, 5]
)
max_combinations = 200  # safety limit on the number of combinations (the number grows as 2**n_receivers)

# Distance types to analyse: None -> all those computed, otherwise a list among "theta", "theta_gamma", "L2_gamma",
# "dcf", "theta_gamma_centered", "L2_gamma_centered" (the centered ones need compare_centered_gamma = True)
distance_types_to_analyse = None

# Contrast of the ambiguity surface = (P - min) / P, P being the percentile `contrast_percentile` of the surface
contrast_percentile = 50

# Ambiguity surfaces
#   combinations_to_plot = None -> all the combinations studied
#   combinations_to_plot = [["obs2", "obs4"], ["obs2", "obs4", "obs5"]] -> only these combinations
# combinations_to_plot = None
combinations_to_plot = [
    ["obs2"],
    ["obs2", "obs4"],
    ["obs2", "obs4", "obs7"],
    ["obs2", "obs4", "obs7", "obs5"],
    ["obs2", "obs4", "obs7", "obs5", "obs6"],
]

# Metrics displayed next to the real distance (always in the first panel) in the ambiguity figures: 3 among
# "theta", "theta_gamma", "L2_gamma", "dcf", "theta_gamma_centered", "L2_gamma_centered"
# (the centered ones need compare_centered_gamma = True), e.g. ["theta_gamma_centered", "L2_gamma_centered", "dcf"]
# ambiguity_metrics = ["theta_gamma", "theta", "dcf"]
ambiguity_metrics = ["theta_gamma_centered", "L2_gamma_centered", "dcf"]

ambiguity_cmap = "Grays_r"
ambiguity_max_quantile = (
    0.95  # quantile of each feature distance used as upper bound of its color scale
)
ambiguity_d_real_vmax = 100  # upper bound of the color scale of the real distance [m]
max_ambiguity_figures = (
    200  # safety limit on the number of figures (comparisons x combinations)
)

# ======================================================================================================================
# Booleans to control behavior
# ======================================================================================================================
# Compute the CPSD of the events (set to False if the events already contain it)
compute_event_cpsd = True
# Also compute the gamma distances with gamma centered along frequency
compare_centered_gamma = True
# Store the ambiguity surfaces of all the comparisons in a NetCDF file next to the event file
save_mfp_to_netcdf = True
# Save the theoretical and estimated intersections (positions) of every comparison and combination (CSV)
save_intersection_positions = True

# Figures
plot_map = True
plot_features_event = True
plot_ambiguity = True
plot_gamma_comparison = False

# Analysis of the combinations
analyse_error_vs_n_rcv = False
analyse_contrast_vs_n_rcv = False
analyse_combination_performance = False
verbose = True

# ======================================================================================================================
# Console messages and figure saving
# ======================================================================================================================
log = make_logger(portion_id, verbose)
save_event_figure = make_figure_saver(ROOT_IMG_PORTION_RESULTS_EVENT, log)

# ======================================================================================================================
# Check the settings right away (before any heavy computation)
# ======================================================================================================================
position_error_mode = library_event_error_mode(error_mode)
if combination_method not in RCV_COMBINATION_METHODS:
    raise ValueError(
        f"Unknown combination_method '{combination_method}'. Valid methods: "
        f"{list(RCV_COMBINATION_METHODS)}."
    )
if not 0 < contrast_percentile < 100:
    raise ValueError(
        f"contrast_percentile must be in ]0, 100[ (got {contrast_percentile})."
    )
if obs_ids is not None and reference_rcv not in obs_ids:
    print(
        f"    WARNING: the reference receiver '{reference_rcv}' is not in obs_ids: it is used as "
        "reference of the pairs anyway."
    )

# ======================================================================================================================
# Load data (lazy: nothing is read in memory until values are actually used)
# ======================================================================================================================
log(
    f"Step 1 - Opening the portion {portion_id} and its events (lazy, not loaded in memory)"
)
portion = open_portion(portion_fpath)
portion_event = open_portion(portion_event_fpath)
log(
    f"{len(portion_event.position.children)} event(s): {list(portion_event.position.children)}",
    level=1,
)

# ======================================================================================================================
# Features of the events (passive recordings)
# ======================================================================================================================
log("Step 2 - Features of the events")
if compute_event_cpsd:
    log(
        f"CPSD (Welch) of {analysis_segment_duration_s} s windows, overlap {analysis_segment_alpha_overlap}, "
        f"band [{fmin}, {fmax}] Hz",
        level=1,
    )
    derive_portion_cpsd_welch(
        portion_tree=portion_event,
        analysis_segment_duration_s=analysis_segment_duration_s,
        analysis_segment_alpha_overlap=analysis_segment_alpha_overlap,
        nfft=nfft,
        nperseg=nperseg,
        noverlap=noverlap,
        obs_ids=None,
        fmin=fmin,
        fmax=fmax,
    )
elif "feature" not in portion_event.children:
    raise RuntimeError(
        f"No 'feature' group in '{portion_event_fpath}': the CPSD of the events has never been "
        "computed. Set compute_event_cpsd = True."
    )

log(f"RTF and DCF features (reference receiver '{reference_rcv}')", level=1)
derive_portion_features(
    portion_event,
    rcv_ref=reference_rcv,
    fmin=fmin,
    fmax=fmax,
    verbose=verbose,
    rtf_method=event_rtf_method,
)

# ======================================================================================================================
# Library of reference features (active emissions of the portion)
# ======================================================================================================================
log("Step 3 - Library of reference features built from the portion")
if "feature" not in portion.children:
    raise RuntimeError(
        f"No 'feature' group in '{portion_fpath}': the CPSD has never been computed for this "
        "portion. Compute it with acouplane_analyse_portion_generate_diag_all_intersections.py "
        "(compute_cpsds = True, save_modification_to_portion = True)."
    )

portion_library = build_portion_library(
    portion,
    sections=library_sections,
    rcv_ref=reference_rcv,
    fmin=fmin,
    fmax=fmax,
    acquisition_mode="active",
    verbose=verbose,
    rtf_method=library_rtf_method,
)
log(f"Library sections: {list(portion_library.feature.children)}", level=1)

# ======================================================================================================================
# Map of the library and of the events
# ======================================================================================================================
if plot_map:
    log("Map of the portion and of the events")
    fig, _ = plot_library_and_event_map(portion, portion_event)
    save_event_figure(fig, "map_portion_and_events.png")

# ======================================================================================================================
# Maps of the features of the events
# ======================================================================================================================
if plot_features_event:
    log("Features (RTF, DCF) of the events")
    for event_name, node in portion_event.feature.children.items():
        if event_sections is not None and event_name not in event_sections:
            continue
        fig_rtf, fig_dcf = plot_event_features(node.to_dataset())
        fig_rtf.suptitle(f"Event {event_name}: RTF amplitude")
        fig_dcf.suptitle(f"Event {event_name}: DCF")
        save_event_figure(fig_rtf, f"event_rtf_amplitude_{event_name}.png")
        save_event_figure(fig_dcf, f"event_dcf_{event_name}.png")

# ======================================================================================================================
# Ambiguity surfaces between the library and the events
# ======================================================================================================================
log("Step 4 - Ambiguity surfaces between the library sections and the events")
portion_mfp = apply_mfp_strategy_to_portions(
    portion_library=portion_library,
    portion_event=portion_event,
    rcv_ref=reference_rcv,
    fmin=fmin,
    fmax=fmax,
    verbose=verbose,
    compare_centered_gamma=compare_centered_gamma,
    library_sections=library_sections,
    event_sections=event_sections,
    obs_ids=obs_ids,
)

if save_mfp_to_netcdf:
    mfp_fpath = portion_event_fpath.replace(".nc", "_mfp.nc")
    portion_mfp.to_netcdf(mfp_fpath)
    log(f"Saved {mfp_fpath}", level=1)

# ======================================================================================================================
# Theoretical intersections: does the trajectory of the event cross the library sections?
# ======================================================================================================================
log("Step 5 - Theoretical intersections (AIS trajectory x library section)")
stored_comparisons = list_stored_intersections(portion_mfp)

datasets_mfp = {}
crossing_records = []
for library_name, event_name in stored_comparisons:
    ds_mfp = portion_mfp[f"/mfp_intersection/{library_name}/{event_name}"].to_dataset()

    # Distance between the library section and the trajectory at their closest samples
    crossing_distance_m = get_crossing_distance_m(ds_mfp)
    is_crossing = crossing_distance_m <= max_crossing_distance_m
    crossing_records.append(
        dict(
            library=library_name,
            event=event_name,
            crossing_distance_m=crossing_distance_m,
            analysed=is_crossing,
        )
    )
    log(
        f"Library '{library_name}' x event '{event_name}': closest samples {crossing_distance_m:.0f} m apart"
        + (
            ""
            if is_crossing
            else f" (> {max_crossing_distance_m} m: no intersection, not analysed)"
        ),
        level=1,
    )

    if is_crossing:
        datasets_mfp[(library_name, event_name)] = ds_mfp

df_crossing = pd.DataFrame(crossing_records)
fpath = os.path.join(ROOT_IMG_PORTION_RESULTS_EVENT, "crossing_distances.csv")
df_crossing.to_csv(fpath, index=False)
log(f"Saved {fpath}", level=2)

if not datasets_mfp:
    raise RuntimeError(
        "No event crosses a library section (closest samples further than "
        f"max_crossing_distance_m = {max_crossing_distance_m} m, see crossing_distances.csv): nothing to analyse. "
        "Select other library sections, other events, or increase max_crossing_distance_m."
    )

# Receivers available for all the comparisons analysed
rcv_lists = [
    [str(rcv_id) for rcv_id in ds["rcv_id_feature"].values]
    for ds in datasets_mfp.values()
]
available_rcv_ids = [
    rcv_id
    for rcv_id in rcv_lists[0]
    if all(rcv_id in rcv_list for rcv_list in rcv_lists)
]

# Combinations of receivers to study
combinations = enumerate_rcv_combinations(
    available_rcv_ids,
    sizes=combination_sizes,
    combinations=rcv_combinations,
    max_combinations=max_combinations,
)
log(
    f"{len(datasets_mfp)} comparison(s) analysed, {len(combinations)} combination(s) of receivers among "
    f"{available_rcv_ids} (reference '{reference_rcv}', method '{combination_method}', "
    f"position error computed on '{error_mode}')",
    level=1,
)

# ======================================================================================================================
# Position error and contrast of each combination of receivers
# ======================================================================================================================
run_analysis = (
    analyse_error_vs_n_rcv
    or analyse_contrast_vs_n_rcv
    or analyse_combination_performance
)
if run_analysis:
    log(
        "Step 6 - Combining the receivers (position error and contrast of the combined ambiguity surfaces)"
    )

    if len(datasets_mfp) < 3:
        print(
            f"    WARNING: only {len(datasets_mfp)} comparison(s) analysed: "
            "the statistics (boxplots, success rate, CDF) are not significant."
        )

    records = []
    for (library_name, event_name), ds_mfp in datasets_mfp.items():
        log(f"Library '{library_name}' x event '{event_name}'", level=2)

        records += rcv_combination_records(
            ds_mfp,
            combinations,
            reference_rcv=ds_mfp.attrs.get("reference_receiver", reference_rcv),
            error_mode=position_error_mode,
            percentile=contrast_percentile,
            method=combination_method,
            intersection=f"{library_name} x {event_name}",
            library=library_name,
            event=event_name,
            crossing_distance_m=get_crossing_distance_m(ds_mfp),
        )

    # Keep the requested distance types and flag the valid localizations (error <= threshold)
    df_records, distance_types = select_and_order_distance_types(
        pd.DataFrame(records), distance_types_to_analyse
    )
    df_records = add_validity_column(df_records, success_threshold_m)

    fpath = os.path.join(
        ROOT_IMG_PORTION_RESULTS_EVENT, "event_combination_records.csv"
    )
    df_records.to_csv(fpath, index=False)
    log(f"Saved {fpath}", level=2)

    figure_title = (
        f"Event vs library, receiver combination ({combination_method}), position error ({error_mode}), "
        f"{len(datasets_mfp)} comparison(s)"
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Position error versus number of combined receivers
    # ------------------------------------------------------------------------------------------------------------------
    if analyse_error_vs_n_rcv:
        log("Position error versus number of combined receivers", level=1)

        fig = plot_metric_vs_n_rcv(
            df_records,
            distance_types,
            value_col="error_m",
            ylabel="Position error [m]",
            title=figure_title,
        )
        save_event_figure(fig, "event_combination_error_vs_n_rcv.png")

    # ------------------------------------------------------------------------------------------------------------------
    # Contrast of the combined ambiguity surfaces versus number of combined receivers
    # ------------------------------------------------------------------------------------------------------------------
    if analyse_contrast_vs_n_rcv:
        log(
            f"Contrast of the combined ambiguity surfaces (P{contrast_percentile:g}; valid: "
            f"error <= {success_threshold_m:g} m)",
            level=1,
        )

        fig = plot_metric_vs_n_rcv(
            df_records,
            distance_types,
            value_col="contrast",
            ylabel=f"Contrast  (P{contrast_percentile:g} - min) / P{contrast_percentile:g}",
            split_valid=True,
            title=figure_title,
        )
        save_event_figure(fig, "event_combination_contrast_vs_n_rcv.png")

    # ------------------------------------------------------------------------------------------------------------------
    # Comparison of the combinations
    # ------------------------------------------------------------------------------------------------------------------
    if analyse_combination_performance:
        log(
            f"Comparison of the combinations (success: error <= {success_threshold_m:g} m)",
            level=1,
        )

        # Statistics per distance type and combination (the combination plays the role of the receiver pair)
        summary = summarize_localization_errors(df_records, success_threshold_m)
        fpath = os.path.join(
            ROOT_IMG_PORTION_RESULTS_EVENT, "event_combination_performance.csv"
        )
        summary.to_csv(fpath, index=False)
        log(f"Saved {fpath}", level=2)

        # Statistics per distance type and number of combined receivers (all combinations of a size pooled)
        df_by_n_rcv = df_records.assign(
            rcv_pair=df_records["n_rcv"].map(lambda n: f"{n} receiver(s)")
        )
        summary_by_n_rcv = summarize_localization_errors(
            df_by_n_rcv, success_threshold_m
        )
        fpath = os.path.join(
            ROOT_IMG_PORTION_RESULTS_EVENT, "event_combination_performance_by_n_rcv.csv"
        )
        summary_by_n_rcv.to_csv(fpath, index=False)
        log(f"Saved {fpath}", level=2)

        # Console summary: median error by number of receivers and best combination of each distance type
        if verbose:
            print(
                "    Median position error [m] (rows: number of combined receivers, columns: distance types)"
            )
            table = pivot_error_summary(summary_by_n_rcv, "median_error_m").round(0)
            print("\n".join("    " + line for line in table.to_string().splitlines()))
            for distance_type, (rcv_pair, value) in best_rcv_pair_per_distance(
                summary
            ).items():
                if rcv_pair is not None:
                    print(
                        f"    Lowest median error for '{distance_type}': "
                        f"{rcv_pair} ({value:.0f} m)"
                    )

        # Heatmaps of the median error and of the success rate, one row per combination
        fig = plot_rcv_pair_performance_heatmaps(
            summary, success_threshold_m, title=figure_title
        )
        save_event_figure(fig, "event_combination_performance_heatmaps.png")

        # Cumulative distribution of the position error, one curve per number of receivers
        fig = plot_error_cdf(
            df_by_n_rcv, distance_types, success_threshold_m, title=figure_title
        )
        save_event_figure(fig, "event_combination_error_cdf_by_n_rcv.png")

# ======================================================================================================================
# Theoretical and estimated intersections (positions)
# ======================================================================================================================
if save_intersection_positions:
    log(
        "Step 6b - Theoretical and estimated intersections of each comparison and combination of receivers"
    )

    position_records = []
    for (library_name, event_name), ds_mfp in datasets_mfp.items():
        for rcv_combination in combinations:
            for record in combination_intersection_positions(
                ds_mfp, rcv_combination, method=combination_method
            ):
                position_records.append(
                    dict(
                        library=library_name,
                        event=event_name,
                        combination=format_rcv_combination(rcv_combination),
                        **record,
                    )
                )

    fpath = os.path.join(
        ROOT_IMG_PORTION_RESULTS_EVENT, "event_intersection_positions.csv"
    )
    pd.DataFrame(position_records).to_csv(fpath, index=False)
    log(f"Saved {fpath}", level=2)

# ======================================================================================================================
# Ambiguity surfaces of the comparisons, for the combinations of receivers
# ======================================================================================================================
if plot_ambiguity or plot_gamma_comparison:
    log("Step 7 - Ambiguity surfaces of the combinations of receivers")

    # Combinations to plot: all those studied, or an explicit selection
    if combinations_to_plot is None:
        combinations_plotted = combinations
    else:
        combinations_plotted = enumerate_rcv_combinations(
            available_rcv_ids,
            sizes=None,
            combinations=combinations_to_plot,
            max_combinations=max_combinations,
        )

    n_figures = (
        len(datasets_mfp)
        * len(combinations_plotted)
        * (int(plot_ambiguity) + int(plot_gamma_comparison))
    )
    if n_figures > max_ambiguity_figures:
        raise ValueError(
            f"{n_figures} ambiguity figures requested ({len(datasets_mfp)} comparison(s) x "
            f"{len(combinations_plotted)} combination(s)), above max_ambiguity_figures = "
            f"{max_ambiguity_figures}: select fewer library sections, events or combinations "
            "(library_sections, event_sections, combinations_to_plot) or raise the limit."
        )
    log(f"{n_figures} figure(s)", level=1)

    for (library_name, event_name), ds_mfp in datasets_mfp.items():
        log(f"Library '{library_name}' x event '{event_name}'", level=2)

        # Names displayed on the axes
        try:
            library_display_name = (
                f"library {section_name_of(portion_library.position[library_name])}"
            )
        except KeyError:
            # The orientation / identifier of the section are not stored: use its label
            library_display_name = f"library {library_name}"
        event_attrs = portion_event.position[event_name].to_dataset().attrs
        event_display_name = f"event {event_attrs.get('event_id', event_name)}"

        # One folder per comparison
        save_ambiguity_figure = make_figure_saver(
            os.path.join(
                ROOT_IMG_PORTION_RESULTS_EVENT,
                "ambiguity",
                f"{library_name}_{event_name}",
            ),
            log,
        )

        for rcv_combination in combinations_plotted:
            slug = format_rcv_combination(rcv_combination, sep="_")

            if plot_ambiguity:
                fig = plot_combined_intersection_ambiguity(
                    ds_mfp,
                    rcv_combination,
                    library_display_name,
                    event_display_name,
                    method=combination_method,
                    cmap=ambiguity_cmap,
                    max_quantile=ambiguity_max_quantile,
                    metrics=ambiguity_metrics,
                    d_real_vmax=ambiguity_d_real_vmax,
                )
                save_ambiguity_figure(
                    fig, ambiguity_figure_filename(ambiguity_metrics, slug)
                )

            if plot_gamma_comparison:
                fig = plot_combined_intersection_gamma_comparison(
                    ds_mfp,
                    rcv_combination,
                    library_display_name,
                    event_display_name,
                    method=combination_method,
                    cmap=ambiguity_cmap,
                    max_quantile=ambiguity_max_quantile,
                )
                save_ambiguity_figure(fig, f"mfp_gamma_centered_comparison_{slug}.png")

# ======================================================================================================================
# Close the files
# ======================================================================================================================
portion.close()
portion_event.close()
log("Done (the portion and the event files are not modified)")
