#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_analyse_portion_test_section.py
@Time    :   2026/10/30 16:00:00
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Localization of a test section within a library (active / active): the library is made of the
             active emissions of a portion from which the test section has been removed, the event is the test
             section itself. The test section is localized by estimating its intersections with the library
             sections (ambiguity surfaces), compared to the theoretical intersections given by the shot positions.
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
    build_portion_event_from_active_section,
    build_portion_library,
    get_crossing_distance_m,
    library_event_crossing_distances,
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
ROOT_IMG_PORTION_RESULTS_TEST = os.path.join(ROOT_IMG_PORTION, "results_test_section")
os.makedirs(ROOT_IMG_PORTION_RESULTS_TEST, exist_ok=True)

portion_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}.nc"

# Reference receiver of the pairs (rcv_i, reference_rcv): the same for the library and for the test section
reference_rcv = "obs1"

# Receivers that can be used (the reference receiver is ignored). None -> all the receivers of the portion
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
fmax = 150

# Estimation of the RTF (pi_ij), chosen separately for the library and for the event
#   "spectral_ratio" -> S_ij / S_jj (active case)
#   "cs_evd"         -> first eigenvector of the spatial covariance matrix (preferable for passive signals)
# The two can differ (e.g. library: "spectral_ratio", passive event: "cs_evd")
library_rtf_method = "spectral_ratio"
event_rtf_method = "spectral_ratio"

# Test sections: each of them is removed from the library in turn and localized (its own run, with the library
# made of the other sections)
test_sections = ["sn2"]
# test_sections = ["sn0", "sn2", "oe7"]

# Sections that can be part of the library (None -> all the sections of the portion). The test section of the
# current run is always removed from it
library_sections = None
# library_sections = ["sn0", "sn3", "oe6", "oe7", "oe8"]

# Position error of the estimated intersection with respect to the theoretical one
#   "library" -> error between the library samples (position along the library section)
#   "event"   -> error between the samples of the test section (position along the test section)
#   "both"    -> sqrt(error_library**2 + error_test**2)
error_mode = "both"
success_threshold_m = 100  # an intersection is successfully localized if its error <= threshold [m]

# The theoretical intersection only exists if the test section crosses the library section: the library sections
# whose closest samples are further apart than this distance are not compared to the test section
max_crossing_distance_m = 200

# Approach used to combine the receivers and combinations to study
#   "mean_surfaces" -> mean of the ambiguity surfaces of the combined receiver pairs
combination_method = "mean_surfaces"
#   rcv_combinations = None  -> all the combinations of the sizes `combination_sizes` among the receivers
#   rcv_combinations = [["obs2"], ["obs2", "obs4"], ...] -> explicit combinations
rcv_combinations = None
combination_sizes = None  # None -> all sizes, from 1 receiver to all of them; e.g. [1, 2, 5]
max_combinations = 200  # safety limit on the number of combinations (the number grows as 2**n_receivers)

# Distance types to analyse: None -> all those computed, otherwise a list among "theta", "theta_gamma", "L2_gamma",
# "dcf", "theta_gamma_centered", "L2_gamma_centered" (the centered ones need compare_centered_gamma = True)
distance_types_to_analyse = None

# Contrast of the ambiguity surface = (P - min) / P, P being the percentile `contrast_percentile` of the surface
contrast_percentile = 50

# Ambiguity surfaces
#   combinations_to_plot = None -> all the combinations studied
#   combinations_to_plot = [["obs2", "obs4"], ["obs2", "obs4", "obs5"]] -> only these combinations
combinations_to_plot = None
# Metrics displayed next to the real distance (always in the first panel): 3 among
# "theta", "theta_gamma", "L2_gamma", "dcf", "theta_gamma_centered", "L2_gamma_centered"
# (the centered ones need compare_centered_gamma = True), e.g. ["theta_gamma_centered", "L2_gamma_centered", "dcf"]
ambiguity_metrics = ["theta_gamma", "theta", "dcf"]
ambiguity_cmap = "Grays_r"
ambiguity_max_quantile = 0.95  # quantile of each feature distance used as upper bound of its color scale
ambiguity_d_real_vmax = 100  # upper bound of the color scale of the real distance [m]
max_ambiguity_figures = 200  # safety limit on the number of figures (comparisons x combinations)

# ======================================================================================================================
# Booleans to control behavior
# ======================================================================================================================
# Also compute the gamma distances with gamma centered along frequency
compare_centered_gamma = True
# Save the theoretical and estimated intersections (positions) of every comparison and combination (CSV)
save_intersection_positions = True

# Figures
plot_map = True
plot_ambiguity = True
plot_gamma_comparison = False

# Analysis of the combinations
analyse_error_vs_n_rcv = True
analyse_contrast_vs_n_rcv = True
analyse_combination_performance = True
verbose = True

# ======================================================================================================================
# Console messages and figure saving
# ======================================================================================================================
log = make_logger(portion_id, verbose)
save_test_figure = make_figure_saver(ROOT_IMG_PORTION_RESULTS_TEST, log)

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
if not test_sections:
    raise ValueError("test_sections is empty: give at least one section to localize.")

# ======================================================================================================================
# Load data (lazy: nothing is read in memory until values are actually used)
# ======================================================================================================================
log(f"Step 1 - Opening the portion {portion_id} (lazy, not loaded in memory)")
portion = open_portion(portion_fpath)

if "feature" not in portion.children:
    raise RuntimeError(
        f"No 'feature' group in '{portion_fpath}': the CPSD has never been computed for this "
        "portion. Compute it with acouplane_analyse_portion_generate_diag_all_intersections.py "
        "(compute_cpsds = True, save_modification_to_portion = True)."
    )

# Check the sections right away
available_sections = list(portion.feature.children)
unknown_sections = [name for name in test_sections if name not in available_sections]
if unknown_sections:
    raise ValueError(
        f"Unknown test section(s) {unknown_sections} (sections of the portion: {available_sections})."
    )
if library_sections is not None:
    unknown_sections = [name for name in library_sections if name not in available_sections]
    if unknown_sections:
        raise ValueError(
            f"Unknown library section(s) {unknown_sections} (sections of the portion: {available_sections})."
        )
library_pool = available_sections if library_sections is None else list(library_sections)

# ======================================================================================================================
# Library of reference features (all the candidate sections, built once)
# ======================================================================================================================
log("Step 2 - Library of reference features built from the portion")
portion_library = build_portion_library(
    portion,
    sections=library_pool,
    rcv_ref=reference_rcv,
    fmin=fmin,
    fmax=fmax,
    acquisition_mode="active",
    verbose=verbose,
    rtf_method=library_rtf_method,
)

# ======================================================================================================================
# Ambiguity surfaces between each test section and the library (without the test section)
# ======================================================================================================================
log("Step 3 - Ambiguity surfaces between each test section and the library sections that cross it")
datasets_mfp = {}
crossing_records = []

for test_section in test_sections:
    log(f"Test section '{test_section}'", level=1)

    # The test section plays the role of the event
    portion_event = build_portion_event_from_active_section(
        portion,
        test_section,
        rcv_ref=reference_rcv,
        fmin=fmin,
        fmax=fmax,
        rtf_method=event_rtf_method,
    )

    # The test section is removed from the library
    candidate_sections = [name for name in library_pool if name != test_section]
    if not candidate_sections:
        raise RuntimeError(
            f"No library section left once the test section '{test_section}' is removed: "
            "add sections to library_sections."
        )

    # Library sections that cross the test section (the others have no intersection to find)
    crossing_distances = library_event_crossing_distances(
        portion_library, portion_event, library_sections=candidate_sections
    )
    crossing_sections = []
    for record in crossing_distances:
        is_crossing = record["crossing_distance_m"] <= max_crossing_distance_m
        crossing_records.append(
            dict(
                test_section=test_section,
                library=record["library"],
                crossing_distance_m=record["crossing_distance_m"],
                analysed=is_crossing,
            )
        )
        log(
            f"Library '{record['library']}': closest samples {record['crossing_distance_m']:.0f} m apart"
            + ("" if is_crossing else f" (> {max_crossing_distance_m} m: no intersection, not compared)"),
            level=2,
        )
        if is_crossing:
            crossing_sections.append(record["library"])

    if not crossing_sections:
        print(
            f"    WARNING: no library section crosses the test section '{test_section}' "
            f"(closest samples further than max_crossing_distance_m = {max_crossing_distance_m} m): not analysed."
        )
        continue

    # Ambiguity surfaces for the crossing library sections only
    portion_mfp = apply_mfp_strategy_to_portions(
        portion_library=portion_library,
        portion_event=portion_event,
        rcv_ref=reference_rcv,
        fmin=fmin,
        fmax=fmax,
        verbose=verbose,
        compare_centered_gamma=compare_centered_gamma,
        library_sections=crossing_sections,
        event_sections=None,
        obs_ids=obs_ids,
    )

    for library_name, event_name in list_stored_intersections(portion_mfp):
        datasets_mfp[(library_name, event_name)] = portion_mfp[
            f"/mfp_intersection/{library_name}/{event_name}"
        ].to_dataset()

    # Map of the portion, the test section is the highlighted "event"
    if plot_map:
        fig, ax = plot_library_and_event_map(portion, portion_event)
        ax.set_title(f"Test section {test_section}")
        save_test_figure(fig, f"map_test_section_{test_section}.png")

df_crossing = pd.DataFrame(crossing_records)
fpath = os.path.join(ROOT_IMG_PORTION_RESULTS_TEST, "crossing_distances.csv")
df_crossing.to_csv(fpath, index=False)
log(f"Saved {fpath}", level=2)

if not datasets_mfp:
    raise RuntimeError(
        "No test section crosses a library section (see crossing_distances.csv): nothing to analyse. "
        "Select other test sections or library sections, or increase max_crossing_distance_m."
    )

# ======================================================================================================================
# Receivers, combinations and analysis of the comparisons
# ======================================================================================================================
log("Step 4 - Position error and contrast of the combinations of receivers")

# Receivers available for all the comparisons analysed
rcv_lists = [
    [str(rcv_id) for rcv_id in ds["rcv_id_feature"].values]
    for ds in datasets_mfp.values()
]
available_rcv_ids = [
    rcv_id for rcv_id in rcv_lists[0] if all(rcv_id in rcv_list for rcv_list in rcv_lists)
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
    analyse_error_vs_n_rcv or analyse_contrast_vs_n_rcv or analyse_combination_performance
)
if run_analysis:
    log("Step 6 - Combining the receivers (position error and contrast of the combined ambiguity surfaces)")

    if len(datasets_mfp) < 3:
        print(
            f"    WARNING: only {len(datasets_mfp)} comparison(s) analysed: "
            "the statistics (boxplots, success rate, CDF) are not significant."
        )

    records = []
    for (library_name, event_name), ds_mfp in datasets_mfp.items():
        log(f"Library '{library_name}' x test section '{event_name}'", level=2)

        records += rcv_combination_records(
            ds_mfp,
            combinations,
            reference_rcv=ds_mfp.attrs.get("reference_receiver", reference_rcv),
            error_mode=position_error_mode,
            percentile=contrast_percentile,
            method=combination_method,
            intersection=f"{library_name} x {event_name}",
            library=library_name,
            test_section=event_name,
            crossing_distance_m=get_crossing_distance_m(ds_mfp),
        )

    # Keep the requested distance types and flag the valid localizations (error <= threshold)
    df_records, distance_types = select_and_order_distance_types(
        pd.DataFrame(records), distance_types_to_analyse
    )
    df_records = add_validity_column(df_records, success_threshold_m)

    fpath = os.path.join(ROOT_IMG_PORTION_RESULTS_TEST, "test_section_combination_records.csv")
    df_records.to_csv(fpath, index=False)
    log(f"Saved {fpath}", level=2)

    figure_title = (
        f"Test section vs library, receiver combination ({combination_method}), position error ({error_mode}), "
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
        save_test_figure(fig, "test_section_combination_error_vs_n_rcv.png")

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
        save_test_figure(fig, "test_section_combination_contrast_vs_n_rcv.png")

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
            ROOT_IMG_PORTION_RESULTS_TEST, "test_section_combination_performance.csv"
        )
        summary.to_csv(fpath, index=False)
        log(f"Saved {fpath}", level=2)

        # Statistics per distance type and number of combined receivers (all combinations of a size pooled)
        df_by_n_rcv = df_records.assign(
            rcv_pair=df_records["n_rcv"].map(lambda n: f"{n} receiver(s)")
        )
        summary_by_n_rcv = summarize_localization_errors(df_by_n_rcv, success_threshold_m)
        fpath = os.path.join(
            ROOT_IMG_PORTION_RESULTS_TEST, "test_section_combination_performance_by_n_rcv.csv"
        )
        summary_by_n_rcv.to_csv(fpath, index=False)
        log(f"Saved {fpath}", level=2)

        # Console summary: median error by number of receivers and best combination of each distance type
        if verbose:
            print("    Median position error [m] (rows: number of combined receivers, columns: distance types)")
            table = pivot_error_summary(summary_by_n_rcv, "median_error_m").round(0)
            print("\n".join("    " + line for line in table.to_string().splitlines()))
            for distance_type, (rcv_pair, value) in best_rcv_pair_per_distance(summary).items():
                if rcv_pair is not None:
                    print(
                        f"    Lowest median error for '{distance_type}': "
                        f"{rcv_pair} ({value:.0f} m)"
                    )

        # Heatmaps of the median error and of the success rate, one row per combination
        fig = plot_rcv_pair_performance_heatmaps(
            summary, success_threshold_m, title=figure_title
        )
        save_test_figure(fig, "test_section_combination_performance_heatmaps.png")

        # Cumulative distribution of the position error, one curve per number of receivers
        fig = plot_error_cdf(
            df_by_n_rcv, distance_types, success_threshold_m, title=figure_title
        )
        save_test_figure(fig, "test_section_combination_error_cdf_by_n_rcv.png")

# ======================================================================================================================
# Theoretical and estimated intersections (positions)
# ======================================================================================================================
if save_intersection_positions:
    log("Step 6b - Theoretical and estimated intersections of each comparison and combination of receivers")

    position_records = []
    for (library_name, event_name), ds_mfp in datasets_mfp.items():
        for rcv_combination in combinations:
            for record in combination_intersection_positions(
                ds_mfp, rcv_combination, method=combination_method
            ):
                position_records.append(
                    dict(
                        library=library_name,
                        test_section=event_name,
                        combination=format_rcv_combination(rcv_combination),
                        **record,
                    )
                )

    fpath = os.path.join(ROOT_IMG_PORTION_RESULTS_TEST, "test_section_intersection_positions.csv")
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
        len(datasets_mfp) * len(combinations_plotted)
        * (int(plot_ambiguity) + int(plot_gamma_comparison))
    )
    if n_figures > max_ambiguity_figures:
        raise ValueError(
            f"{n_figures} ambiguity figures requested ({len(datasets_mfp)} comparison(s) x "
            f"{len(combinations_plotted)} combination(s)), above max_ambiguity_figures = "
            f"{max_ambiguity_figures}: select fewer test sections, library sections or combinations "
            "(test_sections, library_sections, combinations_to_plot) or raise the limit."
        )
    log(f"{n_figures} figure(s)", level=1)

    for (library_name, event_name), ds_mfp in datasets_mfp.items():
        log(f"Library '{library_name}' x test section '{event_name}'", level=2)

        # Names displayed on the axes
        try:
            library_display_name = f"library {section_name_of(portion.position[library_name])}"
        except KeyError:
            # The orientation / identifier of the section are not stored: use its label
            library_display_name = f"library {library_name}"
        try:
            event_display_name = f"test {section_name_of(portion.position[event_name])}"
        except KeyError:
            # The orientation / identifier of the section are not stored: use its label
            event_display_name = f"test section {event_name}"

        # One folder per comparison
        save_ambiguity_figure = make_figure_saver(
            os.path.join(
                ROOT_IMG_PORTION_RESULTS_TEST, "ambiguity", f"{library_name}_{event_name}"
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
                save_ambiguity_figure(fig, ambiguity_figure_filename(ambiguity_metrics, slug))

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
# Close the file
# ======================================================================================================================
portion.close()
log("Done (the portion file is not modified)")
