#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_analyse_portion_combine_rcv.py
@Time    :   2026/10/27 09:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Effect of the combination of receivers on the localization performance of an
             ACOUPLANE portion: the ambiguity surfaces of several receiver pairs
             (rcv_i, reference receiver) are combined, then the position error and the contrast
             of the combined surfaces are compared for every combination of receivers.
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
    enumerate_rcv_combinations,
    format_rcv_combination,
    rcv_combination_records,
)
from real_data_analysis.acouplane.src.acouplane_console import (
    make_figure_saver,
    make_logger,
)
from real_data_analysis.acouplane.src.acouplane_localization import (
    POSITION_ERROR_MODES,
    add_validity_column,
    best_rcv_pair_per_distance,
    pivot_error_summary,
    select_and_order_distance_types,
    summarize_localization_errors,
)
from real_data_analysis.acouplane.src.acouplane_mfp import (
    apply_mfp_strategy_to_portion_intersections,
)
from real_data_analysis.acouplane.src.acouplane_plot_mfp import (
    ambiguity_figure_filename,
    plot_combined_intersection_ambiguity,
    plot_combined_intersection_gamma_comparison,
)
from real_data_analysis.acouplane.src.acouplane_plot_analysis import (
    plot_error_cdf,
    plot_metric_vs_n_rcv,
    plot_rcv_pair_performance_heatmaps,
)
from real_data_analysis.acouplane.src.acouplane_portion_io import (
    list_stored_intersections,
    open_portion,
    restrict_dataset_to_receivers,
    save_portion_inplace,
    section_name_of,
    select_intersections,
)

# ======================================================================================================================
# Constants
# ======================================================================================================================
portion_id = 5
ROOT_IMG_PORTION = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR, f"portion{portion_id}"
)
ROOT_IMG_PORTION_RESULTS_COMBINATION = os.path.join(
    ROOT_IMG_PORTION, "results_rcv_combination"
)
os.makedirs(ROOT_IMG_PORTION_RESULTS_COMBINATION, exist_ok=True)

# Reference receiver of the pairs (rcv_i, reference_rcv)
reference_rcv = "obs1"

# Receivers that can be combined (the reference receiver is ignored)
obs_ids = [
    "obs1",
    "obs2",
    "obs4",
    "obs5",
    "obs6",
    "obs7",
]

# Frequency band (only used if the MFP strategy is applied here)
fmin = 10
fmax = 800

# Intersections to analyse. None -> all intersections. Otherwise a list of (meridional, zonal) section labels
intersections_to_process = [("sn2", "oe6")]
# intersections_to_process = None

# Approach used to combine the receivers
#   "mean_surfaces" -> mean of the ambiguity surfaces of the combined receiver pairs
combination_method = "mean_surfaces"

# Combinations of receivers to study
#   rcv_combinations = None  -> all the combinations of the sizes `combination_sizes` among the receivers of obs_ids
#   rcv_combinations = [["obs2"], ["obs2", "obs4"], ...] -> explicit combinations (`combination_sizes` is then ignored)
# combination_sizes = None -> all sizes, from 1 receiver to all of them; e.g. [1, 2, 5]
rcv_combinations = None
combination_sizes = None
max_combinations = 200  # safety limit on the number of combinations (the number grows as 2**n_receivers)

# Position error of the estimated intersection with respect to the true one
#   "section1" -> shots of the meridional section (error mostly along North-South)
#   "section2" -> shots of the zonal section (error mostly along East-West)
#   "both"     -> sqrt(error_section1**2 + error_section2**2)
error_mode = "section1"
success_threshold_m = 100  # an intersection is successfully localized if its error <= threshold [m]

# Distance types to analyse: None -> all those stored, otherwise a list among "theta", "theta_gamma", "L2_gamma",
# "dcf", "theta_gamma_centered", "L2_gamma_centered" (the centered ones need compare_centered_gamma = True)
distance_types_to_analyse = None

# Contrast of the ambiguity surface = (P - min) / P, P being the percentile `contrast_percentile` of the surface
contrast_percentile = 50

# Ambiguity surfaces of the combinations (one figure per intersection and per combination)
#   combinations_to_plot = None -> all the combinations studied (see rcv_combinations / combination_sizes)
#   combinations_to_plot = [["obs2", "obs4"], ["obs2", "obs4", "obs5"]] -> only these combinations
combinations_to_plot = None
# Metrics displayed next to the real distance (always in the first panel) in the ambiguity figures: 3 among
# "theta", "theta_gamma", "L2_gamma", "dcf", "theta_gamma_centered", "L2_gamma_centered"
# (the centered ones need compare_centered_gamma = True), e.g. ["theta_gamma_centered", "L2_gamma_centered", "dcf"]
ambiguity_metrics = ["theta_gamma", "theta", "dcf"]
ambiguity_cmap = "Grays_r"
ambiguity_max_quantile = 0.95  # quantile of each feature distance used as upper bound of its color scale
max_ambiguity_figures = 200  # safety limit on the number of figures (intersections x combinations)

# ======================================================================================================================
# Booleans to control behavior
# ======================================================================================================================
save_modification_to_portion = False
apply_mfp_strategy_intersections = False
# Also compute the gamma distances with gamma centered along frequency (only if the MFP strategy is applied here)
compare_centered_gamma = True

# Analysis of the combinations
analyse_error_vs_n_rcv = True
analyse_contrast_vs_n_rcv = True
analyse_combination_performance = True

# Plot the ambiguity surfaces of each intersection for the combinations of receivers
plot_combined_ambiguity = False
# Also plot the comparison of the gamma distances with / without centering (needs compare_centered_gamma results)
plot_combined_gamma_comparison = False
verbose = True

# ======================================================================================================================
# Console messages and figure saving
# ======================================================================================================================
log = make_logger(portion_id, verbose)
save_combination_figure = make_figure_saver(ROOT_IMG_PORTION_RESULTS_COMBINATION, log)


# ======================================================================================================================
# Load data (lazy: nothing is read in memory until values are actually used)
# ======================================================================================================================
log(f"Step 1 - Opening file for portion {portion_id} (lazy, not loaded in memory)")
portion_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}.nc"
portion = open_portion(portion_fpath)

# Check the requested intersections right away (before any heavy computation)
selected_intersections = select_intersections(
    list(portion.position.children), intersections_to_process
)
log(
    f"{len(selected_intersections)} intersection(s) selected"
    + (" (all)" if intersections_to_process is None else f": {selected_intersections}"),
    level=1,
)

# Check the settings right away (before any heavy computation)
if error_mode not in POSITION_ERROR_MODES:
    raise ValueError(
        f"Unknown error_mode '{error_mode}'. Valid modes: {list(POSITION_ERROR_MODES)}."
    )
if combination_method not in RCV_COMBINATION_METHODS:
    raise ValueError(
        f"Unknown combination_method '{combination_method}'. Valid methods: "
        f"{list(RCV_COMBINATION_METHODS)}."
    )
if not 0 < contrast_percentile < 100:
    raise ValueError(
        f"contrast_percentile must be in ]0, 100[ (got {contrast_percentile})."
    )

# ======================================================================================================================
# Derive features and apply MFP to each intersection (optional)
# ======================================================================================================================
if apply_mfp_strategy_intersections:
    # The CPSD must have been computed beforehand (see acouplane_analyse_portion_generate_diag_all_intersections.py)
    if "feature" not in portion.children:
        raise RuntimeError(
            f"No 'feature' group in '{portion_fpath}': the CPSD has never been computed for this "
            "portion. Compute it with acouplane_analyse_portion_generate_diag_all_intersections.py "
            "(compute_cpsds = True, save_modification_to_portion = True)."
        )

    log("Step 2 - Applying the MFP strategy to each intersection")
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
    log("Step 2 - MFP strategy: skipped (apply_mfp_strategy_intersections = False), using stored results")

# ======================================================================================================================
# Analysis of the combinations of receivers
# ======================================================================================================================
run_analysis = (
    analyse_error_vs_n_rcv or analyse_contrast_vs_n_rcv or analyse_combination_performance
)
run_ambiguity_plots = plot_combined_ambiguity or plot_combined_gamma_comparison
if run_analysis or run_ambiguity_plots:
    log("Step 3 - Combining the receivers (combined ambiguity surfaces)")

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
            "the statistics (boxplots, success rate, CDF) are not significant."
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Stored ambiguity surfaces, restricted to the receivers of obs_ids (lazy datasets)
    # ------------------------------------------------------------------------------------------------------------------
    datasets_mfp = {}
    warned_missing = set()
    for section1_label, section2_label in analysed_intersections:
        ds_mfp_intersection = portion[
            f"/mfp_intersection/{section1_label}/{section2_label}"
        ].to_dataset()
        stored_reference = ds_mfp_intersection.attrs.get("reference_receiver", reference_rcv)

        # Keep the receivers selected by obs_ids
        try:
            ds_mfp_intersection, missing_rcv = restrict_dataset_to_receivers(
                ds_mfp_intersection, obs_ids, reference=stored_reference
            )
        except ValueError as error:
            raise RuntimeError(f"Intersection '{section1_label}' x '{section2_label}': {error}")
        if missing_rcv and tuple(missing_rcv) not in warned_missing:
            warned_missing.add(tuple(missing_rcv))
            print(
                f"    WARNING: receivers {missing_rcv} are listed in obs_ids but are not stored in "
                "the results: they are not combined."
            )
        datasets_mfp[(section1_label, section2_label)] = ds_mfp_intersection

    # Receivers available for all the analysed intersections
    rcv_lists = [
        [str(rcv_id) for rcv_id in ds["rcv_id_feature"].values]
        for ds in datasets_mfp.values()
    ]
    available_rcv_ids = [
        rcv_id for rcv_id in rcv_lists[0] if all(rcv_id in rcv_list for rcv_list in rcv_lists)
    ]
    if len(available_rcv_ids) < len(set().union(*rcv_lists)):
        print(
            "    WARNING: the receivers stored differ between the intersections: only the "
            f"receivers available for all of them are combined: {available_rcv_ids}."
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Combinations of receivers to study
    # ------------------------------------------------------------------------------------------------------------------
    combinations = enumerate_rcv_combinations(
        available_rcv_ids,
        sizes=combination_sizes,
        combinations=rcv_combinations,
        max_combinations=max_combinations,
    )
    log(
        f"{len(analysed_intersections)} intersection(s) analysed, {len(combinations)} combination(s) of "
        f"receivers among {available_rcv_ids} (reference '{reference_rcv}', method '{combination_method}', "
        f"position error computed on '{error_mode}')",
        level=1,
    )

    if run_analysis:
        # ------------------------------------------------------------------------------------------------------------------
        # Position error and contrast of each combination, for each intersection
        # ------------------------------------------------------------------------------------------------------------------
        records = []
        for (section1_label, section2_label), ds_mfp_intersection in datasets_mfp.items():
            log(f"Intersection '{section1_label}' x '{section2_label}'", level=2)

            records += rcv_combination_records(
                ds_mfp_intersection,
                combinations,
                reference_rcv=ds_mfp_intersection.attrs.get("reference_receiver", reference_rcv),
                error_mode=error_mode,
                percentile=contrast_percentile,
                method=combination_method,
                intersection=f"{section1_label} x {section2_label}",
            )

        # Keep the requested distance types and flag the valid localizations (error <= threshold)
        df_records, distance_types = select_and_order_distance_types(
            pd.DataFrame(records), distance_types_to_analyse
        )
        df_records = add_validity_column(df_records, success_threshold_m)

        # Save the records
        fpath = os.path.join(ROOT_IMG_PORTION_RESULTS_COMBINATION, "rcv_combination_records.csv")
        df_records.to_csv(fpath, index=False)
        log(f"Saved {fpath}", level=2)

        figure_title = (
            f"Receiver combination ({combination_method}), position error ({error_mode}), "
            f"{len(analysed_intersections)} intersection(s)"
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
            save_combination_figure(fig, "rcv_combination_error_vs_n_rcv.png")

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
            save_combination_figure(fig, "rcv_combination_contrast_vs_n_rcv.png")

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
                ROOT_IMG_PORTION_RESULTS_COMBINATION, "rcv_combination_performance.csv"
            )
            summary.to_csv(fpath, index=False)
            log(f"Saved {fpath}", level=2)

            # Statistics per distance type and number of combined receivers (all combinations of a size pooled)
            df_by_n_rcv = df_records.assign(
                rcv_pair=df_records["n_rcv"].map(lambda n: f"{n} receiver(s)")
            )
            summary_by_n_rcv = summarize_localization_errors(df_by_n_rcv, success_threshold_m)
            fpath = os.path.join(
                ROOT_IMG_PORTION_RESULTS_COMBINATION, "rcv_combination_performance_by_n_rcv.csv"
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
            save_combination_figure(fig, "rcv_combination_performance_heatmaps.png")

            # Cumulative distribution of the position error, one curve per number of receivers
            fig = plot_error_cdf(
                df_by_n_rcv, distance_types, success_threshold_m, title=figure_title
            )
            save_combination_figure(fig, "rcv_combination_error_cdf_by_n_rcv.png")


    # ------------------------------------------------------------------------------------------------------------------
    # Ambiguity surfaces of the combinations, for each intersection
    # ------------------------------------------------------------------------------------------------------------------
    if run_ambiguity_plots:
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
            * (int(plot_combined_ambiguity) + int(plot_combined_gamma_comparison))
        )
        if n_figures > max_ambiguity_figures:
            raise ValueError(
                f"{n_figures} ambiguity figures requested ({len(datasets_mfp)} intersection(s) x "
                f"{len(combinations_plotted)} combination(s)), above max_ambiguity_figures = "
                f"{max_ambiguity_figures}: select fewer intersections or combinations "
                "(intersections_to_process, combinations_to_plot) or raise the limit."
            )
        log(f"Ambiguity surfaces of the combinations: {n_figures} figure(s)", level=1)

        for (section1_label, section2_label), ds_mfp_intersection in datasets_mfp.items():
            log(f"Intersection '{section1_label}' x '{section2_label}'", level=2)

            section1_name = section_name_of(portion.position[section1_label])
            section2_name = section_name_of(portion.position[section2_label])

            # One folder per intersection
            save_ambiguity_figure = make_figure_saver(
                os.path.join(
                    ROOT_IMG_PORTION_RESULTS_COMBINATION,
                    "ambiguity",
                    f"{section1_label}_{section2_label}",
                ),
                log,
            )

            for rcv_combination in combinations_plotted:
                slug = format_rcv_combination(rcv_combination, sep="_")

                if plot_combined_ambiguity:
                    fig = plot_combined_intersection_ambiguity(
                        ds_mfp_intersection,
                        rcv_combination,
                        section1_name,
                        section2_name,
                        method=combination_method,
                        cmap=ambiguity_cmap,
                        max_quantile=ambiguity_max_quantile,
                        metrics=ambiguity_metrics,
                    )
                    save_ambiguity_figure(fig, ambiguity_figure_filename(ambiguity_metrics, slug))

                if plot_combined_gamma_comparison:
                    fig = plot_combined_intersection_gamma_comparison(
                        ds_mfp_intersection,
                        rcv_combination,
                        section1_name,
                        section2_name,
                        method=combination_method,
                        cmap=ambiguity_cmap,
                        max_quantile=ambiguity_max_quantile,
                    )
                    save_ambiguity_figure(fig, f"mfp_gamma_centered_comparison_{slug}.png")

# ======================================================================================================================
# Save results (last step: the portion is closed by the save)
# ======================================================================================================================
if save_modification_to_portion:
    log(f"Step 4 - Saving results to '{portion_fpath}'")
    save_portion_inplace(portion, portion_fpath)
else:
    portion.close()
    log("Done (no modification saved to the portion file)")