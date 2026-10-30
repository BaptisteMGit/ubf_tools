#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_analyse_portion_test_section_block.py
@Time    :   2026/10/30 19:00:00
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Localization of a test section (event) within the whole library considered as ONE block of positions
             with features (active / active). No intersection is looked for: a single ambiguity figure is drawn for
             the whole library, and the minima of each feature distance are compared to the minima of the real
             distance (in meters) to detect the "spurious" minima, i.e. those that do not correspond to a real
             proximity between the library sample and the event sample.
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
    enumerate_rcv_combinations,
    format_rcv_combination,
)
from real_data_analysis.acouplane.src.acouplane_console import (
    make_figure_saver,
    make_logger,
)
from real_data_analysis.acouplane.src.acouplane_event import (
    _features_in_band,
    build_portion_event_from_active_section,
    build_portion_library,
)
from real_data_analysis.acouplane.src.acouplane_features import (
    get_distance_between_library_and_event,
)
from real_data_analysis.acouplane.src.acouplane_library_block import (
    build_library_block,
    compute_block_ambiguity,
    compute_block_receiver_distances,
    library_block_minima,
)
from real_data_analysis.acouplane.src.acouplane_localization import (
    DISTANCE_TYPES,
)
from real_data_analysis.acouplane.src.acouplane_plot_event import (
    plot_library_and_event_map,
    plot_library_block_ambiguity,
)
from real_data_analysis.acouplane.src.acouplane_plot_mfp import (
    ambiguity_figure_filename,
)
from real_data_analysis.acouplane.src.acouplane_portion_io import (
    open_portion,
)

# ======================================================================================================================
# Constants
# ======================================================================================================================
portion_id = 5
ROOT_IMG_PORTION = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR, f"portion{portion_id}"
)
ROOT_IMG_PORTION_RESULTS_BLOCK = os.path.join(
    ROOT_IMG_PORTION, "results_test_section_block"
)
os.makedirs(ROOT_IMG_PORTION_RESULTS_BLOCK, exist_ok=True)

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

# Test section (the event). It is removed from the library, which is then ONE block made of all the other sections
test_section = "sn0"

# Sections that can be part of the library block (None -> all the sections of the portion except the test section)
library_sections = None
# library_sections = ["sn0", "sn3", "oe6", "oe7", "oe8"]

# Combinations of receivers (each gives one ambiguity figure)
#   rcv_combinations = [["obs2", "obs4", "obs5"]] -> explicit combinations
#   rcv_combinations = None -> all the combinations of the sizes `combination_sizes` (None -> all sizes)
rcv_combinations = None
combination_sizes = [1, 2, 5]
max_combinations = 200

# Metrics analysed (minima) and displayed next to the real distance (1 to 3, among "theta", "theta_gamma",
# "L2_gamma", "dcf", "theta_gamma_centered", "L2_gamma_centered"; the centered ones need compare_centered_gamma)
# ambiguity_metrics = ["theta_gamma", "theta", "dcf"]
ambiguity_metrics = ["theta_gamma_centered", "L2_gamma_centered", "dcf"]

ambiguity_cmap = "Grays_r"
ambiguity_max_quantile = (
    0.95  # quantile of each feature distance used as upper bound of its color scale
)
ambiguity_d_real_vmax = 500  # upper bound of the color scale of the real distance [m]
max_ambiguity_figures = 50  # safety limit on the number of figures

# Detection of the minima
#   n_minima: maximum number of distinct minima searched in each surface (the lowest ones)
#   minima_quantile: only the deep minima (below this quantile of the surface) are considered
#   minima_exclusion_samples: two minima are at least this many samples apart (library, event)
#   consistency_distance_m: a minimum is consistent if the real distance between the two samples is <= this value;
#   it is spurious otherwise (the features suggest a match between points that are really far apart)
n_minima = 10
minima_quantile = 0.01  # only the minima below this quantile of the surface are kept (None -> the n_minima lowest)
minima_exclusion_samples = (5, 5)
consistency_distance_m = 100

# ======================================================================================================================
# Booleans to control behavior
# ======================================================================================================================
# Also compute the gamma distances with gamma centered along frequency
compare_centered_gamma = True
plot_map = True
plot_ambiguity = True
verbose = True

# ======================================================================================================================
# Console messages and figure saving
# ======================================================================================================================
log = make_logger(portion_id, verbose)
save_block_figure = make_figure_saver(ROOT_IMG_PORTION_RESULTS_BLOCK, log)

# ======================================================================================================================
# Check the settings right away (before any heavy computation)
# ======================================================================================================================
ambiguity_metrics = (
    [ambiguity_metrics]
    if isinstance(ambiguity_metrics, str)
    else list(ambiguity_metrics)
)
unknown_metrics = [key for key in ambiguity_metrics if key not in DISTANCE_TYPES]
if unknown_metrics or not 1 <= len(ambiguity_metrics) <= 3:
    raise ValueError(
        f"ambiguity_metrics must contain 1 to 3 metrics among {list(DISTANCE_TYPES)} (got {ambiguity_metrics})."
    )
centered_metrics = [key for key in ambiguity_metrics if key.endswith("_centered")]
if centered_metrics and not compare_centered_gamma:
    raise ValueError(
        f"The metrics {centered_metrics} need compare_centered_gamma = True."
    )

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

available_sections = list(portion.feature.children)
if test_section not in available_sections:
    raise ValueError(
        f"Unknown test section '{test_section}' (sections of the portion: {available_sections})."
    )
library_pool = (
    available_sections if library_sections is None else list(library_sections)
)
unknown_sections = [name for name in library_pool if name not in available_sections]
if unknown_sections:
    raise ValueError(
        f"Unknown library section(s) {unknown_sections} (sections of the portion: {available_sections})."
    )

# The test section is removed from the library
library_pool = [name for name in library_pool if name != test_section]
if not library_pool:
    raise RuntimeError(
        f"No library section left once the test section '{test_section}' is removed: add sections to library_sections."
    )

# ======================================================================================================================
# Library (one block) and event (test section)
# ======================================================================================================================
log("Step 2 - Library and event features")
portion_library = build_portion_library(
    portion,
    sections=library_pool,
    rcv_ref=reference_rcv,
    fmin=fmin,
    fmax=fmax,
    acquisition_mode="active",
    verbose=verbose,
)
portion_event = build_portion_event_from_active_section(
    portion, test_section, rcv_ref=reference_rcv, fmin=fmin, fmax=fmax
)

# The whole library is one block of samples (positions + features)
ds_library_block = build_library_block(
    portion_library, sections=library_pool, fmin=fmin, fmax=fmax
)
ds_feature_event = _features_in_band(portion_event, test_section, fmin, fmax)
n_library_samples = ds_library_block.sizes["library_datetime"]
n_event_samples = ds_feature_event.sizes["event_datetime"]
log(
    f"Library block: {n_library_samples} samples from {len(library_pool)} section(s) {library_pool}; "
    f"event '{test_section}': {n_event_samples} samples",
    level=1,
)

if plot_map:
    fig, ax = plot_library_and_event_map(portion, portion_event)
    ax.set_title(f"Test section {test_section}")
    save_block_figure(fig, f"map_test_section_{test_section}.png")

# ======================================================================================================================
# Receivers and combinations
# ======================================================================================================================
event_receivers = [str(rcv) for rcv in ds_feature_event["rcv_id_feature"].values]
available_rcv_ids = [
    str(rcv)
    for rcv in ds_library_block["rcv_id_feature"].values
    if str(rcv) in event_receivers and (obs_ids is None or str(rcv) in obs_ids)
]
if not available_rcv_ids:
    raise RuntimeError(
        f"No receiver in common between the library ({list(ds_library_block['rcv_id_feature'].values)}), "
        f"the event ({event_receivers}) and obs_ids ({obs_ids})."
    )
combinations = enumerate_rcv_combinations(
    available_rcv_ids,
    sizes=combination_sizes,
    combinations=rcv_combinations,
    max_combinations=max_combinations,
)
log(
    f"{len(combinations)} combination(s) of receivers among {available_rcv_ids} (reference '{reference_rcv}')",
    level=1,
)
if plot_ambiguity and len(combinations) > max_ambiguity_figures:
    raise ValueError(
        f"{len(combinations)} ambiguity figures would be drawn (max_ambiguity_figures = {max_ambiguity_figures}): "
        "reduce the combinations or increase max_ambiguity_figures (or set plot_ambiguity = False)."
    )

# ======================================================================================================================
# Ambiguity surface of the whole library, minima, comparison with the minima of the real distance
# ======================================================================================================================
log("Step 3 - Ambiguity surfaces of the whole library and minima")

# Computed once: distances per receiver pair and real distance
receiver_distances = compute_block_receiver_distances(
    ds_library_block, ds_feature_event, available_rcv_ids, compare_centered_gamma
)
real_distance = get_distance_between_library_and_event(
    ds_library_block, ds_feature_event
)

metrics_to_analyse = list(ambiguity_metrics)
summary_rows = []
minima_rows = []

for rcv_combination in combinations:
    combination_label = format_rcv_combination(rcv_combination)
    log(f"Combination {combination_label}", level=1)

    ds_surfaces = compute_block_ambiguity(
        ds_library_block,
        ds_feature_event,
        rcv_combination,
        receiver_distances=receiver_distances,
        real_distance=real_distance,
    )
    records, summary = library_block_minima(
        ds_surfaces,
        metrics=metrics_to_analyse,
        n_minima=n_minima,
        exclusion_samples=minima_exclusion_samples,
        consistency_distance_m=consistency_distance_m,
        minima_quantile=minima_quantile,
    )

    for row in summary:
        summary_rows.append(
            dict(test_section=test_section, combination=combination_label, **row)
        )
        log(
            f"{row['metric']}: {row['n_consistent']} consistent / {row['n_spurious']} spurious minima among the "
            f"{row['n_minima']} deep minima"
            + (
                f"; the deepest spurious minimum ({row['best_spurious_value']:.4f}) is "
                + ("deeper" if row["gap"] < 0 else "shallower")
                + f" than the deepest consistent one ({row['best_consistent_value']:.4f})"
                if row["n_spurious"] and row["n_consistent"]
                else (
                    "; NO minimum coincides with a real distance minimum"
                    if not row["n_consistent"]
                    else ""
                )
            ),
            level=2,
        )
    for record in records:
        minima_rows.append(
            dict(
                test_section=test_section,
                combination=combination_label,
                **{key: value for key, value in record.items()},
            )
        )

    if plot_ambiguity:
        fig = plot_library_block_ambiguity(
            ds_surfaces,
            records,
            metrics=ambiguity_metrics,
            cmap=ambiguity_cmap,
            max_quantile=ambiguity_max_quantile,
            d_real_vmax=ambiguity_d_real_vmax,
            event_label=f"Test section {test_section}",
            title=f"Whole library vs test section {test_section} - receivers {combination_label}",
        )
        slug = format_rcv_combination(rcv_combination, sep="_")
        save_block_figure(
            fig,
            ambiguity_figure_filename(
                ambiguity_metrics, f"block_{test_section}_{slug}"
            ).replace("mfp_ambiguity", "library_block_ambiguity"),
        )

# ======================================================================================================================
# Results
# ======================================================================================================================
log("Step 4 - Saving the results")
df_summary = pd.DataFrame(summary_rows)
fpath = os.path.join(
    ROOT_IMG_PORTION_RESULTS_BLOCK, f"minima_summary_{test_section}.csv"
)
df_summary.to_csv(fpath, index=False)
log(f"Saved {fpath}", level=2)

df_minima = pd.DataFrame(minima_rows)
fpath = os.path.join(ROOT_IMG_PORTION_RESULTS_BLOCK, f"minima_{test_section}.csv")
df_minima.to_csv(fpath, index=False)
log(f"Saved {fpath}", level=2)

# Verdict: are there feature minima that do not coincide with a minimum of the real distance?
n_spurious_total = int(df_summary["n_spurious"].sum()) if not df_summary.empty else 0
if n_spurious_total:
    print(
        f"WARNING: {n_spurious_total} minimum(s) of the feature distances do not coincide with a minimum of the "
        f"real distance (real distance > {consistency_distance_m} m); see {fpath}."
    )
else:
    log(
        f"All the deep minima (<= quantile {minima_quantile} of each surface, max {n_minima}) of every feature distance coincide with a real distance <= "
        f"{consistency_distance_m} m."
    )
