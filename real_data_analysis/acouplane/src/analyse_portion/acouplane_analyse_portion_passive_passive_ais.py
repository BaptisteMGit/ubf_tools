#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_analyse_portion_event_vs_event.py
@Time    :   2026/11/17 10:00:00
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Comparison of passive events (vessel passages with an AIS trajectory) between them.
             The library is made of some of the events, and the events compared to the library are
             also taken in the event file (passive / passive case).

             For each pair (library event x tested event):
               1. the features (RTF, DCF) of both events are compared sample by sample (ambiguity surfaces);
               2. the theoretical intersection (closest samples of the two AIS trajectories) is compared
                  to the intersection estimated from the features.

             The long results (CPSD and ambiguity surfaces) are saved and reused when the script is run again
             with the same settings.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from real_data_analysis.acouplane.src.acouplane_cst import (
    ACOUPLANE_PORTION_NC_ROOT_FPATH,
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR,
)
from real_data_analysis.acouplane.src.acouplane_combination import (
    combination_intersection_positions,
    enumerate_rcv_combinations,
    format_rcv_combination,
)
from real_data_analysis.acouplane.src.acouplane_console import (
    make_figure_saver,
    make_logger,
)
from real_data_analysis.acouplane.src.acouplane_event import (
    apply_mfp_strategy_to_portions,
    build_portion_library,
    derive_portion_cpsd_welch,
    derive_portion_features,
    get_crossing_distance_m,
)
from real_data_analysis.acouplane.src.acouplane_plot_mfp import (
    ambiguity_figure_filename,
    plot_combined_intersection_ambiguity,
)
from real_data_analysis.acouplane.src.acouplane_portion_io import (
    file_fingerprint,
    list_stored_intersections,
    load_tree_if_settings_match,
    open_portion,
    save_tree_atomic,
)

# ======================================================================================================================
# Settings
# ======================================================================================================================
portion_id = 5

# Folder of the figures and tables
ROOT_IMG_PORTION = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR, f"portion{portion_id}"
)
ROOT_IMG_PORTION_RESULTS = os.path.join(ROOT_IMG_PORTION, "results_event_vs_event")
os.makedirs(ROOT_IMG_PORTION_RESULTS, exist_ok=True)

# Event file (never modified). The saved CPSD is shared with acouplane_analyse_portion_event_ais.py
portion_event_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}_event_ais.nc"
portion_event_cpsd_fpath = portion_event_fpath.replace(".nc", "_cpsd.nc")
# Ambiguity surfaces of this script (a different file from the one of the event / active library script)
portion_event_mfp_fpath = portion_event_fpath.replace(".nc", "_event_vs_event_mfp.nc")

# Reference receiver of the pairs (rcv_i, reference_rcv): the same for the library and for the events
reference_rcv = "obs1"

# Receivers that can be used (the reference receiver is ignored). None -> all the receivers of the events
obs_ids = ["obs1", "obs2", "obs4", "obs5", "obs6", "obs7"]

# Frequency band of the features [Hz]
fmin = 10
fmax = 500

# ----------------------------------------------------------------------------------------------------------------------
# Which events are in the library, and which events are compared to it
# ----------------------------------------------------------------------------------------------------------------------
# Events used as library (the reference trajectories)
# library_events = ["event_16", "event_18", "event_19"]
library_events = ["event_25", "event_31"]

# Events compared to the library. None -> all the events of the file.
# An event is never compared to itself (so a library event can also be in this list).
# tested_events = None
# tested_events = ["event_34"]
tested_events = ["event_10"]


# ----------------------------------------------------------------------------------------------------------------------
# RTF estimation, chosen separately for the library and for the tested events
#   "spectral_ratio" -> S_ij / S_jj
#   "cs_evd"         -> first eigenvector of the spatial covariance matrix (preferable for passive signals)
# ----------------------------------------------------------------------------------------------------------------------
library_rtf_method = "cs_evd"
event_rtf_method = "cs_evd"

# ----------------------------------------------------------------------------------------------------------------------
# CPSD of the events (Welch method)
# ----------------------------------------------------------------------------------------------------------------------
analysis_segment_duration_s = 10
analysis_segment_alpha_overlap = 0.75
nfft = 2**13
nperseg = 2**13
noverlap = 2**12

# ----------------------------------------------------------------------------------------------------------------------
# Intersections
# ----------------------------------------------------------------------------------------------------------------------
# Two trajectories only have a theoretical intersection if they cross each other. The pairs whose closest samples are
# further apart than this distance [m] are not analysed.
max_crossing_distance_m = 200

# Receivers combined to build the ambiguity surface (the surfaces of the receiver pairs are averaged)
combination_method = "mean_surfaces"
# rcv_combinations = [
#     ["obs2", "obs4", "obs7", "obs5", "obs6"],
# ]
rcv_combinations = [
    ["obs2"],
    ["obs2", "obs4"],
    ["obs2", "obs4", "obs7"],
    ["obs2", "obs4", "obs7", "obs5"],
    ["obs2", "obs4", "obs7", "obs5", "obs6"],
]

# Metrics displayed next to the real distance (always in the first panel): 3 among
# "theta", "theta_gamma", "L2_gamma", "dcf", "theta_gamma_centered", "L2_gamma_centered"
ambiguity_metrics = ["theta_gamma_centered", "L2_gamma_centered", "dcf"]
ambiguity_cmap = "Grays_r"
ambiguity_max_quantile = (
    0.95  # quantile of each feature distance used as upper bound of its color scale
)
ambiguity_d_real_vmax = 100  # upper bound of the color scale of the real distance [m]
max_ambiguity_figures = 200  # safety limit on the number of ambiguity figures

# ======================================================================================================================
# Switches
# ======================================================================================================================
# Compute the CPSD of the events (False if the event file already contains it)
compute_event_cpsd = True
# Also compute the distances with gamma centered along frequency (needed by the "..._centered" metrics)
compare_centered_gamma = True

# Reuse the saved CPSD and ambiguity surfaces if they were computed with exactly the same settings.
# If something differs, the reason is displayed and the result is computed again.
reuse_stored_results = True
save_cpsd_to_netcdf = True
save_mfp_to_netcdf = True

# Outputs
plot_map = True
save_intersection_positions = True
plot_ambiguity = True
verbose = True

# ======================================================================================================================
# Helpers for the console messages and the figures
# ======================================================================================================================
log = make_logger(portion_id, verbose)
save_figure = make_figure_saver(ROOT_IMG_PORTION_RESULTS, log)

# ======================================================================================================================
# Step 1 - Open the event file and check the settings
# ======================================================================================================================
log("Step 1 - Opening the event file (lazy: nothing is loaded in memory yet)")
portion_event = open_portion(portion_event_fpath)

# Names of all the events of the file
event_group = (
    portion_event.feature
    if "feature" in portion_event.children
    else portion_event.pressure
)
all_events = list(event_group.children)
log(f"{len(all_events)} event(s) in the file: {all_events}", level=1)

# Events of the library and events compared to the library
tested_events_resolved = all_events if tested_events is None else list(tested_events)
for name in list(library_events) + tested_events_resolved:
    if name not in all_events:
        raise ValueError(f"Unknown event '{name}' (events of the file: {all_events}).")

# At least one comparison must remain once the pairs (event, same event) are removed
n_pairs = sum(
    1 for lib in library_events for evt in tested_events_resolved if lib != evt
)
if n_pairs == 0:
    raise ValueError(
        "No pair to compare: every tested event is also the only library event. "
        "Add other events to library_events or tested_events."
    )
log(
    f"Library: {len(library_events)} event(s); tested: {len(tested_events_resolved)} event(s); "
    f"{n_pairs} pair(s) to compare (an event is never compared to itself)",
    level=1,
)

# ======================================================================================================================
# Step 2 - Can the saved ambiguity surfaces be reused ?
# ======================================================================================================================
log("Step 2 - Saved ambiguity surfaces")

# Everything the CPSD depends on. A saved CPSD is reused only if all of this is identical.
cpsd_settings = dict(
    computed_by_this_script=compute_event_cpsd,
    event_file=file_fingerprint(portion_event_fpath),
    analysis_segment_duration_s=analysis_segment_duration_s,
    analysis_segment_alpha_overlap=analysis_segment_alpha_overlap,
    nfft=nfft,
    nperseg=nperseg,
    noverlap=noverlap,
    fmin=fmin,
    fmax=fmax,
    obs_ids=None,
)

# Everything the ambiguity surfaces depend on
mfp_settings = dict(
    cpsd=cpsd_settings,
    reference_rcv=reference_rcv,
    fmin=fmin,
    fmax=fmax,
    library_rtf_method=library_rtf_method,
    event_rtf_method=event_rtf_method,
    library_events=sorted(library_events),
    tested_events=sorted(tested_events_resolved),
    obs_ids=None if obs_ids is None else sorted(obs_ids),
    compare_centered_gamma=compare_centered_gamma,
)

portion_mfp = None
if reuse_stored_results:
    portion_mfp, reason = load_tree_if_settings_match(
        portion_event_mfp_fpath, mfp_settings
    )
    log(f"Saved ambiguity surfaces: {reason}", level=1)
mfp_reused = portion_mfp is not None

# ======================================================================================================================
# Step 3 - CPSD of the events (only needed to compute the surfaces)
# ======================================================================================================================
log("Step 3 - CPSD of the events")
cpsd_tree = None  # lazy tree of the saved CPSD, if it is reused

if mfp_reused:
    log("Not needed: the ambiguity surfaces are reused", level=1)

elif compute_event_cpsd:
    # 3a) Reuse the CPSD saved by a previous run
    if reuse_stored_results:
        cpsd_tree, reason = load_tree_if_settings_match(
            portion_event_cpsd_fpath, cpsd_settings
        )
        log(f"Saved CPSD: {reason}", level=1)

    if cpsd_tree is not None:
        # Plug the saved CPSD into the event tree (it stays lazily read from its file)
        for event_name, node in cpsd_tree.feature.children.items():
            portion_event[f"/feature/{event_name}"] = node.to_dataset()
        log(f"CPSD loaded from {portion_event_cpsd_fpath}", level=1)

    # 3b) Otherwise compute it (long), and save it right away
    else:
        log(
            f"Computing the CPSD (Welch) of {analysis_segment_duration_s} s windows, "
            f"overlap {analysis_segment_alpha_overlap}, band [{fmin}, {fmax}] Hz",
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
        if save_cpsd_to_netcdf:
            cpsd_to_save = xr.DataTree()
            for event_name, node in portion_event.feature.children.items():
                cpsd_to_save[f"/feature/{event_name}"] = node.to_dataset()
            save_tree_atomic(cpsd_to_save, portion_event_cpsd_fpath, cpsd_settings)
            log(f"CPSD saved: {portion_event_cpsd_fpath}", level=1)

elif "feature" not in portion_event.children:
    raise RuntimeError(
        f"No 'feature' group in '{portion_event_fpath}': the CPSD of the events has never been computed. "
        "Set compute_event_cpsd = True."
    )
else:
    log("Using the CPSD stored in the event file", level=1)

# ======================================================================================================================
# Step 4 - Features (RTF, DCF) of the library and of the tested events (only needed to compute the surfaces)
# ======================================================================================================================
log("Step 4 - Features of the library and of the events")
portion_library = None

if mfp_reused:
    log("Not needed: the ambiguity surfaces are reused", level=1)

else:
    # The library is built first, from the CPSD of the library events.
    # acquisition_mode="passive": the samples of the library are the time windows of the events.
    log(
        f"Library of {len(library_events)} event(s) (RTF '{library_rtf_method}')",
        level=1,
    )
    portion_library = build_portion_library(
        portion_event,
        sections=library_events,
        rcv_ref=reference_rcv,
        fmin=fmin,
        fmax=fmax,
        acquisition_mode="passive",
        verbose=verbose,
        rtf_method=library_rtf_method,
    )

    # Features of all the events of the file (the tested events are among them)
    log(f"Events (RTF '{event_rtf_method}')", level=1)
    derive_portion_features(
        portion_event,
        rcv_ref=reference_rcv,
        fmin=fmin,
        fmax=fmax,
        verbose=verbose,
        rtf_method=event_rtf_method,
    )

# ======================================================================================================================
# Step 5 - Map of the trajectories
# ======================================================================================================================
if plot_map:
    log("Step 5 - Map of the trajectories")
    fig, ax = plt.subplots(figsize=(12, 12))

    # Library events in red, tested events in blue
    for event_name, node in portion_event.position.children.items():
        if (
            event_name not in library_events
            and event_name not in tested_events_resolved
        ):
            continue
        ds_position = node.to_dataset()
        is_library = event_name in library_events
        ax.plot(
            ds_position["lon"].values,
            ds_position["lat"].values,
            color="tab:red" if is_library else "tab:blue",
            linewidth=3 if is_library else 1.5,
            label=f"{event_name} (library)" if is_library else event_name,
        )

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.legend(ncols=2)
    save_figure(fig, "map_library_and_tested_events.png")

# ======================================================================================================================
# Step 6 - Ambiguity surfaces between the library events and the tested events
# ======================================================================================================================
log("Step 6 - Ambiguity surfaces")

if mfp_reused:
    log(f"Loaded from {portion_event_mfp_fpath}: not computed again", level=1)

else:
    # Every library event is compared to every tested event (long)
    portion_mfp = apply_mfp_strategy_to_portions(
        portion_library=portion_library,
        portion_event=portion_event,
        rcv_ref=reference_rcv,
        fmin=fmin,
        fmax=fmax,
        verbose=verbose,
        compare_centered_gamma=compare_centered_gamma,
        library_sections=library_events,
        event_sections=tested_events_resolved,
        obs_ids=obs_ids,
        skip_identical_names=True,  # never compare an event with itself
    )

    # Saved right away: if the rest of the script fails, the next run reuses the surfaces
    if save_mfp_to_netcdf:
        save_tree_atomic(portion_mfp, portion_event_mfp_fpath, mfp_settings)
        log(f"Ambiguity surfaces saved: {portion_event_mfp_fpath}", level=1)

# ======================================================================================================================
# Step 7 - Theoretical intersections: do the two trajectories cross each other ?
# ======================================================================================================================
log("Step 7 - Theoretical intersections (library trajectory x tested trajectory)")

datasets_mfp = (
    {}
)  # (library event, tested event) -> ambiguity surfaces, for the pairs that cross
crossing_records = []

for library_name, event_name in list_stored_intersections(portion_mfp):
    ds_mfp = portion_mfp[f"/mfp_intersection/{library_name}/{event_name}"].to_dataset()

    # Distance between the two trajectories at their closest samples
    crossing_distance_m = get_crossing_distance_m(ds_mfp)
    is_crossing = crossing_distance_m <= max_crossing_distance_m

    crossing_records.append(
        dict(
            library_event=library_name,
            tested_event=event_name,
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

pd.DataFrame(crossing_records).to_csv(
    os.path.join(ROOT_IMG_PORTION_RESULTS, "crossing_distances.csv"), index=False
)

if not datasets_mfp:
    raise RuntimeError(
        "No tested event crosses a library event (closest samples further than "
        f"max_crossing_distance_m = {max_crossing_distance_m} m, see crossing_distances.csv). "
        "Select other events or increase max_crossing_distance_m."
    )

# Receivers available for all the pairs analysed, and combinations of receivers to study
rcv_lists = [
    [str(rcv) for rcv in ds["rcv_id_feature"].values] for ds in datasets_mfp.values()
]
available_rcv_ids = [
    rcv for rcv in rcv_lists[0] if all(rcv in rcv_list for rcv_list in rcv_lists)
]
combinations = enumerate_rcv_combinations(
    available_rcv_ids,
    sizes=None,
    combinations=rcv_combinations,
    max_combinations=200,
)
log(
    f"{len(datasets_mfp)} pair(s) analysed, {len(combinations)} combination(s) of receivers "
    f"among {available_rcv_ids} (reference '{reference_rcv}')",
    level=1,
)

# ======================================================================================================================
# Step 8 - Theoretical and estimated intersections (positions, as a table)
# ======================================================================================================================
if save_intersection_positions:
    log("Step 8 - Theoretical and estimated intersections")

    position_records = []
    for (library_name, event_name), ds_mfp in datasets_mfp.items():
        for rcv_combination in combinations:
            # One record for the theoretical intersection and one for each type of distance
            for record in combination_intersection_positions(
                ds_mfp, rcv_combination, method=combination_method
            ):
                position_records.append(
                    dict(
                        library_event=library_name,
                        tested_event=event_name,
                        combination=format_rcv_combination(rcv_combination),
                        **record,
                    )
                )

    fpath = os.path.join(
        ROOT_IMG_PORTION_RESULTS, "event_vs_event_intersection_positions.csv"
    )
    pd.DataFrame(position_records).to_csv(fpath, index=False)
    log(f"Saved {fpath}", level=1)

# ======================================================================================================================
# Step 9 - Ambiguity figures (one folder per pair of events)
# ======================================================================================================================
if plot_ambiguity:
    log("Step 9 - Ambiguity figures")

    # Safety: do not generate hundreds of figures by mistake
    n_figures = len(datasets_mfp) * len(combinations)
    if n_figures > max_ambiguity_figures:
        raise ValueError(
            f"{n_figures} ambiguity figures requested ({len(datasets_mfp)} pair(s) x {len(combinations)} "
            f"combination(s)), above max_ambiguity_figures = {max_ambiguity_figures}: select fewer events "
            "or combinations, or raise the limit."
        )
    log(f"{n_figures} figure(s)", level=1)

    for (library_name, event_name), ds_mfp in datasets_mfp.items():
        # Names displayed on the axes
        library_display_name = f"library {library_name}"
        event_display_name = f"event {event_name}"

        # One folder per pair
        save_pair_figure = make_figure_saver(
            os.path.join(
                ROOT_IMG_PORTION_RESULTS, "ambiguity", f"{library_name}_{event_name}"
            ),
            log,
        )

        for rcv_combination in combinations:
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
            slug = format_rcv_combination(rcv_combination, sep="_")
            save_pair_figure(fig, ambiguity_figure_filename(ambiguity_metrics, slug))

# ======================================================================================================================
# Close the files (the event file is never modified)
# ======================================================================================================================
portion_event.close()
if cpsd_tree is not None:
    cpsd_tree.close()
if mfp_reused:
    portion_mfp.close()
log("Done")
