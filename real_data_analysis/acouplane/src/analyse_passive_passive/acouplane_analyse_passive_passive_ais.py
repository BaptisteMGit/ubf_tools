#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_analyse_passive_passive_ais.py
@Time    :   2026/11/18 18:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Comparison of passive events (vessel passages with an AIS trajectory) between them,
             sub-box by sub-box.

             The sub-box datasets are produced by
             acouplane_passive_passive_ais_events_subbox_definition_batch_processing.py.
             In each sub-box, the library is made of the events of the box and the events compared to the
             library are the same events (passive / passive case). For each pair of events:
               1. the features (RTF, DCF) of both events are compared sample by sample (ambiguity surfaces);
               2. the theoretical intersection (closest samples of the two AIS trajectories) is compared
                  to the intersection estimated from the features.

             The long results (CPSD and ambiguity surfaces) are saved next to each sub-box file and reused when
             the script is run again with the same settings. The processing itself is in
             acouplane_passive_passive_compare.py.

             Outputs (same folder as the figures of the sub-box selection: pressure, spectrogram, ...):
               - <ROOT_IMG>/passive_ais_events_subbox/box<i>/passive_passive_comparison/ : map, features/ (RTF and
                 DCF of each event, optional), crossing distances, intersection positions,
                 ambiguity/<library>_<event>/ (one folder per pair of events)
               - <ROOT_IMG>/passive_ais_events_subbox/comparison_boxes_summary.csv and
                 comparison_all_boxes_intersection_positions.csv
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import traceback

import matplotlib.pyplot as plt
import pandas as pd

from real_data_analysis.acouplane.src.acouplane_cst import (
    ACOUPLANE_PASSIVE_NC_ROOT_FPATH,
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR,
)
from real_data_analysis.acouplane.src.analyse_passive_passive.acouplane_passive_passive_compare import (
    PassivePassiveSettings,
    compare_events_of_subbox,
    list_available_subboxes,
)

# ======================================================================================================================
# Settings
# ======================================================================================================================

# Sub-boxes to process. None -> all the sub-boxes whose dataset exists. Useful to resume or debug, e.g. [3, 4]
# plt.pause(180 * 60)
# boxes_to_process = None
boxes_to_process = [40, 41, 42, 43, 44, 45, 46, 47]

settings = PassivePassiveSettings(
    # --- Receivers --------------------------------------------------------------------------------------------------
    # Reference receiver of the pairs (rcv_i, reference_rcv): the same for the library and for the events
    reference_rcv="obs1",
    # Receivers that can be used (the reference receiver is ignored). None -> all the receivers of the events
    obs_ids=["obs1", "obs2", "obs4", "obs5", "obs6", "obs7"],
    # --- Frequency band of the features [Hz] ------------------------------------------------------------------------
    fmin=10,
    fmax=800,
    # --- RTF estimation, chosen separately for the library and for the events ---------------------------------------
    #   "spectral_ratio" -> S_ij / S_jj
    #   "cs_evd"         -> first eigenvector of the spatial covariance matrix (preferable for passive signals)
    library_rtf_method="cs_evd",
    event_rtf_method="cs_evd",
    # --- CPSD of the events (Welch method) --------------------------------------------------------------------------
    analysis_segment_duration_s=10,
    analysis_segment_alpha_overlap=0.75,
    nfft=2**13,
    nperseg=2**13,
    noverlap=2**12,
    # --- Pairs of events --------------------------------------------------------------------------------------------
    # False : each pair of events is computed once (A as library vs B as event).
    # True  : (A, B) and (B, A) are both computed (useful only if the two RTF methods differ).
    both_directions=False,
    # --- Intersections ----------------------------------------------------------------------------------------------
    # Two trajectories only have a theoretical intersection if they cross each other. The pairs whose closest samples
    # are further apart than this distance [m] are not analysed.
    max_crossing_distance_m=200,
    # Receivers combined to build the ambiguity surface (the surfaces of the receiver pairs are averaged)
    combination_method="mean_surfaces",
    # rcv_combinations=[
    #     ["obs2", "obs4", "obs7", "obs5", "obs6"],
    # ],
    rcv_combinations=[
        ["obs2"],
        ["obs4"],
        ["obs7"],
        ["obs5"],
        # ["obs2", "obs4"],
        # ["obs2", "obs7"],
        # ["obs2", "obs5"],
        # ["obs4", "obs7"],
        # ["obs4", "obs5"],
        # ["obs5", "obs7"],
        ["obs2", "obs4", "obs7"],
        ["obs2", "obs4", "obs5"],
        ["obs2", "obs4", "obs7", "obs5"],
        ["obs2", "obs4", "obs7", "obs5", "obs6"],
    ],
    # --- Ambiguity figures ------------------------------------------------------------------------------------------
    # Metrics displayed next to the real distance (always in the first panel): 3 among
    # "theta", "theta_gamma", "L2_gamma", "dcf", "theta_gamma_centered", "L2_gamma_centered"
    ambiguity_metrics=["theta_gamma_centered", "L2_gamma_centered", "dcf"],
    ambiguity_cmap="Grays_r",
    ambiguity_max_quantile=0.95,  # quantile of each feature distance used as upper bound of its color scale
    ambiguity_d_real_vmax=200,  # upper bound of the color scale of the real distance [m]
    max_ambiguity_figures=4000,  # safety limit on the number of ambiguity figures of ONE sub-box
    # Also compute the distances with gamma centered along frequency (needed by the "..._centered" metrics)
    compare_centered_gamma=True,
    # --- Switches ---------------------------------------------------------------------------------------------------
    # Reuse the saved CPSD and ambiguity surfaces if they were computed with exactly the same settings.
    # If something differs, the reason is displayed and the result is computed again.
    reuse_stored_results=True,
    save_cpsd_to_netcdf=True,
    save_mfp_to_netcdf=True,
    plot_map=True,
    # Plot the features (RTF amplitude and DCF) of the events used by the ambiguity surfaces (the events of the pairs
    # whose trajectories cross). Their CPSD and features are then computed or read even if the surfaces are reused.
    plot_features_event=True,
    save_intersection_positions=True,
    plot_ambiguity=True,
    verbose=True,
)

# ======================================================================================================================
# Folders and sub-boxes available
# ======================================================================================================================
# Same root as the selection script (box<i>/ folders already hold the selection, pressure, spectrogram images)
ROOT_IMG_SUBBOX = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR, "passive_ais_events_subbox"
)
os.makedirs(ROOT_IMG_SUBBOX, exist_ok=True)

available_boxes = list_available_subboxes(ACOUPLANE_PASSIVE_NC_ROOT_FPATH)
if not available_boxes:
    raise FileNotFoundError(
        f"No sub-box dataset '{ACOUPLANE_PASSIVE_NC_ROOT_FPATH}_ais_box<i>.nc' found: run "
        "acouplane_passive_passive_ais_events_subbox_definition_batch_processing.py first."
    )
print(f"{len(available_boxes)} sub-box dataset(s) available: {list(available_boxes)}")

if boxes_to_process is not None:
    unknown_boxes = [ibox for ibox in boxes_to_process if ibox not in available_boxes]
    if unknown_boxes:
        raise ValueError(
            f"No dataset for the sub-box(es) {unknown_boxes} (available: {list(available_boxes)})."
        )
    available_boxes = {ibox: available_boxes[ibox] for ibox in boxes_to_process}

# ======================================================================================================================
# Iterate over sub-boxes
# ======================================================================================================================
summary_records = []  # one row per box
all_position_records = []  # intersection positions of all the boxes
failed_boxes = []

for ibox, box_fpath in available_boxes.items():

    print(f"\n===== Sub-box {ibox} ({box_fpath}) =====")
    # The comparison has its own sub-folder in the folder of the box, next to the selection images
    root_img_box = os.path.join(
        ROOT_IMG_SUBBOX, f"box{ibox}", "passive_passive_comparison"
    )

    # One failing box must not stop the whole batch (the saved results of the other boxes are kept)
    try:
        summary, position_records = compare_events_of_subbox(
            box_fpath=box_fpath,
            ibox=ibox,
            settings=settings,
            root_img_box=root_img_box,
        )
        summary_records.append(summary)
        all_position_records.extend(position_records)

    except Exception as error:
        print(f"!!! Sub-box {ibox} failed, going on with the next one :")
        traceback.print_exc()
        plt.close("all")
        failed_boxes.append(ibox)
        summary_records.append(dict(box=ibox, status=f"failed: {error!r}"))

# ======================================================================================================================
# Summary
# ======================================================================================================================
df_summary = pd.DataFrame(summary_records)
summary_fpath = os.path.join(ROOT_IMG_SUBBOX, "comparison_boxes_summary.csv")
df_summary.to_csv(summary_fpath, index=False)
print(f"\nSummary per box saved : {summary_fpath}")
print(df_summary.to_string(index=False))

if all_position_records:
    positions_fpath = os.path.join(
        ROOT_IMG_SUBBOX, "comparison_all_boxes_intersection_positions.csv"
    )
    pd.DataFrame(all_position_records).to_csv(positions_fpath, index=False)
    print(f"Intersection positions of all the boxes saved : {positions_fpath}")

if failed_boxes:
    print(f"\nFailed boxes (to process again with boxes_to_process) : {failed_boxes}")
