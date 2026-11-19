#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_passive_passive_ais_events_subbox_definition_batch_processing.py
@Time    :   2026/11/18 16:01:44
@Author  :   Menetrier Baptiste
@Version :   1.1
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Process each sub-box of a large area to extract the AIS events (ship trajectories)
    and their associated pressure signals.

    Process:
        1. A large box surrounding the receivers is defined and all the AIS trajectories inside are extracted.
        2. The large box is subdivided into small sub-boxes, each crossed by a limited number of trajectories,
           which eases the analysis of the localization results.
        3. Each sub-box is processed independently:
            - filter 1 : keep the events for which every OBS recorded data,
            - filter 2 : keep the events which do not overlap any active emission,
            - filter 3 : remove the events which are suspiciously long (ship not crossing the box),
            - extraction of the pressure signals of the remaining events.
        4. For each sub-box, the figures illustrating the selection and the dataset are saved.

    Outputs:
        - <ROOT_IMG>/passive_ais_events_subbox/large_box_events.png, sub_boxes_map.png, sub_boxes_summary.csv
        - <ROOT_IMG>/passive_ais_events_subbox/box<i>/ : selection maps, pressure, spectrograms, temporal coverage
        - <ACOUPLANE_PASSIVE_NC_ROOT_FPATH>_ais_box<i>.nc : dataset of each sub-box (positions and pressure)
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import traceback

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from publication.publication_figure import color
from real_data_analysis.acouplane.src.acouplane_cst import (
    ACOUPLANE_POSITIONS_NC_FPATH,
    ACOUPLANE_PRESSURE_NC_FPATH,
    ACOUPLANE_PASSIVE_NC_ROOT_FPATH,
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR,
)
from real_data_analysis.acouplane.src.acouplane_console import make_figure_saver
from real_data_analysis.acouplane.src.acouplane_utils import (
    plot_spectrogram_section,
    plot_pressure_section,
)
from real_data_analysis.acouplane.src.analyse_passive_passive.acouplane_passive_passive_utils import (
    extract_ais_in_box,
    ais_to_distinct_events,
    assert_events_is_not_suspiciously_long,
    assert_no_events_during_emissions,
    assert_pressure_data_is_available,
    build_large_box,
    build_portion_events,
    build_sub_boxes,
    count_events,
    extract_pressure_events,
    get_position_nodes,
    plot_events_in_box,
    plot_events_temporal_coverage,
    save_portion_event,
)

# ======================================================================================================================
# Settings
# ======================================================================================================================

# --- Large box ---
# Receivers defining the area (OBS 6 is removed to reduce the area)
rcv_id_for_area_def = ["obs1", "obs2", "obs4", "obs7", "obs5"]
# Relative margin added around the receivers
large_box_offset_percentage = 0.004

# --- Sub-boxes ---
# Size of a sub-box, in degrees
subbox_lon_width = subbox_lat_width = 0.042

# --- Filters ---
# An event longer than this is not a ship crossing the box
max_event_duration_min = 60

# --- Spectrograms ---
spectrogram_nperseg = 2**10
spectrogram_noverlap = 2**9

# --- Switches ---
# Boxes to process (None = all). Useful to resume or debug, e.g. [3, 4]
boxes_to_process = None
# boxes_to_process = [6, 9]

# If False, a box whose dataset file already exists is skipped
overwrite_existing_boxes = True
plot_pressure = True
plot_spectrogram = True

# Maximum event labels allowed
max_number_of_event_labels = 10

# ======================================================================================================================
# Folders
# ======================================================================================================================
ROOT_IMG_SUBBOX = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR, "passive_ais_events_subbox"
)
os.makedirs(ROOT_IMG_SUBBOX, exist_ok=True)
save_root_figure = make_figure_saver(ROOT_IMG_SUBBOX)

# ======================================================================================================================
# Load useful data
# ======================================================================================================================
# Position dataset
ds_pos = xr.open_dataset(ACOUPLANE_POSITIONS_NC_FPATH)
# Pressure dataset (lazily loaded)
ds_pressure = xr.open_dataset(ACOUPLANE_PRESSURE_NC_FPATH)

# ======================================================================================================================
# Get large box
# ======================================================================================================================
large_box = build_large_box(
    ds_pos, rcv_ids=rcv_id_for_area_def, offset_percentage=large_box_offset_percentage
)
print(f"Large box : {large_box}")

# Extract all the AIS events of the large box
ds_ais_large_box = extract_ais_in_box(ds_pos, box=large_box)
events_large_box = ais_to_distinct_events(ds_ais_box=ds_ais_large_box)
portion_event_large_box = build_portion_events(events_large_box, box=large_box)
print(f"{count_events(portion_event_large_box)} events in the large box")

# Plot trajectories in the large box
fig, ax = plot_events_in_box(
    ds_pos,
    portion_event_large_box,
    title="AIS events in the large box",
    add_event_labels=False,  # hundreds of events : no label
)
save_root_figure(fig, "large_box_events.png")

# ======================================================================================================================
# Define sub-boxes
# ======================================================================================================================
sub_boxes = build_sub_boxes(
    large_box, subbox_lon_width=subbox_lon_width, subbox_lat_width=subbox_lat_width
)
print(f"{len(sub_boxes)} sub-boxes defined")

# Illustrate on map
fig, ax = plot_events_in_box(
    ds_pos,
    portion_event_large_box,
    title=f"{len(sub_boxes)} sub-boxes",
    add_event_labels=False,
)

for ibox, box in enumerate(sub_boxes):
    ax.add_patch(
        Rectangle(
            (box["lon_min"], box["lat_min"]),
            box["lon_max"] - box["lon_min"],
            box["lat_max"] - box["lat_min"],
            fc="none",
            ec=color(ibox),
            lw=5,
            # NOTE : too many boxes to add labels
            zorder=500,
        )
    )
    # Box number at the center, to match the saved folders
    ax.text(
        (box["lon_min"] + box["lon_max"]) / 2,
        (box["lat_min"] + box["lat_max"]) / 2,
        str(ibox),
        ha="center",
        va="center",
        fontsize=16,
        zorder=600,
    )
save_root_figure(fig, "sub_boxes_map.png")

# ======================================================================================================================
# Iterate over sub-boxes
# ======================================================================================================================

summary_records = []  # one row per box : number of events after each step
failed_boxes = []

for ibox, box in enumerate(sub_boxes):

    if boxes_to_process is not None and ibox not in boxes_to_process:
        continue

    print(f"\n===== Sub-box {ibox}/{len(sub_boxes) - 1} =====")

    box_data_fpath = f"{ACOUPLANE_PASSIVE_NC_ROOT_FPATH}_ais_box{ibox}.nc"
    if os.path.exists(box_data_fpath) and not overwrite_existing_boxes:
        print(f"Dataset {box_data_fpath} already exists, box skipped.")
        continue

    # Dedicated image folder of the box
    root_img_box = os.path.join(ROOT_IMG_SUBBOX, f"box{ibox}")
    save_box_figure = make_figure_saver(root_img_box)

    # One failing box must not stop the whole batch
    try:
        n_events = dict(box=ibox)

        # --------------------------------------------------------------------------------------------------------------
        # Extract the events of the current box
        # --------------------------------------------------------------------------------------------------------------
        ds_ais_box = extract_ais_in_box(ds_pos, box=box)
        events_box = ais_to_distinct_events(ds_ais_box=ds_ais_box)
        portion_event_box = build_portion_events(events_box, box=box)
        n_events["all"] = count_events(portion_event_box)
        print(f"{n_events['all']} events in the box")

        if n_events["all"] == 0:
            print("No event in this box, nothing to save.")
            summary_records.append(n_events)
            continue

        fig, ax = plot_events_in_box(
            ds_pos, portion_event_box, title=f"Box {ibox} : all events"
        )
        save_box_figure(fig, "selection_0_all_events.png")

        # --------------------------------------------------------------------------------------------------------------
        # First filter : pressure data available on every OBS
        # --------------------------------------------------------------------------------------------------------------
        portion_event_box = assert_pressure_data_is_available(
            portion_event_box, ds_pressure, verbose=False
        )
        n_events["pressure_available"] = count_events(portion_event_box)
        print(
            f"{n_events['pressure_available']} events after filter 1 (pressure available)"
        )

        add_event_labels = n_events["pressure_available"] <= max_number_of_event_labels
        fig, ax = plot_events_in_box(
            ds_pos,
            portion_event_box,
            title=f"Box {ibox} : pressure data available",
            add_event_labels=add_event_labels,
        )
        save_box_figure(fig, "selection_1_pressure_available.png")

        # --------------------------------------------------------------------------------------------------------------
        # Second filter : no event during the active emissions
        # --------------------------------------------------------------------------------------------------------------
        portion_event_box = assert_no_events_during_emissions(
            portion_event_box, ds_pos, verbose=False
        )
        n_events["no_emission"] = count_events(portion_event_box)
        print(f"{n_events['no_emission']} events after filter 2 (no emission)")

        add_event_labels = n_events["no_emission"] <= max_number_of_event_labels
        fig, ax = plot_events_in_box(
            ds_pos,
            portion_event_box,
            title=f"Box {ibox} : no emission during events",
            add_event_labels=add_event_labels,
        )
        save_box_figure(fig, "selection_2_no_emission.png")

        # --------------------------------------------------------------------------------------------------------------
        # Third filter : events not suspiciously long
        # --------------------------------------------------------------------------------------------------------------
        portion_event_box = assert_events_is_not_suspiciously_long(
            portion_event_box, verbose=False, max_duration_min=max_event_duration_min
        )
        n_events["not_too_long"] = count_events(portion_event_box)
        print(f"{n_events['not_too_long']} events after filter 3 (duration)")

        add_event_labels = n_events["not_too_long"] <= max_number_of_event_labels
        fig, ax = plot_events_in_box(
            ds_pos,
            portion_event_box,
            title=rf"Box {ibox} : duration $\leq$ {max_event_duration_min} min",
            add_event_labels=add_event_labels,
        )
        save_box_figure(fig, "selection_3_not_too_long.png")

        summary_records.append(n_events)

        if n_events["not_too_long"] == 0:
            print("No event left after the filters, nothing to save.")
            continue

        # --------------------------------------------------------------------------------------------------------------
        # Extract the associated pressure signals
        # --------------------------------------------------------------------------------------------------------------
        extract_pressure_events(portion_event_box, ds_pressure)

        # Pressure signals (one figure per event)
        if plot_pressure:
            single_fig = False
            for name, node in portion_event_box.pressure.children.items():
                ds_pressure_section = node.to_dataset()
                fig, axs = plot_pressure_section(
                    ds_pressure_section=ds_pressure_section, single_fig=single_fig
                )
                fig.suptitle(f"Event {ds_pressure_section.event_id}")
                save_box_figure(fig, os.path.join("pressure", f"{name}.png"))

        # Spectrograms (one figure per event)
        if plot_spectrogram:
            for name, node in portion_event_box.pressure.children.items():
                ds_pressure_section = node.to_dataset()
                try:
                    fig, axs = plot_spectrogram_section(
                        ds_pressure_section,
                        nperseg=spectrogram_nperseg,
                        noverlap=spectrogram_noverlap,
                        cmap="magma",
                    )
                    fig.suptitle(f"Event {ds_pressure_section.event_id}")
                    save_box_figure(fig, os.path.join("spectrogram", f"{name}.png"))
                except Exception as error:
                    # Typically OBS signals with different lengths
                    print(f"Spectrogram not plotted for {name} : {error!r}")
                    plt.close("all")

        # Temporal coverage of the events
        fig, axs = plot_events_temporal_coverage(portion_event_box)
        if fig is not None:
            save_box_figure(fig, "events_temporal_coverage.png")

        # --------------------------------------------------------------------------------------------------------------
        # Save the dataset of the box
        # --------------------------------------------------------------------------------------------------------------
        selection_settings = dict(
            box_index=ibox,
            box=box,
            large_box=large_box,
            max_event_duration_min=max_event_duration_min,
            rcv_id_for_area_def=rcv_id_for_area_def,
        )
        save_portion_event(portion_event_box, box_data_fpath, selection_settings)
        print(f"Dataset saved : {box_data_fpath}")

    except Exception:
        print(f"!!! Sub-box {ibox} failed, going on with the next one :")
        traceback.print_exc()
        plt.close("all")
        failed_boxes.append(ibox)

# ======================================================================================================================
# Summary
# ======================================================================================================================
df_summary = pd.DataFrame(summary_records)
summary_fpath = os.path.join(ROOT_IMG_SUBBOX, "sub_boxes_summary.csv")
df_summary.to_csv(summary_fpath, index=False)
print(f"\nSummary of the number of events per box saved : {summary_fpath}")
print(df_summary.to_string(index=False))

if failed_boxes:
    print(f"\nFailed boxes (to process again with boxes_to_process) : {failed_boxes}")
