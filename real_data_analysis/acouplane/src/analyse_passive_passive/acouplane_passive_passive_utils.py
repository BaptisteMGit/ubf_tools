#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_passive_passive_utils.py
@Time    :   2026/11/18 15:48:19
@Author  :   Menetrier Baptiste
@Version :   1.1
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Utilities to extract, filter and save the AIS events (ship trajectories) of a
    geographical box, together with the pressure signals recorded during these events.

    Structure of a "portion event" (xarray.DataTree):
        - root attributes : box_lon_min, box_lon_max, box_lat_min, box_lat_max
        - position/event_<i> : one dataset per event (datetime, lon, lat, start/stop datetime)
        - pressure/event_<i> : the pressure signals of the OBS during the event (optional)
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from real_data_analysis.acouplane.src.acouplane_utils import (
    plot_obs_pos,
    extract_pressure_section,
    convert_raw_pressure_to_physical_units,
)
from real_data_analysis.acouplane.src.acouplane_portion_io import save_tree_atomic

# ======================================================================================================================
# Box definition
# ======================================================================================================================


def build_large_box(ds_pos, rcv_ids, offset_percentage=0.0025):
    """Build the large box surrounding a set of receivers.

    The box is the bounding box of the receivers, enlarged by a small relative
    margin on each side.

    Parameters
    ----------
    ds_pos : xarray.Dataset
        Position dataset with the variables ``rcv_lon`` and ``rcv_lat`` and the
        coordinate ``rcv_id``.
    rcv_ids : list of str
        Receivers used to define the area (e.g. ``["obs1", "obs2"]``).
    offset_percentage : float, optional
        Relative margin added to the bounding box (applied as a multiplicative
        factor on the coordinates, half on each side), by default 0.0025.

    Returns
    -------
    box : dict
        Box with the keys ``lon_min``, ``lon_max``, ``lat_min``, ``lat_max``
        (degrees).
    """
    ds_pos_area = ds_pos.sel(rcv_id=rcv_ids)

    lon_min = float(ds_pos_area.rcv_lon.min().values) * (1 - offset_percentage / 2)
    lon_max = float(ds_pos_area.rcv_lon.max().values) * (1 + offset_percentage / 2)
    lat_min = float(ds_pos_area.rcv_lat.min().values) * (1 - offset_percentage / 2)
    lat_max = float(ds_pos_area.rcv_lat.max().values) * (1 + offset_percentage / 2)

    return dict(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)


def build_sub_boxes(large_box, subbox_lon_width, subbox_lat_width):
    """Subdivide a large box into a regular grid of smaller boxes.

    The boxes are ordered row by row (increasing latitude, then increasing
    longitude). The boxes of the last row / column are clipped to the large box
    so they can be smaller than requested.

    Parameters
    ----------
    large_box : dict
        Box to subdivide, with the keys ``lon_min``, ``lon_max``, ``lat_min``,
        ``lat_max`` (degrees).
    subbox_lon_width : float
        Width of a sub-box in longitude (degrees).
    subbox_lat_width : float
        Height of a sub-box in latitude (degrees).

    Returns
    -------
    sub_boxes : list of dict
        Sub-boxes, each with the same keys as ``large_box``. The index of a
        box in this list is its ``ibox`` identifier.
    """
    large_box_lon_width = large_box["lon_max"] - large_box["lon_min"]
    large_box_lat_width = large_box["lat_max"] - large_box["lat_min"]

    # Round before the ceil : a ratio of 4.0000000001 (float error) would otherwise
    # create an extra box of null width
    n_box_lon = int(np.ceil(round(large_box_lon_width / subbox_lon_width, 6)))
    n_box_lat = int(np.ceil(round(large_box_lat_width / subbox_lat_width, 6)))

    sub_boxes = []
    for i_subbox_lat in range(n_box_lat):
        lat_min = large_box["lat_min"] + i_subbox_lat * subbox_lat_width
        lat_max = min(lat_min + subbox_lat_width, large_box["lat_max"])

        for i_subbox_lon in range(n_box_lon):
            lon_min = large_box["lon_min"] + i_subbox_lon * subbox_lon_width
            lon_max = min(lon_min + subbox_lon_width, large_box["lon_max"])

            sub_boxes.append(
                dict(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
            )

    return sub_boxes


# ======================================================================================================================
# Extraction of the AIS events
# ======================================================================================================================


def extract_ais_in_box(ds_pos, box):
    """Extract the AIS positions located inside a box.

    Positions outside the box are set to NaN, then the times and the vessels
    (MMSI) without any position in the box are dropped. The ``mmsi``
    coordinate is replaced by an integer index to ease the selection.

    Parameters
    ----------
    ds_pos : xarray.Dataset
        Position dataset with the variables ``ais_lon``, ``ais_lat`` and
        ``ais_datetime`` (dimensions ``ais_datetime`` and ``mmsi``).
    box : dict
        Box with the keys ``lon_min``, ``lon_max``, ``lat_min``, ``lat_max``
        (degrees).

    Returns
    -------
    ds_ais_in_area : xarray.Dataset
        AIS positions inside the box (NaN outside the box for the vessels which
        only cross it). It can be empty if no vessel crosses the box.
    """
    lon_min = box.get("lon_min")
    lon_max = box.get("lon_max")
    lat_min = box.get("lat_min")
    lat_max = box.get("lat_max")

    # Keep only the AIS variables
    ds_ais = ds_pos.copy().drop_vars(
        [
            v
            for v in list(ds_pos.variables)
            if v not in ["ais_lon", "ais_lat", "ais_datetime"]
        ]
    )

    mask = (
        (ds_ais["ais_lon"] >= lon_min)
        & (ds_ais["ais_lon"] <= lon_max)
        & (ds_ais["ais_lat"] >= lat_min)
        & (ds_ais["ais_lat"] <= lat_max)
    )

    # Mask and drop coordinates that have no positions in the area
    ds_ais_in_area = (
        ds_ais.where(mask).dropna("ais_datetime", how="all").dropna("mmsi", how="all")
    )
    # Add mmsi to ease selection
    ds_ais_in_area = ds_ais_in_area.assign_coords(
        mmsi=np.arange(ds_ais_in_area.sizes["mmsi"])
    )
    return ds_ais_in_area


def ais_to_distinct_events(ds_ais_box):
    """Cut the AIS positions of each vessel into distinct events.

    A vessel can enter and leave the box several times. Each continuous stay
    in the box (a run of non-NaN positions) is one event.

    Parameters
    ----------
    ds_ais_box : xarray.Dataset
        AIS positions inside a box (see ``extract_ais_in_box``).

    Returns
    -------
    event_sections : list of xarray.Dataset
        One dataset per event, with the dimension ``ais_datetime`` and the
        scalar coordinate ``mmsi`` (index of the vessel in ``ds_ais_box``).
        Empty if no vessel is present.
    """
    event_sections = []
    for m in ds_ais_box.mmsi.values:
        pos_mmsi = ds_ais_box.sel(mmsi=m)

        # 1 where the vessel is outside the box (NaN position), 0 inside
        nan_indicator = np.isnan(pos_mmsi.ais_lon * pos_mmsi.ais_lat)
        edge_indicator = nan_indicator.astype(int)

        # The vessel leaves the box at a rising edge and enters it at a falling edge
        rising_edge = (
            np.diff(edge_indicator.values, append=edge_indicator.values[-1]) == 1
        )
        rising_edge_dt = pos_mmsi.ais_datetime.values[rising_edge]
        falling_edge = (
            np.diff(edge_indicator.values, append=edge_indicator.values[-1]) == -1
        )
        # Roll to avoid taking the previous point which is still a NaN (diff = a[i+1] - a[i])
        falling_edge_dt = pos_mmsi.ais_datetime.values[np.roll(falling_edge, +1)]

        n_section_for_current_mmsi = max(rising_edge_dt.size, falling_edge_dt.size)

        if n_section_for_current_mmsi == 0:
            continue

        # Edges might not be visible (vessel already in the box at the start of the
        # record, or still in the box at the end) : use the record bounds
        if falling_edge_dt.size < n_section_for_current_mmsi:
            falling_edge_dt = np.concatenate(
                [[pos_mmsi.ais_datetime.values[0]], falling_edge_dt]
            )
        if rising_edge_dt.size < n_section_for_current_mmsi:
            rising_edge_dt = np.concatenate(
                [rising_edge_dt, [pos_mmsi.ais_datetime.values[-1]]]
            )

        # One event per (enter, leave) pair
        for isec in range(n_section_for_current_mmsi):
            pos_sec = pos_mmsi.sel(
                ais_datetime=slice(falling_edge_dt[isec], rising_edge_dt[isec])
            )
            event_sections.append(pos_sec)

    return event_sections


def build_portion_events(events, box):
    """Gather the events in a portion event (DataTree).

    Parameters
    ----------
    events : list of xarray.Dataset
        Events, as returned by ``ais_to_distinct_events``.
    box : dict
        Box in which the events were extracted (``lon_min``, ``lon_max``,
        ``lat_min``, ``lat_max``, degrees). Stored, rounded, in the root
        attributes ``box_lon_min`` ... ``box_lat_max``.

    Returns
    -------
    portion_event : xarray.DataTree
        Tree with the variable ``event_id`` and one dataset ``position/event_<i>``
        per event, with the variables ``lon``, ``lat``, ``start_datetime``,
        ``stop_datetime`` (dimension ``datetime``) and the attributes
        ``event_id`` and ``MMSI``. Without ``position`` node if ``events`` is empty.
    """
    lon_min = box.get("lon_min")
    lon_max = box.get("lon_max")
    lat_min = box.get("lat_min")
    lat_max = box.get("lat_max")

    portion_event = xr.DataTree()
    event_ids = np.arange(len(events))
    portion_event["event_id"] = event_ids
    portion_event.attrs = dict(
        box_lon_min=np.round(lon_min, 4),
        box_lon_max=np.round(lon_max, 4),
        box_lat_min=np.round(lat_min, 4),
        box_lat_max=np.round(lat_max, 4),
    )

    for isec, pos_sec in enumerate(events):
        # Use the same names as the rest of the ACOUPLANE code
        ds_event_i = pos_sec.rename(
            dict(ais_datetime="datetime", ais_lon="lon", ais_lat="lat")
        )
        ds_event_i = ds_event_i.drop_vars("mmsi")
        ds_event_i["start_datetime"] = ds_event_i.datetime.min().values
        ds_event_i["stop_datetime"] = ds_event_i.datetime.max().values
        ds_event_i.attrs["event_id"] = isec
        ds_event_i.attrs["MMSI"] = pos_sec.mmsi.values

        path = f"position/event_{isec}"
        portion_event[path] = ds_event_i

    return portion_event


# ======================================================================================================================
# Accessors
# ======================================================================================================================


def get_position_nodes(portion_event):
    """Return the position nodes of a portion event, even when it has none.

    The filters can drop all the events, in which case the ``position`` node
    does not exist any more.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Portion event (see ``build_portion_events``).

    Returns
    -------
    nodes : dict
        ``{"event_<i>": DataTree node}``, empty if there is no event.
    """
    if "position" not in portion_event.children:
        return {}
    return dict(portion_event["position"].children)


def count_events(portion_event):
    """Count the events of a portion event.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Portion event (see ``build_portion_events``).

    Returns
    -------
    n_events : int
        Number of events in the ``position`` node (0 if none).
    """
    return len(get_position_nodes(portion_event))


# ======================================================================================================================
# Plots
# ======================================================================================================================


def plot_events_in_box(ds_pos, portion_event, title=None, add_event_labels=True):
    """Plot the receivers, the box and the trajectories of the events on a map.

    Parameters
    ----------
    ds_pos : xarray.Dataset
        Position dataset (receivers, see ``plot_obs_pos``).
    portion_event : xarray.DataTree
        Portion event, with the box in its root attributes.
    title : str or None, optional
        Title of the figure. By default none.
    add_event_labels : bool, optional
        If True (default), each event is labelled in the legend. Set it to False
        when there are too many events (e.g. the large box) : the legend then only
        contains the receivers and the box.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    _ = plot_obs_pos(ds_pos, ax=ax, add_legend=False, add_title=False)

    # Selected area
    ax.add_patch(
        Rectangle(
            (portion_event.box_lon_min, portion_event.box_lat_min),
            portion_event.box_lon_max - portion_event.box_lon_min,
            portion_event.box_lat_max - portion_event.box_lat_min,
            fc="none",
            ec="k",
            lw=5,
            label="Selected area",
        )
    )

    # One line per event
    for name, node in get_position_nodes(portion_event).items():
        ds_position_section = node.to_dataset()
        ax.plot(
            ds_position_section.lon,
            ds_position_section.lat,
            "-x",
            markersize=2,
            label=f"Event {ds_position_section.event_id}" if add_event_labels else None,
        )

    ax.legend(ncols=2, loc="upper right")

    if title is not None:
        ax.set_title(title)

    return fig, ax


def get_events_start_days(portion_event):
    """Get the distinct days (at midnight) on which the events start.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Portion event with at least one event.

    Returns
    -------
    days : numpy.ndarray of datetime64[ns]
        Sorted distinct days, at midnight.
    """
    start_days = [
        pd.Timestamp(node.to_dataset().start_datetime.values).normalize()
        for node in get_position_nodes(portion_event).values()
    ]
    return np.sort(np.unique(np.asarray(start_days, dtype="datetime64[ns]")))


def plot_events_temporal_coverage(portion_event):
    """Plot the time span of each event, with one panel per day.

    An event is drawn on the panel of the day on which it starts.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Portion event (see ``build_portion_events``).

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        None (with nothing plotted) if the portion has no event.
    axs : numpy.ndarray of matplotlib.axes.Axes or None
        One axis per day.
    """
    nodes = get_position_nodes(portion_event)
    if len(nodes) == 0:
        print("plot_events_temporal_coverage: no event to plot.")
        return None, None

    day_bounds = get_events_start_days(portion_event)
    fig, axs = plt.subplots(day_bounds.size, 1, figsize=(16, 16), sharex=False)
    axs = np.atleast_1d(axs)

    for name, node in nodes.items():
        ds_position_event = node.to_dataset()

        # Panel of the day on which the event starts
        start_day = pd.Timestamp(ds_position_event.start_datetime.values).normalize()
        i_day = int(np.searchsorted(day_bounds, np.datetime64(start_day, "ns")))

        axs[i_day].plot(
            ds_position_event.datetime,
            np.ones(ds_position_event.datetime.size) + ds_position_event.event_id + 1,
            linewidth=4,
            label=name,
        )

    fig.suptitle("Event AIS")
    for i_day, ax in enumerate(axs):
        ax.legend(fontsize=12, ncols=4)

        # Add date at the bottom-right corner
        ax.text(
            0.99,
            0.02,
            np.datetime_as_string(day_bounds[i_day], unit="D"),
            transform=ax.transAxes,
            fontsize=18,
            ha="right",
            va="bottom",
            color="black",
        )

    return fig, axs


# ======================================================================================================================
# Filters
# ======================================================================================================================


def filter_valid_events(portion_event, valid_events, reorder_events=False):
    """Rebuild a portion event keeping only some events.

    Only the ``position`` node is rebuilt, so the filters must be applied before
    the pressure extraction (``extract_pressure_events``).

    Parameters
    ----------
    portion_event : xarray.DataTree
        Portion event to filter.
    valid_events : list of str
        Names (``"event_<i>"``) of the events to keep.
    reorder_events : bool, optional
        If True the kept events are renumbered from 0. If False (default) they
        keep their original number, so ``event_id`` stays comparable between
        the successive filters.

    Returns
    -------
    portion_event_filter : xarray.DataTree
        New portion event, with the same root attributes. Without ``position``
        node if no event is kept.
    """
    portion_event_filter = xr.DataTree()

    for e_id_new, event_name in enumerate(valid_events):
        if not reorder_events:
            e_id_new = int(event_name.split("_")[-1])

        new_path = f"position/event_{e_id_new}"
        old = portion_event[f"position/{event_name}"].to_dataset()
        old.attrs["event_id"] = e_id_new
        portion_event_filter[new_path] = old

    portion_event_filter["event_id"] = np.arange(len(valid_events))
    portion_event_filter.attrs = portion_event.attrs

    return portion_event_filter


def _get_recording_bounds(ds_pressure, obs_id):
    """Get the start and stop datetimes of the recording of one OBS.

    Parameters
    ----------
    ds_pressure : xarray.Dataset
        Pressure dataset (``start_datetime`` per ``obs_id``, sampling frequency
        ``fs`` and the dimension ``time_<obs_id>``).
    obs_id : str
        Receiver identifier.

    Returns
    -------
    start : numpy.datetime64
        First sample of the recording.
    stop : numpy.datetime64
        End of the recording (start + number of samples / fs).
    """
    start = ds_pressure.start_datetime.sel(obs_id=obs_id).values
    stop = start + np.timedelta64(
        int(1 / ds_pressure.fs * ds_pressure.sizes[f"time_{obs_id}"] * 1e6),
        "us",
    )
    return start, stop


def assert_pressure_data_is_available(portion_event, ds_pressure, verbose=False):
    """Drop the events for which at least one OBS has no recording.

    An event is kept only if every OBS of ``ds_pressure`` recorded data
    during part of the event.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Portion event (see ``build_portion_events``).
    ds_pressure : xarray.Dataset
        Pressure dataset (see ``_get_recording_bounds``).
    verbose : bool, optional
        If True, print the number of dropped events. By default False.

    Returns
    -------
    portion_event : xarray.DataTree
        Filtered portion event (see ``filter_valid_events``).
    """
    nodes = get_position_nodes(portion_event)

    valid_events = []
    for name, node in nodes.items():
        ds_position_event = node.to_dataset()

        event_start = ds_position_event.start_datetime.values
        event_stop = ds_position_event.stop_datetime.values

        invalid_event = False
        for obs_id in ds_pressure.obs_id.values:
            start, stop = _get_recording_bounds(ds_pressure, obs_id)

            # Check if any data is available
            event_stops_before_recording = start >= event_stop
            event_starts_after_recording = stop <= event_start
            if event_stops_before_recording or event_starts_after_recording:
                invalid_event = True

        if not invalid_event:
            valid_events.append(name)

    if verbose:
        n_dropped = len(nodes) - len(valid_events)
        print(
            f"{n_dropped}/{len(nodes)} events dropped (no data available) "
            f"({len(valid_events)} events remaining)"
        )

    return filter_valid_events(portion_event, valid_events)


def assert_no_events_during_emissions(portion_event, ds_pos, verbose=False):
    """Drop the events which overlap the active emissions.

    An event is kept only if it ends before the first emission or starts after the
    last emission, so it cannot be polluted by the active source.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Portion event (see ``build_portion_events``).
    ds_pos : xarray.Dataset
        Position dataset with the variable ``src_datetime`` (emission times).
    verbose : bool, optional
        If True, print the number of remaining events. By default False.

    Returns
    -------
    portion_event : xarray.DataTree
        Filtered portion event (see ``filter_valid_events``).
    """
    valid_events = []
    for name, node in get_position_nodes(portion_event).items():
        ds_position_event = node.to_dataset()

        event_start = ds_position_event.start_datetime.values
        event_stop = ds_position_event.stop_datetime.values

        # Filter to avoid any emission
        event_stops_before_first_emission = ds_pos.src_datetime.values[0] >= event_stop
        event_starts_after_last_emission = ds_pos.src_datetime.values[-1] <= event_start

        if event_stops_before_first_emission or event_starts_after_last_emission:
            valid_events.append(name)

    if verbose:
        print(f"{len(valid_events)} events remaining")

    return filter_valid_events(portion_event, valid_events)


def assert_events_is_not_suspiciously_long(
    portion_event, verbose=False, max_duration_min=30
):
    """Remove the events with a too long duration.

    For small boxes it makes no sense for a ship to take hours to cross the
    box (it is most probably stopped, e.g. fishing or anchored).

    Parameters
    ----------
    portion_event : xarray.DataTree
        Portion event (see ``build_portion_events``).
    verbose : bool, optional
        If True, print the number of remaining events. By default False.
    max_duration_min : float, optional
        Maximum duration of an event, in minutes. By default 30.

    Returns
    -------
    portion_event : xarray.DataTree
        Filtered portion event (see ``filter_valid_events``).
    """
    valid_events = []
    for name, node in get_position_nodes(portion_event).items():
        ds_position_event = node.to_dataset()

        event_start = ds_position_event.start_datetime.values
        event_stop = ds_position_event.stop_datetime.values

        if (event_stop - event_start) <= np.timedelta64(max_duration_min, "m"):
            valid_events.append(name)

    if verbose:
        print(f"{len(valid_events)} events remaining")

    return filter_valid_events(portion_event, valid_events)


# ======================================================================================================================
# Pressure
# ======================================================================================================================


def extract_pressure_events(portion_event, ds_pressure):
    """Extract the pressure signals of each event and store them in the portion event.

    The portion event is modified in place: the node ``pressure/event_<i>``
    is added for each event, in physical units. The events should already
    have been checked with ``assert_pressure_data_is_available``; the check done
    here is only an extra level of security.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Portion event (see ``build_portion_events``), modified in place.
    ds_pressure : xarray.Dataset
        Raw pressure dataset.

    Returns
    -------
    events_without_pressure : list of str
        Events for which an OBS had no data. They are NOT removed from the
        ``position`` node, but have no ``pressure`` node.
    """
    events_without_pressure = []
    for name, node in get_position_nodes(portion_event).items():
        ds_position_event = node.to_dataset()

        event_start = ds_position_event.start_datetime.values
        event_stop = ds_position_event.stop_datetime.values

        invalid_event = False
        for obs_id in ds_pressure.obs_id.values:
            start, stop = _get_recording_bounds(ds_pressure, obs_id)

            event_stops_before_recording = start >= event_stop
            event_starts_after_recording = stop <= event_start
            if event_stops_before_recording or event_starts_after_recording:
                print(
                    f"OBS '{obs_id}' has no data between {event_start} and {event_stop} "
                    f"(recording spans {start} -> {stop})."
                )
                invalid_event = True

        if invalid_event:
            print(f"Event {name} has no pressure signal")
            events_without_pressure.append(name)
            continue

        ds_pressure_event = extract_pressure_section(
            event_start,
            event_stop,
            ds_pressure,
            obs_datetimes=None,
        )
        ds_pressure_event = convert_raw_pressure_to_physical_units(
            ds_pressure=ds_pressure_event
        )

        ds_pressure_event.attrs["event_id"] = ds_position_event.event_id

        # Store in event portion
        path = f"pressure/event_{ds_position_event.event_id}"
        portion_event[path] = ds_pressure_event

    return events_without_pressure


# ======================================================================================================================
# Saving
# ======================================================================================================================


def save_portion_event(portion_event, fpath, settings):
    """Save a portion event as a NetCDF file, atomically.

    The path of the file is stored in the root attribute ``portion_filepath``
    and the settings of the selection (box, filters) in ``computation_settings``
    (see ``save_tree_atomic``). A crash while writing never leaves a truncated file.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Portion event to save. Its root attributes are modified.
    fpath : str
        Destination file (the directory is created if needed).
    settings : dict
        Settings of the selection, kept in the file for traceability.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    portion_event.attrs["portion_filepath"] = fpath
    save_tree_atomic(portion_event, fpath, settings)


if __name__ == "__main__":
    pass
