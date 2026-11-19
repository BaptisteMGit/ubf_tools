#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_passive_passive_compare.py
@Time    :   2026/11/18 18:00:00
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Compare, within each sub-box, the passive events (AIS trajectories) between them.

    The sub-box files (``<ACOUPLANE_PASSIVE_NC_ROOT_FPATH>_ais_box<i>.nc``) are produced by
    ``acouplane_passive_passive_ais_events_subbox_definition_batch_processing.py``. They contain the groups
    ``position`` and ``pressure`` of the events kept by the selection filters.

    For one sub-box (``compare_events_of_subbox``), the library is made of the events of the box and the events
    compared to the library are the same events. Each pair of events is compared (an event is never compared to
    itself, and by default each pair is computed once). Only the pairs whose AIS trajectories can cross each other are
    kept, a check made from the real distance d_real between the analysis windows BEFORE the RTF / DCF features and the
    ambiguity surfaces (the long computation):
        1. the features (RTF, DCF) of the two events are compared sample by sample (ambiguity surfaces),
        2. the theoretical intersection (closest samples of the two AIS trajectories) is compared to the
           intersection estimated from the features.

    The long results (CPSD and ambiguity surfaces) are saved next to the sub-box file and reused when the
    comparison is run again with the same settings. The sub-box file itself is never modified.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import re
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

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
    get_event_window_positions,
)
from real_data_analysis.acouplane.src.acouplane_features import (
    get_distance_between_library_and_event,
)
from real_data_analysis.acouplane.src.acouplane_plot_event import plot_event_features
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


@dataclass
class PassivePassiveSettings:
    """Settings of the comparison of the events of a sub-box.

    Attributes
    ----------
    reference_rcv : str
        Reference receiver of the pairs (rcv_i, reference_rcv), the same for both events.
    obs_ids : list of str or None
        Receivers that can be used (the reference receiver is ignored). None -> all
        the receivers of the events.
    fmin, fmax : float
        Frequency band of the features [Hz].
    library_rtf_method, event_rtf_method : str
        RTF estimator of the library and of the tested events (``"spectral_ratio"`` or
        ``"cs_evd"``, see ``acouplane_features.RTF_METHODS``).
    analysis_segment_duration_s, analysis_segment_alpha_overlap, nfft, nperseg, noverlap
        Welch CPSD estimation of the events (analysis window duration [s], overlap
        between windows, FFT length, Welch segment length and overlap [samples]).
    both_directions : bool
        If False (default), each pair of events is computed once (A as library vs B as
        event). If True, both (A, B) and (B, A) are computed: useful only when the
        library and event RTF methods differ.
    max_crossing_distance_m : float
        Pairs whose closest samples are further apart than this distance [m] have no
        theoretical intersection and are not analysed.
    combination_method : str
        Combination of the receiver pairs (``"mean_surfaces"``: surfaces averaged).
    rcv_combinations : list of list of str
        Combinations of receivers studied.
    ambiguity_metrics : list of str
        Metrics shown next to the real distance (3 among ``"theta"``, ``"theta_gamma"``,
        ``"L2_gamma"``, ``"dcf"``, ``"theta_gamma_centered"``, ``"L2_gamma_centered"``).
    ambiguity_cmap : str
        Colormap of the ambiguity figures.
    ambiguity_max_quantile : float
        Quantile of each feature distance used as upper bound of its color scale.
    ambiguity_d_real_vmax : float
        Upper bound of the color scale of the real distance [m].
    max_ambiguity_figures : int
        Safety limit on the number of ambiguity figures of ONE sub-box. If exceeded,
        the figures of the box are skipped (the tables and surfaces are kept).
    compare_centered_gamma : bool
        Also compute the distances with gamma centered along frequency.
    reuse_stored_results, save_cpsd_to_netcdf, save_mfp_to_netcdf : bool
        Reuse the saved CPSD / surfaces when computed with identical settings; save them.
    plot_map, plot_features_event, save_intersection_positions, plot_ambiguity, verbose : bool
        Outputs and console messages. ``plot_features_event`` saves the RTF amplitude and the
        DCF of the events used by the ambiguity surfaces (the events of the pairs whose trajectories cross) only.
        Their CPSD and features are then needed even if the surfaces are reused.
    """

    reference_rcv: str = "obs1"
    obs_ids: list = None
    fmin: float = 10
    fmax: float = 500
    library_rtf_method: str = "cs_evd"
    event_rtf_method: str = "cs_evd"
    analysis_segment_duration_s: float = 10
    analysis_segment_alpha_overlap: float = 0.75
    nfft: int = 2**13
    nperseg: int = 2**13
    noverlap: int = 2**12
    both_directions: bool = False
    max_crossing_distance_m: float = 200
    combination_method: str = "mean_surfaces"
    rcv_combinations: list = field(
        default_factory=lambda: [["obs2", "obs4", "obs7", "obs5", "obs6"]]
    )
    ambiguity_metrics: list = field(
        default_factory=lambda: ["theta_gamma_centered", "L2_gamma_centered", "dcf"]
    )
    ambiguity_cmap: str = "Grays_r"
    ambiguity_max_quantile: float = 0.95
    ambiguity_d_real_vmax: float = 100
    max_ambiguity_figures: int = 200
    compare_centered_gamma: bool = True
    reuse_stored_results: bool = True
    save_cpsd_to_netcdf: bool = True
    save_mfp_to_netcdf: bool = True
    plot_map: bool = True
    plot_features_event: bool = False
    save_intersection_positions: bool = True
    plot_ambiguity: bool = True
    verbose: bool = True


# ======================================================================================================================
# Files of the sub-boxes
# ======================================================================================================================


def list_available_subboxes(root_fpath):
    """List the sub-box files produced by the selection script.

    Parameters
    ----------
    root_fpath : str
        Root of the file names (``ACOUPLANE_PASSIVE_NC_ROOT_FPATH``): the files are
        ``<root_fpath>_ais_box<i>.nc``. Results saved next to them (``..._cpsd.nc``,
        ``..._mfp.nc``) are ignored.

    Returns
    -------
    available_boxes : dict
        ``{i: file path}`` sorted by box index; empty if no file exists.
    """
    folder = os.path.dirname(root_fpath) or "."
    prefix = os.path.basename(root_fpath) + "_ais_box"
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.nc")

    available_boxes = {}
    if not os.path.isdir(folder):
        return available_boxes

    for file_name in os.listdir(folder):
        match = pattern.fullmatch(file_name)
        if match:
            available_boxes[int(match.group(1))] = os.path.join(folder, file_name)

    return dict(sorted(available_boxes.items()))


def get_subbox_result_paths(box_fpath):
    """Get the paths of the results saved next to a sub-box file.

    Parameters
    ----------
    box_fpath : str
        Sub-box file (``..._ais_box<i>.nc``).

    Returns
    -------
    cpsd_fpath : str
        File of the CPSD of the events (``..._ais_box<i>_cpsd.nc``).
    mfp_fpath : str
        File of the ambiguity surfaces (``..._ais_box<i>_passive_passive_mfp.nc``).
    """
    base, ext = os.path.splitext(box_fpath)
    return f"{base}_cpsd{ext}", f"{base}_passive_passive_mfp{ext}"


# ======================================================================================================================
# Events and pairs
# ======================================================================================================================


def _natural_event_key(event_name):
    """Sort key putting ``event_9`` before ``event_10``."""
    return int(event_name.split("_")[-1])


def list_box_events(portion_event):
    """List the events of a sub-box dataset.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Sub-box dataset (opened with ``open_portion``).

    Returns
    -------
    event_names : list of str
        Names (``"event_<i>"``) of the events that have a pressure signal, in increasing
        event number.

    Raises
    ------
    ValueError
        If the dataset has no ``pressure`` group (no signal to analyse).
    """
    if "pressure" not in portion_event.children:
        raise ValueError(
            "No 'pressure' group in the sub-box file: the pressure signals were not extracted."
        )
    return sorted(portion_event.pressure.children, key=_natural_event_key)


def build_event_pairs(event_names, both_directions=False):
    """Build the (library event, tested event) pairs comparing the events between them.

    Parameters
    ----------
    event_names : list of str
        Events of the sub-box.
    both_directions : bool, optional
        If False (default), each pair of events appears once, as (A, B) with A before B.
        If True, (B, A) is added too.

    Returns
    -------
    pairs : list of tuple of str
        ``(library event, tested event)``; never an event with itself.
    library_events : list of str
        Distinct library events of the pairs, in order.
    tested_events : list of str
        Distinct tested events of the pairs, in order.
    """
    names = sorted(event_names, key=_natural_event_key)

    pairs = []
    for i_first, first in enumerate(names):
        for second in names[i_first + 1 :]:
            pairs.append((first, second))
            if both_directions:
                pairs.append((second, first))

    library_events = [name for name in names if any(p[0] == name for p in pairs)]
    tested_events = [name for name in names if any(p[1] == name for p in pairs)]
    return pairs, library_events, tested_events


# ======================================================================================================================
# Pre-selection of the pairs from the real distance between the analysis windows (before the ambiguity surfaces)
# ======================================================================================================================


def select_crossing_event_pairs(window_positions, pairs, max_crossing_distance_m):
    """Keep the pairs of events whose trajectories cross, before computing any CPSD or ambiguity surface.

    For each pair, the real-space distance ``d_real`` between every analysis window of the library event and every
    analysis window of the tested event is computed from the positions of the windows (see
    ``get_distance_between_library_and_event``) and its minimum is the crossing distance. This is exactly the value
    that ``get_crossing_distance_m`` reads in the ambiguity surfaces, but it needs only the positions of the windows
    (``get_event_window_positions``): no CPSD, no RTF, no DCF, no feature distance.

    Parameters
    ----------
    window_positions : dict
        ``{event name: xarray.Dataset}`` with ``event_lon`` / ``event_lat`` of dimension ``event_datetime`` (see
        ``get_event_window_positions``), for every event of ``pairs``.
    pairs : list of tuple of str
        ``(library event, tested event)`` pairs to examine.
    max_crossing_distance_m : float
        A pair is kept if its crossing distance is at most this value [m].

    Returns
    -------
    kept_pairs : list of tuple of str
        Pairs whose trajectories cross, in the order of ``pairs``.
    records : list of dict
        One record per examined pair: ``library_event``, ``tested_event``, ``crossing_distance_m`` (NaN if no
        valid position) and ``kept``.
    """
    kept_pairs = []
    records = []
    for library_name, event_name in pairs:
        d_real = get_distance_between_library_and_event(
            window_positions[library_name], window_positions[event_name]
        )
        crossing_distance_m = float(np.nanmin(d_real)) if d_real.size else float("nan")

        kept = bool(crossing_distance_m <= max_crossing_distance_m)  # False for NaN
        records.append(
            dict(
                library_event=library_name,
                tested_event=event_name,
                crossing_distance_m=crossing_distance_m,
                kept=kept,
            )
        )
        if kept:
            kept_pairs.append((library_name, event_name))

    return kept_pairs, records


def events_in_pairs(pairs, event_names):
    """Distinct library events and tested events of a list of pairs.

    Parameters
    ----------
    pairs : list of tuple of str
        ``(library event, tested event)`` pairs.
    event_names : list of str
        Events of the sub-box, which gives the order of the result.

    Returns
    -------
    library_events : list of str
        Events appearing as library in at least one pair.
    tested_events : list of str
        Events appearing as tested event in at least one pair.
    """
    names = sorted(event_names, key=_natural_event_key)
    library_events = [name for name in names if any(pair[0] == name for pair in pairs)]
    tested_events = [name for name in names if any(pair[1] == name for pair in pairs)]
    return library_events, tested_events


# ======================================================================================================================
# Steps of the comparison
# ======================================================================================================================


def _prepare_cpsd(portion_event, needed_events, cpsd_fpath, cpsd_settings, settings, log):
    """Plug the CPSD of the needed events into the sub-box tree (read from the file and/or computed, then saved).

    The CPSD file may hold only some events of the box. The events already in the file (saved with the same
    settings) are read, the missing ones are computed (long) and the file is rewritten with all of them.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Sub-box tree, modified in memory only: the group ``feature`` is added, with the needed events only.
    needed_events : list of str
        Events whose CPSD is needed.
    cpsd_fpath : str
        File where the CPSD is saved / read.
    cpsd_settings : dict
        Everything the CPSD depends on (a saved CPSD is reused only if identical). The events are NOT part of it.
    settings : PassivePassiveSettings
        Comparison settings.
    log : callable
        Progress function (see ``make_logger``).

    Returns
    -------
    cpsd_tree : xarray.DataTree or None
        Lazy tree of the CPSD file when all the needed events were read from it (to close at the end), else None.
    cpsd_reused : bool
        True if no CPSD had to be computed.
    """
    stored_tree = None
    stored_events = []
    if settings.reuse_stored_results:
        stored_tree, reason = load_tree_if_settings_match(cpsd_fpath, cpsd_settings)
        log(f"Saved CPSD: {reason}", level=1)
        if stored_tree is not None:
            stored_events = list(stored_tree.feature.children)

    events_from_file = [name for name in needed_events if name in stored_events]
    events_to_compute = [name for name in needed_events if name not in stored_events]
    log(
        f"CPSD of {len(needed_events)} event(s): {len(events_from_file)} in the saved file, "
        f"{len(events_to_compute)} to compute",
        level=1,
    )

    # Everything is in the file: plug it (it stays lazily read from its file)
    if not events_to_compute:
        for event_name in events_from_file:
            portion_event[f"/feature/{event_name}"] = stored_tree.feature[event_name].to_dataset()
        log(f"CPSD loaded from {cpsd_fpath}", level=1)
        return stored_tree, True

    # Some events are missing: the file will be rewritten, so the events it holds are loaded in memory first
    # (a file cannot be replaced while it is open)
    stored_datasets = {}
    if stored_tree is not None:
        for event_name in stored_events:
            stored_datasets[event_name] = stored_tree.feature[event_name].to_dataset().load()
        stored_tree.close()
    for event_name in events_from_file:
        portion_event[f"/feature/{event_name}"] = stored_datasets[event_name]

    # Compute the missing events (long)
    log(
        f"Computing the CPSD (Welch) of {settings.analysis_segment_duration_s} s windows, "
        f"overlap {settings.analysis_segment_alpha_overlap}, band [{settings.fmin}, {settings.fmax}] Hz",
        level=1,
    )
    derive_portion_cpsd_welch(
        portion_tree=portion_event,
        analysis_segment_duration_s=settings.analysis_segment_duration_s,
        analysis_segment_alpha_overlap=settings.analysis_segment_alpha_overlap,
        nfft=settings.nfft,
        nperseg=settings.nperseg,
        noverlap=settings.noverlap,
        obs_ids=None,
        fmin=settings.fmin,
        fmax=settings.fmax,
        event_sections=events_to_compute,
    )

    # Save right away, together with the events already in the file
    if settings.save_cpsd_to_netcdf:
        cpsd_to_save = xr.DataTree()
        for event_name, dataset in stored_datasets.items():
            cpsd_to_save[f"/feature/{event_name}"] = dataset
        for event_name in events_to_compute:
            cpsd_to_save[f"/feature/{event_name}"] = portion_event.feature[event_name].to_dataset()
        save_tree_atomic(cpsd_to_save, cpsd_fpath, cpsd_settings)
        log(f"CPSD saved: {cpsd_fpath}", level=1)

    return None, False


def _plot_box_map(portion_event, event_names, ibox, save_figure):
    """Plot and save the map of the trajectories of the events of a sub-box.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Sub-box tree with the group ``position``.
    event_names : list of str
        Events to draw.
    ibox : int
        Sub-box index (title).
    save_figure : callable
        ``save_figure(fig, filename)`` (see ``make_figure_saver``).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    for event_name in event_names:
        ds_position = portion_event.position[event_name].to_dataset()
        ax.plot(
            ds_position["lon"].values,
            ds_position["lat"].values,
            linewidth=2,
            label=event_name,
        )

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Sub-box {ibox}")
    ax.legend(ncols=2)
    save_figure(fig, "map_events.png")


def _plot_events_features(portion_event, event_names, root_img_box, log):
    """Plot and save the features (RTF amplitude, DCF) of every event of a sub-box.

    Parameters
    ----------
    portion_event : xarray.DataTree
        Sub-box tree with the group ``feature`` (see ``derive_portion_features``).
    event_names : list of str
        Events to plot.
    root_img_box : str
        Folder of the sub-box results (figures in ``features/``).
    log : callable
        Progress function.

    Returns
    -------
    None
    """
    save_features_figure = make_figure_saver(os.path.join(root_img_box, "features"), log)

    for event_name in event_names:
        fig_rtf, fig_dcf = plot_event_features(
            portion_event.feature[event_name].to_dataset()
        )
        fig_rtf.suptitle(f"Event {event_name}: RTF amplitude")
        fig_dcf.suptitle(f"Event {event_name}: DCF")
        save_features_figure(fig_rtf, f"event_rtf_amplitude_{event_name}.png")
        save_features_figure(fig_dcf, f"event_dcf_{event_name}.png")


def _select_crossing_pairs(portion_mfp, settings, root_img_box, log):
    """Keep the pairs of events whose trajectories cross each other.

    The distance between the closest samples of the two trajectories is computed for
    every compared pair and saved in ``crossing_distances.csv``.

    Parameters
    ----------
    portion_mfp : xarray.DataTree
        Ambiguity surfaces of all the pairs.
    settings : PassivePassiveSettings
        Uses ``max_crossing_distance_m``.
    root_img_box : str
        Folder of the sub-box results.
    log : callable
        Progress function.

    Returns
    -------
    datasets_mfp : dict
        ``{(library event, tested event): ambiguity surfaces}`` for the pairs that cross.
    """
    datasets_mfp = {}
    crossing_records = []

    for library_name, event_name in list_stored_intersections(portion_mfp):
        ds_mfp = portion_mfp[f"/mfp_intersection/{library_name}/{event_name}"].to_dataset()

        # Distance between the two trajectories at their closest samples
        crossing_distance_m = get_crossing_distance_m(ds_mfp)
        is_crossing = crossing_distance_m <= settings.max_crossing_distance_m

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
                else f" (> {settings.max_crossing_distance_m} m: no intersection, not analysed)"
            ),
            level=1,
        )
        if is_crossing:
            datasets_mfp[(library_name, event_name)] = ds_mfp

    pd.DataFrame(crossing_records).to_csv(
        os.path.join(root_img_box, "crossing_distances.csv"), index=False
    )
    return datasets_mfp


def _intersection_position_records(datasets_mfp, combinations, settings):
    """Theoretical and estimated intersections of every analysed pair, as records.

    Parameters
    ----------
    datasets_mfp : dict
        ``{(library event, tested event): ambiguity surfaces}``.
    combinations : list of list of str
        Combinations of receivers.
    settings : PassivePassiveSettings
        Uses ``combination_method``.

    Returns
    -------
    position_records : list of dict
        One record for the theoretical intersection and one for each type of distance, for
        each pair and each combination (see ``combination_intersection_positions``).
    """
    position_records = []
    for (library_name, event_name), ds_mfp in datasets_mfp.items():
        for rcv_combination in combinations:
            for record in combination_intersection_positions(
                ds_mfp, rcv_combination, method=settings.combination_method
            ):
                position_records.append(
                    dict(
                        library_event=library_name,
                        tested_event=event_name,
                        combination=format_rcv_combination(rcv_combination),
                        **record,
                    )
                )
    return position_records


def _plot_ambiguity_figures(
    datasets_mfp, combinations, settings, root_img_box, log
):
    """Plot and save the ambiguity figures (one folder per pair of events).

    Parameters
    ----------
    datasets_mfp : dict
        ``{(library event, tested event): ambiguity surfaces}``.
    combinations : list of list of str
        Combinations of receivers (one figure per pair and per combination).
    settings : PassivePassiveSettings
        Figure settings and ``max_ambiguity_figures``.
    root_img_box : str
        Folder of the sub-box results (figures in ``ambiguity/<library>_<event>/``).
    log : callable
        Progress function.

    Returns
    -------
    n_figures : int
        Number of figures saved (0 if the safety limit is exceeded: nothing is plotted).
    """
    n_figures = len(datasets_mfp) * len(combinations)

    # Safety: do not generate hundreds of figures by mistake
    if n_figures > settings.max_ambiguity_figures:
        log(
            f"WARNING: {n_figures} ambiguity figures requested ({len(datasets_mfp)} pair(s) x "
            f"{len(combinations)} combination(s)), above max_ambiguity_figures = "
            f"{settings.max_ambiguity_figures}: figures of this box skipped (select fewer "
            "combinations or raise the limit).",
            level=1,
        )
        return 0
    log(f"{n_figures} figure(s)", level=1)

    for (library_name, event_name), ds_mfp in datasets_mfp.items():
        # One folder per pair
        save_pair_figure = make_figure_saver(
            os.path.join(root_img_box, "ambiguity", f"{library_name}_{event_name}"), log
        )
        for rcv_combination in combinations:
            fig = plot_combined_intersection_ambiguity(
                ds_mfp,
                rcv_combination,
                f"library {library_name}",  # names displayed on the axes
                f"event {event_name}",
                method=settings.combination_method,
                cmap=settings.ambiguity_cmap,
                max_quantile=settings.ambiguity_max_quantile,
                metrics=settings.ambiguity_metrics,
                d_real_vmax=settings.ambiguity_d_real_vmax,
            )
            slug = format_rcv_combination(rcv_combination, sep="_")
            save_pair_figure(
                fig, ambiguity_figure_filename(settings.ambiguity_metrics, slug)
            )

    return n_figures


# ======================================================================================================================
# Comparison of the events of one sub-box
# ======================================================================================================================


def compare_events_of_subbox(box_fpath, ibox, settings, root_img_box):
    """Compare the events of one sub-box between them.

    Steps: pre-selection of the pairs from the positions of the analysis windows, CPSD of the events used by the
    selected pairs only (reused if saved with the same settings), features (plotted if
    ``settings.plot_features_event``),
    ambiguity surfaces of each pair of events (reused if saved with the same settings),
    theoretical intersections (pairs whose trajectories cross), intersection positions
    table and ambiguity figures. The sub-box file is only read.

    Parameters
    ----------
    box_fpath : str
        Sub-box file (``..._ais_box<i>.nc``, groups ``position`` and ``pressure``).
    ibox : int
        Sub-box index (messages, tables).
    settings : PassivePassiveSettings
        Settings of the comparison.
    root_img_box : str
        Folder where the figures and tables of this sub-box are saved.

    Returns
    -------
    summary : dict
        ``box``, ``n_events``, ``n_events_used`` (events of the pairs computed: the only ones with a CPSD), ``n_pairs`` (all the pairs), ``n_pairs_to_compute`` (pairs whose
        trajectories cross, the only ones computed), ``n_crossing_pairs`` (after the final check), ``n_figures``,
        ``cpsd_reused``, ``mfp_reused`` and ``status`` (``"done"``, ``"skipped: ..."``).
    position_records : list of dict
        Theoretical and estimated intersections, with the column ``box`` (empty if
        nothing was analysed). See ``_intersection_position_records``.

    Raises
    ------
    ValueError
        If the file has no ``pressure`` group, or if the receivers / frequencies of the
        events are not compatible (see ``apply_mfp_strategy_to_portions``).
    """
    log = make_logger(None, settings.verbose, label=f"box {ibox}")
    save_figure = make_figure_saver(root_img_box, log)
    cpsd_fpath, mfp_fpath = get_subbox_result_paths(box_fpath)

    summary = dict(
        box=ibox,
        n_events=0,
        n_pairs=0,
        n_pairs_to_compute=0,
        n_events_used=0,
        n_crossing_pairs=0,
        n_figures=0,
        cpsd_reused=False,
        mfp_reused=False,
        status="done",
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Step 1 - Events of the sub-box and pairs to compare
    # ------------------------------------------------------------------------------------------------------------------
    log("Step 1 - Opening the sub-box file (lazy: nothing is loaded in memory yet)")
    portion_event = open_portion(box_fpath)
    cpsd_tree = None
    portion_mfp = None

    try:
        event_names = list_box_events(portion_event)
        summary["n_events"] = len(event_names)
        log(f"{len(event_names)} event(s): {event_names}", level=1)

        # Every pair of events of the box
        candidate_pairs, _, _ = build_event_pairs(event_names, settings.both_directions)
        summary["n_pairs"] = len(candidate_pairs)
        if candidate_pairs:
            log(f"{len(candidate_pairs)} pair(s) of events in the box", level=1)
        else:
            summary["status"] = "skipped: less than 2 events"
            log("Less than 2 events: nothing to compare", level=1)
            return summary, []

        if settings.plot_map:
            _plot_box_map(portion_event, event_names, ibox, save_figure)

        # --------------------------------------------------------------------------------------------------------------
        # Step 2 - Can the saved ambiguity surfaces be reused ?
        # --------------------------------------------------------------------------------------------------------------
        log("Step 2 - Saved ambiguity surfaces")

        # Everything the CPSD depends on. A saved CPSD is reused only if all of this is identical.
        cpsd_settings = dict(
            box_file=file_fingerprint(box_fpath),
            analysis_segment_duration_s=settings.analysis_segment_duration_s,
            analysis_segment_alpha_overlap=settings.analysis_segment_alpha_overlap,
            nfft=settings.nfft,
            nperseg=settings.nperseg,
            noverlap=settings.noverlap,
            fmin=settings.fmin,
            fmax=settings.fmax,
        )
        # Everything the ambiguity surfaces depend on. The surfaces are only computed for the pairs whose trajectories
        # cross (step 4), so the pairs requested and the crossing distance are part of the settings.
        mfp_settings = dict(
            cpsd=cpsd_settings,
            reference_rcv=settings.reference_rcv,
            fmin=settings.fmin,
            fmax=settings.fmax,
            library_rtf_method=settings.library_rtf_method,
            event_rtf_method=settings.event_rtf_method,
            pairs=[f"{library}|{event}" for library, event in candidate_pairs],
            max_crossing_distance_m=settings.max_crossing_distance_m,
            obs_ids=None if settings.obs_ids is None else sorted(settings.obs_ids),
            compare_centered_gamma=settings.compare_centered_gamma,
        )

        mfp_reused = False
        if not candidate_pairs:
            log("No pair: no ambiguity surface", level=1)
        elif settings.reuse_stored_results:
            portion_mfp, reason = load_tree_if_settings_match(mfp_fpath, mfp_settings)
            log(f"Saved ambiguity surfaces: {reason}", level=1)
            mfp_reused = portion_mfp is not None
        summary["mfp_reused"] = mfp_reused

        # Pairs with surfaces: read from the saved file if reused, else selected at step 3
        pairs = list_stored_intersections(portion_mfp) if mfp_reused else []
        need_surfaces = len(candidate_pairs) > 0 and not mfp_reused

        # --------------------------------------------------------------------------------------------------------------
        # Step 3 - Which pairs can have an intersection ? (real distance between the windows, before any CPSD)
        # --------------------------------------------------------------------------------------------------------------
        if need_surfaces:
            log("Step 3 - Pre-selection of the pairs whose trajectories cross (minimum of d_real, no CPSD needed)")
            library_candidates, tested_candidates = events_in_pairs(candidate_pairs, event_names)
            window_positions = get_event_window_positions(
                portion_event,
                analysis_segment_duration_s=settings.analysis_segment_duration_s,
                analysis_segment_alpha_overlap=settings.analysis_segment_alpha_overlap,
                obs_ids=None,
                event_sections=sorted(set(library_candidates) | set(tested_candidates), key=_natural_event_key),
            )
            pairs, crossing_records = select_crossing_event_pairs(
                window_positions, candidate_pairs, settings.max_crossing_distance_m
            )
            pd.DataFrame(crossing_records).to_csv(
                os.path.join(root_img_box, "preselection_crossing_distances.csv"), index=False
            )
            for record in crossing_records:
                log(
                    f"Library '{record['library_event']}' x event '{record['tested_event']}': closest windows "
                    f"{record['crossing_distance_m']:.0f} m apart"
                    + ("" if record["kept"] else " (no intersection: surface not computed)"),
                    level=1,
                )
            log(
                f"{len(pairs)}/{len(candidate_pairs)} pair(s) to compute "
                f"(closest windows at most {settings.max_crossing_distance_m} m apart)",
                level=1,
            )
        elif mfp_reused:
            log(f"Step 3 - Not needed: {len(pairs)} pair(s) stored with the saved surfaces")
        summary["n_pairs_to_compute"] = len(pairs)

        has_pairs = len(pairs) > 0
        library_events, tested_events = events_in_pairs(pairs, event_names)
        # Only the events of the pairs are worth a CPSD, features and plots
        events_used = sorted(set(library_events) | set(tested_events), key=_natural_event_key)
        summary["n_events_used"] = len(events_used)

        if need_surfaces and not has_pairs:
            summary["status"] = (
                f"done: no pair crosses (closest windows > {settings.max_crossing_distance_m} m apart)"
            )
            log("No pair of trajectories crosses: no CPSD, no surface to compute", level=1)
            return summary, []

        compute_surfaces = need_surfaces and has_pairs

        # --------------------------------------------------------------------------------------------------------------
        # Step 4 - CPSD of the events used (only if needed)
        # --------------------------------------------------------------------------------------------------------------
        need_cpsd = compute_surfaces or (settings.plot_features_event and has_pairs)
        if need_cpsd:
            log(f"Step 4 - CPSD of the {len(events_used)} event(s) used: {events_used}")
            cpsd_tree, summary["cpsd_reused"] = _prepare_cpsd(
                portion_event, events_used, cpsd_fpath, cpsd_settings, settings, log
            )
        else:
            log("Step 4 - Not needed")

        # --------------------------------------------------------------------------------------------------------------
        # Step 5 - Features of the events used (the library only if the surfaces are computed)
        # --------------------------------------------------------------------------------------------------------------
        portion_library = None
        if need_cpsd:
            log("Step 5 - Features of the library and of the events")
            if compute_surfaces:
                # The library is built from the CPSD of the library events
                # (acquisition_mode="passive": its samples are the time windows of the events)
                portion_library = build_portion_library(
                    portion_event,
                    sections=library_events,
                    rcv_ref=settings.reference_rcv,
                    fmin=settings.fmin,
                    fmax=settings.fmax,
                    acquisition_mode="passive",
                    verbose=settings.verbose,
                    rtf_method=settings.library_rtf_method,
                )
            # Features of the events that have a CPSD (the events used)
            derive_portion_features(
                portion_event,
                rcv_ref=settings.reference_rcv,
                fmin=settings.fmin,
                fmax=settings.fmax,
                verbose=settings.verbose,
                rtf_method=settings.event_rtf_method,
            )
        else:
            log("Step 5 - Not needed")

        # Features of the events (RTF amplitude and DCF), one figure each per event
        if settings.plot_features_event and need_cpsd:
            log(f"Features (RTF, DCF) of the {len(events_used)} event(s) used")
            _plot_events_features(portion_event, events_used, root_img_box, log)

        # --------------------------------------------------------------------------------------------------------------
        # Step 6 - Ambiguity surfaces
        # --------------------------------------------------------------------------------------------------------------
        if compute_surfaces:
            log("Step 6 - Ambiguity surfaces")
            portion_mfp = apply_mfp_strategy_to_portions(
                portion_library=portion_library,
                portion_event=portion_event,
                rcv_ref=settings.reference_rcv,
                fmin=settings.fmin,
                fmax=settings.fmax,
                verbose=settings.verbose,
                compare_centered_gamma=settings.compare_centered_gamma,
                library_sections=library_events,
                event_sections=tested_events,
                obs_ids=settings.obs_ids,
                skip_identical_names=True,
                pairs=pairs,  # each pair of events once
            )
            # Saved right away: if the rest fails, the next run reuses the surfaces
            if settings.save_mfp_to_netcdf:
                save_tree_atomic(portion_mfp, mfp_fpath, mfp_settings)
                log(f"Ambiguity surfaces saved: {mfp_fpath}", level=1)
        elif has_pairs:
            log("Step 6 - Not needed: the ambiguity surfaces are reused")

        if not has_pairs:
            return summary, []

        # --------------------------------------------------------------------------------------------------------------
        # Step 7 - Crossing distances of the computed surfaces (table). Same criterion as the pre-selection of step 3:
        # a pair that does not cross can only appear here for a saved file computed with another threshold.
        # --------------------------------------------------------------------------------------------------------------
        log("Step 7 - Theoretical intersections (closest windows of the two events)")
        datasets_mfp = _select_crossing_pairs(portion_mfp, settings, root_img_box, log)
        summary["n_crossing_pairs"] = len(datasets_mfp)

        if not datasets_mfp:
            summary["status"] = (
                f"done: no pair crosses (closest samples > {settings.max_crossing_distance_m} m)"
            )
            log("No pair of trajectories crosses: nothing more to analyse", level=1)
            return summary, []

        # Receivers available for all the pairs analysed, and combinations of receivers to study
        rcv_lists = [
            [str(rcv) for rcv in ds["rcv_id_feature"].values]
            for ds in datasets_mfp.values()
        ]
        available_rcv_ids = [
            rcv for rcv in rcv_lists[0] if all(rcv in rcv_list for rcv_list in rcv_lists)
        ]
        combinations = enumerate_rcv_combinations(
            available_rcv_ids,
            sizes=None,
            combinations=settings.rcv_combinations,
            max_combinations=200,
        )
        log(
            f"{len(datasets_mfp)} pair(s) analysed, {len(combinations)} combination(s) of receivers "
            f"among {available_rcv_ids} (reference '{settings.reference_rcv}')",
            level=1,
        )

        # --------------------------------------------------------------------------------------------------------------
        # Step 8 - Theoretical and estimated intersections (positions, as a table)
        # --------------------------------------------------------------------------------------------------------------
        position_records = []
        if settings.save_intersection_positions:
            log("Step 8 - Theoretical and estimated intersections")
            position_records = _intersection_position_records(
                datasets_mfp, combinations, settings
            )
            fpath = os.path.join(root_img_box, "intersection_positions.csv")
            pd.DataFrame(position_records).to_csv(fpath, index=False)
            log(f"Saved {fpath}", level=1)
            position_records = [dict(box=ibox, **record) for record in position_records]

        # --------------------------------------------------------------------------------------------------------------
        # Step 9 - Ambiguity figures (one folder per pair of events)
        # --------------------------------------------------------------------------------------------------------------
        if settings.plot_ambiguity:
            log("Step 9 - Ambiguity figures")
            summary["n_figures"] = _plot_ambiguity_figures(
                datasets_mfp, combinations, settings, root_img_box, log
            )

        return summary, position_records

    finally:
        # Close the files (the sub-box file is never modified)
        portion_event.close()
        if cpsd_tree is not None:
            cpsd_tree.close()
        if portion_mfp is not None and summary["mfp_reused"]:
            portion_mfp.close()
