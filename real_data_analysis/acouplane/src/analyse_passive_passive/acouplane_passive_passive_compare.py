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
    itself, and by default each pair is computed once):
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
        DCF of every event of the box (the CPSD and the features are then needed even if the
        ambiguity surfaces are reused, and boxes with a single event are processed too).
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
# Steps of the comparison
# ======================================================================================================================


def _prepare_cpsd(
    portion_event, cpsd_fpath, cpsd_settings, settings, log
):
    """Plug the CPSD of the events into the sub-box tree (reused from file, or computed and saved).

    Parameters
    ----------
    portion_event : xarray.DataTree
        Sub-box tree, modified in memory only: the group ``feature`` is added.
    cpsd_fpath : str
        File where the CPSD is saved / read.
    cpsd_settings : dict
        Everything the CPSD depends on (a saved CPSD is reused only if identical).
    settings : PassivePassiveSettings
        Comparison settings.
    log : callable
        Progress function (see ``make_logger``).

    Returns
    -------
    cpsd_tree : xarray.DataTree or None
        Lazy tree of the reused CPSD file (to close at the end), or None if it was computed.
    cpsd_reused : bool
        True if the CPSD was read from the file.
    """
    cpsd_tree = None
    if settings.reuse_stored_results:
        cpsd_tree, reason = load_tree_if_settings_match(cpsd_fpath, cpsd_settings)
        log(f"Saved CPSD: {reason}", level=1)

    if cpsd_tree is not None:
        # Plug the saved CPSD into the event tree (it stays lazily read from its file)
        for event_name, node in cpsd_tree.feature.children.items():
            portion_event[f"/feature/{event_name}"] = node.to_dataset()
        log(f"CPSD loaded from {cpsd_fpath}", level=1)
        return cpsd_tree, True

    # Otherwise compute it (long), and save it right away
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
    )
    if settings.save_cpsd_to_netcdf:
        cpsd_to_save = xr.DataTree()
        for event_name, node in portion_event.feature.children.items():
            cpsd_to_save[f"/feature/{event_name}"] = node.to_dataset()
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

    Steps: CPSD of the events (reused if saved with the same settings), features (plotted if
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
        ``box``, ``n_events``, ``n_pairs``, ``n_crossing_pairs``, ``n_figures``,
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

        pairs, library_events, tested_events = build_event_pairs(
            event_names, settings.both_directions
        )
        summary["n_pairs"] = len(pairs)
        has_pairs = len(pairs) > 0
        if has_pairs:
            log(f"{len(pairs)} pair(s) of events to compare", level=1)
        else:
            summary["status"] = "skipped: less than 2 events"
            log("Less than 2 events: nothing to compare", level=1)
            if not settings.plot_features_event:
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
        # Everything the ambiguity surfaces depend on
        mfp_settings = dict(
            cpsd=cpsd_settings,
            reference_rcv=settings.reference_rcv,
            fmin=settings.fmin,
            fmax=settings.fmax,
            library_rtf_method=settings.library_rtf_method,
            event_rtf_method=settings.event_rtf_method,
            pairs=[f"{library}|{event}" for library, event in pairs],
            obs_ids=None if settings.obs_ids is None else sorted(settings.obs_ids),
            compare_centered_gamma=settings.compare_centered_gamma,
        )

        mfp_reused = False
        if not has_pairs:
            log("No pair: no ambiguity surface", level=1)
        elif settings.reuse_stored_results:
            portion_mfp, reason = load_tree_if_settings_match(mfp_fpath, mfp_settings)
            log(f"Saved ambiguity surfaces: {reason}", level=1)
            mfp_reused = portion_mfp is not None
        summary["mfp_reused"] = mfp_reused

        # The surfaces must be computed unless they are reused. The CPSD and the features of the events are needed
        # for the surfaces, or to plot the features.
        need_surfaces = has_pairs and not mfp_reused
        need_features = need_surfaces or settings.plot_features_event

        # --------------------------------------------------------------------------------------------------------------
        # Steps 3 and 4 - CPSD and features (only if needed)
        # --------------------------------------------------------------------------------------------------------------
        portion_library = None
        if not need_features:
            log("Steps 3-4 - Not needed: the ambiguity surfaces are reused")
        else:
            log("Step 3 - CPSD of the events")
            cpsd_tree, summary["cpsd_reused"] = _prepare_cpsd(
                portion_event, cpsd_fpath, cpsd_settings, settings, log
            )

            log("Step 4 - Features of the library and of the events")
            if need_surfaces:
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
            # Features of all the events of the box (the tested events are among them)
            derive_portion_features(
                portion_event,
                rcv_ref=settings.reference_rcv,
                fmin=settings.fmin,
                fmax=settings.fmax,
                verbose=settings.verbose,
                rtf_method=settings.event_rtf_method,
            )

        # Features of the events (RTF amplitude and DCF), one figure each per event
        if settings.plot_features_event:
            log("Features (RTF, DCF) of the events")
            _plot_events_features(portion_event, event_names, root_img_box, log)

        # --------------------------------------------------------------------------------------------------------------
        # Step 5 - Ambiguity surfaces
        # --------------------------------------------------------------------------------------------------------------
        if need_surfaces:
            log("Step 5 - Ambiguity surfaces")
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
            log("Step 5 - Not needed: the ambiguity surfaces are reused")

        if not has_pairs:
            return summary, []

        # --------------------------------------------------------------------------------------------------------------
        # Step 6 - Theoretical intersections: do the two trajectories cross each other ?
        # --------------------------------------------------------------------------------------------------------------
        log("Step 6 - Theoretical intersections (library trajectory x tested trajectory)")
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
        # Step 7 - Theoretical and estimated intersections (positions, as a table)
        # --------------------------------------------------------------------------------------------------------------
        position_records = []
        if settings.save_intersection_positions:
            log("Step 7 - Theoretical and estimated intersections")
            position_records = _intersection_position_records(
                datasets_mfp, combinations, settings
            )
            fpath = os.path.join(root_img_box, "intersection_positions.csv")
            pd.DataFrame(position_records).to_csv(fpath, index=False)
            log(f"Saved {fpath}", level=1)
            position_records = [dict(box=ibox, **record) for record in position_records]

        # --------------------------------------------------------------------------------------------------------------
        # Step 8 - Ambiguity figures (one folder per pair of events)
        # --------------------------------------------------------------------------------------------------------------
        if settings.plot_ambiguity:
            log("Step 8 - Ambiguity figures")
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
