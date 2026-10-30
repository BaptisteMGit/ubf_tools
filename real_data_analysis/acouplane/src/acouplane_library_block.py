#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_library_block.py
@Time    :   2026/10/30 18:00:00
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Library considered as a single block of positions with features, compared to an event:
             one ambiguity surface for the whole library, and detection of the minima that do not
             coincide with a minimum of the real distance.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import xarray as xr

from real_data_analysis.acouplane.src.acouplane_event import (
    LIBRARY_FEATURE_VARS,
    LIBRARY_RECEIVER_VARS,
    check_library_and_event_compatibility,
)
from real_data_analysis.acouplane.src.acouplane_features import (
    feature_distances_between_datasets,
    get_distance_between_library_and_event,
)
from real_data_analysis.acouplane.src.acouplane_localization import (
    DISTANCE_TYPES,
)
from real_data_analysis.acouplane.src.acouplane_mfp import (
    DISTANCE_VARIABLE_ATTRS,
)

BLOCK_DIMS_FEATURES = ("rcv_id_feature", "library_datetime", "freq")


# ======================================================================================================================
# Library block
# ======================================================================================================================
def build_library_block(portion_library, sections=None, fmin=None, fmax=None):
    """Gather the sections of a library in a single block of samples.

    The samples of the selected sections are concatenated (in the order of
    ``sections``) along the dimension ``library_datetime``. Each sample keeps
    its position (``library_lon`` / ``library_lat``), its features and the label
    of its section (``library_section``). The block can be used as any library
    section, e.g. in ``get_distance_between_library_and_event``. Datetimes of
    different sections are not required to be distinct: samples are always
    identified by their position in the block.

    Parameters
    ----------
    portion_library : xarray.DataTree
        Library, see ``build_portion_library``.
    sections : list of str or None, optional
        Sections gathered in the block. If None, all the sections of the library.
    fmin, fmax : float or None, optional
        Frequency band kept, in Hz (same band as the event). None: the whole
        frequency axis of the library.

    Returns
    -------
    xarray.Dataset
        Block with the variables ``pi_ij``, ``gamma_ij``, ``sij_scot``
        (dimensions ``(rcv_id_feature, library_datetime, freq)``),
        ``library_lon``, ``library_lat``, ``library_section`` (dimension
        ``library_datetime``) and the receiver positions if available. The
        attributes give the reference receiver, the acquisition mode and the
        sections (comma separated).

    Raises
    ------
    ValueError
        If the library has no feature group, if a section is unknown, or if the
        sections do not have the same receivers, frequencies or reference receiver.
    """
    if "feature" not in portion_library.children:
        raise ValueError("No 'feature' group in the library.")

    available_sections = list(portion_library.feature.children)
    if sections is None:
        sections = available_sections
    sections = list(sections)
    unknown_sections = [name for name in sections if name not in available_sections]
    if unknown_sections:
        raise ValueError(
            f"Unknown library section(s) {unknown_sections} (available: {available_sections})."
        )
    if not sections:
        raise ValueError("No library section to gather in the block.")

    datasets = [portion_library.feature[name].to_dataset() for name in sections]
    if fmin is not None or fmax is not None:
        datasets = [ds.sel(freq=slice(fmin, fmax)) for ds in datasets]
    reference_ds = datasets[0]

    # The sections must be comparable: same reference receiver, receivers and frequencies
    for name, ds in zip(sections, datasets):
        if ds.attrs.get("reference_receiver") != reference_ds.attrs.get("reference_receiver"):
            raise ValueError(
                f"The section '{name}' was built with another reference receiver than '{sections[0]}'."
            )
        if list(ds["rcv_id_feature"].values) != list(reference_ds["rcv_id_feature"].values):
            raise ValueError(
                f"The section '{name}' has not the same receivers as '{sections[0]}'."
            )
        if ds["freq"].values.shape != reference_ds["freq"].values.shape or not np.allclose(
            ds["freq"].values, reference_ds["freq"].values, rtol=0, atol=1e-6
        ):
            raise ValueError(
                f"The section '{name}' has not the same frequencies as '{sections[0]}'."
            )

    # Concatenate the samples of the sections (positional, no index alignment)
    data_vars = {}
    for var in LIBRARY_FEATURE_VARS:
        data_vars[var] = (
            list(BLOCK_DIMS_FEATURES),
            np.concatenate(
                [ds[var].transpose(*BLOCK_DIMS_FEATURES).values for ds in datasets], axis=1
            ),
        )
    for var in ("library_lon", "library_lat"):
        data_vars[var] = (
            ["library_datetime"],
            np.concatenate([ds[var].values for ds in datasets]),
        )
    data_vars["library_section"] = (
        ["library_datetime"],
        np.concatenate(
            [np.full(ds["library_datetime"].values.size, name) for name, ds in zip(sections, datasets)]
        ),
    )
    for var in LIBRARY_RECEIVER_VARS:
        if var in reference_ds:
            data_vars[var] = reference_ds[var]

    return xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            rcv_id_feature=reference_ds["rcv_id_feature"].values,
            library_datetime=np.concatenate([ds["library_datetime"].values for ds in datasets]),
            freq=reference_ds["freq"].values,
        ),
        attrs=dict(
            description="Library gathered in a single block of samples",
            section_label="library_block",
            acquisition_mode=reference_ds.attrs.get("acquisition_mode", "unknown"),
            reference_receiver=reference_ds.attrs.get("reference_receiver", "unknown"),
            library_sections=",".join(sections),
        ),
    )


# ======================================================================================================================
# Ambiguity surface of the whole library
# ======================================================================================================================
def compute_block_receiver_distances(
    ds_library_block, ds_feature_event, rcv_ids, compare_centered_gamma=False
):
    """Feature distances between the whole library and an event, for each receiver pair.

    Computed once and reused by ``compute_block_ambiguity`` for every combination
    of receivers (the cost of a combination is then only a mean of surfaces).

    Parameters
    ----------
    ds_library_block : xarray.Dataset
        Library block, see ``build_library_block``.
    ds_feature_event : xarray.Dataset
        Feature dataset of the event.
    rcv_ids : sequence of str
        Receivers (pairs ``(rcv, reference receiver)``) to compute.
    compare_centered_gamma : bool, optional
        If True, also compute the distances with gamma centered along frequency.

    Returns
    -------
    dict
        ``{rcv_id: {"dist_<type>": 2D array (n_library_samples, n_event_samples)}}``.

    Raises
    ------
    ValueError
        If the library and the event are not compatible, or a receiver is not
        available for both.
    """
    rcv_ids = [str(rcv) for rcv in rcv_ids]
    common_receivers = check_library_and_event_compatibility(ds_library_block, ds_feature_event)
    missing = [rcv for rcv in rcv_ids if rcv not in common_receivers]
    if missing:
        raise ValueError(
            f"Receiver(s) {missing} are not available for both the library and the event "
            f"(common receivers: {common_receivers})."
        )

    receiver_distances = {}
    for rcv_id in rcv_ids:
        feature_distances = feature_distances_between_datasets(
            ds_library_block.sel(rcv_id_feature=[rcv_id]),
            ds_feature_event.sel(rcv_id_feature=[rcv_id]),
            compare_centered_gamma=compare_centered_gamma,
        )
        # One receiver: drop the receiver axis
        receiver_distances[rcv_id] = {key: dist[0] for key, dist in feature_distances.items()}

    return receiver_distances


def compute_block_ambiguity(
    ds_library_block,
    ds_feature_event,
    rcv_combination,
    compare_centered_gamma=False,
    receiver_distances=None,
    real_distance=None,
):
    """Ambiguity surfaces between the whole library and an event, for a combination of receivers.

    The surface of a combination is the mean of the surfaces of its receiver
    pairs (NaN propagating, as ``combine_distances_over_receivers``).

    Parameters
    ----------
    ds_library_block : xarray.Dataset
        Library block, see ``build_library_block``.
    ds_feature_event : xarray.Dataset
        Feature dataset of the event (dimension ``event_datetime``, features
        ``pi_ij``, ``gamma_ij`` and ``sij_scot``).
    rcv_combination : sequence of str
        Receivers combined.
    compare_centered_gamma : bool, optional
        If True, also compute the distances with gamma centered along frequency
        (ignored if ``receiver_distances`` is given).
    receiver_distances : dict or None, optional
        Result of ``compute_block_receiver_distances`` covering the receivers of
        the combination. If None, computed here.
    real_distance : numpy.ndarray or None, optional
        Real distance in meters ``(n_library_samples, n_event_samples)``. If
        None, computed here (geodesic, WGS84).

    Returns
    -------
    xarray.Dataset
        Dataset with ``d_real`` (m) and the combined distances ``d_<type>``,
        of dimensions ``(library_datetime, event_datetime)``, with the sample
        positions of both sides and ``library_section``. The variables have the
        names and units used by the figures. The attribute ``combination`` gives
        the receivers.

    Raises
    ------
    ValueError
        If the combination is empty, if the library and the event are not
        compatible or if a receiver of the combination is missing.
    """
    rcv_combination = [str(rcv) for rcv in rcv_combination]
    if not rcv_combination:
        raise ValueError("The combination of receivers is empty.")

    if receiver_distances is None:
        receiver_distances = compute_block_receiver_distances(
            ds_library_block, ds_feature_event, rcv_combination, compare_centered_gamma
        )
    missing = [rcv for rcv in rcv_combination if rcv not in receiver_distances]
    if missing:
        raise ValueError(f"No distances computed for the receiver(s) {missing}.")

    # Distance in real space between every library sample and every event sample
    if real_distance is None:
        real_distance = get_distance_between_library_and_event(ds_library_block, ds_feature_event)

    # Mean of the surfaces of the receivers
    combined = {}
    for key in receiver_distances[rcv_combination[0]]:
        combined[key] = sum(receiver_distances[rcv][key] for rcv in rcv_combination) / len(
            rcv_combination
        )

    dims = ["library_datetime", "event_datetime"]
    data_vars = dict(
        d_real=(dims, real_distance),
        library_lon=(["library_datetime"], ds_library_block["library_lon"].values),
        library_lat=(["library_datetime"], ds_library_block["library_lat"].values),
        library_section=(["library_datetime"], ds_library_block["library_section"].values),
        event_lon=(["event_datetime"], ds_feature_event["event_lon"].values),
        event_lat=(["event_datetime"], ds_feature_event["event_lat"].values),
    )
    for key in DISTANCE_TYPES:
        if f"dist_{key}" in combined:
            data_vars[f"d_{key}"] = (dims, combined[f"dist_{key}"])

    ds_surfaces = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            library_datetime=ds_library_block["library_datetime"].values,
            event_datetime=ds_feature_event["event_datetime"].values,
        ),
        attrs=dict(
            description="Ambiguity surfaces between the whole library and an event",
            combination=" + ".join(rcv_combination),
            reference_receiver=ds_library_block.attrs.get("reference_receiver", "unknown"),
            library_sections=ds_library_block.attrs.get("library_sections", "unknown"),
        ),
    )

    # Names and units of the variables (labels of the colorbars)
    for var in data_vars:
        if var in DISTANCE_VARIABLE_ATTRS:
            ds_surfaces[var].attrs = dict(DISTANCE_VARIABLE_ATTRS[var])

    return ds_surfaces


# ======================================================================================================================
# Minima of the surfaces
# ======================================================================================================================
def find_surface_minima(surface, n_minima=10, exclusion_samples=(5, 5), max_value=None):
    """Find the lowest distinct minima of a 2D surface.

    Greedy search: the global minimum is taken, a window around it is masked,
    and the search goes on with the remaining values. Two minima are therefore
    never closer than the exclusion window, which avoids listing the
    neighbouring samples of a same minimum.

    Parameters
    ----------
    surface : numpy.ndarray
        2D array ``(n_library_samples, n_event_samples)``; NaN and infinite
        values are ignored.
    n_minima : int, optional
        Maximum number of minima returned.
    exclusion_samples : tuple of int, optional
        Half-width of the masked window around each minimum, in samples along
        the library and along the event.
    max_value : float or None, optional
        If given, only the minima whose value is lower or equal are returned.

    Returns
    -------
    list of tuple
        ``(i_library, j_event, value)`` of each minimum, from the lowest to the
        highest value.
    """
    work = np.array(surface, dtype=float)
    work[~np.isfinite(work)] = np.inf
    half_library, half_event = exclusion_samples

    minima = []
    for _ in range(int(n_minima)):
        flat_index = int(np.argmin(work))
        i_min, j_min = np.unravel_index(flat_index, work.shape)
        value = work[i_min, j_min]

        # No valid value left, or above the limit
        if not np.isfinite(value) or (max_value is not None and value > max_value):
            break

        minima.append((int(i_min), int(j_min), float(value)))

        # Mask the neighbourhood of the minimum
        work[
            max(i_min - half_library, 0) : i_min + half_library + 1,
            max(j_min - half_event, 0) : j_min + half_event + 1,
        ] = np.inf

    return minima


def _quantile_value(surface, quantile):
    """Quantile of the finite values of a surface (None if ``quantile`` is None or no finite value)."""
    if quantile is None:
        return None
    values = np.asarray(surface, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, quantile)) if values.size else None


def library_block_minima(
    ds_surfaces,
    metrics,
    n_minima=10,
    exclusion_samples=(5, 5),
    consistency_distance_m=100.0,
    minima_quantile=0.01,
):
    """List the minima of the surfaces and check that they coincide with minima of the real distance.

    The reference minima are those of the real distance (in meters) lower than
    ``consistency_distance_m``: the places where the event really passes close
    to a sample of the library. A minimum of a feature surface is *consistent*
    if the real distance between the library sample and the event sample is
    lower than ``consistency_distance_m``, and *spurious* otherwise: the
    features then suggest a match between samples that are far apart.

    Parameters
    ----------
    ds_surfaces : xarray.Dataset
        Surfaces, see ``compute_block_ambiguity``.
    metrics : sequence of str
        Distance types analysed (e.g. ``["theta_gamma", "theta", "dcf"]``).
    n_minima : int, optional
        Number of minima searched for each surface.
    exclusion_samples : tuple of int, optional
        Half-width of the window separating two minima, in samples (library,
        event).
    consistency_distance_m : float, optional
        Maximum real distance for a minimum to be consistent, in meters.
    minima_quantile : float or None, optional
        Only the minima whose value is below this quantile of the feature
        surface are kept (default 1 %, the *deep* minima that could be taken
        for a localization). This avoids counting as spurious the shallow
        minima that any noisy surface has. None: the ``n_minima`` lowest are
        always kept.

    Returns
    -------
    records : list of dict
        One record per minimum (``kind`` is ``"real"`` for the minima of the
        real distance, the distance type otherwise) with ``rank``,
        ``library_section``, ``i_library`` (index in the block),
        ``library_sample`` (index in its section), ``j_event``, the event
        datetime, ``value``, ``real_distance_m`` and ``consistent``.
    summary : list of dict
        One record per metric with ``n_consistent`` and ``n_spurious`` among the
        ``n_minima`` lowest minima, ``first_spurious_rank`` (NaN if none),
        ``best_consistent_value``, ``best_spurious_value`` and ``gap`` (best
        spurious value - best consistent value: positive when the true match is
        the deepest minimum).

    Raises
    ------
    ValueError
        If a metric is not in the dataset.
    """
    metrics = list(metrics)
    missing = [key for key in metrics if f"d_{key}" not in ds_surfaces]
    if missing:
        raise ValueError(
            f"Metric(s) {missing} not in the surfaces "
            "(the centered distances need compare_centered_gamma = True)."
        )

    dist_real = np.asarray(ds_surfaces["d_real"].values, dtype=float)
    sections = np.asarray(ds_surfaces["library_section"].values)
    event_datetimes = ds_surfaces["event_datetime"].values

    # Index of each sample in its own section
    sample_in_section = np.zeros(sections.size, dtype=int)
    for name in dict.fromkeys(sections.tolist()):
        mask = sections == name
        sample_in_section[mask] = np.arange(mask.sum())

    def make_record(kind, rank, i_library, j_event, value):
        real_distance = float(dist_real[i_library, j_event])
        return dict(
            kind=kind,
            rank=rank,
            library_section=str(sections[i_library]),
            i_library=i_library,
            library_sample=int(sample_in_section[i_library]),
            j_event=j_event,
            event_datetime=event_datetimes[j_event],
            value=value,
            real_distance_m=real_distance,
            consistent=bool(real_distance <= consistency_distance_m),
        )

    # Minima of the real distance: the places where the event really crosses the library
    records = [
        make_record("real", rank, i, j, value)
        for rank, (i, j, value) in enumerate(
            find_surface_minima(
                dist_real, n_minima, exclusion_samples, max_value=consistency_distance_m
            ),
            start=1,
        )
    ]

    # Minima of each feature surface
    summary = []
    for key in metrics:
        metric_records = [
            make_record(key, rank, i, j, value)
            for rank, (i, j, value) in enumerate(
                find_surface_minima(
                    ds_surfaces[f"d_{key}"].values,
                    n_minima,
                    exclusion_samples,
                    max_value=_quantile_value(ds_surfaces[f"d_{key}"].values, minima_quantile),
                ),
                start=1,
            )
        ]
        records += metric_records

        consistent_values = [r["value"] for r in metric_records if r["consistent"]]
        spurious = [r for r in metric_records if not r["consistent"]]
        best_consistent = min(consistent_values) if consistent_values else np.nan
        best_spurious = min((r["value"] for r in spurious), default=np.nan)
        summary.append(
            dict(
                metric=key,
                n_minima=len(metric_records),
                n_consistent=len(consistent_values),
                n_spurious=len(spurious),
                first_spurious_rank=spurious[0]["rank"] if spurious else np.nan,
                best_consistent_value=best_consistent,
                best_spurious_value=best_spurious,
                gap=best_spurious - best_consistent,
            )
        )

    return records, summary
