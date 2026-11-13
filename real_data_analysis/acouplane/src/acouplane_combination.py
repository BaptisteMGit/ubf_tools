#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_combination.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Combination of receivers: combinations to study, combined ambiguity surfaces, error and contrast.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import itertools
import numpy as np
import xarray as xr

from real_data_analysis.acouplane.src.acouplane_conventions import (
    get_mfp_axis_names,
    get_mfp_shot_positions,
)
from real_data_analysis.acouplane.src.acouplane_mfp import (
    DISTANCE_VARIABLE_ATTRS,
)
from real_data_analysis.acouplane.src.acouplane_localization import (
    DISTANCE_TYPES,
    _argmin_2d,
    _distances_from_mfp_dataset,
    _geodesic_distance_m,
    ambiguity_contrast_from_distances,
    errors_to_records,
    localization_errors_from_distances,
)


# Approaches available to combine the receiver pairs
#   "mean_surfaces": mean of the ambiguity surfaces (feature distances) of the combined receiver pairs
RCV_COMBINATION_METHODS = ("mean_surfaces",)


def format_rcv_combination(rcv_combination, sep=" + "):
    """Label of a combination of receivers, e.g. ``"obs2 + obs4"``.

    Parameters
    ----------
    rcv_combination : sequence of str
        Receivers (each one forms a pair with the reference receiver).
    sep : str, optional
        Separator between the receivers (e.g. ``"_"`` to build a file name).

    Returns
    -------
    str
        Receivers joined by ``sep``.
    """
    return sep.join(str(rcv_id) for rcv_id in rcv_combination)


def enumerate_rcv_combinations(
    rcv_ids, sizes=None, combinations=None, max_combinations=200
):
    """List the combinations of receivers to study.

    Either the combinations are given explicitly, or all the combinations of the
    requested sizes are generated from ``rcv_ids``.

    Parameters
    ----------
    rcv_ids : sequence of str
        Receivers available (reference receiver excluded).
    sizes : sequence of int or None, optional
        Numbers of receivers per combination (None: from 1 to ``len(rcv_ids)``).
        Ignored if ``combinations`` is given.
    combinations : sequence of sequence of str or None, optional
        Explicit combinations, e.g. ``[["obs2"], ["obs2", "obs4"]]``. Receivers
        are reordered as in ``rcv_ids``; duplicated combinations are ignored.
    max_combinations : int, optional
        Maximum number of combinations (guard against a combinatorial explosion).

    Returns
    -------
    list of tuple of str
        Combinations, sorted by number of receivers then in the order of
        ``rcv_ids``.

    Raises
    ------
    ValueError
        If a receiver or a size is invalid, if a combination is empty, or if
        there are more than ``max_combinations`` combinations.
    """
    rcv_ids = [str(rcv_id) for rcv_id in rcv_ids]
    if not rcv_ids:
        raise ValueError("No receiver available to build combinations.")

    if combinations is not None:
        # Explicit combinations: check receivers and put them in the order of rcv_ids
        selected = []
        for combination in combinations:
            combination = [str(rcv_id) for rcv_id in combination]
            if not combination:
                raise ValueError("A combination of receivers cannot be empty.")
            unknown = [rcv_id for rcv_id in combination if rcv_id not in rcv_ids]
            if unknown:
                raise ValueError(
                    f"Invalid combination {combination}: receivers {unknown} are not "
                    f"available (available receivers: {rcv_ids})."
                )
            selected.append(tuple(rcv_id for rcv_id in rcv_ids if rcv_id in combination))
        selected = list(dict.fromkeys(selected))
    else:
        # All the combinations of the requested sizes
        sizes = list(range(1, len(rcv_ids) + 1)) if sizes is None else list(sizes)
        invalid = [k for k in sizes if not 1 <= k <= len(rcv_ids)]
        if invalid:
            raise ValueError(
                f"Invalid combination size(s) {invalid}: must be between 1 and "
                f"{len(rcv_ids)} (number of available receivers)."
            )
        selected = [
            combination
            for k in sorted(set(sizes))
            for combination in itertools.combinations(rcv_ids, k)
        ]

    if not selected:
        raise ValueError("No combination of receivers to study.")
    if len(selected) > max_combinations:
        raise ValueError(
            f"{len(selected)} combinations of receivers requested (maximum "
            f"{max_combinations}): reduce the number of receivers or the sizes, or give the "
            "combinations explicitly."
        )

    return sorted(selected, key=len)


def combine_distances_over_receivers(
    distance_section12, rcv_ids, rcv_combination, method="mean_surfaces"
):
    """Combine the ambiguity surfaces of several receiver pairs.

    With the method ``"mean_surfaces"``, the combined surface of each distance
    type is the mean, over the receivers of the combination, of the feature
    distances computed for each pair ``(rcv_i, reference receiver)``. A cell of
    the combined surface is valid only if it is valid for all the receivers of
    the combination (NaN is propagated, as in a plain mean).

    Parameters
    ----------
    distance_section12 : dict
        Distances as returned by ``distances_single_intersection``: key
        ``dist_in_real_space`` and ``dist_<type>`` of shape
        ``(n_rcv, n_emission1, n_emission2)``.
    rcv_ids : sequence of str
        Receiver of each row of the distance arrays.
    rcv_combination : sequence of str
        Receivers to combine.
    method : {"mean_surfaces"}, optional
        Combination approach, see ``RCV_COMBINATION_METHODS``.

    Returns
    -------
    dict
        Same layout as ``distance_section12`` with a single receiver (shape
        ``(1, n_emission1, n_emission2)``), so that the result can be given to
        ``localization_errors_from_distances`` and
        ``ambiguity_contrast_from_distances``.

    Raises
    ------
    ValueError
        If the method is unknown, or if a receiver of the combination is not in
        ``rcv_ids``.
    """
    if method not in RCV_COMBINATION_METHODS:
        raise ValueError(
            f"Unknown combination method '{method}'. Valid methods: "
            f"{list(RCV_COMBINATION_METHODS)}."
        )

    rcv_ids = [str(rcv_id) for rcv_id in rcv_ids]
    unknown = [str(r) for r in rcv_combination if str(r) not in rcv_ids]
    if unknown:
        raise ValueError(
            f"Receivers {unknown} are not in the results (available receivers: {rcv_ids})."
        )
    rows = [rcv_ids.index(str(rcv_id)) for rcv_id in rcv_combination]

    # The real-space distance does not depend on the receivers
    combined = {"dist_in_real_space": distance_section12["dist_in_real_space"]}

    for key in DISTANCE_TYPES:
        dist = distance_section12.get(f"dist_{key}")
        if dist is None:
            continue

        # Mean of the surfaces of the selected receivers, kept with a receiver axis of size 1
        combined[f"dist_{key}"] = np.mean(
            np.asarray(dist, dtype=float)[rows], axis=0, keepdims=True
        )

    return combined


def rcv_combination_records(
    ds_mfp_intersection,
    rcv_combinations,
    reference_rcv,
    error_mode="section1",
    percentile=50.0,
    method="mean_surfaces",
    **extra,
):
    """Position error and contrast of each combination of receivers, for one stored intersection.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection (see ``build_mfp_intersection_dataset``).
    rcv_combinations : sequence of sequence of str
        Combinations of receivers to evaluate (see ``enumerate_rcv_combinations``).
    reference_rcv : str
        Reference receiver of the pairs.
    error_mode : {"section1", "section2", "both"}, optional
        How the position error is computed, see
        ``localization_errors_from_distances``.
    percentile : float, optional
        Percentile used in the contrast, see ``ambiguity_contrast_from_distances``.
    method : {"mean_surfaces"}, optional
        Combination approach, see ``combine_distances_over_receivers``.
    **extra
        Additional fields copied in each record (e.g. ``intersection="sn2 x oe6"``).

    Returns
    -------
    list of dict
        One record per combination and distance type, with the fields of
        ``errors_to_records`` (``rcv_pair`` is ``"(<combination>, <reference>)"``)
        and the extra fields ``combination`` (label), ``n_rcv`` (number of
        receivers) and ``method``.
    """
    distance_section12 = _distances_from_mfp_dataset(ds_mfp_intersection)
    rcv_ids = [str(rcv_id) for rcv_id in ds_mfp_intersection["rcv_id_feature"].values]

    # Shot positions of the two sides of the intersection
    positions = get_mfp_shot_positions(ds_mfp_intersection)

    records = []
    for rcv_combination in rcv_combinations:
        label = format_rcv_combination(rcv_combination)

        # Combined ambiguity surfaces
        combined = combine_distances_over_receivers(
            distance_section12, rcv_ids, rcv_combination, method=method
        )

        # Position error and contrast of the combined surfaces
        errors = localization_errors_from_distances(
            combined, error_mode=error_mode, **positions
        )
        contrasts = ambiguity_contrast_from_distances(combined, percentile=percentile)

        records += errors_to_records(
            errors,
            [label],
            reference_rcv,
            contrasts=contrasts,
            combination=label,
            n_rcv=len(rcv_combination),
            method=method,
            **extra,
        )

    return records


def build_combined_mfp_dataset(
    ds_mfp_intersection, rcv_combination, method="mean_surfaces"
):
    """Build the dataset of the combined ambiguity surfaces of one intersection.

    The combined surfaces replace the receiver axis: the result has the layout of
    the dataset of one receiver pair (``d_real`` and ``d_<type>`` of dimensions
    ``(src_datetime_section1, src_datetime_section2)``), so that it can be given
    to the figures of the ambiguity surfaces.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection (see ``build_mfp_intersection_dataset``).
    rcv_combination : sequence of str
        Receivers to combine.
    method : {"mean_surfaces"}, optional
        Combination approach, see ``combine_distances_over_receivers``.

    Returns
    -------
    xarray.Dataset
        Dataset with ``d_real`` and the combined distances present in
        ``ds_mfp_intersection``, and the attributes ``combination``, ``method``
        and ``reference_receiver``.

    Raises
    ------
    ValueError
        If the method is unknown or if a receiver of the combination is not in
        the dataset.
    """
    axis_names = get_mfp_axis_names(ds_mfp_intersection)
    dim1, dim2 = axis_names["dim1"], axis_names["dim2"]
    rcv_ids = [str(rcv_id) for rcv_id in ds_mfp_intersection["rcv_id_feature"].values]

    # Combined surfaces
    combined = combine_distances_over_receivers(
        _distances_from_mfp_dataset(ds_mfp_intersection),
        rcv_ids,
        rcv_combination,
        method=method,
    )

    # Same layout as the dataset of a single receiver pair
    data_vars = {"d_real": ([dim1, dim2], combined["dist_in_real_space"])}
    for key in DISTANCE_TYPES:
        if f"dist_{key}" in combined:
            data_vars[f"d_{key}"] = ([dim1, dim2], combined[f"dist_{key}"][0])

    ds_combined = xr.Dataset(
        data_vars=data_vars,
        coords={
            dim1: ds_mfp_intersection[dim1].values,
            dim2: ds_mfp_intersection[dim2].values,
        },
        attrs=dict(
            description="Combined MFP ambiguity surfaces of one intersection",
            combination=format_rcv_combination(rcv_combination),
            method=method,
            reference_receiver=ds_mfp_intersection.attrs.get("reference_receiver", "unknown"),
        ),
    )

    # Names and units of the variables (labels of the colorbars)
    for var in data_vars:
        ds_combined[var].attrs = dict(DISTANCE_VARIABLE_ATTRS[var])

    return ds_combined


def combination_intersection_positions(
    ds_mfp_intersection, rcv_combination, method="mean_surfaces"
):
    """Theoretical and estimated intersection of one comparison, for a combination of receivers.

    The theoretical intersection is the pair of samples of the two sides
    (section 1 / section 2, or library / event) which are the closest in real
    space. For each distance type, the estimated intersection is the pair of
    samples minimizing the combined ambiguity surface.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection (two sections, or a library section and
        an event).
    rcv_combination : sequence of str
        Receivers to combine, e.g. ``("obs2", "obs4")``.
    method : {"mean_surfaces"}, optional
        Combination approach, see ``combine_distances_over_receivers``.

    Returns
    -------
    list of dict
        One record for the theoretical intersection (``kind = "theoretical"``)
        followed by one record per distance type. Each record gives the
        datetimes (``datetime1``, ``datetime2``) and the positions (``lon1``,
        ``lat1``, ``lon2``, ``lat2``) of the two samples, the real distance
        between them (``real_distance_m``) and, for the estimates, the position
        errors along side 1 (``error_side1_m``) and side 2 (``error_side2_m``).
        The values are NaN if a surface has no valid value.

    Raises
    ------
    ValueError
        If the real-space distance has no valid value (the theoretical
        intersection cannot be determined).
    """
    axis_names = get_mfp_axis_names(ds_mfp_intersection)
    positions = get_mfp_shot_positions(ds_mfp_intersection)
    datetimes1 = ds_mfp_intersection[axis_names["dim1"]].values
    datetimes2 = ds_mfp_intersection[axis_names["dim2"]].values
    rcv_ids = [str(rcv_id) for rcv_id in ds_mfp_intersection["rcv_id_feature"].values]

    # Surfaces combined over the receivers
    combined = combine_distances_over_receivers(
        _distances_from_mfp_dataset(ds_mfp_intersection),
        rcv_ids,
        rcv_combination,
        method=method,
    )

    # Theoretical intersection: closest samples in real space
    target = _argmin_2d(combined["dist_in_real_space"])
    if target is None:
        raise ValueError(
            "The real-space distance has no valid value: the theoretical intersection "
            "cannot be determined."
        )
    i_target, j_target = int(target[0]), int(target[1])

    def sample_record(kind, i1, j1):
        """Record of the pair of samples (i1 on side 1, j1 on side 2)."""
        return dict(
            kind=kind,
            datetime1=datetimes1[i1],
            datetime2=datetimes2[j1],
            lon1=positions["lon1"][i1],
            lat1=positions["lat1"][i1],
            lon2=positions["lon2"][j1],
            lat2=positions["lat2"][j1],
            real_distance_m=float(combined["dist_in_real_space"][i1, j1]),
        )

    records = [sample_record("theoretical", i_target, j_target)]

    # Estimated intersection of each distance type: minimum of the combined surface
    for key in DISTANCE_TYPES:
        if f"dist_{key}" not in combined:
            continue

        guess = _argmin_2d(combined[f"dist_{key}"][0])
        if guess is None:
            records.append(
                dict(
                    kind=key,
                    datetime1=np.datetime64("NaT"),
                    datetime2=np.datetime64("NaT"),
                    lon1=np.nan,
                    lat1=np.nan,
                    lon2=np.nan,
                    lat2=np.nan,
                    real_distance_m=np.nan,
                    error_side1_m=np.nan,
                    error_side2_m=np.nan,
                )
            )
            continue

        i_guess, j_guess = int(guess[0]), int(guess[1])
        record = sample_record(key, i_guess, j_guess)

        # Position error along each side with respect to the theoretical intersection
        record["error_side1_m"] = _geodesic_distance_m(
            positions["lon1"][i_target], positions["lat1"][i_target],
            positions["lon1"][i_guess], positions["lat1"][i_guess],
        )
        record["error_side2_m"] = _geodesic_distance_m(
            positions["lon2"][j_target], positions["lat2"][j_target],
            positions["lon2"][j_guess], positions["lat2"][j_guess],
        )
        records.append(record)

    return records
