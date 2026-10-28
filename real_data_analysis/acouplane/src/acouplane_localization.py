#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_localization.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Localization performance: position error, ambiguity contrast, records and statistics.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import functools
import numpy as np


# Distance types compared in the results analysis, in display order
DISTANCE_TYPES = (
    "theta",
    "theta_gamma",
    "L2_gamma",
    "dcf",
    "theta_gamma_centered",
    "L2_gamma_centered",
)


DISTANCE_TYPE_LABELS = {
    "theta": r"$\theta_{\Pi}$",
    "theta_gamma": r"$\theta_{\gamma}$",
    "L2_gamma": r"$L_{2_{\gamma}}$",
    "dcf": r"$C_{el}$",
    "theta_gamma_centered": r"$\theta_{\gamma}$ (centered)",
    "L2_gamma_centered": r"$L_{2_{\gamma}}$ (centered)",
}


# Ways of computing the position error of an estimated intersection
POSITION_ERROR_MODES = ("section1", "section2", "both")


def get_distance_type_label(distance_type):
    """Return the display label of a distance type.

    Parameters
    ----------
    distance_type : str
        One of ``DISTANCE_TYPES`` (e.g. ``"theta"``, ``"dcf"``).

    Returns
    -------
    str
        Label (mathtext) used in the figures, or ``distance_type`` itself if it
        is unknown.
    """
    return DISTANCE_TYPE_LABELS.get(distance_type, distance_type)


def select_distance_types(available, requested=None):
    """Choose the distance types to analyse among the available ones.

    Parameters
    ----------
    available : iterable of str
        Distance types present in the results.
    requested : list of str or None, optional
        Distance types wished by the user. If None, all the available ones are
        returned.

    Returns
    -------
    list of str
        Selected distance types: in the order of ``requested`` (duplicates
        removed) or, if ``requested`` is None, in the canonical order of
        ``DISTANCE_TYPES``.

    Raises
    ------
    ValueError
        If a requested type is not available, or if the selection is empty.
    """
    ordered = [key for key in DISTANCE_TYPES if key in set(available)]
    if requested is None:
        selected = ordered
    else:
        selected = list(dict.fromkeys(requested))
        unknown = [key for key in selected if key not in ordered]
        if unknown:
            raise ValueError(
                f"Distance type(s) {unknown} not available in the results. "
                f"Available distance types: {ordered}."
            )
    if not selected:
        raise ValueError(
            f"No distance type to analyse (available distance types: {ordered})."
        )
    return selected


@functools.lru_cache(maxsize=None)
def _get_wgs84_geod():
    """Return a (cached) pyproj ``Geod`` object on the WGS84 ellipsoid."""
    from pyproj import Geod

    return Geod(ellps="WGS84")


def _geodesic_distance_m(lon1, lat1, lon2, lat2):
    """Geodesic distance between two points of the WGS84 ellipsoid.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float
        Longitudes and latitudes in degrees.

    Returns
    -------
    float
        Distance in meters, NaN if it cannot be computed (invalid coordinates).
    """
    # pyproj returns (forward azimuth, back azimuth, distance)
    _, _, dist = _get_wgs84_geod().inv(lon1, lat1, lon2, lat2)
    dist = float(dist)

    return dist if np.isfinite(dist) else np.nan


def _argmin_2d(dist):
    """Index of the minimum of a 2D array, ignoring NaN and infinite values.

    Parameters
    ----------
    dist : numpy.ndarray
        2D distance matrix.

    Returns
    -------
    tuple of int or None
        ``(row, column)`` of the first minimum, or None if the matrix has no
        finite value.
    """
    # Ignore NaN and infinite values
    valid = np.isfinite(dist)
    if not valid.any():
        return None

    # Invalid values are replaced by inf so that they can never be the minimum
    flat_idx = np.argmin(np.where(valid, dist, np.inf))

    return np.unravel_index(flat_idx, dist.shape)


def localization_errors_from_distances(
    distance_section12, lon1, lat1, lon2, lat2, error_mode="section1"
):
    """Position error of the estimated intersection, for each distance type and receiver pair.

    The true intersection is the pair of shots ``(i1, i2)`` minimizing the
    real-space distance. For each distance type and receiver pair, the estimated
    intersection is the pair of shots ``(j1, j2)`` minimizing the feature
    distance (NaN values are ignored). The position error is the geodesic
    distance, on the WGS84 ellipsoid, between the true and the estimated shots:

    - ``"section1"``: shots ``i1`` and ``j1`` on section 1 (meridional section);
    - ``"section2"``: shots ``i2`` and ``j2`` on section 2 (zonal section);
    - ``"both"``: ``sqrt(error_section1**2 + error_section2**2)``.

    Section 1 being meridional, its error is essentially along the North-South
    axis, and the error of the zonal section 2 along the East-West axis: the
    ``"both"`` mode is thus the only one accounting for both directions.

    Parameters
    ----------
    distance_section12 : dict
        Distances as returned by ``distances_single_intersection``: key
        ``dist_in_real_space`` (shape ``(n_emission1, n_emission2)``) and, for
        each available distance type, ``dist_<type>`` (shape
        ``(n_rcv, n_emission1, n_emission2)``).
    lon1, lat1 : array_like
        Shot positions of section 1, in degrees, shape ``(n_emission1,)``.
    lon2, lat2 : array_like
        Shot positions of section 2, in degrees, shape ``(n_emission2,)``.
    error_mode : {"section1", "section2", "both"}, optional
        How the position error is computed (see above).

    Returns
    -------
    dict of numpy.ndarray
        One array of shape ``(n_rcv,)`` per available distance type (keys of
        ``DISTANCE_TYPES``), position errors in meters. NaN when the feature
        distance has no valid value for the receiver pair.

    Raises
    ------
    ValueError
        If ``error_mode`` is unknown, if the shapes are inconsistent, or if the
        real-space distance has no valid value.
    """
    if error_mode not in POSITION_ERROR_MODES:
        raise ValueError(
            f"Unknown error_mode '{error_mode}'. Valid modes: {list(POSITION_ERROR_MODES)}."
        )

    # Check the inputs
    d_real = np.asarray(distance_section12["dist_in_real_space"], dtype=float)
    lon1, lat1, lon2, lat2 = (
        np.asarray(arr, dtype=float) for arr in (lon1, lat1, lon2, lat2)
    )
    if (
        d_real.ndim != 2
        or lon1.shape != (d_real.shape[0],)
        or lat1.shape != (d_real.shape[0],)
        or lon2.shape != (d_real.shape[1],)
        or lat2.shape != (d_real.shape[1],)
    ):
        raise ValueError(
            f"Inconsistent shapes: real-space distance {d_real.shape}, section 1 positions "
            f"{lon1.shape}/{lat1.shape}, section 2 positions {lon2.shape}/{lat2.shape}."
        )

    # True intersection: pair of shots closest in real space
    target = _argmin_2d(d_real)
    if target is None:
        raise ValueError(
            "The real-space distance has no valid value: the true intersection "
            "cannot be determined."
        )
    i1_t, i2_t = target

    errors = {}
    for key in DISTANCE_TYPES:
        # Distance types that have not been computed are skipped
        dist = distance_section12.get(f"dist_{key}")
        if dist is None:
            continue

        dist = np.asarray(dist, dtype=float)
        if dist.ndim != 3 or dist.shape[1:] != d_real.shape:
            raise ValueError(
                f"Distance 'dist_{key}' has shape {dist.shape}, expected "
                f"(n_rcv, {d_real.shape[0]}, {d_real.shape[1]})."
            )

        err = np.full(dist.shape[0], np.nan)
        for i_rcv in range(dist.shape[0]):
            # Estimated intersection: pair of shots minimizing the feature distance
            guess = _argmin_2d(dist[i_rcv])
            if guess is None:
                continue
            j1, j2 = guess

            # Geodesic distance between true and estimated shots, on each section
            err1 = err2 = np.nan
            if error_mode in ("section1", "both"):
                err1 = _geodesic_distance_m(lon1[i1_t], lat1[i1_t], lon1[j1], lat1[j1])
            if error_mode in ("section2", "both"):
                err2 = _geodesic_distance_m(lon2[i2_t], lat2[i2_t], lon2[j2], lat2[j2])

            # Position error according to the requested mode
            if error_mode == "section1":
                err[i_rcv] = err1
            elif error_mode == "section2":
                err[i_rcv] = err2
            else:
                err[i_rcv] = np.hypot(err1, err2)

        errors[key] = err

    return errors


def _distances_from_mfp_dataset(ds_mfp_intersection):
    """Extract the distance arrays of a stored intersection, in the layout of ``distances_single_intersection``.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection (see ``build_mfp_intersection_dataset``).

    Returns
    -------
    dict
        Keys ``dist_in_real_space`` (shape ``(n_emission1, n_emission2)``) and
        ``dist_<type>`` (shape ``(n_rcv, n_emission1, n_emission2)``) for each
        distance type present in the dataset.
    """
    rcv_dim = "rcv_id_feature"
    dim1, dim2 = "src_datetime_section1", "src_datetime_section2"

    # Real-space distance between the shots of the two sections
    distance_section12 = {
        "dist_in_real_space": ds_mfp_intersection["d_real"].transpose(dim1, dim2).values
    }

    # Feature distances (only those stored in the dataset)
    for key in DISTANCE_TYPES:
        var = f"d_{key}"
        if var in ds_mfp_intersection:
            distance_section12[f"dist_{key}"] = (
                ds_mfp_intersection[var].transpose(rcv_dim, dim1, dim2).values
            )

    return distance_section12


def localization_errors_from_mfp_dataset(ds_mfp_intersection, error_mode="section1"):
    """Position errors of a stored intersection (dataset built by ``build_mfp_intersection_dataset``).

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection, with the variables ``d_real`` and
        ``d_<type>``, and the shot positions ``tir_src_lon_section1``,
        ``tir_src_lat_section1``, ``tir_src_lon_section2``,
        ``tir_src_lat_section2``.
    error_mode : {"section1", "section2", "both"}, optional
        How the position error is computed, see
        ``localization_errors_from_distances``.

    Returns
    -------
    errors : dict of numpy.ndarray
        Position errors in meters, one array of shape ``(n_rcv,)`` per
        distance type present in the dataset.
    rcv_ids : list of str
        Receiver of each error (``rcv_id_feature`` coordinate).
    """
    rcv_dim = "rcv_id_feature"

    # Position errors from the stored ambiguity surfaces and the shot positions
    errors = localization_errors_from_distances(
        _distances_from_mfp_dataset(ds_mfp_intersection),
        lon1=ds_mfp_intersection["tir_src_lon_section1"].values,
        lat1=ds_mfp_intersection["tir_src_lat_section1"].values,
        lon2=ds_mfp_intersection["tir_src_lon_section2"].values,
        lat2=ds_mfp_intersection["tir_src_lat_section2"].values,
        error_mode=error_mode,
    )
    rcv_ids = [str(rcv_id) for rcv_id in ds_mfp_intersection[rcv_dim].values]

    return errors, rcv_ids


def ambiguity_contrast_from_distances(distance_section12, percentile=50.0):
    """Contrast of the ambiguity surfaces, for each distance type and receiver pair.

    All the feature distances are non-negative and equal to 0 for identical
    features, so the best match is the minimum of the surface. The contrast
    compares this minimum ``d_min`` with a reference level ``d_q``, the
    ``percentile`` of the surface (its median by default)::

        contrast = (d_q - d_min) / d_q

    It lies in [0, 1]: 0 for a flat surface (no distinguishable minimum), close
    to 1 for a sharp, deep minimum. It is dimensionless, so the distance types
    can be compared. Non-finite values are ignored; the contrast is NaN when
    the surface has no finite value or when ``d_q`` is 0.

    Parameters
    ----------
    distance_section12 : dict
        Distances as returned by ``distances_single_intersection``
        (``dist_<type>`` of shape ``(n_rcv, n_emission1, n_emission2)``).
    percentile : float, optional
        Percentile (in ]0, 100[) used as reference level.

    Returns
    -------
    dict of numpy.ndarray
        One array of shape ``(n_rcv,)`` per available distance type (keys of
        ``DISTANCE_TYPES``).

    Raises
    ------
    ValueError
        If ``percentile`` is not in ]0, 100[ or if a distance array does not
        have 3 dimensions.
    """
    if not 0 < percentile < 100:
        raise ValueError(f"The percentile must be in ]0, 100[ (got {percentile}).")

    contrasts = {}
    for key in DISTANCE_TYPES:
        # Distance types that have not been computed are skipped
        dist = distance_section12.get(f"dist_{key}")
        if dist is None:
            continue

        dist = np.asarray(dist, dtype=float)
        if dist.ndim != 3:
            raise ValueError(
                f"Distance 'dist_{key}' has {dist.ndim} dimension(s), expected 3 "
                "(n_rcv, n_emission1, n_emission2)."
            )

        contrast = np.full(dist.shape[0], np.nan)
        for i_rcv in range(dist.shape[0]):
            # Finite values of the ambiguity surface
            values = dist[i_rcv][np.isfinite(dist[i_rcv])]
            if values.size == 0:
                continue

            # Minimum and reference level (percentile) of the surface
            d_min = values.min()
            d_ref = np.percentile(values, percentile)

            # Undefined if the reference level is null
            if d_ref > 0:
                contrast[i_rcv] = (d_ref - d_min) / d_ref

        contrasts[key] = contrast

    return contrasts


def ambiguity_contrast_from_mfp_dataset(ds_mfp_intersection, percentile=50.0):
    """Contrast of the stored ambiguity surfaces of an intersection.

    See ``ambiguity_contrast_from_distances`` for the definition.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection (see ``build_mfp_intersection_dataset``).
    percentile : float, optional
        Percentile (in ]0, 100[) of the surface used as reference level.

    Returns
    -------
    dict of numpy.ndarray
        One array of shape ``(n_rcv,)`` per distance type present in the dataset.
    """
    return ambiguity_contrast_from_distances(
        _distances_from_mfp_dataset(ds_mfp_intersection), percentile=percentile
    )


def errors_to_records(errors, rcv_ids, reference_rcv, contrasts=None, **extra):
    """Convert position errors to a list of records (one per distance type and receiver pair).

    Parameters
    ----------
    errors : dict of numpy.ndarray
        Position errors per distance type, as returned by
        ``localization_errors_from_distances``.
    rcv_ids : sequence of str
        Receiver of each error.
    reference_rcv : str
        Reference receiver of the pairs ``(rcv_id, reference_rcv)``.
    contrasts : dict of numpy.ndarray or None, optional
        Ambiguity contrast per distance type (same layout as ``errors``, see
        ``ambiguity_contrast_from_distances``). If given, a ``contrast`` field is
        added to each record (NaN when the distance type is missing).
    **extra
        Additional fields copied in each record (e.g. ``intersection="sn2 x oe6"``).

    Returns
    -------
    list of dict
        Records with the keys of ``extra`` and ``distance_type``, ``rcv_id``,
        ``rcv_pair`` (``"(<rcv_id>, <reference_rcv>)"``), ``error_m`` and, if
        ``contrasts`` is given, ``contrast``. They can be concatenated and given to ``pandas.DataFrame``.

    Raises
    ------
    ValueError
        If the number of errors does not match the number of receivers.
    """
    records = []

    for key, err in errors.items():
        if len(err) != len(rcv_ids):
            raise ValueError(
                f"{len(err)} errors for distance type '{key}' but {len(rcv_ids)} receivers."
            )

        # Contrast of the current distance type (None if not provided)
        contrast_key = None if contrasts is None else contrasts.get(key)

        for i_rcv, (rcv_id, error) in enumerate(zip(rcv_ids, err)):
            # One record per distance type and receiver pair
            record = dict(
                **extra,
                distance_type=key,
                rcv_id=str(rcv_id),
                rcv_pair=f"({rcv_id}, {reference_rcv})",
                error_m=float(error),
            )
            if contrasts is not None:
                record["contrast"] = (
                    np.nan if contrast_key is None else float(contrast_key[i_rcv])
                )
            records.append(record)

    return records


def add_validity_column(df_records, success_threshold_m):
    """Add a boolean ``valid`` column: the position error is not greater than the threshold.

    Parameters
    ----------
    df_records : pandas.DataFrame
        Records with an ``error_m`` column.
    success_threshold_m : float
        Maximum position error of a valid localization, in meters.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df_records`` with the ``valid`` column. A NaN error (no
        estimate) is not valid.
    """
    df = df_records.copy()

    # NaN <= threshold is False: a missing estimate is not valid
    df["valid"] = (df["error_m"] <= success_threshold_m).to_numpy()

    return df


def summarize_localization_errors(df_errors, success_threshold_m=100.0):
    """Statistics of the position errors, per distance type and receiver pair.

    Parameters
    ----------
    df_errors : pandas.DataFrame
        Records with the columns ``distance_type``, ``rcv_pair`` and
        ``error_m`` (one row per intersection), see ``errors_to_records``.
    success_threshold_m : float, optional
        An intersection is successfully localized if its error is not larger
        than this threshold, in meters.

    Returns
    -------
    pandas.DataFrame
        One row per ``(distance_type, rcv_pair)``, in order of appearance, with
        the columns ``n_intersections``, ``n_valid`` (errors that could be
        computed), ``mean_error_m``, ``median_error_m``, ``std_error_m`` (not
        corrected for the degrees of freedom), ``rms_error_m``, ``p90_error_m``
        and ``success_rate``. The statistics ignore the invalid errors (NaN),
        except ``success_rate`` where they count as failures
        (``success_rate = n_success / n_intersections``).

    Raises
    ------
    ValueError
        If ``df_errors`` is empty.
    """
    if len(df_errors) == 0:
        raise ValueError("No position error to summarize.")

    # Statistics ignoring the invalid errors (NaN)
    def _rms(x):
        x = x.dropna()
        return float(np.sqrt(np.mean(x**2))) if len(x) else np.nan

    def _std(x):
        return x.std(ddof=0)

    def _p90(x):
        return x.quantile(0.9)

    # NaN errors count as failures
    def _success_rate(x):
        return float((x <= success_threshold_m).sum()) / len(x)

    # One row per (distance type, receiver pair), in order of appearance
    summary = (
        df_errors.groupby(["distance_type", "rcv_pair"], sort=False)["error_m"]
        .agg(
            n_intersections="size",
            n_valid="count",
            mean_error_m="mean",
            median_error_m="median",
            std_error_m=_std,
            rms_error_m=_rms,
            p90_error_m=_p90,
            success_rate=_success_rate,
        )
        .reset_index()
    )

    return summary


def pivot_error_summary(summary, value="median_error_m"):
    """Table of one statistic: receiver pairs in rows, distance types in columns.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of ``summarize_localization_errors``.
    value : str, optional
        Column of ``summary`` to tabulate (e.g. ``"median_error_m"`` or
        ``"success_rate"``).

    Returns
    -------
    pandas.DataFrame
        Pivot table, rows and columns in order of appearance in ``summary``.
    """
    rows = list(summary["rcv_pair"].drop_duplicates())
    cols = list(summary["distance_type"].drop_duplicates())

    # Keep the order of appearance (pivot would sort alphabetically)
    table = summary.pivot(index="rcv_pair", columns="distance_type", values=value)

    return table.reindex(index=rows, columns=cols)


def best_rcv_pair_per_distance(summary, value="median_error_m", minimize=True):
    """Best receiver pair of each distance type.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of ``summarize_localization_errors``.
    value : str, optional
        Statistic used to rank the pairs.
    minimize : bool, optional
        True if a lower value is better (errors), False otherwise (success rate).

    Returns
    -------
    dict
        ``{distance_type: (rcv_pair, value)}``; ``(None, nan)`` if the statistic
        is not available for any pair.
    """
    table = pivot_error_summary(summary, value)

    best = {}
    for distance_type in table.columns:
        col = table[distance_type].to_numpy(dtype=float)

        # No valid statistic for this distance type
        if not np.isfinite(col).any():
            best[distance_type] = (None, np.nan)
            continue

        # First pair reaching the best value
        idx = np.nanargmin(col) if minimize else np.nanargmax(col)
        best[distance_type] = (table.index[idx], float(col[idx]))

    return best


def select_and_order_distance_types(df_errors, requested=None):
    """Keep, and order, the distance types to analyse in a table of position errors.

    Parameters
    ----------
    df_errors : pandas.DataFrame
        Records with a ``distance_type`` column.
    requested : list of str or None, optional
        Distance types wished by the user (None: all the available ones), see
        ``select_distance_types``.

    Returns
    -------
    df_selected : pandas.DataFrame
        Records of the selected distance types, sorted by distance type in the
        order of ``distance_types`` (the order of the other rows is preserved).
    distance_types : list of str
        Selected distance types.

    Raises
    ------
    ValueError
        If a requested type is not available, or if the selection is empty.
    """
    distance_types = select_distance_types(
        df_errors["distance_type"].unique(), requested
    )

    # Keep the selected types and sort them as requested (stable sort: other rows keep their order)
    order = {key: i for i, key in enumerate(distance_types)}
    df_selected = df_errors[df_errors["distance_type"].isin(distance_types)]
    df_selected = df_selected.sort_values(
        "distance_type", key=lambda col: col.map(order), kind="stable"
    ).reset_index(drop=True)

    return df_selected, distance_types
