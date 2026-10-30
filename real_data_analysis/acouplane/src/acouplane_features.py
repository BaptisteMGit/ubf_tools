#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_features.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Features (RTF, DCF) and distances between the shots of two sections.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import warnings
import numpy as np

from real_data_analysis.acouplane.src.acouplane_conventions import (
    get_sample_axis_names,
)


# Maximum number of elements of the temporary 3D arrays built when comparing two sections
# (see derive_feature_distance_between_sections). Keeps the RAM usage bounded.
_MAX_BROADCAST_ELEMENTS = 20_000_000


def get_distance_between_sections(ds_feature_section1, ds_feature_section2):
    """Compute the geodesic distance between every shot of two sections.

    Distances are computed on the WGS84 ellipsoid between the source positions
    ``tir_src_lon`` / ``tir_src_lat`` of the two sections.

    Parameters
    ----------
    ds_feature_section1, ds_feature_section2 : xarray.Dataset
        Feature datasets of two sections (dimension ``src_datetime``).

    Returns
    -------
    numpy.ndarray
        Distances in meters, shape ``(n_emission1, n_emission2)``.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")

    src_lon1, src_lat1 = (
        ds_feature_section1.tir_src_lon.values,
        ds_feature_section1.tir_src_lat.values,
    )
    src_lon2, src_lat2 = (
        ds_feature_section2.tir_src_lon.values,
        ds_feature_section2.tir_src_lat.values,
    )

    dist_12 = []
    for i in range(src_lon2.size):

        # Cast to src pos shape
        lon2 = np.full(src_lon1.size, src_lon2[i])
        lat2 = np.full(src_lon1.size, src_lat2[i])

        # Compute geodesic distance
        _, _, horiz_d = geod.inv(lons1=src_lon1, lats1=src_lat1, lons2=lon2, lats2=lat2)

        dist_12.append(horiz_d)

    dist_12 = np.asarray(dist_12)  # (n_emission2, n_emission1)
    dist_12 = dist_12.T  # (n_emission1, n_emission2)

    return dist_12


def get_distance_between_library_and_event(ds_feature_library, ds_feature_event):
    """Compute the geodesic distance between every library sample and every event sample.

    Distances are computed on the WGS84 ellipsoid between the source positions of
    the library section (one position per library sample) and of the event (one
    position per event sample). The datasets can come from any acquisition mode
    (active emissions or passive windows): the names of the sample dimension and
    of the position variables are detected with ``get_sample_axis_names``.

    Parameters
    ----------
    ds_feature_library : xarray.Dataset
        Feature dataset of a library section (sample dimension
        ``library_datetime``, positions ``library_lon`` / ``library_lat``).
    ds_feature_event : xarray.Dataset
        Feature dataset of an event (sample dimension ``event_datetime``,
        positions ``event_lon`` / ``event_lat``).

    Returns
    -------
    numpy.ndarray
        Distances in meters, shape ``(n_library_samples, n_event_samples)``.

    Raises
    ------
    ValueError
        If a dataset follows none of the sample naming conventions, or if a
        position is missing (NaN) so that the distance cannot be computed.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")

    # Positions of the source at each sample of the library section and of the event
    axis_library = get_sample_axis_names(ds_feature_library)
    axis_event = get_sample_axis_names(ds_feature_event)

    library_lon = ds_feature_library[axis_library["lon_var"]].values
    library_lat = ds_feature_library[axis_library["lat_var"]].values
    event_lon = ds_feature_event[axis_event["lon_var"]].values
    event_lat = ds_feature_event[axis_event["lat_var"]].values

    for name, values in (
        ("library", library_lon),
        ("library", library_lat),
        ("event", event_lon),
        ("event", event_lat),
    ):
        if np.isnan(values).any():
            raise ValueError(
                f"The {name} positions contain NaN values: the distance in real space "
                "cannot be computed (check the position of the source along the trajectory)."
            )

    # One column per event sample, computed against all the library samples
    dist_library_event = []
    for i_event in range(event_lon.size):

        # Cast the event position to the shape of the library positions
        lon_event_i = np.full(library_lon.size, event_lon[i_event])
        lat_event_i = np.full(library_lon.size, event_lat[i_event])

        # Compute geodesic distance
        _, _, horizontal_distance = geod.inv(
            lons1=library_lon, lats1=library_lat, lons2=lon_event_i, lats2=lat_event_i
        )

        dist_library_event.append(horizontal_distance)

    dist_library_event = np.asarray(dist_library_event)  # (n_event, n_library)
    dist_library_event = dist_library_event.T  # (n_library, n_event)

    return dist_library_event


def _gamma_distances_block(gamma_a, gamma_b):
    """Compute the unnormalized distances between gamma (RTF level) vectors.

    Inputs are already broadcastable: ``gamma_a`` has shape ``(nt1, 1, nf)`` and
    ``gamma_b`` has shape ``(1, nt2, nf)``. NaN values are ignored (``nansum``).

    Parameters
    ----------
    gamma_a, gamma_b : numpy.ndarray
        RTF level in dB (real).

    Returns
    -------
    dist_theta_gamma : numpy.ndarray
        Euclidean angle between the gamma vectors, rescaled to [0, 1], shape
        ``(nt1, nt2)``.
    dist_L2_gamma_raw : numpy.ndarray
        L2 distance between the gamma vectors, shape ``(nt1, nt2)``. Not yet
        normalized to [0, 1].
    """
    # L2 distance (normalized by the caller)
    dist_L2_gamma = np.sqrt(np.nansum(np.abs(gamma_a - gamma_b) ** 2, axis=-1))

    # Theta
    inner_prod = np.nansum(gamma_a * gamma_b, axis=-1)
    norm_a = np.sqrt(np.nansum(gamma_a**2, axis=-1))
    norm_b = np.sqrt(np.nansum(gamma_b**2, axis=-1))
    # Clip to [-1, 1] for stability
    cos_angle = np.clip(inner_prod / (norm_a * norm_b), -1.0, 1.0)
    dist_theta_gamma = (1 - cos_angle) / 2  # [-1, 1] to [0, 1]

    return dist_theta_gamma, dist_L2_gamma


def _feature_distances_block(pi_a, gamma_a, sij_scot_a, pi_b, gamma_b, sij_scot_b):
    """Compute the unnormalized feature distances between two sets of shots.

    Inputs are already broadcastable: ``*_a`` have shape ``(nt1, 1, nf)`` and
    ``*_b`` have shape ``(1, nt2, nf)``. NaN values are ignored (``nansum``).

    Parameters
    ----------
    pi_a, pi_b : numpy.ndarray
        Complex RTF.
    gamma_a, gamma_b : numpy.ndarray
        RTF level in dB (real).
    sij_scot_a, sij_scot_b : numpy.ndarray
        Complex SCOT-weighted normalized cross spectrum (DCF).

    Returns
    -------
    dist_theta, dist_L2_gamma_raw, dist_theta_gamma, dist_dcf : numpy.ndarray
        Distance matrices of shape ``(nt1, nt2)``. ``dist_L2_gamma_raw`` is not
        yet normalized to [0, 1].
    """
    # Compute theta with pi
    inner_prod = np.nansum(pi_a * np.conj(pi_b), axis=-1)
    norm_a = np.sqrt(np.nansum(pi_a * np.conj(pi_a), axis=-1))
    norm_b = np.sqrt(np.nansum(pi_b * np.conj(pi_b), axis=-1))
    # Clip to [0, 1] for stability
    cos_angle = np.clip(np.abs(inner_prod) / np.abs(norm_a * norm_b), 0, 1.0)
    dist_theta = 1 - cos_angle  # to [0, 1]

    # Distances on gamma (L2 is normalized by the caller)
    dist_theta_gamma, dist_L2_gamma = _gamma_distances_block(gamma_a, gamma_b)

    # DCF
    dist_dcf = np.abs(np.nansum(sij_scot_a * np.conj(sij_scot_b), axis=-1))
    dist_dcf = 1 - dist_dcf / (
        np.sqrt(np.nansum(np.abs(sij_scot_a) ** 2, axis=-1))
        * np.sqrt(np.nansum(np.abs(sij_scot_b) ** 2, axis=-1))
    )

    return dist_theta, dist_L2_gamma, dist_theta_gamma, dist_dcf


def derive_feature_distance_between_sections(features_section1, features_section2):
    """Compute the distance in feature space between all shots of two sections.

    Four distances are computed between every shot of section 1 and every shot
    of section 2, for a single receiver pair:

    - ``dist_theta``: Hermitian angle between the complex RTF vectors, in [0, 1];
    - ``dist_theta_gamma``: Euclidean angle between the RTF level (gamma)
      vectors, rescaled to [0, 1];
    - ``dist_L2_gamma``: L2 distance between the gamma vectors, divided by its
      maximum so that it lies in [0, 1];
    - ``dist_dcf``: 1 minus the normalized correlation between the DCF vectors.

    To bound the memory usage, section 1 is processed by blocks of rows so that
    the temporary arrays never exceed about ``_MAX_BROADCAST_ELEMENTS``
    elements. The results are identical to a fully broadcast computation.

    Parameters
    ----------
    features_section1, features_section2 : dict
        Features of each section, with the keys ``pi_ij`` (complex RTF),
        ``gamma_ij`` (RTF level in dB) and ``sij_scot`` (DCF), each of shape
        ``(n_emission, n_freq)``. Both sections must share the same ``n_freq``.

    Returns
    -------
    dist_theta, dist_theta_gamma, dist_L2_gamma, dist_dcf : numpy.ndarray
        Distance matrices of shape ``(n_emission1, n_emission2)``.

    Raises
    ------
    ValueError
        If the two sections do not have the same number of frequency bins.
    """
    pi_ij_a, gamma_a, sij_scot_a = (
        features_section1["pi_ij"],
        features_section1["gamma_ij"],
        features_section1["sij_scot"],
    )
    pi_ij_b, gamma_b, sij_scot_b = (
        features_section2["pi_ij"],
        features_section2["gamma_ij"],
        features_section2["sij_scot"],
    )

    if pi_ij_a.shape[-1] != pi_ij_b.shape[-1]:
        raise ValueError(
            "The two sections do not have the same number of frequency bins "
            f"({pi_ij_a.shape[-1]} vs {pi_ij_b.shape[-1]}): recompute the CPSD "
            "of both sections with the same fmin / fmax / nfft."
        )

    nt1, nf = pi_ij_a.shape
    nt2 = pi_ij_b.shape[0]

    # Reshape for broadcasting
    pi_b = pi_ij_b[np.newaxis, :, :]  # (1, nt2, nf)
    gamma_b = gamma_b[np.newaxis, :, :]  # (1, nt2, nf)
    sij_scot_b = sij_scot_b[np.newaxis, :, :]  # (1, nt2, nf)

    block = max(1, _MAX_BROADCAST_ELEMENTS // max(1, nt2 * nf))

    dist_theta = np.empty((nt1, nt2))
    dist_L2_gamma = np.empty((nt1, nt2))
    dist_theta_gamma = np.empty((nt1, nt2))
    dist_dcf = np.empty((nt1, nt2))

    # 0/0 occurs for shots with no valid frequency (all NaN): the distance is then NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        for start in range(0, nt1, block):
            sl = slice(start, min(start + block, nt1))
            d_theta, d_l2, d_theta_gamma, d_dcf = _feature_distances_block(
                pi_ij_a[sl, np.newaxis, :],  # (block, 1, nf)
                gamma_a[sl, np.newaxis, :],
                sij_scot_a[sl, np.newaxis, :],
                pi_b,
                gamma_b,
                sij_scot_b,
            )
            dist_theta[sl] = d_theta
            dist_L2_gamma[sl] = d_l2
            dist_theta_gamma[sl] = d_theta_gamma
            dist_dcf[sl] = d_dcf

    # Normalize L2 to [0, 1]
    dist_L2_gamma = _normalize_by_max(dist_L2_gamma)

    return dist_theta, dist_theta_gamma, dist_L2_gamma, dist_dcf


def _non_finite_to_nan(arr):
    """Replace the non finite values (inf, -inf, NaN) of an array by NaN.

    Parameters
    ----------
    arr : numpy.ndarray
        Real or complex array.

    Returns
    -------
    numpy.ndarray
        Copy of ``arr`` where every non finite element (for a complex array, an
        element with a non finite real or imaginary part) is NaN.
    """
    arr = np.array(arr, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _normalize_by_max(dist):
    """Divide a distance matrix by its maximum, ignoring NaN values.

    Parameters
    ----------
    dist : numpy.ndarray
        Distance matrix, possibly containing NaN.

    Returns
    -------
    numpy.ndarray
        ``dist / nanmax(dist)``. If ``dist`` has no valid value, the result is
        all NaN. No warning is emitted.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN matrix
        dmax = np.nanmax(dist)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dist / dmax


def _center_along_freq(gamma):
    """Remove from each shot the mean of its gamma over the frequencies.

    NaN values are ignored in the mean (``np.nanmean``) and stay NaN in the
    result. A shot with only NaN stays entirely NaN.

    Parameters
    ----------
    gamma : numpy.ndarray
        RTF level in dB, shape ``(n_emission, n_freq)``.

    Returns
    -------
    numpy.ndarray
        Centered gamma, same shape.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
        return gamma - np.nanmean(gamma, axis=1, keepdims=True)


def derive_gamma_distance_between_sections(gamma_section1, gamma_section2):
    """Compute the gamma distances (theta_gamma and L2) between all shots of two sections.

    Same computation as in ``derive_feature_distance_between_sections`` but
    restricted to the gamma (RTF level) vectors, which allows comparing variants
    of gamma (e.g. centered along frequency) without storing them. Processed by
    blocks of rows to bound the memory usage.

    Parameters
    ----------
    gamma_section1, gamma_section2 : numpy.ndarray
        RTF level in dB of each section, shape ``(n_emission, n_freq)``.

    Returns
    -------
    dist_theta_gamma, dist_L2_gamma : numpy.ndarray
        Distance matrices of shape ``(n_emission1, n_emission2)``;
        ``dist_L2_gamma`` is normalized by its maximum.

    Raises
    ------
    ValueError
        If the two sections do not have the same number of frequency bins.
    """
    if gamma_section1.shape[-1] != gamma_section2.shape[-1]:
        raise ValueError(
            "The two sections do not have the same number of frequency bins "
            f"({gamma_section1.shape[-1]} vs {gamma_section2.shape[-1]})."
        )
    nt1, nf = gamma_section1.shape
    nt2 = gamma_section2.shape[0]
    gamma_b = gamma_section2[np.newaxis, :, :]
    block = max(1, _MAX_BROADCAST_ELEMENTS // max(1, nt2 * nf))

    dist_theta_gamma = np.empty((nt1, nt2))
    dist_L2_gamma = np.empty((nt1, nt2))
    # 0/0 occurs for shots with no valid frequency (all NaN): the distance is then NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        for start in range(0, nt1, block):
            sl = slice(start, min(start + block, nt1))
            dist_theta_gamma[sl], dist_L2_gamma[sl] = _gamma_distances_block(
                gamma_section1[sl, np.newaxis, :], gamma_b
            )

    return dist_theta_gamma, _normalize_by_max(dist_L2_gamma)


def derive_section_features(
    ds_feature_section, fmin=5, fmax=50, rcv_ref="obs1", rcv_i="obs2"
):
    """Derive the RTF and DCF features of a section for one receiver pair.

    With ``i = rcv_i`` and ``j = rcv_ref``, for every shot and frequency:

    - RTF: ``pi_ij = S_ij / S_jj`` and its level ``gamma_ij = 20 log10 |pi_ij|``;
    - DCF: ``S_ij / sqrt(S_ii * S_jj)`` (SCOT weighting).

    Parameters
    ----------
    ds_feature_section : xarray.Dataset
        Feature dataset of a section, with the variable ``Sxy`` of dimensions
        ``(src_datetime, rcv_i, rcv_j, freq)``.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.
    rcv_ref : str, optional
        Reference receiver (``j``).
    rcv_i : str, optional
        Other receiver of the pair (``i``).

    Returns
    -------
    pi_ij : numpy.ndarray
        Complex RTF, shape ``(n_emission, n_freq)``.
    gamma_ij : numpy.ndarray
        RTF level in dB, same shape.
    sij_scot : numpy.ndarray
        Complex SCOT-weighted cross spectrum, same shape.

    Notes
    -----
    Non finite values (``inf`` / ``-inf`` / ``NaN``), which appear at frequency
    bins where a spectrum is exactly zero (division by zero, ``log10(0)``), are
    all replaced by NaN in the three outputs, so that they are consistently
    ignored by the distance computations.

    Raises
    ------
    ValueError
        If ``rcv_ref`` or ``rcv_i`` is not a receiver of the dataset.
    """
    available = list(ds_feature_section.rcv_i.values)
    for rcv in (rcv_ref, rcv_i):
        if rcv not in available:
            raise ValueError(
                f"Receiver '{rcv}' not found in the feature dataset "
                f"(available receivers: {available})."
            )

    freq_slice = slice(fmin, fmax)
    # Sii, Sjj and Sij
    sii = ds_feature_section.sel(rcv_i=rcv_i, rcv_j=rcv_i, freq=freq_slice).Sxy
    sjj = ds_feature_section.sel(rcv_i=rcv_ref, rcv_j=rcv_ref, freq=freq_slice).Sxy
    sij = ds_feature_section.sel(rcv_i=rcv_i, rcv_j=rcv_ref, freq=freq_slice).Sxy

    # Frequency bins where a spectrum is exactly zero give 0/0 or log10(0): these
    # invalid values are expected and are replaced by NaN below
    with np.errstate(divide="ignore", invalid="ignore"):
        # RTF
        pi_ij = sij.values / sjj.values
        gamma_ij = 20 * np.log10(np.abs(pi_ij))  # (nt, nf)

        # DCF
        sij_scot = sij.values / np.sqrt(sii.values * sjj.values)

    return (
        _non_finite_to_nan(pi_ij),
        _non_finite_to_nan(gamma_ij),
        _non_finite_to_nan(sij_scot),
    )


def all_features_single_section(ds_feature_section, rcv_ref="obs1", fmin=5, fmax=50):
    """Derive the RTF and DCF features of a section for all receivers but the reference.

    The RTF level ``gamma`` is NOT centered along the frequency axis (see
    ``distances_single_intersection`` to compare with a centered ``gamma``).

    Parameters
    ----------
    ds_feature_section : xarray.Dataset
        Feature dataset of a section (variable ``Sxy``).
    rcv_ref : str, optional
        Reference receiver.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.

    Returns
    -------
    dict
        ``rcv_id`` (list of the non-reference receivers) and the arrays
        ``pi_ij``, ``gamma_ij`` and ``sij_scot`` of shape
        ``(n_rcv - 1, n_emission, n_freq)``.

    Raises
    ------
    ValueError
        If ``rcv_ref`` is not a receiver of the dataset.
    """
    rcv_i_arr = list(ds_feature_section.rcv_i.values)
    if rcv_ref not in rcv_i_arr:
        raise ValueError(
            f"Reference receiver '{rcv_ref}' not found in the feature dataset "
            f"(available receivers: {rcv_i_arr})."
        )
    rcv_i_arr.remove(rcv_ref)

    pi_1_arr = []
    gamma_1_arr = []
    sij_scot_1_arr = []

    for rcv_id_i in rcv_i_arr:
        # Compute features (RTF, DCF) of the section
        pi_ij_a, gamma_a, sij_scot_a = derive_section_features(
            ds_feature_section=ds_feature_section,
            fmin=fmin,
            fmax=fmax,
            rcv_ref=rcv_ref,
            rcv_i=rcv_id_i,
        )

        # Store
        pi_1_arr.append(pi_ij_a)
        gamma_1_arr.append(gamma_a)
        sij_scot_1_arr.append(sij_scot_a)

    # Convert to arrays and group (nrcv, n_emission, nfreq)
    features_section = dict(
        rcv_id=rcv_i_arr,
        pi_ij=np.asarray(pi_1_arr),
        gamma_ij=np.asarray(gamma_1_arr),
        sij_scot=np.asarray(sij_scot_1_arr),
    )

    return features_section


def feature_distances_between_datasets(
    ds_feature_a, ds_feature_b, compare_centered_gamma=False
):
    """Compute the feature distances between the samples of two feature datasets.

    For each receiver pair ``(rcv_id_feature, reference receiver)``, the four
    feature distances of ``derive_feature_distance_between_sections`` are
    computed between every sample of ``ds_feature_a`` and every sample of
    ``ds_feature_b``. Only the variables ``pi_ij``, ``gamma_ij`` and
    ``sij_scot`` are used, so that the function is the same for two active
    sections, a library section and an event, or two passive events.

    Optionally, the gamma distances are also computed with a ``gamma`` centered
    along frequency (see ``distances_single_intersection``).

    Parameters
    ----------
    ds_feature_a, ds_feature_b : xarray.Dataset
        Feature datasets with the variables ``pi_ij``, ``gamma_ij`` and
        ``sij_scot`` of dimensions ``(rcv_id_feature, <sample dimension>, freq)``
        and the same receivers and frequencies.
    compare_centered_gamma : bool, optional
        If True, also compute ``dist_theta_gamma_centered`` and
        ``dist_L2_gamma_centered``.

    Returns
    -------
    dict
        ``dist_theta``, ``dist_theta_gamma``, ``dist_L2_gamma`` and ``dist_dcf``
        (shape ``(n_rcv, n_samples_a, n_samples_b)``), plus
        ``dist_theta_gamma_centered`` and ``dist_L2_gamma_centered`` if
        ``compare_centered_gamma`` is True.
    """
    dist_theta_arr = []
    dist_theta_gamma_arr = []
    dist_L2_gamma_arr = []
    dist_dcf_arr = []
    dist_theta_gamma_c_arr = []
    dist_L2_gamma_c_arr = []

    # Loop over the receiver pairs (rcv_id_i, reference receiver)
    for rcv_id_i in ds_feature_a.rcv_id_feature.values:

        ds_feature_a_rcv_i = ds_feature_a.sel(rcv_id_feature=rcv_id_i)
        ds_feature_b_rcv_i = ds_feature_b.sel(rcv_id_feature=rcv_id_i)
        # Extract features
        features_a = dict(
            pi_ij=ds_feature_a_rcv_i.pi_ij.values,
            gamma_ij=ds_feature_a_rcv_i.gamma_ij.values,
            sij_scot=ds_feature_a_rcv_i.sij_scot.values,
        )
        features_b = dict(
            pi_ij=ds_feature_b_rcv_i.pi_ij.values,
            gamma_ij=ds_feature_b_rcv_i.gamma_ij.values,
            sij_scot=ds_feature_b_rcv_i.sij_scot.values,
        )
        # Compute distances between the samples of both datasets
        dist_theta, dist_theta_gamma, dist_L2_gamma, dist_dcf = (
            derive_feature_distance_between_sections(
                features_a, features_b
            )
        )

        # Store
        dist_theta_arr.append(dist_theta)
        dist_theta_gamma_arr.append(dist_theta_gamma)
        dist_L2_gamma_arr.append(dist_L2_gamma)
        dist_dcf_arr.append(dist_dcf)

        # Same gamma distances with gamma centered along frequency (not stored)
        if compare_centered_gamma:
            gamma1_c = _center_along_freq(features_a["gamma_ij"])
            gamma2_c = _center_along_freq(features_b["gamma_ij"])
            d_theta_gamma_c, d_L2_gamma_c = derive_gamma_distance_between_sections(
                gamma1_c, gamma2_c
            )
            dist_theta_gamma_c_arr.append(d_theta_gamma_c)
            dist_L2_gamma_c_arr.append(d_L2_gamma_c)

    # Convert to arrays and group (nrcv, n_samples_a, n_samples_b)
    feature_distances = dict(
        dist_theta=np.asarray(dist_theta_arr),
        dist_theta_gamma=np.asarray(dist_theta_gamma_arr),
        dist_L2_gamma=np.asarray(dist_L2_gamma_arr),
        dist_dcf=np.asarray(dist_dcf_arr),
    )
    if compare_centered_gamma:
        feature_distances["dist_theta_gamma_centered"] = np.asarray(
            dist_theta_gamma_c_arr
        )
        feature_distances["dist_L2_gamma_centered"] = np.asarray(dist_L2_gamma_c_arr)

    return feature_distances


def distances_single_intersection(
    ds_feature_section1, ds_feature_section2, compare_centered_gamma=False
):
    """Compute all distances between the shots of two intersecting sections.

    The geodesic distance between shots ("real space") and, for each receiver
    pair ``(rcv_id_feature, reference receiver)``, the four feature distances of
    ``derive_feature_distance_between_sections`` are computed.

    Optionally, the gamma distances are also computed with a ``gamma`` centered
    along frequency (mean of each shot removed, NaN values ignored in the mean,
    see ``_center_along_freq``). The centered gamma is computed on the fly and
    is never stored. ``theta``
    (complex RTF) and ``dcf`` do not depend on ``gamma`` and are therefore
    unchanged by the centering.

    Parameters
    ----------
    ds_feature_section1, ds_feature_section2 : xarray.Dataset
        Feature datasets of two sections, with the variables ``pi_ij``,
        ``gamma_ij`` and ``sij_scot`` of dimensions
        ``(rcv_id_feature, src_datetime, freq)`` (see
        ``apply_mfp_strategy_to_portion_intersections``).
    compare_centered_gamma : bool, optional
        If True, also compute ``dist_theta_gamma_centered`` and
        ``dist_L2_gamma_centered``.

    Returns
    -------
    dict
        Keys ``dist_in_real_space`` (shape ``(n_emission1, n_emission2)``, m)
        and ``dist_theta``, ``dist_theta_gamma``, ``dist_L2_gamma``,
        ``dist_dcf`` (shape ``(n_rcv, n_emission1, n_emission2)``). If
        ``compare_centered_gamma`` is True, also ``dist_theta_gamma_centered``
        and ``dist_L2_gamma_centered`` (same shape).
    """
    # Compute real distance in meters between emissions from section 1 to emission from section 2
    dist_in_real_space_12 = get_distance_between_sections(
        ds_feature_section1, ds_feature_section2
    )

    feature_distances = feature_distances_between_datasets(
        ds_feature_section1, ds_feature_section2, compare_centered_gamma
    )

    # Group (nrcv, n_emission1, n_emission2)
    distance_section12 = dict(dist_in_real_space=dist_in_real_space_12)
    distance_section12.update(feature_distances)

    return distance_section12


def distances_library_and_event(
    ds_feature_library, ds_feature_event, compare_centered_gamma=False
):
    """Compute all distances between the samples of a library section and of an event.

    The geodesic distance between samples ("real space", see
    ``get_distance_between_library_and_event``) and the feature distances (see
    ``feature_distances_between_datasets``) are computed. This is the
    counterpart of ``distances_single_intersection`` for a library / event
    comparison.

    Parameters
    ----------
    ds_feature_library : xarray.Dataset
        Feature dataset of a library section.
    ds_feature_event : xarray.Dataset
        Feature dataset of an event, with the same receivers (``rcv_id_feature``)
        and the same frequencies as the library.
    compare_centered_gamma : bool, optional
        If True, also compute ``dist_theta_gamma_centered`` and
        ``dist_L2_gamma_centered``.

    Returns
    -------
    dict
        Keys ``dist_in_real_space`` (shape ``(n_library_samples,
        n_event_samples)``, m) and the feature distances, shape ``(n_rcv,
        n_library_samples, n_event_samples)``, see
        ``distances_single_intersection``.
    """
    # Distance in meters between the library samples and the event samples
    dist_in_real_space = get_distance_between_library_and_event(
        ds_feature_library, ds_feature_event
    )

    # Distances in the feature space, for each receiver pair
    feature_distances = feature_distances_between_datasets(
        ds_feature_library, ds_feature_event, compare_centered_gamma
    )

    distance_library_event = dict(dist_in_real_space=dist_in_real_space)
    distance_library_event.update(feature_distances)

    return distance_library_event
