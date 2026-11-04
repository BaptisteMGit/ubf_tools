#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_mfp.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Matched field processing of the intersections: ambiguity surfaces, true and estimated intersections.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import xarray as xr

from real_data_analysis.acouplane.src.acouplane_conventions import (
    get_mfp_axis_names,
)
from real_data_analysis.acouplane.src.acouplane_features import (
    DEFAULT_RTF_METHOD,
    all_features_single_section,
    check_rtf_method,
    distances_single_intersection,
)
from real_data_analysis.acouplane.src.acouplane_portion_io import (
    select_intersections,
)


# Names and descriptions of the variables of the ambiguity surfaces (used for the labels of the figures)
DISTANCE_VARIABLE_ATTRS = {
    "d_real": dict(units="m", long_name="Distance"),
    "d_theta": dict(
        long_name=r"$\theta_{\Pi}$",
        description="Hermitian angle between RTF vectors",
        minval=0,
        maxval=1,
    ),
    "d_theta_gamma": dict(
        long_name=r"$\theta_{\gamma}$",
        description="Euclidean angle between RTF gamma vectors",
        minval=0,
        maxval=1,
    ),
    "d_L2_gamma": dict(
        long_name=r"$L_{2_{\gamma}}$",
        description="Normalized euclidean distance (L2) between RTF gamma vectors",
        minval=0,
        maxval=1,
    ),
    "d_dcf": dict(
        long_name=r"$C_{el}$",
        description="Normalized intercorrelation between intercorrelation vectors (DCF)",
        minval=0,
        maxval=1,
    ),
    "d_theta_gamma_centered": dict(
        long_name=r"$\theta_{\gamma}$ (centered)",
        description="Euclidean angle between RTF gamma vectors centered along frequency",
        minval=0,
        maxval=1,
    ),
    "d_L2_gamma_centered": dict(
        long_name=r"$L_{2_{\gamma}}$ (centered)",
        description="Normalized euclidean distance (L2) between RTF gamma vectors centered along frequency",
        minval=0,
        maxval=1,
    ),
}


def add_centered_gamma_distances(ds_mfp, distance_section12, dims):
    """Add the distances computed with gamma centered along frequency to an MFP dataset (if available).

    Parameters
    ----------
    ds_mfp : xarray.Dataset
        MFP dataset of one intersection (modified in place).
    distance_section12 : dict
        Distances as returned by ``distances_single_intersection`` or
        ``distances_library_and_event``; the centered distances
        ``dist_theta_gamma_centered`` and ``dist_L2_gamma_centered`` are added
        only if they are present.
    dims : list of str
        Dimensions of the distances: receiver pair and the two shot dimensions.

    Returns
    -------
    None
    """
    for var in ("d_theta_gamma_centered", "d_L2_gamma_centered"):
        key = var.replace("d_", "dist_", 1)
        if key in distance_section12:
            ds_mfp[var] = (
                dims,
                distance_section12[key],
            )
            ds_mfp[var].attrs = dict(DISTANCE_VARIABLE_ATTRS[var])


def add_distance_attrs(ds_mfp):
    """Add the names and descriptions of the distances to the variables of an MFP dataset.

    Parameters
    ----------
    ds_mfp : xarray.Dataset
        MFP dataset of one intersection (modified in place), with the variables
        ``d_real``, ``d_theta``, ``d_theta_gamma``, ``d_L2_gamma`` and ``d_dcf``.

    Returns
    -------
    None
    """
    for var in ("d_real", "d_theta", "d_theta_gamma", "d_L2_gamma", "d_dcf"):
        ds_mfp[var].attrs = dict(DISTANCE_VARIABLE_ATTRS[var])


def build_mfp_intersection_dataset(
    distance_section12,
    ds_feature_section1,
    ds_feature_section2,
):
    """Build the dataset holding the MFP ambiguity surfaces of one intersection.

    Parameters
    ----------
    distance_section12 : dict
        Distances as returned by ``distances_single_intersection``.
    ds_feature_section1, ds_feature_section2 : xarray.Dataset
        Feature datasets of the two intersecting sections (section 1 is the
        meridional one, section 2 the zonal one).

    Returns
    -------
    xarray.Dataset
        Dataset with the receiver information, the shot positions of both
        sections, the real distance ``d_real`` (dimensions
        ``(src_datetime_section1, src_datetime_section2)``) and the feature
        distances ``d_theta``, ``d_theta_gamma``, ``d_L2_gamma`` and ``d_dcf``
        (dimensions ``(rcv_id_feature, src_datetime_section1,
        src_datetime_section2)``). The attribute values that are missing from
        the feature datasets are set to ``"unknown"``.
    """
    attrs1 = ds_feature_section1.attrs

    ds_mfp = xr.Dataset(
        data_vars=dict(
            rcv_lon=ds_feature_section1.rcv_lon,
            rcv_lat=ds_feature_section1.rcv_lat,
            rcv_immersion=ds_feature_section1.rcv_immersion,
            tir_src_lon_section1=(
                ["src_datetime_section1"],
                ds_feature_section1.tir_src_lon.values,
            ),
            tir_src_lat_section1=(
                ["src_datetime_section1"],
                ds_feature_section1.tir_src_lat.values,
            ),
            tir_src_lon_section2=(
                ["src_datetime_section2"],
                ds_feature_section2.tir_src_lon.values,
            ),
            tir_src_lat_section2=(
                ["src_datetime_section2"],
                ds_feature_section2.tir_src_lat.values,
            ),
            d_real=(
                ["src_datetime_section1", "src_datetime_section2"],
                distance_section12["dist_in_real_space"],
            ),
            d_theta=(
                ["rcv_id_feature", "src_datetime_section1", "src_datetime_section2"],
                distance_section12["dist_theta"],
            ),
            d_theta_gamma=(
                ["rcv_id_feature", "src_datetime_section1", "src_datetime_section2"],
                distance_section12["dist_theta_gamma"],
            ),
            d_L2_gamma=(
                ["rcv_id_feature", "src_datetime_section1", "src_datetime_section2"],
                distance_section12["dist_L2_gamma"],
            ),
            d_dcf=(
                ["rcv_id_feature", "src_datetime_section1", "src_datetime_section2"],
                distance_section12["dist_dcf"],
            ),
        ),
        coords=dict(
            rcv_id_feature=ds_feature_section1.rcv_id_feature.values,
            src_datetime_section1=ds_feature_section1.src_datetime.values,
            src_datetime_section2=ds_feature_section2.src_datetime.values,
        ),
        attrs=dict(
            description="MFP result for a single intersection",
            campagne=attrs1.get("campagne", "unknown"),
            section1=attrs1.get("section_label", "unknown"),
            section2=ds_feature_section2.attrs.get("section_label", "unknown"),
            reference_receiver=attrs1.get("reference_receiver", "unknown"),
            obs_position=attrs1.get("obs_position", "unknown"),
            obs_position_filepath=attrs1.get("obs_position_filepath", "unknown"),
        ),
    )

    # Optional comparison with gamma centered along frequency
    add_centered_gamma_distances(
        ds_mfp,
        distance_section12,
        dims=["rcv_id_feature", "src_datetime_section1", "src_datetime_section2"],
    )

    # Add attrs to variables
    add_distance_attrs(ds_mfp)

    return ds_mfp


def apply_mfp_strategy_to_portion_intersections(
    portion,
    rcv_ref="obs1",
    fmin=5,
    fmax=50,
    verbose=False,
    compare_centered_gamma=False,
    intersections=None,
    rtf_method=DEFAULT_RTF_METHOD,
):
    """Compute the MFP ambiguity surfaces for meridional / zonal intersections.

    Step 1: the intersections to process are selected (see ``select_intersections``).
    Step 2: the RTF and DCF features of the sections involved in these
    intersections are derived (``all_features_single_section``) and stored in
    the ``feature`` group (variables ``pi_ij``, ``gamma_ij``, ``sij_scot``).
    Sections that are not involved are left untouched.
    Step 3: each selected intersection (meridional section, label starting with
    ``"sn"``, x zonal section, label starting with ``"oe"``) is processed and
    the result is stored in ``/mfp_intersection/<meridional>/<zonal>``.

    Parameters
    ----------
    portion : xarray.DataTree
        Portion tree with a ``feature`` group (see ``derive_portion_cpsd``);
        modified in place.
    rcv_ref : str, optional
        Reference receiver.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.
    verbose : bool, optional
        If True, print the progress.
    compare_centered_gamma : bool, optional
        If True, the intersection datasets also contain ``d_theta_gamma_centered``
        and ``d_L2_gamma_centered`` (distances with ``gamma`` centered along
        frequency). The centered gamma itself is not stored.
    intersections : list of tuple of str or None, optional
        Intersections to process, as ``(meridional_label, zonal_label)`` pairs,
        e.g. ``[("sn0", "oe1"), ("sn0", "oe3")]``. If None (default), all
        intersections are processed.
    rtf_method : {"spectral_ratio", "cs_evd"}, optional
        Estimation of the RTF (see ``all_features_single_section``): the ratio
        ``S_ij / S_jj`` (active case, default) or the first eigenvector of the
        spatial covariance matrix (passive case). It is stored in the attribute
        ``rtf_method`` and must be the same for the library and the event.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the ``feature`` group is missing, if the portion has no meridional
        or no zonal section (no intersection can be built), or if
        ``intersections`` contains an invalid pair (see ``select_intersections``),
        or if ``rtf_method`` is unknown.
    """
    check_rtf_method(rtf_method)
    if "feature" not in portion.children:
        raise ValueError(
            "No 'feature' group in the portion: the CPSD has not been computed yet. "
            "Run with compute_cpsds = True first."
        )

    # Step 1 : list available sections and select the intersections to process
    sections = list(portion.feature.children)
    intersections_to_process = select_intersections(sections, intersections)
    sections_needed = {sec for pair in intersections_to_process for sec in pair}

    # Step 2 : derive the features of the needed sections only (avoids duplicated
    # calls to derive_section_features)
    for section_name, feature_node in portion.feature.children.items():
        if section_name not in sections_needed:
            continue
        if verbose:
            print(f"    Features of section '{section_name}' (RTF '{rtf_method}')")
        ds_feature_section = feature_node.to_dataset()

        # Slice frequency band
        ds_feature_section = ds_feature_section.sel(freq=slice(fmin, fmax))

        # Derive features
        features_section = all_features_single_section(
            ds_feature_section=ds_feature_section,
            rcv_ref=rcv_ref,
            fmin=fmin,
            fmax=fmax,
            rtf_method=rtf_method,
        )  # (nrcv, n_emission, nfreq)

        # Add coords
        ds_feature_section["rcv_id_feature"] = features_section["rcv_id"]

        # Store in dataset
        for var in ("pi_ij", "gamma_ij", "sij_scot"):
            ds_feature_section[var] = (
                ["rcv_id_feature", "src_datetime", "freq"],
                features_section[var],
            )

        # Add attributes
        ds_feature_section.attrs["reference_receiver"] = rcv_ref
        ds_feature_section.attrs["rtf_method"] = rtf_method

        # Update portion
        feature_node.ds = ds_feature_section

    # Step 3 : intersections are between a meridional and a zonal section
    for i_inter, (section_name1, section_name2) in enumerate(intersections_to_process):
        if verbose:
            print(
                f"    Intersection '{section_name1}' x '{section_name2}' "
                f"({i_inter + 1}/{len(intersections_to_process)})"
            )
        ds_feature_section1 = portion.feature[section_name1].to_dataset()
        ds_feature_section2 = portion.feature[section_name2].to_dataset()

        # Compute distance (in real space and in features space)
        distance_section12 = distances_single_intersection(
            ds_feature_section1=ds_feature_section1,
            ds_feature_section2=ds_feature_section2,
            compare_centered_gamma=compare_centered_gamma,
        )

        # Build dataset to save
        ds_mfp = build_mfp_intersection_dataset(
            distance_section12=distance_section12,
            ds_feature_section1=ds_feature_section1,
            ds_feature_section2=ds_feature_section2,
        )

        # Add to portion
        new_path = f"/mfp_intersection/{section_name1}/{section_name2}"
        portion[new_path] = ds_mfp


def get_intersection_target_position(ds_mfp_intersection):
    """Find the true intersection of two sections from the real-space distance.

    The "target" is the pair of shots (one on each section) which are the
    closest to each other in real space, i.e. the position where the two
    trajectories actually cross.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection (see ``build_mfp_intersection_dataset``),
        with the variable ``d_real`` of dimensions
        ``(src_datetime_section1, src_datetime_section2)``.

    Returns
    -------
    intersection_dt1 : numpy.datetime64
        Datetime, along side 1 (section 1 or library), of the closest pair.
    intersection_dt2 : numpy.datetime64
        Datetime, along side 2 (section 2 or event), of the closest pair.
    """
    axis_names = get_mfp_axis_names(ds_mfp_intersection)
    dims = [axis_names["dim1"], axis_names["dim2"]]

    idx_target_pos = ds_mfp_intersection.d_real.argmin(dim=dims)
    ds_target = ds_mfp_intersection.isel(idx_target_pos)
    intersection_dt1, intersection_dt2 = (
        ds_target[axis_names["dim1"]].values,
        ds_target[axis_names["dim2"]].values,
    )

    return intersection_dt1, intersection_dt2


def get_intersection_guess_positions(ds_mfp_intersection):
    """Find the estimated intersection for each feature-space distance.

    For each distance (``d_theta``, ``d_theta_gamma``, ``d_L2_gamma`` and
    ``d_dcf``, plus ``d_theta_gamma_centered`` and ``d_L2_gamma_centered`` when
    they are present in the dataset) and each receiver pair, the pair of shots
    minimizing the distance is the estimated (guessed) intersection.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection, with the distances of dimensions
        ``(rcv_id_feature, src_datetime_section1, src_datetime_section2)``.

    Returns
    -------
    dict
        One entry per distance: ``"theta"``, ``"theta_gamma"``, ``"L2_gamma"``
        and ``"dcf"``, plus ``"theta_gamma_centered"`` and
        ``"L2_gamma_centered"`` if available. Each entry is a dict with the keys
        ``dt1`` and ``dt2`` (xarray.DataArray of dimension ``rcv_id_feature``):
        emission datetimes of the estimated intersection along section 1 and
        section 2.
    """
    axis_names = get_mfp_axis_names(ds_mfp_intersection)
    dims = [axis_names["dim1"], axis_names["dim2"]]
    metrics = dict(
        theta="d_theta",
        theta_gamma="d_theta_gamma",
        L2_gamma="d_L2_gamma",
        dcf="d_dcf",
        theta_gamma_centered="d_theta_gamma_centered",
        L2_gamma_centered="d_L2_gamma_centered",
    )

    intersections = {}
    for key, var in metrics.items():
        if var not in ds_mfp_intersection:
            continue
        idx_guess_pos = ds_mfp_intersection[var].argmin(dim=dims)
        ds_guess = ds_mfp_intersection.isel(idx_guess_pos)
        intersections[key] = dict(
            dt1=ds_guess[axis_names["dim1"]],
            dt2=ds_guess[axis_names["dim2"]],
        )

    return intersections
