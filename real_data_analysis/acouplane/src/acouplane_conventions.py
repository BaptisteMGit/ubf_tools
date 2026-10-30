#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_conventions.py
@Time    :   2026/10/30 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.1
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Names of the dimensions / variables of the feature and MFP datasets (active, event, library).
"""

# ======================================================================================================================
# Conventions of the feature datasets (one dataset per section / per event)
# ======================================================================================================================
# A "sample" is one observation of the source along a trajectory: an active emission (shot) or a passive
# window (segment of a recording). Each convention gives the name of the sample dimension and of the
# variables holding the position of the source at each sample.
SAMPLE_AXIS_CONVENTIONS = {
    "active": dict(
        datetime_dim="src_datetime", lon_var="tir_src_lon", lat_var="tir_src_lat"
    ),
    "event": dict(
        datetime_dim="event_datetime", lon_var="event_lon", lat_var="event_lat"
    ),
    "library": dict(
        datetime_dim="library_datetime", lon_var="library_lon", lat_var="library_lat"
    ),
}


def get_sample_axis_names(ds_feature):
    """Find the names of the sample dimension and of the source position variables of a feature dataset.

    Parameters
    ----------
    ds_feature : xarray.Dataset
        Feature dataset of a section (active emissions), of an event (passive
        windows) or of a library section.

    Returns
    -------
    dict
        Keys ``kind`` (``"active"``, ``"event"`` or ``"library"``),
        ``datetime_dim``, ``lon_var`` and ``lat_var``.

    Raises
    ------
    ValueError
        If the dataset follows none of the conventions of
        ``SAMPLE_AXIS_CONVENTIONS``.
    """
    for kind, names in SAMPLE_AXIS_CONVENTIONS.items():
        if names["datetime_dim"] in ds_feature.sizes:
            return dict(kind=kind, **names)

    expected = [names["datetime_dim"] for names in SAMPLE_AXIS_CONVENTIONS.values()]
    raise ValueError(
        "Cannot find the sample dimension of the feature dataset: none of the "
        f"dimensions {expected} is present (dimensions: {list(ds_feature.sizes)})."
    )


# ======================================================================================================================
# Conventions of the MFP datasets (one dataset per intersection)
# ======================================================================================================================
# Section 1 / section 2 are the two sides of an intersection: meridional / zonal section for two active
# sections, library section / event for a library compared to an event.
MFP_AXIS_CONVENTIONS = {
    "section_section": dict(
        dim1="src_datetime_section1",
        dim2="src_datetime_section2",
        lon1="tir_src_lon_section1",
        lat1="tir_src_lat_section1",
        lon2="tir_src_lon_section2",
        lat2="tir_src_lat_section2",
    ),
    "library_event": dict(
        dim1="library_datetime",
        dim2="event_datetime",
        lon1="library_lon",
        lat1="library_lat",
        lon2="event_lon",
        lat2="event_lat",
    ),
}


def get_mfp_axis_names(ds_mfp_intersection):
    """Find the names of the two shot dimensions and of the shot positions of an MFP dataset.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection (two sections, or a library section and
        an event).

    Returns
    -------
    dict
        Keys ``kind`` (``"section_section"`` or ``"library_event"``), ``dim1``,
        ``dim2`` (dimensions of the two sides of the intersection), and
        ``lon1``, ``lat1``, ``lon2``, ``lat2`` (names of the position variables).

    Raises
    ------
    ValueError
        If the dataset follows none of the conventions of
        ``MFP_AXIS_CONVENTIONS``.
    """
    for kind, names in MFP_AXIS_CONVENTIONS.items():
        if names["dim1"] in ds_mfp_intersection.sizes and names["dim2"] in ds_mfp_intersection.sizes:
            return dict(kind=kind, **names)

    expected = [(names["dim1"], names["dim2"]) for names in MFP_AXIS_CONVENTIONS.values()]
    raise ValueError(
        "Cannot find the shot dimensions of the MFP dataset: none of the pairs of "
        f"dimensions {expected} is present (dimensions: {list(ds_mfp_intersection.sizes)})."
    )


def get_mfp_shot_positions(ds_mfp_intersection):
    """Positions of the shots (samples) of the two sides of an MFP intersection dataset.

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of one intersection, see ``get_mfp_axis_names``.

    Returns
    -------
    dict
        Arrays ``lon1``, ``lat1`` (side 1) and ``lon2``, ``lat2`` (side 2), in
        the layout expected by ``localization_errors_from_distances``.
    """
    names = get_mfp_axis_names(ds_mfp_intersection)

    return {
        key: ds_mfp_intersection[names[key]].values
        for key in ("lon1", "lat1", "lon2", "lat2")
    }


# ======================================================================================================================
# Position error modes of a library / event comparison
# ======================================================================================================================
# The error modes of the localization functions refer to the sections of an intersection ("section1",
# "section2", "both"). For a library / event comparison, section 1 is the library and section 2 the event.
LIBRARY_EVENT_ERROR_MODES = {
    "library": "section1",
    "event": "section2",
    "both": "both",
}


def library_event_error_mode(error_mode):
    """Translate an error mode expressed for a library / event comparison.

    Parameters
    ----------
    error_mode : {"library", "event", "both"}
        ``"library"``: error measured between the library samples (position
        along the library section); ``"event"``: error measured between the
        event samples (position along the event trajectory); ``"both"``:
        ``sqrt(error_library**2 + error_event**2)``.

    Returns
    -------
    str
        The corresponding mode of ``localization_errors_from_distances``.

    Raises
    ------
    ValueError
        If ``error_mode`` is unknown.
    """
    if error_mode not in LIBRARY_EVENT_ERROR_MODES:
        raise ValueError(
            f"Unknown error_mode '{error_mode}' for a library / event comparison. "
            f"Valid modes: {list(LIBRARY_EVENT_ERROR_MODES)}."
        )

    return LIBRARY_EVENT_ERROR_MODES[error_mode]
