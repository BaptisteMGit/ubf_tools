#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_event.py
@Time    :   2026/10/30 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.1
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Comparison of an event (passive recording, AIS trajectory) to a library of reference features.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import xarray as xr

from real_data_analysis.acouplane.src.acouplane_conventions import (
    get_sample_axis_names,
)
from real_data_analysis.acouplane.src.acouplane_features import (
    all_features_single_section,
    distances_library_and_event,
)
from real_data_analysis.acouplane.src.acouplane_mfp import (
    add_centered_gamma_distances,
    add_distance_attrs,
)
from real_data_analysis.acouplane.src.acouplane_portion_io import (
    restrict_dataset_to_receivers,
)

# Naming of the library samples (common to all the acquisition modes of the library)
LIBRARY_DATETIME_DIM = "library_datetime"
LIBRARY_LON_VAR = "library_lon"
LIBRARY_LAT_VAR = "library_lat"

# Variables of a feature dataset kept in the library: the features and the receiver positions
LIBRARY_RECEIVER_VARS = ("rcv_lon", "rcv_lat", "rcv_immersion")
LIBRARY_FEATURE_VARS = ("pi_ij", "gamma_ij", "sij_scot")

ACQUISITION_MODES = ("active", "passive")


# ======================================================================================================================
# Features of an event (passive recording)
# ======================================================================================================================
def derive_portion_cpsd_welch(
    portion_tree,
    analysis_segment_duration_s=5,
    analysis_segment_alpha_overlap=0.75,
    nfft=2**14,
    nperseg=2**13,
    noverlap=2**12,
    obs_ids=None,
    fmin=None,
    fmax=None,
):
    """Compute the CPSD matrix of every analysis window of every event and store it as features.

    The recording of each event is cut into overlapping analysis windows. For
    each window, the cross-power spectral density (CPSD) matrix between all the
    receivers is estimated with the Welch method (no noise model, the noise is
    assumed negligible). The result is stored in the group ``/feature/<event>``
    of the tree, as the variable ``Sxy`` of dimensions ``(event_datetime,
    rcv_i, rcv_j, freq)``, together with the position of the source (AIS) at the
    center of each window (``event_lon`` / ``event_lat``).

    Only the pressure windows are read from disk, but the resulting ``Sxy``
    arrays are held in memory until the tree is saved.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Event tree with the groups ``position`` (variables ``lon`` / ``lat`` of
        dimension ``datetime``) and ``pressure``; modified in place: the group
        ``feature`` is added.
    analysis_segment_duration_s : float, optional
        Duration of an analysis window, in seconds.
    analysis_segment_alpha_overlap : float, optional
        Overlap between two successive analysis windows, between 0 and 1.
    nfft : int, optional
        Number of points of the FFT of the Welch estimator.
    nperseg : int, optional
        Length of the Welch sub-segments, in samples.
    noverlap : int, optional
        Overlap of the Welch sub-segments, in samples.
    obs_ids : array_like of str or None, optional
        OBS used. If None, all OBS of the first pressure section are used.
    fmin, fmax : float or None, optional
        Frequency band kept, in Hz. If None, the lowest / highest frequency is used.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If no frequency bin is in ``[fmin, fmax]``, or if no analysis window of
        an event is complete on all the OBS.
    """
    # Imported here: this external tool is only needed to compute the CPSD
    from source.cov_manager import CovManager

    analysis_segment_duration_ms = int(round(analysis_segment_duration_s * 1000))
    covariance_manager = CovManager(nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # Receivers and sampling frequency, taken from the first event
    first_section_name = list(portion_tree.pressure.children)[0]
    if obs_ids is None:
        obs_ids = (
            portion_tree.pressure[first_section_name].to_dataset().obs_id.values
        )
    obs_ids = np.asarray(obs_ids)

    fs = portion_tree.pressure[first_section_name].to_dataset().attrs["fs"]
    n_samples_expected = int(analysis_segment_duration_s * fs)

    if nfft is None:
        nfft = n_samples_expected

    # Frequency axis of the FFT, restricted to the requested band
    freq = np.fft.rfftfreq(nfft, d=1 / fs)
    if fmin is None:
        fmin = np.min(freq)
    if fmax is None:
        fmax = np.max(freq)

    freq_band_mask = np.logical_and(freq >= fmin, freq <= fmax)
    if not freq_band_mask.any():
        raise ValueError(
            f"No frequency bin in [{fmin}, {fmax}] Hz (frequency range: "
            f"{freq.min()} - {freq.max()} Hz, resolution {freq[1] - freq[0]:.4g} Hz)."
        )
    freq = freq[freq_band_mask]

    # Shift between the start of two successive analysis windows
    window_shift_ms = analysis_segment_duration_ms * (
        1 - analysis_segment_alpha_overlap
    )

    # 1) Iterate over the events of the tree
    for event_name, node in portion_tree.pressure.children.items():
        ds_pressure_section = node.to_dataset()
        ds_position_section = portion_tree.position[event_name].to_dataset()

        # NOTE : all the receivers are assumed to have the same time axis
        signal_datetime = ds_pressure_section[f"time_{obs_ids[0]}"].values
        elapsed_time_ms = (signal_datetime - signal_datetime[0]) / np.timedelta64(
            1, "ms"
        )

        # Number of analysis windows that fit in the recording
        n_window = int(
            np.floor(
                (
                    elapsed_time_ms.max()
                    - analysis_segment_duration_ms * analysis_segment_alpha_overlap
                )
                / window_shift_ms
            )
        )

        # First window: [window_start, window_stop[ (one sample is removed to
        # get exactly the expected number of samples)
        window_start = signal_datetime[0]
        window_stop = (
            window_start
            + np.timedelta64(analysis_segment_duration_ms, "ms")
            - np.timedelta64(int(1 / fs * 1e9), "ns")
        )

        cpsd_per_window = []
        valid_window_datetimes = []

        # 2) Iterate over the analysis windows
        for _ in range(n_window):

            # Datetime of the center of the window
            window_center = window_start + np.timedelta64(
                int(analysis_segment_duration_ms / 2), "ms"
            )

            # 3) Extract the signal of every receiver on the window
            signals = []
            cropped_signal = False

            for rcv_id in obs_ids:
                signal = ds_pressure_section[f"pressure_{rcv_id}"].sel(
                    {f"time_{rcv_id}": slice(window_start, window_stop)}
                )

                # Incomplete window (end of the recording)
                if signal.size != n_samples_expected:
                    cropped_signal = True
                    break

                # Remove the mean of the signal
                signal = signal - np.nanmean(signal)
                signals.append(signal.values)

            # Move to the next window
            window_start += np.timedelta64(int(window_shift_ms * 1e6), "ns")
            window_stop += np.timedelta64(int(window_shift_ms * 1e6), "ns")

            if cropped_signal:
                continue

            valid_window_datetimes.append(window_center)

            # 4) CPSD matrix of the window, restricted to the requested band
            signals = np.asarray(signals)  # (n_rcv, n_samples)
            _, cpsd = covariance_manager.get_signal_csdm(
                y=signals.T,
                fs=fs,
                add_identity=False,
            )
            cpsd = cpsd[freq_band_mask, ...].T  # (n_rcv, n_rcv, n_freq)

            cpsd_per_window.append(cpsd)

        if len(cpsd_per_window) == 0:
            raise ValueError(
                f"No signal window of event '{event_name}' has a complete "
                f"{analysis_segment_duration_s} s window on all of the OBS {list(obs_ids)}."
            )
        cpsd_per_window = np.array(cpsd_per_window)  # (n_window, n_rcv, n_rcv, n_freq)

        # Position of the source (AIS) at the center of each window
        ds_position_windows = ds_position_section.sel(
            datetime=valid_window_datetimes, method="nearest"
        )

        # Store in the feature dataset of the event
        ds_feature = xr.Dataset(
            data_vars=dict(
                Sxy=(["event_datetime", "rcv_i", "rcv_j", "freq"], cpsd_per_window),
                event_lon=(["event_datetime"], ds_position_windows.lon.data),
                event_lat=(["event_datetime"], ds_position_windows.lat.data),
            ),
            coords=dict(
                event_datetime=valid_window_datetimes,
                rcv_i=obs_ids,
                rcv_j=obs_ids,
                freq=freq,
            ),
            attrs=dict(
                description="Event localisation features",
                section_label=event_name,
                acquisition_mode="passive",
            ),
        )

        # Update tree
        portion_tree[f"/feature/{event_name}"] = ds_feature


# ======================================================================================================================
# Features (RTF, DCF) of the library sections and of the events: common processing
# ======================================================================================================================
def derive_features_dataset(ds_feature_section, rcv_ref="obs1", fmin=5, fmax=50):
    """Derive the RTF and DCF features of a feature dataset (section of the library, or event).

    The processing does not depend on the acquisition mode (active emissions or
    passive windows): it only needs the CPSD matrix ``Sxy``. The sample
    dimension (``src_datetime``, ``event_datetime`` or ``library_datetime``) is
    detected with ``get_sample_axis_names``.

    Parameters
    ----------
    ds_feature_section : xarray.Dataset
        Feature dataset with the variable ``Sxy`` of dimensions ``(<sample
        dimension>, rcv_i, rcv_j, freq)``.
    rcv_ref : str, optional
        Reference receiver of the receiver pairs.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.

    Returns
    -------
    xarray.Dataset
        The dataset restricted to ``[fmin, fmax]``, with the additional
        variables ``pi_ij``, ``gamma_ij`` and ``sij_scot`` of dimensions
        ``(rcv_id_feature, <sample dimension>, freq)``, and the attribute
        ``reference_receiver``.

    Raises
    ------
    ValueError
        If the dataset has no ``Sxy`` variable, or if ``rcv_ref`` is not one of
        its receivers.
    """
    if "Sxy" not in ds_feature_section:
        raise ValueError(
            "The feature dataset has no CPSD matrix 'Sxy': the RTF and DCF features "
            "cannot be derived (compute the CPSD first)."
        )
    datetime_dim = get_sample_axis_names(ds_feature_section)["datetime_dim"]

    # Slice frequency band
    ds_feature_section = ds_feature_section.sel(freq=slice(fmin, fmax))

    # Remove features stored by a previous run (other receivers / band): they are recomputed
    ds_feature_section = ds_feature_section.drop_vars(
        [*LIBRARY_FEATURE_VARS, "rcv_id_feature"], errors="ignore"
    )

    # Derive features for all the receivers but the reference
    features_section = all_features_single_section(
        ds_feature_section=ds_feature_section,
        rcv_ref=rcv_ref,
        fmin=fmin,
        fmax=fmax,
    )  # (n_rcv, n_samples, n_freq)

    # Add coords
    ds_feature_section["rcv_id_feature"] = features_section["rcv_id"]

    # Store in dataset
    for var in LIBRARY_FEATURE_VARS:
        ds_feature_section[var] = (
            ["rcv_id_feature", datetime_dim, "freq"],
            features_section[var],
        )

    # Add attributes
    ds_feature_section.attrs["reference_receiver"] = rcv_ref

    return ds_feature_section


def derive_portion_features(portion_tree, rcv_ref="obs1", fmin=5, fmax=50, verbose=False):
    """Derive the RTF and DCF features of every section of the ``feature`` group of a tree.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Tree with a ``feature`` group whose datasets have the variable ``Sxy``
        (library or event); modified in place.
    rcv_ref : str, optional
        Reference receiver of the receiver pairs. It must be the same for the
        library and the events that are compared.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.
    verbose : bool, optional
        If True, print the progress.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the tree has no ``feature`` group, see also ``derive_features_dataset``.
    """
    if "feature" not in portion_tree.children:
        raise ValueError(
            "No 'feature' group in the tree: the CPSD has not been computed yet."
        )

    for section_name, feature_node in portion_tree.feature.children.items():
        if verbose:
            print(f"    Features of '{section_name}' (reference receiver '{rcv_ref}')")

        feature_node.ds = derive_features_dataset(
            feature_node.to_dataset(), rcv_ref=rcv_ref, fmin=fmin, fmax=fmax
        )


# ======================================================================================================================
# Library of reference features
# ======================================================================================================================
def _to_library_section(ds_feature_section, rcv_ref, fmin, fmax, acquisition_mode):
    """Convert a feature dataset (active section or passive event) into a library section.

    Only the information needed to compare the section to an event is kept: the
    features, the position of the source at each sample and the position of the
    receivers. The samples are renamed with the library names
    (``library_datetime``, ``library_lon``, ``library_lat``), which makes the
    library independent of the acquisition mode.

    Parameters
    ----------
    ds_feature_section : xarray.Dataset
        Feature dataset with the CPSD matrix ``Sxy``.
    rcv_ref : str
        Reference receiver of the receiver pairs.
    fmin, fmax : float
        Frequency band kept, in Hz.
    acquisition_mode : {"active", "passive"}
        Acquisition mode of the section, stored in the attributes.

    Returns
    -------
    xarray.Dataset
        Library section with the variables ``pi_ij``, ``gamma_ij``,
        ``sij_scot``, ``library_lon``, ``library_lat`` (and the receiver
        positions if available).
    """
    # Names of the samples in the original dataset
    axis_names = get_sample_axis_names(ds_feature_section)

    # Features (RTF, DCF) of the section, with the common processing
    ds_features = derive_features_dataset(
        ds_feature_section, rcv_ref=rcv_ref, fmin=fmin, fmax=fmax
    )

    # Keep the useful variables only (the CPSD matrix is not needed anymore)
    kept_vars = list(LIBRARY_FEATURE_VARS)
    kept_vars += [axis_names["lon_var"], axis_names["lat_var"]]
    kept_vars += [var for var in LIBRARY_RECEIVER_VARS if var in ds_features]
    ds_library = ds_features[kept_vars]

    # Rename the samples with the library names
    ds_library = ds_library.rename(
        {
            axis_names["datetime_dim"]: LIBRARY_DATETIME_DIM,
            axis_names["lon_var"]: LIBRARY_LON_VAR,
            axis_names["lat_var"]: LIBRARY_LAT_VAR,
        }
    )

    # Attributes: those of the section + acquisition mode and reference receiver
    ds_library.attrs = dict(ds_features.attrs)
    ds_library.attrs["acquisition_mode"] = acquisition_mode
    ds_library.attrs["reference_receiver"] = rcv_ref

    return ds_library


def library_section_from_active_section(
    ds_feature_active_section, rcv_ref="obs1", fmin=5, fmax=50
):
    """Build a library section from the active emissions of a section.

    Parameters
    ----------
    ds_feature_active_section : xarray.Dataset
        Feature dataset of an active section (dimension ``src_datetime``,
        variables ``tir_src_lon`` / ``tir_src_lat`` and ``Sxy``).
    rcv_ref : str, optional
        Reference receiver of the receiver pairs.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.

    Returns
    -------
    xarray.Dataset
        Library section, see ``_to_library_section``.
    """
    return _to_library_section(
        ds_feature_active_section, rcv_ref, fmin, fmax, acquisition_mode="active"
    )


def library_section_from_passive_event(
    ds_feature_event, rcv_ref="obs1", fmin=5, fmax=50
):
    """Build a library section from the windows of a passive event.

    This is the entry point for a passive / passive comparison: an event whose
    trajectory is known (AIS) becomes a section of the library.

    Parameters
    ----------
    ds_feature_event : xarray.Dataset
        Feature dataset of an event (dimension ``event_datetime``, variables
        ``event_lon`` / ``event_lat`` and ``Sxy``).
    rcv_ref : str, optional
        Reference receiver of the receiver pairs.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.

    Returns
    -------
    xarray.Dataset
        Library section, see ``_to_library_section``.
    """
    return _to_library_section(
        ds_feature_event, rcv_ref, fmin, fmax, acquisition_mode="passive"
    )


def build_portion_library(
    portion,
    sections=None,
    rcv_ref="obs1",
    fmin=5,
    fmax=50,
    acquisition_mode="active",
    verbose=False,
):
    """Build the library ``portion_library`` from the sections of a portion.

    The library has the same structure as an event tree (``portion_event``):

    - ``/position/<section>``: dimension ``datetime``, variables ``lon`` and
      ``lat`` (position of the source at each sample);
    - ``/feature/<section>``: dimension ``library_datetime``, variables
      ``library_lon``, ``library_lat``, ``pi_ij``, ``gamma_ij`` and
      ``sij_scot`` (dimensions ``(rcv_id_feature, library_datetime, freq)``),
      attribute ``reference_receiver``.

    The other groups of the portion (pressure, CPSD, MFP results...) are not
    copied. The portion is read lazily: only the CPSD of the selected sections
    are loaded.

    Parameters
    ----------
    portion : xarray.DataTree
        Portion tree with the groups ``position`` and ``feature`` (CPSD).
        Not modified.
    sections : list of str or None, optional
        Labels of the sections kept in the library. If None, all the sections
        of the ``feature`` group are used.
    rcv_ref : str, optional
        Reference receiver of the receiver pairs. It must be the one used for
        the events compared to this library.
    fmin, fmax : float, optional
        Frequency band kept, in Hz.
    acquisition_mode : {"active", "passive"}, optional
        Acquisition mode of the sections of ``portion``: ``"active"`` for the
        shots of a portion, ``"passive"`` for the windows of events.
    verbose : bool, optional
        If True, print the progress.

    Returns
    -------
    xarray.DataTree
        The library.

    Raises
    ------
    ValueError
        If the portion has no ``feature`` group, if a requested section is
        unknown, or if ``acquisition_mode`` is unknown.
    """
    if acquisition_mode not in ACQUISITION_MODES:
        raise ValueError(
            f"Unknown acquisition_mode '{acquisition_mode}'. Valid modes: {list(ACQUISITION_MODES)}."
        )
    if "feature" not in portion.children:
        raise ValueError(
            "No 'feature' group in the portion: the CPSD has never been computed. "
            "Compute it first (see acouplane_analyse_portion_generate_diag_all_intersections.py)."
        )

    # Sections kept in the library
    available_sections = list(portion.feature.children)
    if sections is None:
        sections = available_sections
    unknown_sections = [name for name in sections if name not in available_sections]
    if unknown_sections:
        raise ValueError(
            f"Unknown section(s) {unknown_sections} for the library "
            f"(available sections: {available_sections})."
        )

    # Converter of the section, depending on the acquisition mode
    to_library_section = {
        "active": library_section_from_active_section,
        "passive": library_section_from_passive_event,
    }[acquisition_mode]

    portion_library = xr.DataTree()
    portion_library.attrs["description"] = "Library of reference features"
    portion_library.attrs["acquisition_mode"] = acquisition_mode
    portion_library.attrs["reference_receiver"] = rcv_ref

    for section_name in sections:
        if verbose:
            print(f"    Library section '{section_name}' ({acquisition_mode})")

        # Features of the section (only this section is read)
        ds_library = to_library_section(
            portion.feature[section_name].to_dataset(),
            rcv_ref=rcv_ref,
            fmin=fmin,
            fmax=fmax,
        )
        ds_library.attrs["section_label"] = section_name

        # Position group, with the same layout as the events
        ds_position = xr.Dataset(
            data_vars=dict(
                lon=(["datetime"], ds_library[LIBRARY_LON_VAR].values),
                lat=(["datetime"], ds_library[LIBRARY_LAT_VAR].values),
            ),
            coords=dict(datetime=ds_library[LIBRARY_DATETIME_DIM].values),
            attrs=dict(ds_library.attrs),
        )

        portion_library[f"/position/{section_name}"] = ds_position
        portion_library[f"/feature/{section_name}"] = ds_library

    return portion_library


# ======================================================================================================================
# Consistency between the library and an event
# ======================================================================================================================
def check_library_and_event_compatibility(
    ds_feature_library, ds_feature_event, frequency_tolerance_hz=1e-6
):
    """Check that a library section and an event can be compared, and list their common receivers.

    The features can only be compared if they were built with the same
    reference receiver, on the same frequencies, for the same receivers.

    Parameters
    ----------
    ds_feature_library : xarray.Dataset
        Feature dataset of a library section.
    ds_feature_event : xarray.Dataset
        Feature dataset of an event.
    frequency_tolerance_hz : float, optional
        Maximum difference between two frequencies considered as equal.

    Returns
    -------
    list of str
        Receivers (``rcv_id_feature``) present in both datasets, in the order
        of the library.

    Raises
    ------
    ValueError
        If the reference receivers differ, if the frequency axes differ, or if
        the datasets have no receiver in common.
    """
    # Same reference receiver: otherwise the features describe different receiver pairs
    reference_library = ds_feature_library.attrs.get("reference_receiver")
    reference_event = ds_feature_event.attrs.get("reference_receiver")
    if reference_library != reference_event:
        raise ValueError(
            f"The reference receiver of the library ('{reference_library}') differs from "
            f"the one of the event ('{reference_event}'): rebuild both with the same "
            "reference receiver."
        )

    # Same frequencies: the features are compared bin by bin
    freq_library = np.asarray(ds_feature_library["freq"].values, dtype=float)
    freq_event = np.asarray(ds_feature_event["freq"].values, dtype=float)
    if freq_library.size != freq_event.size or not np.allclose(
        freq_library, freq_event, rtol=0, atol=frequency_tolerance_hz
    ):
        raise ValueError(
            "The frequency axes of the library and of the event differ "
            f"({freq_library.size} bins in [{freq_library.min():g}, {freq_library.max():g}] Hz "
            f"for the library, {freq_event.size} bins in [{freq_event.min():g}, "
            f"{freq_event.max():g}] Hz for the event): use the same sampling frequency, "
            "nfft and frequency band to compute the CPSD of both."
        )

    # Common receivers
    receivers_library = [str(rcv) for rcv in ds_feature_library["rcv_id_feature"].values]
    receivers_event = [str(rcv) for rcv in ds_feature_event["rcv_id_feature"].values]
    common_receivers = [rcv for rcv in receivers_library if rcv in receivers_event]
    if not common_receivers:
        raise ValueError(
            f"The library (receivers {receivers_library}) and the event (receivers "
            f"{receivers_event}) have no receiver in common."
        )

    return common_receivers


# ======================================================================================================================
# MFP: ambiguity surfaces between the library and an event
# ======================================================================================================================
def build_mfp_library_event_dataset(
    distance_library_event,
    ds_feature_library,
    ds_feature_event,
):
    """Build the dataset holding the MFP ambiguity surfaces of a library section and an event.

    Parameters
    ----------
    distance_library_event : dict
        Distances as returned by ``distances_library_and_event``.
    ds_feature_library : xarray.Dataset
        Feature dataset of the library section (side 1 of the comparison).
    ds_feature_event : xarray.Dataset
        Feature dataset of the event (side 2 of the comparison).

    Returns
    -------
    xarray.Dataset
        Dataset with the receiver information, the source positions of the
        library samples (``library_lon`` / ``library_lat``) and of the event
        samples (``event_lon`` / ``event_lat``), the real distance ``d_real``
        (dimensions ``(library_datetime, event_datetime)``) and the feature
        distances ``d_theta``, ``d_theta_gamma``, ``d_L2_gamma`` and ``d_dcf``
        (dimensions ``(rcv_id_feature, library_datetime, event_datetime)``),
        plus the centered ones if they were computed. The attribute values
        that are missing from the feature datasets are set to ``"unknown"``.
    """
    attrs_library = ds_feature_library.attrs
    attrs_event = ds_feature_event.attrs
    dims_2d = ["library_datetime", "event_datetime"]
    dims_3d = ["rcv_id_feature"] + dims_2d

    # Receiver information (taken from the library)
    data_vars = {
        var: ds_feature_library[var]
        for var in LIBRARY_RECEIVER_VARS
        if var in ds_feature_library
    }

    # Positions of the samples of both sides
    data_vars.update(
        library_lon=(["library_datetime"], ds_feature_library.library_lon.values),
        library_lat=(["library_datetime"], ds_feature_library.library_lat.values),
        event_lon=(["event_datetime"], ds_feature_event.event_lon.values),
        event_lat=(["event_datetime"], ds_feature_event.event_lat.values),
    )

    # Ambiguity surfaces
    data_vars.update(
        d_real=(dims_2d, distance_library_event["dist_in_real_space"]),
        d_theta=(dims_3d, distance_library_event["dist_theta"]),
        d_theta_gamma=(dims_3d, distance_library_event["dist_theta_gamma"]),
        d_L2_gamma=(dims_3d, distance_library_event["dist_L2_gamma"]),
        d_dcf=(dims_3d, distance_library_event["dist_dcf"]),
    )

    ds_mfp = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            rcv_id_feature=ds_feature_library.rcv_id_feature.values,
            library_datetime=ds_feature_library.library_datetime.values,
            event_datetime=ds_feature_event.event_datetime.values,
        ),
        attrs=dict(
            description="MFP result for a library section and an event",
            campagne=attrs_library.get("campagne", "unknown"),
            section1=attrs_library.get("section_label", "unknown"),
            section2=attrs_event.get("section_label", "unknown"),
            library_acquisition_mode=attrs_library.get("acquisition_mode", "unknown"),
            event_acquisition_mode=attrs_event.get("acquisition_mode", "unknown"),
            reference_receiver=attrs_library.get("reference_receiver", "unknown"),
            obs_position=attrs_library.get("obs_position", "unknown"),
            obs_position_filepath=attrs_library.get("obs_position_filepath", "unknown"),
        ),
    )

    # Optional comparison with gamma centered along frequency
    add_centered_gamma_distances(ds_mfp, distance_library_event, dims=dims_3d)

    # Add attrs to variables
    add_distance_attrs(ds_mfp)

    return ds_mfp


def apply_mfp_strategy_to_portions(
    portion_library,
    portion_event,
    rcv_ref="obs1",
    fmin=5,
    fmax=50,
    verbose=False,
    compare_centered_gamma=False,
    library_sections=None,
    event_sections=None,
    obs_ids=None,
):
    """Compute the MFP ambiguity surfaces between the sections of a library and the events.

    Every selected library section is compared to every selected event. The
    features (``pi_ij``, ``gamma_ij``, ``sij_scot``) must have been derived
    before for both trees (see ``build_portion_library`` and
    ``derive_portion_features``) with the same reference receiver.

    Parameters
    ----------
    portion_library : xarray.DataTree
        Library, see ``build_portion_library``. Not modified.
    portion_event : xarray.DataTree
        Event tree with a ``feature`` group whose datasets have the features.
        Not modified.
    rcv_ref : str, optional
        Reference receiver; it is checked against the one stored in both trees.
    fmin, fmax : float, optional
        Frequency band compared, in Hz.
    verbose : bool, optional
        If True, print the progress.
    compare_centered_gamma : bool, optional
        If True, the datasets also contain ``d_theta_gamma_centered`` and
        ``d_L2_gamma_centered`` (distances with ``gamma`` centered along
        frequency).
    library_sections : list of str or None, optional
        Library sections to compare. If None, all of them.
    event_sections : list of str or None, optional
        Events to compare. If None, all of them.
    obs_ids : sequence of str or None, optional
        Receivers kept for the comparison (the reference receiver is ignored).
        If None, all the receivers common to the library and the event are used.

    Returns
    -------
    xarray.DataTree
        Tree with one dataset per comparison in
        ``/mfp_intersection/<library section>/<event>`` (see
        ``build_mfp_library_event_dataset``).

    Raises
    ------
    ValueError
        If the features have not been derived, if a section is unknown, or if
        the library and an event are not compatible (reference receiver,
        frequencies, receivers, see ``check_library_and_event_compatibility``).
    """
    for name, tree in (("library", portion_library), ("event", portion_event)):
        if "feature" not in tree.children:
            raise ValueError(f"No 'feature' group in the {name} tree.")

    # Sections to compare
    library_names = _select_sections(portion_library, library_sections, "library")
    event_names = _select_sections(portion_event, event_sections, "event")

    portion_mfp = xr.DataTree()
    portion_mfp.attrs["description"] = "MFP between a library and events"
    portion_mfp.attrs["reference_receiver"] = rcv_ref

    n_comparisons = len(library_names) * len(event_names)
    i_comparison = 0

    for library_name in library_names:
        ds_feature_library = _features_in_band(portion_library, library_name, fmin, fmax)

        for event_name in event_names:
            i_comparison += 1
            if verbose:
                print(
                    f"    Library '{library_name}' x event '{event_name}' "
                    f"({i_comparison}/{n_comparisons})"
                )
            ds_feature_event = _features_in_band(portion_event, event_name, fmin, fmax)

            # The reference receiver requested must be the one used for the features
            for tree_name, ds_feature in (
                ("library", ds_feature_library),
                ("event", ds_feature_event),
            ):
                stored_reference = ds_feature.attrs.get("reference_receiver")
                if stored_reference != rcv_ref:
                    raise ValueError(
                        f"The features of the {tree_name} were built with the reference receiver "
                        f"'{stored_reference}', not '{rcv_ref}': derive them again with rcv_ref='{rcv_ref}'."
                    )

            # Library and event must be comparable: reference, frequencies, receivers
            common_receivers = check_library_and_event_compatibility(
                ds_feature_library, ds_feature_event
            )

            # Keep the requested receivers, in the same order for both
            if obs_ids is not None:
                wished = [str(rcv) for rcv in obs_ids]
                common_receivers = [rcv for rcv in common_receivers if rcv in wished]
                if not common_receivers:
                    raise ValueError(
                        f"None of the receivers {wished} is available for both the library "
                        f"'{library_name}' and the event '{event_name}'."
                    )
            ds_feature_library, _ = restrict_dataset_to_receivers(
                ds_feature_library, common_receivers, reference=rcv_ref
            )
            ds_feature_event, _ = restrict_dataset_to_receivers(
                ds_feature_event, common_receivers, reference=rcv_ref
            )

            # Distances in real space and in feature space
            distance_library_event = distances_library_and_event(
                ds_feature_library=ds_feature_library,
                ds_feature_event=ds_feature_event,
                compare_centered_gamma=compare_centered_gamma,
            )

            # Build dataset to store
            ds_mfp = build_mfp_library_event_dataset(
                distance_library_event=distance_library_event,
                ds_feature_library=ds_feature_library,
                ds_feature_event=ds_feature_event,
            )

            portion_mfp[f"/mfp_intersection/{library_name}/{event_name}"] = ds_mfp

    return portion_mfp


def _select_sections(tree, requested_sections, tree_name):
    """List the sections of the ``feature`` group of a tree to process.

    Parameters
    ----------
    tree : xarray.DataTree
        Tree with a ``feature`` group.
    requested_sections : list of str or None
        Requested sections; None for all of them.
    tree_name : str
        ``"library"`` or ``"event"``, used in the error message.

    Returns
    -------
    list of str
        The sections to process.

    Raises
    ------
    ValueError
        If a requested section is unknown or if the selection is empty.
    """
    available_sections = list(tree.feature.children)
    if requested_sections is None:
        requested_sections = available_sections

    unknown_sections = [name for name in requested_sections if name not in available_sections]
    if unknown_sections:
        raise ValueError(
            f"Unknown {tree_name} section(s) {unknown_sections} "
            f"(available: {available_sections})."
        )
    if len(requested_sections) == 0:
        raise ValueError(f"No {tree_name} section to process.")

    return list(requested_sections)


def _features_in_band(tree, section_name, fmin, fmax):
    """Feature dataset of a section restricted to a frequency band (lazy selection).

    Parameters
    ----------
    tree : xarray.DataTree
        Tree with a ``feature`` group.
    section_name : str
        Section (or event) label.
    fmin, fmax : float
        Frequency band, in Hz.

    Returns
    -------
    xarray.Dataset
        Feature dataset restricted to ``[fmin, fmax]``.

    Raises
    ------
    ValueError
        If the features ``pi_ij``, ``gamma_ij`` and ``sij_scot`` are missing.
    """
    ds_feature = tree.feature[section_name].to_dataset().sel(freq=slice(fmin, fmax))

    missing = [var for var in LIBRARY_FEATURE_VARS if var not in ds_feature]
    if missing:
        raise ValueError(
            f"The features {missing} of '{section_name}' have not been derived: "
            "call derive_portion_features (events) or build_portion_library (library) first."
        )

    return ds_feature


def get_crossing_distance_m(ds_mfp_intersection):
    """Distance, in meters, between the library section and the event trajectory at their closest samples.

    This is the value of the real-space distance at the theoretical
    intersection. It is small when the trajectory of the event really crosses
    the library section, and large when it does not (no intersection to find).

    Parameters
    ----------
    ds_mfp_intersection : xarray.Dataset
        MFP dataset of a library section and an event (see
        ``build_mfp_library_event_dataset``).

    Returns
    -------
    float
        Minimum of ``d_real`` in meters (NaN if the dataset has no valid value).
    """
    d_real = np.asarray(ds_mfp_intersection["d_real"].values, dtype=float)
    if np.isnan(d_real).all():
        return float("nan")

    return float(np.nanmin(d_real))
