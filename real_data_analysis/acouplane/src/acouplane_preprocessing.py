#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_preprocessing.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Preprocessing of a portion: receiver position correction, propagation times and CPSD matrices.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import pandas as pd

from real_data_analysis.acouplane.src.acouplane_cst import ROOT_ACOUPLANE_PROCESSED_DATA_DIR


def compute_src_rcv_geodesic_dist(ds_pos_sections):
    """Add the horizontal source-receiver geodesic distance to position datasets.

    Distances are computed on the WGS84 ellipsoid. The datasets are modified in
    place (and also returned for convenience).

    Parameters
    ----------
    ds_pos_sections : list of xarray.Dataset
        Position datasets, each with the variables ``tir_src_lon``,
        ``tir_src_lat`` (dimension ``src_datetime``) and ``rcv_lon``,
        ``rcv_lat`` (dimension ``rcv_id``).

    Returns
    -------
    list of xarray.Dataset
        The same datasets, with the new variable ``src_rcv_geodesic_dist`` in
        meters, of dimensions ``(rcv_id, src_datetime)``.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")

    for ds_pos_section in ds_pos_sections:

        horizontal_distance = []

        slon, slat = (
            ds_pos_section.tir_src_lon.values,
            ds_pos_section.tir_src_lat.values,
        )
        for rcv_id in ds_pos_section.rcv_id.values:
            rlon = float(ds_pos_section.sel(rcv_id=rcv_id).rcv_lon.values)
            rlat = float(ds_pos_section.sel(rcv_id=rcv_id).rcv_lat.values)

            # Cast to src pos shape
            rlon_ = np.full(slon.size, rlon)
            rlat_ = np.full(slon.size, rlat)
            # Compute geodesic distance (only the distance is kept)
            _, _, horiz_d = geod.inv(lons1=slon, lats1=slat, lons2=rlon_, lats2=rlat_)
            horizontal_distance.append(horiz_d)

        horizontal_distance = np.array(horizontal_distance)

        # Add to dataset
        ds_pos_section["src_rcv_geodesic_dist"] = (
            ["rcv_id", "src_datetime"],
            horizontal_distance,
        )

    return ds_pos_sections


def apply_rcv_position_correction_to_portion(
    portion_tree, rcv_pos_corr_fname="rcv_position_corrected.csv"
):
    """Replace the OBS positions of a portion by corrected (LMS inversion) ones.

    The corrected positions are read from a CSV file with the columns ``lon``
    and ``lat``, one row per OBS in the same order as the ``rcv_id`` coordinate
    of the portion. The source-receiver geodesic distance is recomputed and the
    attributes ``obs_position`` and ``obs_position_filepath`` are updated.

    Only the ``position`` group is updated. Features (``feature`` group)
    already computed with the previous positions are NOT updated and must be
    recomputed.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree with a ``position`` group (modified in place).
    rcv_pos_corr_fname : str, optional
        Name of the CSV file, located in ``ROOT_ACOUPLANE_PROCESSED_DATA_DIR``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV file lacks the ``lon``/``lat`` columns or does not contain
        one row per receiver.
    """
    rcv_pos_fpath = os.path.join(ROOT_ACOUPLANE_PROCESSED_DATA_DIR, rcv_pos_corr_fname)
    if not os.path.isfile(rcv_pos_fpath):
        raise FileNotFoundError(
            f"Corrected OBS position file not found: '{rcv_pos_fpath}'."
        )
    df_rcv_pos_corr = pd.read_csv(rcv_pos_fpath)

    missing_cols = {"lon", "lat"} - set(df_rcv_pos_corr.columns)
    if missing_cols:
        raise ValueError(
            f"Columns {sorted(missing_cols)} missing in '{rcv_pos_fpath}' "
            f"(columns found: {list(df_rcv_pos_corr.columns)})."
        )

    # Update tree
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()

        n_rcv = ds_pos_section.sizes["rcv_id"]
        if len(df_rcv_pos_corr) != n_rcv:
            raise ValueError(
                f"'{rcv_pos_fpath}' has {len(df_rcv_pos_corr)} rows but section '{name}' "
                f"has {n_rcv} receivers ({list(ds_pos_section.rcv_id.values)}): "
                "one row per receiver, in the same order, is expected."
            )

        # Update ds_pos_section
        ds_pos_section.update(
            {
                "rcv_lon": ("rcv_id", df_rcv_pos_corr["lon"].to_numpy()),
                "rcv_lat": ("rcv_id", df_rcv_pos_corr["lat"].to_numpy()),
            }
        )
        # No need to update immersion as it is not changed (2D LMS estimation)

        # Update geodetic distance between source and receiver
        ds_pos_section = compute_src_rcv_geodesic_dist(
            ds_pos_sections=[ds_pos_section]
        )[0]

        # Add attributes to track modifications
        ds_pos_section.attrs["obs_position"] = (
            "OBS position corrected using LMS inversion."
        )
        ds_pos_section.attrs["obs_position_filepath"] = rcv_pos_fpath

        # Update tree
        node.ds = ds_pos_section


def derive_portion_propagation_time(portion_tree, c0=1508):
    """Compute the theoretical propagation time and reception datetime of every shot.

    A straight line propagation path at constant celerity ``c0`` is assumed:
    ``t = sqrt(horizontal_distance**2 + rcv_immersion**2) / c0``. The results
    are stored in each ``position`` section as ``tps_propagation_s`` (seconds)
    and ``tir_reception_datetime``, both of dimensions ``(rcv_id, src_datetime)``.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree whose ``position`` sections contain
        ``src_rcv_geodesic_dist`` and ``rcv_immersion`` (modified in place).
    c0 : float, optional
        Equivalent sound celerity in m/s.

    Returns
    -------
    None
    """
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()

        for var in ("src_rcv_geodesic_dist", "rcv_immersion"):
            if var not in ds_pos_section:
                raise KeyError(
                    f"Variable '{var}' missing in position section '{name}' "
                    f"(variables found: {list(ds_pos_section.data_vars)})."
                )

        # Propagation distance assuming strait line propagation path
        dpropa_direct = np.sqrt(
            ds_pos_section.src_rcv_geodesic_dist**2 + ds_pos_section.rcv_immersion**2
        )
        # Compute propagation time
        tps_propagation_s = dpropa_direct / c0
        ds_pos_section["tps_propagation_s"] = tps_propagation_s
        ds_pos_section["tps_propagation_s"].attrs = dict(
            units="s", long_name="Propagation time"
        )

        # Derive associated arrival dt
        tps_propagation_ns = (tps_propagation_s * 1e9).round()
        tps_propagation_td = tps_propagation_ns.astype("timedelta64[ns]")
        ds_pos_section["tir_reception_datetime"] = (
            tps_propagation_td + ds_pos_section.src_datetime
        )
        ds_pos_section["tir_reception_datetime"].attrs = dict(
            long_name="Reception datetime"
        )

        # Update tree
        node.ds = ds_pos_section


def compute_cpsd_matrix(signal_arr, nfft, freq_band_mask):
    """Compute the cross power spectral density matrix of several signals.

    ``Sxy[i, j] = X_i * conj(X_j)`` where ``X`` is the zero-padded FFT of the
    signals, restricted to the frequency band selected by ``freq_band_mask``.
    The diagonal is the PSD of each signal and ``Sxy[j, i] = conj(Sxy[i, j])``.
    No averaging is performed (single window estimate).

    Parameters
    ----------
    signal_arr : numpy.ndarray
        Signals of the receivers, shape ``(n_rcv, n_samples)``.
    nfft : int
        FFT length (must be at least ``n_samples``).
    freq_band_mask : numpy.ndarray of bool
        Mask of the kept FFT bins, shape ``(nfft // 2 + 1,)``.

    Returns
    -------
    numpy.ndarray
        CPSD matrix of dtype complex64, shape ``(n_rcv, n_rcv, n_freq)``, where
        ``n_freq`` is the number of True values in ``freq_band_mask``.

    Raises
    ------
    ValueError
        If ``nfft`` is smaller than the number of samples (the FFT would
        silently truncate the signals).
    """
    n_samples = signal_arr.shape[1]
    if nfft < n_samples:
        raise ValueError(
            f"nfft={nfft} is smaller than the signal length ({n_samples} samples): "
            "the signals would be truncated. Use nfft >= signal length."
        )

    rfft = np.fft.rfft(signal_arr, n=nfft, axis=1)  # (nrcv, nfft // 2 + 1)
    rfft = rfft[:, freq_band_mask]  # (nrcv, nf)

    # Sxy[i, j, f] = X_i(f) * conj(X_j(f))
    sxy = rfft[:, np.newaxis, :] * np.conj(rfft[np.newaxis, :, :])
    return sxy.astype(np.complex64)


def derive_portion_cpsd(
    portion_tree,
    offset_ms=100,
    signal_duration_s=3,
    nfft=None,
    obs_ids=None,
    fmin=None,
    fmax=None,
):
    """Compute the CPSD matrix of every shot of every section and store it as features.

    For each shot, a window of ``signal_duration_s`` seconds starting
    ``offset_ms`` before the theoretical reception time is extracted on each
    OBS, centered, and the CPSD matrix is computed (see ``compute_cpsd_matrix``).
    Shots whose window is incomplete on at least one OBS are discarded.

    The result is stored in the group ``/feature/<section>`` of the tree, as the
    variable ``Sxy`` of dimensions ``(src_datetime, rcv_i, rcv_j, freq)``,
    together with the position information of the section. Requires
    ``derive_portion_propagation_time`` to have been called beforehand.

    Only the pressure windows are read from disk, but the resulting ``Sxy``
    arrays are held in memory until the tree is saved.

    Parameters
    ----------
    portion_tree : xarray.DataTree
        Portion tree with the groups ``position`` and ``pressure`` (modified in
        place: the group ``feature`` is added).
    offset_ms : int, optional
        Time before the theoretical reception included in the window, in ms.
    signal_duration_s : float, optional
        Window duration, in seconds.
    nfft : int or None, optional
        FFT length. If None, the number of samples of the window is used.
    obs_ids : array_like of str or None, optional
        OBS used. If None, all OBS of the first pressure section are used.
    fmin, fmax : float or None, optional
        Frequency band kept, in Hz. If None, the lowest / highest frequency is used.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If ``tir_reception_datetime`` is missing from a position section.
    ValueError
        If ``nfft`` is too small, or if no shot has a complete window in a section.
    """
    signal_duration_ms = int(round(signal_duration_s * 1000))

    first_section_name = list(portion_tree.pressure.children)[0]
    if obs_ids is None:
        obs_ids = portion_tree.pressure[first_section_name].to_dataset().obs_id.values
    obs_ids = np.asarray(obs_ids)

    fs = portion_tree.pressure[first_section_name].to_dataset().attrs["fs"]
    n_samples_expected = int(signal_duration_s * fs)

    if nfft is None:
        nfft = n_samples_expected
    if nfft < n_samples_expected:
        raise ValueError(
            f"nfft={nfft} is smaller than the window length "
            f"({n_samples_expected} samples = {signal_duration_s} s at {fs} Hz)."
        )
    # Init freq vector
    freq = np.fft.rfftfreq(nfft, d=1 / fs)

    if fmin is None:
        fmin = np.min(freq)
    if fmax is None:
        fmax = np.max(freq)

    # Slice required frequency band
    freq_band_mask = np.logical_and(freq >= fmin, freq <= fmax)
    if not freq_band_mask.any():
        raise ValueError(
            f"No frequency bin in [{fmin}, {fmax}] Hz (frequency range: "
            f"{freq.min()} - {freq.max()} Hz, resolution {freq[1] - freq[0]:.4g} Hz)."
        )
    freq = freq[freq_band_mask]

    # 1) Iterate over portion's sections
    for name, node in portion_tree.pressure.children.items():
        ds_pressure_section = node.to_dataset()
        # Associated ds_pos
        ds_pos_section = portion_tree.position[name].to_dataset()

        if "tir_reception_datetime" not in ds_pos_section:
            raise KeyError(
                f"Variable 'tir_reception_datetime' missing in section '{name}': "
                "call derive_portion_propagation_time() first."
            )

        Sxy_data = []
        valid_emission_dt = []
        # 2) Iterate over emissions
        for tir_emission_dt in ds_pos_section.src_datetime.values:
            tir_reception_dt = ds_pos_section["tir_reception_datetime"].sel(
                src_datetime=tir_emission_dt
            )

            # 3) Build list of signal_i
            signal_arr = []
            cropped_signal = False
            for rcv_id_i in obs_ids:
                # Reception datetime for current rcv i
                tir_reception_dt_rcvi = tir_reception_dt.sel(rcv_id=rcv_id_i)

                # Select associated data window
                signal_i = ds_pressure_section[f"pressure_{rcv_id_i}"].sel(
                    {
                        f"time_{rcv_id_i}": slice(
                            tir_reception_dt_rcvi.data - np.timedelta64(offset_ms, "ms"),
                            tir_reception_dt_rcvi.data
                            + np.timedelta64(signal_duration_ms - offset_ms, "ms"),
                        )
                    }
                )

                if signal_i.size != n_samples_expected:
                    cropped_signal = True
                    break

                # Ensure signal is centered
                signal_i = signal_i - np.nanmean(signal_i)

                # Store in array
                signal_arr.append(signal_i.values)

            if cropped_signal:
                continue

            valid_emission_dt.append(tir_emission_dt)
            signal_arr = np.asarray(signal_arr)  # (nrcv, nt)

            # Store for current tir
            Sxy_data.append(compute_cpsd_matrix(signal_arr, nfft, freq_band_mask))

        if len(Sxy_data) == 0:
            raise ValueError(
                f"No shot of section '{name}' has a complete {signal_duration_s} s window "
                f"on all of the OBS {list(obs_ids)}."
            )
        Sxy_data = np.array(Sxy_data)  # (ntir, nr, nr, nf)

        # Store in ds_feature
        ds_feature = ds_pos_section.copy()
        ds_feature = ds_feature.drop_vars(
            [
                "atl_lon",
                "atl_lat",
                "atl_datetime",
                "ais_lon",
                "ais_lat",
                "ais_datetime",
                "tir_SEGD",
                "tir_ECOS",
                "tir_vessel_lon",
                "tir_vessel_lat",
            ],
            errors="ignore",
        )
        attrs = dict(ds_feature.attrs)
        attrs.pop("interpolation_time_step", None)
        attrs["description"] = "Localisation features"
        ds_feature.attrs = attrs

        # Sel only valid tir
        ds_feature = ds_feature.sel(src_datetime=valid_emission_dt)

        # Add coords rcv_i, rcv_j and freq
        ds_feature = ds_feature.assign_coords(
            dict(rcv_i=obs_ids, rcv_j=obs_ids, freq=freq)
        )  # NOTE we need to declare new coords because we cant use duplicated coords

        ds_feature["Sxy"] = (["src_datetime", "rcv_i", "rcv_j", "freq"], Sxy_data)

        # Update tree
        portion_tree[f"/feature/{name}"] = ds_feature
