#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_frequency_bands.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Frequency bands and ranges: definition, slicing and position errors recomputed on each range.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np

from real_data_analysis.acouplane.src.acouplane_features import (
    distances_single_intersection,
)
from real_data_analysis.acouplane.src.acouplane_localization import (
    errors_to_records,
    localization_errors_from_distances,
)


def build_frequency_bands(fmin, fmax, bandwidth):
    """Split ``[fmin, fmax]`` in frequency bands aligned on multiples of ``bandwidth``.

    The band limits are ``fmin``, the multiples of ``bandwidth`` strictly
    between ``fmin`` and ``fmax``, and ``fmax``. The first and last bands can
    therefore be narrower than ``bandwidth`` (e.g. ``[10, 400]`` and
    ``[400, 800]`` for ``fmin=10``, ``fmax=800`` and ``bandwidth=400``). Bands
    are contiguous and cover the whole interval.

    Parameters
    ----------
    fmin, fmax : float
        Limits of the interval, in Hz.
    bandwidth : float
        Nominal width of the bands, in Hz.

    Returns
    -------
    list of tuple of float
        ``(f_low, f_high)`` of each band, in increasing order.

    Raises
    ------
    ValueError
        If ``bandwidth`` is not positive or if ``fmax <= fmin``.
    """
    if bandwidth <= 0:
        raise ValueError(f"The frequency bandwidth must be positive (got {bandwidth}).")
    if fmax <= fmin:
        raise ValueError(f"fmax ({fmax}) must be greater than fmin ({fmin}).")

    # Inner limits: multiples of the bandwidth strictly between fmin and fmax
    k_first = int(np.floor(fmin / bandwidth)) + 1
    k_last = int(np.ceil(fmax / bandwidth))
    inner = [
        k * bandwidth for k in range(k_first, k_last) if fmin < k * bandwidth < fmax
    ]

    # Contiguous bands between consecutive limits
    edges = [fmin, *inner, fmax]

    return [(float(lo), float(hi)) for lo, hi in zip(edges[:-1], edges[1:])]


def frequency_band_label(f_low, f_high):
    """Label of a frequency band, e.g. ``"10-400"`` (Hz)."""
    return f"{f_low:g}-{f_high:g}"


def slice_frequency_band(ds_feature_section, f_low, f_high, include_upper=False):
    """Select the frequency bins of a band, without counting a bin in two adjacent bands.

    ``xarray`` label slicing (``sel(freq=slice(a, b))``) includes both limits,
    so a bin lying exactly on a limit would belong to two adjacent bands. Here
    the band is ``[f_low, f_high[`` unless ``include_upper`` is True.

    Parameters
    ----------
    ds_feature_section : xarray.Dataset
        Dataset with a ``freq`` dimension.
    f_low, f_high : float
        Limits of the band, in Hz.
    include_upper : bool, optional
        If True, the bins equal to ``f_high`` are kept (use it for the last band
        of a series).

    Returns
    -------
    xarray.Dataset
        Dataset restricted to the band; its ``freq`` dimension can be empty.
    """
    freq = np.asarray(ds_feature_section["freq"].values)

    # Half-open band [f_low, f_high[ unless the upper limit is requested
    mask = freq >= f_low
    mask &= (freq <= f_high) if include_upper else (freq < f_high)

    return ds_feature_section.isel(freq=np.flatnonzero(mask))


def build_cumulative_frequency_bands(fmin, fmax, step):
    """Frequency ranges of increasing width, all starting at ``fmin``.

    The ranges are ``[fmin, f]`` for ``f`` taking the multiples of ``step``
    greater than ``fmin`` and not greater than ``fmax``, and ``fmax`` itself if
    it is not a multiple of ``step`` (e.g. ``[10, 50]``, ``[10, 100]``,
    ``[10, 150]``... for ``fmin=10`` and ``step=50``).

    Parameters
    ----------
    fmin, fmax : float
        Limits of the interval, in Hz.
    step : float
        Increment of the upper limit, in Hz.

    Returns
    -------
    list of tuple of float
        ``(fmin, f_high)`` of each range, by increasing ``f_high``.

    Raises
    ------
    ValueError
        If ``step`` is not positive or if ``fmax <= fmin``.
    """
    if step <= 0:
        raise ValueError(f"The frequency step must be positive (got {step}).")
    if fmax <= fmin:
        raise ValueError(f"fmax ({fmax}) must be greater than fmin ({fmin}).")

    # Upper limits: multiples of the step in ]fmin, fmax]
    k_first = int(np.floor(fmin / step)) + 1
    k_last = int(np.floor(fmax / step))
    highs = [k * step for k in range(k_first, k_last + 1) if fmin < k * step <= fmax]

    # The whole interval is always covered by the last range
    if not highs or highs[-1] < fmax:
        highs.append(fmax)

    return [(float(fmin), float(high)) for high in highs]


def collect_errors_over_frequency_ranges(
    portion,
    freq_ranges,
    intersections,
    close_all_ranges,
    reference_rcv,
    error_mode="section1",
    compare_centered_gamma=False,
    rcv_filter=None,
    log=None,
):
    """Position errors of the intersections, with the distances recomputed on each frequency range.

    For each intersection, the stored features (``pi_ij``, ``gamma_ij``, ``sij_scot``) of the two
    sections are restricted to each range, the distances are recomputed and the position errors
    are derived (see ``localization_errors_from_distances``).

    Parameters
    ----------
    portion : xarray.DataTree
        Portion tree (lazy), with the ``feature`` group.
    freq_ranges : list of tuple of float
        ``(f_low, f_high)`` of each range, in Hz.
    intersections : list of tuple of str
        ``(section1_label, section2_label)`` of the intersections to analyse.
    close_all_ranges : bool
        If True, the bins equal to ``f_high`` are kept in every range (nested ranges); otherwise
        only in the last one (contiguous bands, no bin counted twice).
    reference_rcv : str
        Reference receiver, used if the features do not store it.
    error_mode : {"section1", "section2", "both"}, optional
        How the position error is computed, see ``localization_errors_from_distances``.
    compare_centered_gamma : bool, optional
        Also compute the distances with gamma centered along frequency.
    rcv_filter : callable or None, optional
        ``rcv_filter(ds, reference)`` returning the dataset restricted to the receivers to analyse
        (None: all the stored receivers).
    log : callable or None, optional
        Progress function ``log(message, level=0)`` (None: silent).

    Returns
    -------
    list of dict
        Records (see ``errors_to_records``) with the extra fields ``intersection``, ``band_index``,
        ``band_label``, ``band_center``, ``f_low`` and ``f_high``.

    Raises
    ------
    RuntimeError
        If the features of a section are missing from the portion.
    """
    if log is None:
        log = lambda message, level=0: None

    records = []
    for section1_label, section2_label in intersections:
        log(f"Intersection '{section1_label}' x '{section2_label}'", level=2)

        # Intersections are between a meridional and a zonal section
        ds_feature_section1 = portion.feature[section1_label].to_dataset()
        ds_feature_section2 = portion.feature[section2_label].to_dataset()
        for label, ds_feature in (
            (section1_label, ds_feature_section1),
            (section2_label, ds_feature_section2),
        ):
            if "pi_ij" not in ds_feature:
                raise RuntimeError(
                    f"The features (RTF, DCF) of section '{label}' are not available in the "
                    "portion: derive them (apply_mfp_strategy_intersections = True, and "
                    "save_modification_to_portion = True to keep them) before analysing "
                    "the frequency ranges."
                )

        # Keep the receivers selected by obs_ids
        reference_stored = ds_feature_section1.attrs.get("reference_receiver", reference_rcv)
        if rcv_filter is not None:
            ds_feature_section1 = rcv_filter(ds_feature_section1, reference_stored)
            ds_feature_section2 = rcv_filter(ds_feature_section2, reference_stored)

        # Shot positions
        lon1, lat1 = (
            ds_feature_section1.tir_src_lon.values,
            ds_feature_section1.tir_src_lat.values,
        )
        lon2, lat2 = (
            ds_feature_section2.tir_src_lon.values,
            ds_feature_section2.tir_src_lat.values,
        )

        for i_band, (f_low, f_high) in enumerate(freq_ranges):
            band_label = frequency_band_label(f_low, f_high)
            include_upper = close_all_ranges or i_band == len(freq_ranges) - 1

            # Slice the frequency range of interest and recompute the distances
            ds_feature_band1 = slice_frequency_band(
                ds_feature_section1, f_low, f_high, include_upper=include_upper
            )
            ds_feature_band2 = slice_frequency_band(
                ds_feature_section2, f_low, f_high, include_upper=include_upper
            )
            if ds_feature_band1.sizes["freq"] == 0 or ds_feature_band2.sizes["freq"] == 0:
                print(
                    f"    WARNING: no stored frequency bin in the range {band_label} Hz "
                    f"for the intersection '{section1_label}' x '{section2_label}': "
                    "range skipped."
                )
                continue
            log(f"Range {band_label} Hz", level=3)

            distance_section12 = distances_single_intersection(
                ds_feature_section1=ds_feature_band1,
                ds_feature_section2=ds_feature_band2,
                compare_centered_gamma=compare_centered_gamma,
            )
            errors = localization_errors_from_distances(
                distance_section12,
                lon1=lon1,
                lat1=lat1,
                lon2=lon2,
                lat2=lat2,
                error_mode=error_mode,
            )
            records += errors_to_records(
                errors,
                ds_feature_band1.rcv_id_feature.values,
                reference_stored,
                intersection=f"{section1_label} x {section2_label}",
                band_index=i_band,
                band_label=band_label,
                band_center=(f_low + f_high) / 2,
                f_low=f_low,
                f_high=f_high,
            )

    return records
