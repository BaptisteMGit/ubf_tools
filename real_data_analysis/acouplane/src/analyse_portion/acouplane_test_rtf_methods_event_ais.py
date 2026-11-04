#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_test_rtf_methods_event_ais.py
@Time    :   2026/10/30 20:00:00
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Test script: RTF of the passive AIS events of a portion estimated with the two methods
             ("spectral_ratio": S_ij / S_jj, "cs_evd": first eigenvector of the spatial covariance matrix) and
             comparison of the two estimators (amplitude, phase, relative difference), and, for a single window (the
             closest to the library in real space), comparison of the two gamma estimates with the gamma of the
             library at the closest position.
             Nothing is written in the portion nor in the event file.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from real_data_analysis.acouplane.src.acouplane_cst import (
    ACOUPLANE_PORTION_NC_ROOT_FPATH,
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR,
)
from real_data_analysis.acouplane.src.acouplane_console import (
    make_figure_saver,
    make_logger,
)
from real_data_analysis.acouplane.src.acouplane_event import (
    build_portion_library,
    derive_features_dataset,
    derive_portion_cpsd_welch,
)
from real_data_analysis.acouplane.src.acouplane_features import (
    compare_rtf_methods,
    get_distance_between_library_and_event,
)
from real_data_analysis.acouplane.src.acouplane_portion_io import (
    open_portion,
)

# ======================================================================================================================
# Constants
# ======================================================================================================================
portion_id = 5
ROOT_IMG_PORTION = os.path.join(
    ROOT_ACOUPLANE_PROCESSED_IMG_DIR, f"portion{portion_id}"
)
ROOT_IMG_PORTION_RESULTS_RTF = os.path.join(ROOT_IMG_PORTION, "results_event_ais", "rtf_methods")
os.makedirs(ROOT_IMG_PORTION_RESULTS_RTF, exist_ok=True)

portion_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}.nc"
portion_event_fpath = f"{ACOUPLANE_PORTION_NC_ROOT_FPATH}_{portion_id}_event_ais.nc"

# Reference receiver of the pairs (rcv_i, reference_rcv)
reference_rcv = "obs1"

# Frequency band of the features
fmin = 10
fmax = 150

# Events to compare (None -> all the events of the event file)
event_sections = None
# Maximum number of events compared (safety, the eigen decomposition is done for every window and frequency)
max_events = 5

# Library (active emissions of the portion) used for the single-window comparison: the window of each event closest
# in real space to a library sample is compared to the gamma of this library sample
library_sections = ["sn2"]
# RTF estimator of the library (the active case uses the spectral ratio)
library_rtf_method = "spectral_ratio"

# CPSD of the events (Welch method, see derive_portion_cpsd_welch); same settings as acouplane_analyse_portion_event_ais.py
analysis_segment_duration_s = 6
analysis_segment_alpha_overlap = 0.75
nfft = 2**13
nperseg = 2**13
noverlap = 2**12

# Sanity check on an active section of the portion (None -> skipped). With a single dominant source (active
# emissions) both estimators must almost coincide: it validates the conventions of the cs-evd implementation
active_section_check = "sn2"

# Figures: RTF amplitude bounds [dB] of the comparison plot
gamma_ylim_db = (-20, 20)

# ======================================================================================================================
# Booleans to control behavior
# ======================================================================================================================
# Compute the CPSD of the events (set to False if the events already contain it)
compute_event_cpsd = True
plot_comparison = True
plot_single_window = True
verbose = True

log = make_logger(portion_id, verbose)
save_rtf_figure = make_figure_saver(ROOT_IMG_PORTION_RESULTS_RTF, log)


def wrapped_phase_difference(pi_a, pi_b):
    """Phase of ``pi_a`` relative to ``pi_b``, in degrees, wrapped in [-180, 180].

    Parameters
    ----------
    pi_a, pi_b : numpy.ndarray
        Complex RTF of the same shape.

    Returns
    -------
    numpy.ndarray
        ``angle(pi_a * conj(pi_b))`` in degrees (NaN where an RTF is NaN).
    """
    return np.degrees(np.angle(pi_a * np.conj(pi_b)))


# ======================================================================================================================
# Load data (lazy)
# ======================================================================================================================
log(f"Step 1 - Opening the events of the portion {portion_id} (lazy)")
portion_event = open_portion(portion_event_fpath)

if compute_event_cpsd:
    log("CPSD (Welch) of the events", level=1)
    derive_portion_cpsd_welch(
        portion_tree=portion_event,
        analysis_segment_duration_s=analysis_segment_duration_s,
        analysis_segment_alpha_overlap=analysis_segment_alpha_overlap,
        nfft=nfft,
        nperseg=nperseg,
        noverlap=noverlap,
        obs_ids=None,
        fmin=fmin,
        fmax=fmax,
    )
elif "feature" not in portion_event.children:
    raise RuntimeError(
        f"No 'feature' group in '{portion_event_fpath}': set compute_event_cpsd = True."
    )

event_names = list(portion_event.feature.children)
if event_sections is not None:
    unknown = [name for name in event_sections if name not in event_names]
    if unknown:
        raise ValueError(f"Unknown event(s) {unknown} (available events: {event_names}).")
    event_names = list(event_sections)
if len(event_names) > max_events:
    print(f"    WARNING: {len(event_names)} events, only the first {max_events} are compared (max_events).")
    event_names = event_names[:max_events]

# ======================================================================================================================
# Library (gamma of the reference positions)
# ======================================================================================================================
if plot_single_window:
    log(f"Library sections {library_sections} (RTF '{library_rtf_method}')", level=1)
    portion = open_portion(portion_fpath)
    if "feature" not in portion.children:
        raise RuntimeError(f"No 'feature' group in '{portion_fpath}': the CPSD has never been computed.")
    portion_library = build_portion_library(
        portion,
        sections=library_sections,
        rcv_ref=reference_rcv,
        fmin=fmin,
        fmax=fmax,
        acquisition_mode="active",
        verbose=verbose,
        rtf_method=library_rtf_method,
    )
    portion.close()

# ======================================================================================================================
# Comparison of the two estimators on every event
# ======================================================================================================================
log("Step 2 - RTF of the events with both methods")
rows = []
rows_window = []
for event_name in event_names:
    log(f"Event '{event_name}'", level=1)
    ds_cpsd = portion_event.feature[event_name].to_dataset()

    # The features are derived in memory only (nothing is stored in the event file)
    ds_ratio = derive_features_dataset(
        ds_cpsd, rcv_ref=reference_rcv, fmin=fmin, fmax=fmax, rtf_method="spectral_ratio"
    )
    ds_evd = derive_features_dataset(
        ds_cpsd, rcv_ref=reference_rcv, fmin=fmin, fmax=fmax, rtf_method="cs_evd"
    )
    freq = ds_ratio["freq"].values
    rcv_ids = [str(rcv) for rcv in ds_ratio["rcv_id_feature"].values]

    pi_ratio_all = ds_ratio["pi_ij"].values  # (n_rcv, n_windows, n_freq)
    pi_evd_all = ds_evd["pi_ij"].values

    for i_rcv, rcv_id in enumerate(rcv_ids):
        pi_ratio = pi_ratio_all[i_rcv]
        pi_evd = pi_evd_all[i_rcv]

        with np.errstate(divide="ignore", invalid="ignore"):
            gamma_difference_db = 20 * np.log10(np.abs(pi_evd) / np.abs(pi_ratio))
            phase_difference_deg = wrapped_phase_difference(pi_evd, pi_ratio)
            relative_difference = np.abs(pi_evd - pi_ratio) / np.abs(pi_ratio)
            relative_difference_conjugate = np.abs(np.conj(pi_evd) - pi_ratio) / np.abs(pi_ratio)

        rows.append(
            dict(
                event=event_name,
                rcv_id=rcv_id,
                n_windows=pi_ratio.shape[0],
                median_gamma_difference_db=float(np.nanmedian(gamma_difference_db)),
                median_abs_gamma_difference_db=float(np.nanmedian(np.abs(gamma_difference_db))),
                median_phase_difference_deg=float(np.nanmedian(phase_difference_deg)),
                median_abs_phase_difference_deg=float(np.nanmedian(np.abs(phase_difference_deg))),
                median_relative_difference=float(np.nanmedian(relative_difference)),
                median_relative_difference_conjugate=float(np.nanmedian(relative_difference_conjugate)),
                nan_fraction_ratio=float(np.mean(np.isnan(pi_ratio))),
                nan_fraction_evd=float(np.mean(np.isnan(pi_evd))),
            )
        )
        log(
            f"{rcv_id}: |gamma_evd - gamma_ratio| median {rows[-1]['median_abs_gamma_difference_db']:.2f} dB, "
            f"|phase difference| median {rows[-1]['median_abs_phase_difference_deg']:.1f} deg, "
            f"relative difference {rows[-1]['median_relative_difference']:.2f} "
            f"(with conjugate: {rows[-1]['median_relative_difference_conjugate']:.2f})",
            level=2,
        )

    # Figure: median RTF level along frequency and phase difference map, for each receiver pair
    if plot_comparison:
        n_pairs = len(rcv_ids)
        fig, axs = plt.subplots(nrows=n_pairs, ncols=2, figsize=(16, 3 * n_pairs), squeeze=False, sharex=True)
        for i_rcv, rcv_id in enumerate(rcv_ids):
            ax_level, ax_phase = axs[i_rcv]
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma_ratio = 20 * np.log10(np.abs(pi_ratio_all[i_rcv]))
                gamma_evd = 20 * np.log10(np.abs(pi_evd_all[i_rcv]))
            ax_level.plot(freq, np.nanmedian(gamma_ratio, axis=0), label=r"$S_{ij}/S_{jj}$")
            ax_level.plot(freq, np.nanmedian(gamma_evd, axis=0), "--", label="cs-evd")
            ax_level.set_ylim(*gamma_ylim_db)
            ax_level.set_ylabel(f"{rcv_id}\nMedian RTF level [dB]")
            mesh = ax_phase.pcolormesh(
                freq,
                np.arange(pi_ratio_all.shape[1]),
                wrapped_phase_difference(pi_evd_all[i_rcv], pi_ratio_all[i_rcv]),
                cmap="twilight",
                vmin=-180,
                vmax=180,
                shading="auto",
            )
            ax_phase.set_ylabel("Window")
            fig.colorbar(mesh, ax=ax_phase, label="Phase difference [deg]")
        axs[0, 0].legend()
        axs[-1, 0].set_xlabel("Frequency [Hz]")
        axs[-1, 1].set_xlabel("Frequency [Hz]")
        fig.suptitle(f"RTF of event {event_name}: cs-evd versus spectral ratio (reference {reference_rcv})")
        save_rtf_figure(fig, f"rtf_methods_{event_name}.png")

    # Single window: the one closest (real distance) to a sample of the library; gamma only
    if plot_single_window:
        # Real distance between every window of the event and every sample of every library section
        best = None
        for library_name in portion_library.feature.children:
            ds_library = portion_library.feature[library_name].to_dataset()
            distance_m = get_distance_between_library_and_event(ds_library, ds_ratio)  # (n_library, n_windows)
            i_library, i_window = np.unravel_index(np.nanargmin(distance_m), distance_m.shape)
            if best is None or distance_m[i_library, i_window] < best["distance_m"]:
                best = dict(
                    library_name=library_name,
                    i_library=int(i_library),
                    i_window=int(i_window),
                    distance_m=float(distance_m[i_library, i_window]),
                    ds_library=ds_library,
                )
        i_window = best["i_window"]
        i_library = best["i_library"]
        window_datetime = ds_ratio["event_datetime"].values[i_window]
        log(
            f"Closest window: #{i_window} ({window_datetime}), library '{best['library_name']}' sample "
            f"#{i_library}, real distance {best['distance_m']:.0f} m",
            level=2,
        )

        # Common receivers (the library and the event can have different receiver sets)
        library_rcv_ids = [str(rcv) for rcv in best["ds_library"]["rcv_id_feature"].values]
        common_rcv_ids = [rcv for rcv in rcv_ids if rcv in library_rcv_ids]
        if not common_rcv_ids:
            raise ValueError(f"No receiver in common between the library ({library_rcv_ids}) and the event ({rcv_ids}).")

        fig, axs = plt.subplots(
            nrows=len(common_rcv_ids), ncols=1, figsize=(14, 3 * len(common_rcv_ids)), squeeze=False, sharex=True
        )
        for ax, rcv_id in zip(axs[:, 0], common_rcv_ids):
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma_ratio = 20 * np.log10(np.abs(pi_ratio_all[rcv_ids.index(rcv_id), i_window]))
                gamma_evd = 20 * np.log10(np.abs(pi_evd_all[rcv_ids.index(rcv_id), i_window]))
            gamma_library = best["ds_library"]["gamma_ij"].sel(rcv_id_feature=rcv_id).values[i_library]
            ax.plot(freq, gamma_ratio, linewidth=3, label=rf"Event, $S_{{ij}}/S_{{jj}}$")
            ax.plot(freq, gamma_evd, ":", color="tab:red", linewidth=2, label="Event, cs-evd")
            ax.plot(freq, gamma_library, "k--", label=f"Library ({best['library_name']}, closest position)")
            ax.set_ylabel(f"{rcv_id}\n" + r"$\gamma$ [dB]")
            ax.grid(alpha=0.3)
        axs[0, 0].legend(loc="upper right")
        axs[-1, 0].set_xlabel("Frequency [Hz]")
        fig.suptitle(
            f"Event {event_name}, window {i_window} ({str(window_datetime)[:19]}): gamma of both estimators and of the "
            f"library at the closest position ({best['distance_m']:.0f} m)"
        )
        save_rtf_figure(fig, f"rtf_methods_single_window_{event_name}.png")

        # Gamma differences with the library for this window
        for rcv_id in common_rcv_ids:
            gamma_library = best["ds_library"]["gamma_ij"].sel(rcv_id_feature=rcv_id).values[i_library]
            for method, pi_all in (("spectral_ratio", pi_ratio_all), ("cs_evd", pi_evd_all)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    gamma_event = 20 * np.log10(np.abs(pi_all[rcv_ids.index(rcv_id), i_window]))
                rows_window.append(
                    dict(
                        event=event_name,
                        window=i_window,
                        library_section=best["library_name"],
                        library_sample=i_library,
                        real_distance_m=best["distance_m"],
                        rcv_id=rcv_id,
                        event_method=method,
                        median_abs_gamma_difference_to_library_db=float(np.nanmedian(np.abs(gamma_event - gamma_library))),
                    )
                )
                log(
                    f"{rcv_id}, {method}: median |gamma_event - gamma_library| = "
                    f"{rows_window[-1]['median_abs_gamma_difference_to_library_db']:.2f} dB",
                    level=3,
                )

df_rtf = pd.DataFrame(rows)
fpath = os.path.join(ROOT_IMG_PORTION_RESULTS_RTF, "rtf_methods_comparison.csv")
df_rtf.to_csv(fpath, index=False)
log(f"Saved {fpath}", level=2)

if rows_window:
    fpath = os.path.join(ROOT_IMG_PORTION_RESULTS_RTF, "rtf_methods_single_window.csv")
    pd.DataFrame(rows_window).to_csv(fpath, index=False)
    log(f"Saved {fpath}", level=2)

# ======================================================================================================================
# Sanity check on an active section
# ======================================================================================================================
if active_section_check is not None:
    log(f"Step 3 - Sanity check on the active section '{active_section_check}' (both estimators must be close)")
    portion = open_portion(portion_fpath)
    if "feature" not in portion.children or active_section_check not in portion.feature.children:
        raise ValueError(
            f"Section '{active_section_check}' (with its CPSD) not found in '{portion_fpath}'."
        )
    comparison_active = compare_rtf_methods(
        portion.feature[active_section_check].to_dataset(), reference_rcv, fmin, fmax
    )
    rows_active = []
    for rcv_id, values in comparison_active.items():
        rows_active.append(dict(section=active_section_check, rcv_id=rcv_id, **values))
        convention_issue = values["relative_difference_conjugate"] < 0.5 * values["relative_difference"]
        log(
            f"{rcv_id}: relative difference {values['relative_difference']:.3f}, with the conjugate "
            f"{values['relative_difference_conjugate']:.3f}"
            + ("  -> the cs-evd RTF looks conjugated: check the convention of the covariance matrix" if convention_issue else ""),
            level=2,
        )
    fpath = os.path.join(ROOT_IMG_PORTION_RESULTS_RTF, f"rtf_methods_active_{active_section_check}.csv")
    pd.DataFrame(rows_active).to_csv(fpath, index=False)
    log(f"Saved {fpath}", level=2)
    portion.close()

portion_event.close()
