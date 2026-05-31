#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_mfp_misc.py
@Time    :   2026/05/18 13:31:56
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import scipy.signal as sp


def extract_replica_and_features(ds_library, ds_event):
    # Library RTFs
    library_replicas = ds_library.rtf_amp * np.exp(1j * ds_library.rtf_phase)

    # Event RTFs
    event_feature = ds_event.rtf_amp * np.exp(1j * ds_event.rtf_phase)
    # Reshape to 4D array to be able to apply distance function : (n_rcv, n_freq, n_segment_dt) -> (n_rcv, n_freq, n_segment_dt, 1)
    event_feature_4d = event_feature.values[..., np.newaxis]

    return library_replicas, event_feature, event_feature_4d


def get_psd(signal, **kwargs):
    fs = kwargs.get("fs", 2000)
    nperseg = kwargs.get("nperseg", None)
    noverlap = kwargs.get("noverlap", None)
    fmin = kwargs.get("fmin", 100)
    fmax = kwargs.get("fmax", 900)

    if nperseg is None or noverlap is None:
        # Raise an error if nperseg or noverlap is not provided
        raise ValueError(
            "nperseg and noverlap must be provided when using weights_type = 'psd'."
        )

    # Compute PSD of the signal
    ff, Pxx_seg = sp.welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
    )

    # Select frequency band of interest
    idx_ff_in_band = np.logical_and(
        (ff >= fmin),
        (ff <= fmax),
    )
    ff = ff[idx_ff_in_band]
    Pxx_seg = Pxx_seg[idx_ff_in_band]

    return Pxx_seg


def get_weights_psd(feature_psd, freq_axis: int = 0):

    # Compute PSD of the signal

    # Convert to dB
    # Pxx = 10 * np.log10(feature_psd)

    # Compute weights (normalized PSD)
    # w_k = (Pxx + np.abs(np.min(Pxx))) / np.max(
    #     Pxx + np.abs(np.min(Pxx))
    # )

    # w_k = Pxx / np.max(Pxx)

    # Shape : (f_rtf, replica_id)
    # Axis 0 -> frequency
    gamma = 0.15
    w_k = (feature_psd / np.max(feature_psd, axis=freq_axis)) ** gamma

    # 2. Normalisation robuste
    scale = np.percentile(w_k, 99.9, axis=freq_axis)
    w_k = w_k / scale
    w_k = np.clip(w_k, 0, 1)

    alpha = 10
    threshold = 0.3
    w_k_soft = 1 / (1 + np.exp(-alpha * (w_k - threshold)))

    w_k = w_k_soft

    # # 1. Compression
    # alpha = 0.45
    # Pxx_comp = Pxx_seg**alpha

    # # 2. Normalisation robuste
    # scale = np.percentile(Pxx_comp, 99)
    # w_k = Pxx_comp / scale
    # w_k = np.clip(w_k, 0, 1)

    # # 3. Sigmoïde
    # alpha_sig = 15
    # threshold = 0.5
    # w_k = 1 / (1 + np.exp(-alpha_sig * (w_k - threshold)))

    # w_k = (w_k - min(w_k)) / (max(w_k) - min(w_k))
    # w_k[w_k <= 0.3] = 0

    # plt.figure()
    # plt.plot(ff, w_k)
    # # plt.plot(ff, w_k_soft)
    # plt.savefig("test1")

    # plt.figure()
    # # plt.plot(ff, w_k)
    # plt.plot(ff, Pxx_seg)
    # plt.savefig("test1")

    # plt.figure()
    # # plt.plot(ff, w_k)
    # plt.plot(ff, 10 * np.log10(Pxx_seg))
    # plt.savefig("test2")

    # plt.figure()
    # # plt.plot(ff, w_k)
    # plt.plot(ff, 10 * np.log10(10 * np.log10(Pxx_seg)))
    # plt.savefig("test3")

    return w_k


def filter_ais(
    ais_event,
    dlon_box=10,
    dlat_box=10,
    box_center_lon=-81.7504,
    box_center_lat=18.3794,
    verbose=False,
):
    if verbose:
        print("\tFiltering AIS data ...")
    # Filter AIS data in area
    lat0 = box_center_lat
    lon0 = box_center_lon

    mmsi_in_box = []
    for mmsi in ais_event.mmsi.values:
        ship = ais_event.sel(mmsi=mmsi)

        ship_in_box = ship.sel(
            lat=slice(
                lat0 - dlat_box,
                lat0 + dlat_box,
            ),
            lon=slice(
                lon0 - dlon_box,
                lon0 + dlon_box,
            ),
        )

        if np.any(ship_in_box):
            mmsi_in_box.append(mmsi)

    ais_event = ais_event.sel(mmsi=mmsi_in_box)

    return ais_event


if __name__ == "__main__":
    pass
