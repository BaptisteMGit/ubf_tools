#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_mfp_dist_utils.py
@Time    :   2026/05/18 13:31:08
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist

from propa.rtf.rtf_utils import D_hermitian_angle_fast
from source.rtf_mfp.rtf_mfp_misc import (
    extract_replica_and_features,
    get_weights_psd,
)
from misc import cast_matrix_to_target_shape


def get_hermitian_angle_dist(
    ds_library, ds_event, use_weighted_mean=False, verbose=False
):

    # Get RTF features for library and event
    library_replicas, event_feature, event_feature_4d = extract_replica_and_features(
        ds_library=ds_library, ds_event=ds_event
    )

    dist_kwargs = {
        "ax_rcv": 0,
        "ax_f": 1,
        "apply_mean": False,
        "apply_median": True,
        "data_space": "complex",
    }
    comment = "Compute distance using hermitian angle distance (in C^Nrcv)"
    if verbose:
        print(comment)

    rtf_distances = []

    if use_weighted_mean:
        dist_kwargs["apply_mean"] = False

        # Compute weights here
        # library_weights = ds_library.feature_weights.values
        # event_weights = ds_event.feature_weights.values
        library_weights = get_weights_psd(
            feature_psd=ds_library.feature_psd.values, freq_axis=0
        )
        event_weights = get_weights_psd(
            feature_psd=ds_event.feature_psd.values, freq_axis=0
        )

        # # Renormalize according to the selected frequency band
        # library_weights = (library_weights - np.min(library_weights, axis=0)) / (
        #     np.max(library_weights, axis=0) - np.min(library_weights, axis=0)
        # )
        # event_weights = (event_weights - np.min(event_weights, axis=0)) / (
        #     np.max(event_weights, axis=0) - np.min(event_weights, axis=0)
        # )

        # library_weights_rep_i = ds_library.feature_weights.sel(
        #     replica_id=rep_id
        # ).values
        # weights = (
        #     library_weights_rep_i[:, np.newaxis]
        #     + ds_event.feature_weights.values
        # )

    # Iterate over each replica of the library
    for i_rep, rep_id in enumerate(library_replicas.replica_id.values):

        replica_i = library_replicas.sel(replica_id=rep_id)

        # Define loc features to use
        rtf_ref = replica_i.values
        rtf = event_feature_4d

        if use_weighted_mean:
            w_k_e = event_weights
            w_k_l = library_weights[:, i_rep][:, np.newaxis]
            alpha = 1
            beta = 1
            weights = (w_k_e**alpha) * (w_k_l**beta)

            # Compute element wise distance
            dist = D_hermitian_angle_fast(
                rtf_ref=rtf_ref,
                rtf=rtf,
                **dist_kwargs,
            )

            idx_nan = np.isnan(dist)
            weights[idx_nan] = np.nan
            # Compute weighted mean
            dist = np.nansum(dist * weights, axis=0) * 1 / (np.nansum(weights, axis=0))

        else:

            # Compute element wise distance
            dist = D_hermitian_angle_fast(
                rtf_ref=rtf_ref,
                rtf=rtf,
                **dist_kwargs,
            )

        rtf_distances.append(dist)

    dist_output = {
        "dist": np.array(rtf_distances),
        "name": r"$\theta$",
        "unit": "°",
    }

    return dist_output


def get_hermitian_angle_module_dist(ds_library, ds_event, verbose=False):

    # Get RTF features for library and event
    library_replicas, event_feature, event_feature_4d = extract_replica_and_features(
        ds_library=ds_library, ds_event=ds_event
    )

    dist_kwargs = {
        "ax_rcv": 0,
        "ax_f": 1,
        "apply_mean": False,
        "apply_median": True,
        "data_space": "real",
    }
    comment = "Compute distance using euclidian angle between RTF modules (in R^Nrcv)"
    if verbose:
        print(comment)

    rtf_distances = []

    # Iterate over each replica of the library
    for i_rep, rep_id in enumerate(library_replicas.replica_id.values):

        # Get current replica
        replica_i = library_replicas.sel(replica_id=rep_id)

        # Define loc features to use
        rtf_ref = np.abs(replica_i.values)
        rtf = np.abs(event_feature_4d)

        # Compute element wise distance
        dist = D_hermitian_angle_fast(
            rtf_ref=rtf_ref,
            rtf=rtf,
            **dist_kwargs,
        )

        rtf_distances.append(dist)

    dist_output = {
        "dist": np.array(rtf_distances),
        "name": r"$\theta_{\text{mod}}$",
        "unit": "°",
    }

    return dist_output


def get_hermitian_angle_along_freq_axis_dist(ds_library, ds_event, verbose=False):

    # Get RTF features for library and event
    library_replicas, event_feature, event_feature_4d = extract_replica_and_features(
        ds_library=ds_library, ds_event=ds_event
    )

    dist_kwargs = {
        "ax_rcv": 1,
        "ax_f": 0,
        "apply_mean": False,
        "apply_median": True,
        "data_space": "complex",
    }  # Note that rcv and freq axis have been deliberately reversed
    comment = (
        "Compute distance using hermitian angle distance along frequency axis (in C^Nf)"
    )
    if verbose:
        print(comment)

    rtf_distances = []

    # Iterate over each replica of the library
    for i_rep, rep_id in enumerate(library_replicas.replica_id.values):

        # Get current replica
        replica_i = library_replicas.sel(replica_id=rep_id)

        # Define loc features to use
        rtf_ref = replica_i.values
        rtf = event_feature_4d

        # Compute element wise distance
        dist = D_hermitian_angle_fast(
            rtf_ref=rtf_ref,
            rtf=rtf,
            **dist_kwargs,
        )

        rtf_distances.append(dist)

    dist_output = {
        "dist": np.array(rtf_distances),
        "name": r"$\theta_{\text{freq}}$",
        "unit": "°",
    }

    return dist_output


def get_norm_L1_dist(ds_library, ds_event, verbose=False):

    # Get RTF features for library and event
    library_replicas, event_feature, event_feature_4d = extract_replica_and_features(
        ds_library=ds_library, ds_event=ds_event
    )

    comment = "Compute L1 norm (in C^Nrcv)"
    if verbose:
        print(comment)

    rtf_distances = []

    # Iterate over each replica of the library
    for i_rep, rep_id in enumerate(library_replicas.replica_id.values):

        # Get current replica
        replica_i = library_replicas.sel(replica_id=rep_id)

        # Define loc features to use
        rtf_ref = replica_i.values
        rtf = event_feature_4d
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        # Compute element wise distance
        d_L1 = np.sum(np.abs(rtf_ref_expanded - rtf), axis=0)
        dist = np.median(d_L1, axis=0).squeeze()  # Median along f axis

        rtf_distances.append(dist)

    dist_output = {
        "dist": np.array(rtf_distances),
        "name": r"$\text{L}_1$ norm",
        "unit": "",
    }

    return dist_output


def get_norm_L1_module_dist(ds_library, ds_event, verbose=False):

    # Get RTF features for library and event
    library_replicas, event_feature, event_feature_4d = extract_replica_and_features(
        ds_library=ds_library, ds_event=ds_event
    )

    comment = "Compute L1 norm for RTF module (in R^Nrcv)"
    if verbose:
        print(comment)

    rtf_distances = []

    # Iterate over each replica of the library
    for i_rep, rep_id in enumerate(library_replicas.replica_id.values):

        # Get current replica
        replica_i = library_replicas.sel(replica_id=rep_id)

        # Define loc features to use
        rtf_ref = np.abs(replica_i.values)
        rtf = np.abs(event_feature_4d)
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        # Compute element wise distance
        d_L1 = np.sum(np.abs(rtf_ref_expanded - rtf), axis=0)
        dist = np.median(d_L1, axis=0).squeeze()  # Median along f axis

        rtf_distances.append(dist)

    dist_output = {
        "dist": np.array(rtf_distances),
        "name": r"$\text{L}_1$ norm (mod)",
        "unit": "",
    }

    return dist_output


def get_norm_L2_dist(ds_library, ds_event, verbose=False):

    # Get RTF features for library and event
    library_replicas, event_feature, event_feature_4d = extract_replica_and_features(
        ds_library=ds_library, ds_event=ds_event
    )

    comment = "Compute L2 norm (in C^Nrcv)"
    if verbose:
        print(comment)

    rtf_distances = []

    # Iterate over each replica of the library
    for i_rep, rep_id in enumerate(library_replicas.replica_id.values):

        # Get current replica
        replica_i = library_replicas.sel(replica_id=rep_id)

        # Define loc features to use
        rtf_ref = replica_i.values
        rtf = event_feature_4d
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        # Compute element wise distance
        d_L2 = np.sqrt(np.sum(np.abs(rtf_ref_expanded - rtf) ** 2, axis=0))
        dist = np.median(d_L2, axis=0).squeeze()

        rtf_distances.append(dist)

    dist_output = {
        "dist": np.array(rtf_distances),
        "name": r"$\text{L}_2$ norm",
        "unit": "",
    }

    return dist_output


def get_norm_L2_module_dist(ds_library, ds_event, verbose=False):

    # Get RTF features for library and event
    library_replicas, event_feature, event_feature_4d = extract_replica_and_features(
        ds_library=ds_library, ds_event=ds_event
    )

    comment = "Compute L2 norm for RTF module (in R^Nrcv)"
    if verbose:
        print(comment)

    rtf_distances = []

    # Iterate over each replica of the library
    for i_rep, rep_id in enumerate(library_replicas.replica_id.values):

        # Get current replica
        replica_i = library_replicas.sel(replica_id=rep_id)

        # Define loc features to use
        rtf_ref = np.abs(replica_i.values)
        rtf = np.abs(event_feature_4d)
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        # Compute element wise distance
        d_L2 = np.sqrt(np.sum(np.abs(rtf_ref_expanded - rtf) ** 2, axis=0))
        dist = np.median(d_L2, axis=0).squeeze()

        rtf_distances.append(dist)

    dist_output = {
        "dist": np.array(rtf_distances),
        "name": r"$\text{L}_2$ norm (mod)",
        "unit": "",
    }

    return dist_output


def get_norm_L2_normalized_dist(ds_library, ds_event, verbose=False):

    # Get RTF features for library and event
    library_replicas, event_feature, event_feature_4d = extract_replica_and_features(
        ds_library=ds_library, ds_event=ds_event
    )

    comment = "Compute L2 norm normalized with ref RTF (in C^Nrcv)"
    if verbose:
        print(comment)

    rtf_distances = []

    # Iterate over each replica of the library
    for i_rep, rep_id in enumerate(library_replicas.replica_id.values):

        # Get current replica
        replica_i = library_replicas.sel(replica_id=rep_id)

        # Define loc features to use
        rtf_ref = replica_i.values
        rtf = event_feature_4d
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        # Compute element wise distance
        d_L2_normalized = np.sqrt(
            np.sum(np.abs(rtf_ref_expanded - rtf) ** 2, axis=0)
        ) / np.sqrt(np.sum(np.abs(rtf_ref_expanded) ** 2, axis=0))
        dist = np.median(d_L2_normalized, axis=0).squeeze()

        rtf_distances.append(dist)

    dist_output = {
        "dist": np.array(rtf_distances),
        "name": r"$\text{L}_2$ norm (normalized by ref)",
        "unit": "",
    }

    return dist_output


def get_norm_L2_module_normalized_dist(ds_library, ds_event, verbose=False):

    # Get RTF features for library and event
    library_replicas, event_feature, event_feature_4d = extract_replica_and_features(
        ds_library=ds_library, ds_event=ds_event
    )

    comment = "Compute L2 norm for RTF module normalized with ref RTF (in R^Nrcv)"
    if verbose:
        print(comment)

    rtf_distances = []

    # Iterate over each replica of the library
    for i_rep, rep_id in enumerate(library_replicas.replica_id.values):

        # Get current replica
        replica_i = library_replicas.sel(replica_id=rep_id)

        # Define loc features to use
        rtf_ref = np.abs(replica_i.values)
        rtf = np.abs(event_feature_4d)

        # rtf_ref = 10 * np.log10(rtf_ref)
        # rtf = 10 * np.log10(rtf)
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        # Compute element wise distance
        # Normalize by ref RTF
        # d_L2_normalized = np.sqrt(
        #     np.sum(np.abs(rtf_ref_expanded - rtf) ** 2, axis=0)
        # ) / np.sqrt(np.sum(np.abs(rtf_ref_expanded) ** 2, axis=0))
        # Alternative normalization by event RTF
        # d_L2_normalized = np.sqrt(
        #     np.sum(np.abs(rtf_ref_expanded - rtf) ** 2, axis=0)
        # ) / np.sqrt(np.sum(np.abs(rtf) ** 2, axis=0))
        # # Alternative normalization by both ref and event RTF
        # d_L2_normalized = np.sqrt(
        #     np.sum(np.abs(rtf_ref_expanded - rtf) ** 2, axis=0)
        # ) / (
        #     np.sqrt(np.sum(np.abs(rtf) ** 2, axis=0))
        #     * np.sqrt(np.sum(np.abs(rtf_ref_expanded) ** 2, axis=0))
        # )

        d_L2_normalized = np.sqrt(
            np.sum(np.abs(rtf_ref_expanded - rtf) ** 2, axis=0)
        ) / np.sqrt(np.sum(np.abs(rtf_ref_expanded) ** 2, axis=0)) + np.sqrt(
            np.sum(np.abs(rtf_ref_expanded - rtf) ** 2, axis=0)
        ) / np.sqrt(
            np.sum(np.abs(rtf) ** 2, axis=0)
        )

        dist = np.median(d_L2_normalized, axis=0).squeeze()

        rtf_distances.append(dist)

    dist_output = {
        "dist": np.array(rtf_distances),
        "name": r"$\text{L}_2$ norm (mod normalized by ref)",
        "unit": "",
    }

    return dist_output


def get_intercorr_max_module_dist(ds_library, ds_event, verbose=False):

    # Get RTF features for library and event
    library_replicas, event_feature, event_feature_4d = extract_replica_and_features(
        ds_library=ds_library, ds_event=ds_event
    )

    comment = "Compute L2 norm for RTF module normalized with ref RTF (in R^Nrcv)"
    if verbose:
        print(comment)

    rtf_distances = []

    # Iterate over each replica of the library
    for i_rep, rep_id in enumerate(library_replicas.replica_id.values):

        # Get current replica
        replica_i = library_replicas.sel(replica_id=rep_id)

        # Define loc features to use
        rtf_ref = np.abs(replica_i.values)
        rtf = np.abs(event_feature_4d)

        # rtf_ref = 10 * np.log10(rtf_ref)
        # rtf = 10 * np.log10(rtf)
        rtf_ref_expanded = cast_matrix_to_target_shape(rtf_ref, rtf.shape)

        x = np.abs(rtf_ref_expanded)
        y = np.abs(rtf)

        x = x - np.mean(x, axis=1, keepdims=True)
        y = y - np.mean(y, axis=1, keepdims=True)

        x_fft = np.fft.fft(x, axis=1)
        y_fft = np.fft.fft(y, axis=1)
        # df = ds.f.values[1] - ds.f.values[0]
        s_xy = x_fft * np.conj(y_fft)
        c_xy = np.fft.ifft(s_xy, axis=1)
        d_intercorr = np.fft.fftshift(np.real(c_xy), axes=1)
        d_intercorr_max = np.max(d_intercorr, axis=1)  # Max along delta_f axis

        dist = np.sum(d_intercorr_max, axis=0).squeeze()  # Sum along rcv axis

        rtf_distances.append(dist)

    rtf_distances = np.array(rtf_distances)
    rtf_distances = (rtf_distances - rtf_distances.min()) / (
        rtf_distances.max() - rtf_distances.min()
    )
    rtf_distances = 1 - rtf_distances

    dist_output = {
        "dist": rtf_distances,
        "name": r"$\max_{\delta_f} C_{xy}(\delta_f)$",
        "unit": "",
    }

    return dist_output


def ambiguity(
    ds_library,
    ds_event,
    fmin=100,
    fmax=900,
    dist_type="hermitian_angle",
    use_weighted_mean=False,
    verbose=False,
):

    # Define comon frequency band to use
    fmin_common_band = max(fmin, max(ds_library.f_rtf.min(), ds_event.f_rtf.min()))
    fmax_common_band = min(fmax, min(ds_library.f_rtf.max(), ds_event.f_rtf.max()))

    # Slice common band
    ds_library = ds_library.sel(f_rtf=slice(fmin_common_band, fmax_common_band))
    ds_event = ds_event.sel(f_rtf=slice(fmin_common_band, fmax_common_band))

    # Compute dist
    if dist_type == "hermitian_angle":
        dist = get_hermitian_angle_dist(
            ds_library=ds_library,
            ds_event=ds_event,
            use_weighted_mean=use_weighted_mean,
            verbose=verbose,
        )

    elif dist_type == "hermitian_angle_module":
        dist = get_hermitian_angle_module_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )

    elif dist_type == "hermitian_angle_along_freq":
        dist = get_hermitian_angle_along_freq_axis_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )

    elif dist_type == "norm_L1":
        dist = get_norm_L1_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )

    elif dist_type == "norm_L1_module":
        dist = get_norm_L1_module_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )

    elif dist_type == "norm_L2":
        dist = get_norm_L2_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )

    elif dist_type == "norm_L2_module":
        dist = get_norm_L2_module_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )

    elif dist_type == "norm_L2_normalized":
        dist = get_norm_L2_normalized_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )

    elif dist_type == "norm_L2_module_normalized":
        dist = get_norm_L2_module_normalized_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )

    elif dist_type == "combined_norm_L2_module_normalized_hermitian_angle_module":
        dist_L2_mod_norm = get_norm_L2_module_normalized_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )["dist"]
        dist_herm_mod = get_hermitian_angle_module_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )["dist"]
        # Convert distances to same scale (0-1)
        dist_L2_mod_norm_scaled = (dist_L2_mod_norm - np.min(dist_L2_mod_norm)) / (
            np.max(dist_L2_mod_norm) - np.min(dist_L2_mod_norm)
        )
        dist_herm_mod_scaled = (dist_herm_mod - np.min(dist_herm_mod)) / (
            np.max(dist_herm_mod) - np.min(dist_herm_mod)
        )
        # Combine distances with equal weights
        alpha = 0.5
        dist_combined = (
            alpha * dist_L2_mod_norm_scaled + (1 - alpha) * dist_herm_mod_scaled
        )
        dist = {
            "dist": dist_combined,
            "name": r"$\alpha \cdot \text{L}_2\text{ norm (mod normalized by ref)} + (1-\alpha) \cdot \theta_{\text{mod}}$",
            "unit": "",
        }

    elif dist_type == "intercorr_max":
        dist = get_intercorr_max_module_dist(
            ds_library=ds_library, ds_event=ds_event, verbose=verbose
        )

    rtf_dist = dist["dist"]
    rtf_dist = (
        rtf_dist.T
    )  # Transpose to have shape (n_event_feature, n_library_replica)

    # Spatial distance between library and event replicas
    event_e = ds_event["e_replica"].values
    event_n = ds_event["n_replica"].values
    libray_e = ds_library["e_replica"].values
    libray_n = ds_library["n_replica"].values
    event_coords = np.column_stack((event_e, event_n))
    library_coords = np.column_stack((libray_e, libray_n))

    spatial_dist = cdist(event_coords, library_coords, metric="euclidean")

    # Build results dataset
    ds_results = xr.Dataset(
        data_vars={
            "rtf_dist": (("event_replica_id", "library_replica_id"), rtf_dist),
            "spatial_dist": (("event_replica_id", "library_replica_id"), spatial_dist),
        },
        coords={
            "event_replica_id": ds_event.replica_id.values,
            "library_replica_id": ds_library.replica_id.values,
        },
    )

    # Add attributes to the dataset
    ds_results.attrs = {
        "description": f"Distance matrix between library and event replicas computed using {dist_type} distance for the RTF features and euclidean distance for the spatial coordinates.",
        "rtf_dist_type": dist_type,
        "spatial_dist_type": "euclidean",
        "library_id": ds_library.id,
        "event_id": ds_event.id,
        "fmin": fmin,
        "fmax": fmax,
    }
    # Add attributes to variables
    ds_results["rtf_dist"].attrs = {
        "description": f"Distance between library and event replicas computed using {dist_type} distance on the RTF features.",
        "units": dist["unit"],
        "long_name": dist["name"],
    }
    ds_results["spatial_dist"].attrs = {
        "description": "Euclidean distance between library and event replicas in the spatial domain.",
        "units": "m",
        "long_name": "Spatial distance",
    }

    # Add attributes to coordinates
    ds_results["event_replica_id"].attrs = {
        "description": "ID of the event replica",
        "long_name": "Event replica ID",
    }
    ds_results["library_replica_id"].attrs = {
        "description": "ID of the library replica",
        "long_name": "Library replica ID",
    }

    return ds_results


if __name__ == "__main__":
    pass
