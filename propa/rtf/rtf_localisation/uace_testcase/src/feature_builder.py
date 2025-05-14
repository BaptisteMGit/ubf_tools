#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   feature_builder.py
@Time    :   2025/04/07 14:07:28
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to build features for localisation.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import gc
import os
import numpy as np
import xarray as xr
import dask.array as da
import scipy.signal as sp
import matplotlib.pyplot as plt

from time import time

from propa.rtf.rtf_utils import D_hermitian_angle_fast

import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from source.feature_processor import FeatureProcessor
from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.rtf.rtf_localisation.uace_testcase.src.data_builder import DataBuilder

from misc import mult_along_axis
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import (
    library_src_spectrum,
    event_src_spectrum,
    params,
)
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_plot_utils import (
    check_signal_noise,
    check_signal_noise_stft,
    check_gcc_features,
    check_rtf_features,
)


class FeatureBuilder:
    """
    Class to build features for localisation.
    """

    def __init__(
        self,
        simulation: Simulation = None,
        rtf_method: str = p.rtf_method,
        gcc_method: str = p.gcc_method,
        use_dask: bool = True,
    ):
        """
        Init the class.
        """
        self.simulation = simulation
        self.rtf_method = rtf_method
        self.gcc_method = gcc_method

        self.check = self.simulation.check_features
        self.use_dask = use_dask

        self.data_builder = DataBuilder(simulation=self.simulation)

    def build_features_from_time_signal(
        self,
        snr_dB=0,
    ):
        """
        Step 4.2 : build localisation features
            -> GCC for the DCF method
            -> RTF for the RTF-MFP method

        """

        t_start = time()

        # Load params
        dx = self.simulation.dx
        dy = self.simulation.dy

        # fpath = os.path.join(p.root_data, fname)
        ds_sig = xr.open_dataset(self.simulation.library_dataset_fpath)
        attrs = ds_sig.attrs

        # Derive noise dataset
        ds_noise = self.data_builder.derive_received_noise(
            s_library=ds_sig.s_l,
            s_event=ds_sig.s_e,
            snr_dB=snr_dB,
        )  # (nr, nt, ny, nx)

        # NOTE : different noise dataset to simulate the fact that in real life the noise CSDM is
        # estimated from a different segment of the signal than the signal + noise CSDM (assuming noise is stationnary)
        ds_noise_bis = self.data_builder.derive_received_noise(
            s_library=ds_sig.s_l,
            s_event=ds_sig.s_e,
            snr_dB=snr_dB,
        )  # (nr, nt, ny, nx)

        # Noisy signals
        noisy_signal_library = ds_noise.n_l + ds_sig.s_l  # (nrcv, nt, ny, nx)
        noisy_signal_event = ds_noise.n_e + ds_sig.s_e  # (nrcv, nt)

        # We don't need all datasets anymore
        ds_noise.close()
        ds_sig.close()

        # Plot signal and noise at source position -> library
        if self.check:
            ds_sig_noise = xr.Dataset(
                data_vars=dict(
                    x_l=(["idx_rcv", "t", "y", "x"], noisy_signal_library.values),
                    n_l=(["idx_rcv", "t", "y", "x"], ds_noise.n_l.values),
                    s_l=(["idx_rcv", "t", "y", "x"], ds_sig.s_l.values),
                    x_e=(["idx_rcv", "t"], noisy_signal_event.values),
                    n_e=(["idx_rcv", "t"], ds_noise.n_e.values),
                    s_e=(["idx_rcv", "t"], ds_sig.s_e.values),
                ),
                coords=dict(
                    t=ds_sig.t,
                    x=ds_sig.x,
                    y=ds_sig.y,
                    idx_rcv=ds_sig.idx_rcv,
                ),
                attrs=dict(
                    std_ref_event=ds_noise.std_ref_event,
                    std_ref_library=ds_noise.std_ref_library,
                    snr=snr_dB,
                    xs=self.simulation.event_ship_x,
                    ys=self.simulation.event_ship_y,
                    dx=dx,
                    dy=dy,
                    root_img=os.path.join(
                        self.simulation.root_img_from_sig,
                        f"snr_{snr_dB:.1f}dB",
                    ),
                ),
            )

            self.check_signal_noise(ds_sig_noise)
            self.check_signal_noise_stft(ds_sig_noise)

        # List of potential reference receivers to test
        idx_rcv_refs = (
            self.simulation.antenna.rcv_idx
        )  # General case -> all receivers are used as reference to build the ambiguity surface for all couples in array (required for DCF method)

        nperseg = 2**11
        noverlap = nperseg // 2

        ds_feature_estimate = xr.Dataset(
            data_vars=dict(
                x_l=(["idx_rcv", "t", "y", "x"], noisy_signal_library.data),
                n_l_bis=(["idx_rcv", "t", "y", "x"], ds_noise_bis.n_l.data),
                x_e=(["idx_rcv", "t"], noisy_signal_event.data),
                n_e_bis=(["idx_rcv", "t"], ds_noise_bis.n_e.data),
            ),
            coords=dict(
                t=ds_sig.t,
                x=ds_sig.x,
                y=ds_sig.y,
                idx_rcv=ds_sig.idx_rcv,
            ),
        )

        # ====================================================================================================
        # Optimized version: RTF and GCC are computed directly on the whole 4D array (nrcv, nt, ny, nx)
        # ====================================================================================================

        xl_4D = ds_feature_estimate.x_l.data
        vl_4D = ds_feature_estimate.n_l_bis.data

        fp = FeatureProcessor(
            fs=self.simulation.fs,
            idx_rcv_ref=0,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )

        if self.use_dask:

            ### RTF ###
            # Define optimal chunk size to fit with the number of workers
            # shape = xl_4D.shape
            # optimal_chunks = compute_chunks(shape, N_WORKERS)
            optimal_chunks = (
                -1,
                -1,
                4,
                4,
            )  # Seems to be one one the fastest configuration with 20 workers

            # Convert to dask arrays
            xl_4D = da.from_array(xl_4D, chunks=optimal_chunks)
            xl_4D = xl_4D.persist()
            vl_4D = da.from_array(vl_4D, chunks=optimal_chunks)
            vl_4D = vl_4D.persist()

            # print("Total number of chunks : ", xl_4D.numblocks[-2]*xl_4D.numblocks[-1])
            # t0 = time()
            output_chunks = (xl_4D.shape[0],) + xl_4D.shape
            rtf_dask = da.map_blocks(
                fp.dask_rtf_4D,
                xl_4D,
                vl_4D,
                dtype=complex,
                chunks=output_chunks,
                idx_rcv_refs=idx_rcv_refs,
            )
            rtf_library = rtf_dask.compute()

            # We will use the same 4D function for the event signal so we need to add dimensions (single positions x, y)
            xe_4D = ds_feature_estimate.x_e.values[..., np.newaxis, np.newaxis]
            ve_4D = ds_feature_estimate.n_e_bis.values[..., np.newaxis, np.newaxis]

            dims_order = {"r": 0, "t": 1, "y": 2, "x": 3}
            ff, rtf_event = (
                fp.rtf_estimator.covariance_subtraction_major_eigen_vector_4D(
                    xe_4D,
                    ve_4D,
                    dims_order,
                    idx_rcv_refs,
                    return_csdm=False,
                )
            )

            # Squeeze to remove dummy dimensions
            rtf_event = np.squeeze(rtf_event)

            # Restict to the frequency band of interest
            idx_band = (ff >= self.simulation.fmin) & (ff <= self.simulation.fmax)
            rtf_library = rtf_library[:, :, idx_band, ...]
            rtf_event = rtf_event[..., idx_band]
            f_rtf = ff[idx_band]

            ### DCF ###
            gcc_dask = da.map_blocks(
                fp.dask_gcc_4D,
                xl_4D,
                dtype=complex,
                chunks=output_chunks,
                idx_rcv_refs=idx_rcv_refs,
            )
            gcc_library = gcc_dask.compute()

            xe_4D = ds_feature_estimate.x_e.values[..., np.newaxis, np.newaxis]
            ff, gcc_event = fp.gcc_estimator.gcc_4D(
                x_4D=xe_4D,
                idx_rcv_refs=idx_rcv_refs,
            )
            gcc_event = np.squeeze(gcc_event)

            # Restict to the frequency band of interest
            idx_band = (ff >= self.simulation.fmin) & (ff <= self.simulation.fmax)
            gcc_library = gcc_library[:, :, idx_band, ...]
            gcc_event = gcc_event[:, :, idx_band]
            f_gcc = ff[idx_band]

            # Clean up memory
            del xl_4D, vl_4D, xe_4D, ve_4D, gcc_dask, rtf_dask
            gc.collect()

        else:
            ### RTF ###
            # TODO debug dims order missing 
            # t0 = time()
            xl_4D = ds_feature_estimate.x_l.values
            vl_4D = ds_feature_estimate.n_l_bis.values

            dims_order = {"r": 0, "t": 1, "y": 2, "x": 3}
            ff, rtf_library = (
                fp.rtf_estimator.covariance_subtraction_major_eigen_vector_4D(
                    x_4D=xl_4D,
                    v_4D=vl_4D,
                    dims_order=dims_order,
                    idx_rcv_refs=idx_rcv_refs,
                    return_csdm=False,
                )
            )

            # We will use the same 4D function for the event signal so we need to add dimensions (single positions x, y)
            xe_4D = ds_feature_estimate.x_e.values[..., np.newaxis, np.newaxis]
            ve_4D = ds_feature_estimate.n_e_bis.values[..., np.newaxis, np.newaxis]
            dims_order = {"r": 0, "t": 1, "y": 2, "x": 3}
            ff, rtf_event = (
                fp.rtf_estimator.covariance_subtraction_major_eigen_vector_4D(
                    x_4D=xe_4D,
                    v_4D=ve_4D,
                    dims_order=dims_order,
                    idx_rcv_refs=idx_rcv_refs,
                    return_csdm=False,
                )
            )

            # Squeeze to remove dummy dimensions
            rtf_event = np.squeeze(rtf_event)

            # Restict to the frequency band of interest
            idx_band = (ff >= self.simulation.fmin) & (ff <= self.simulation.fmax)
            rtf_library = rtf_library[:, :, idx_band, ...]
            rtf_event = rtf_event[..., idx_band]
            f_rtf = ff[idx_band]

            ### GCC ###
            xl_4D = ds_feature_estimate.x_l.values
            xe_4D = ds_feature_estimate.x_e.values[..., np.newaxis, np.newaxis]

            ff, gcc_library = fp.gcc_estimator.gcc_4D(
                x_4D=xl_4D,
                idx_rcv_refs=idx_rcv_refs,
            )

            _, gcc_event = fp.gcc_estimator.gcc_4D(
                x_4D=xe_4D,
                idx_rcv_refs=idx_rcv_refs,
            )
            gcc_event = np.squeeze(gcc_event)

            # Restict to the frequency band of interest
            idx_band = (ff >= self.simulation.fmin) & (ff <= self.simulation.fmax)
            gcc_library = gcc_library[:, :, idx_band, ...]
            gcc_event = gcc_event[:, :, idx_band]
            f_gcc = ff[idx_band]

            # print("GCC dask matches GCC ", np.allclose(gcc_library, gcc_library_dask))

        # print(f"GCCs computed in {time()-t0} s")

        # Create dataset to store results
        attrs.update(
            dict(
                # std_ref_event=ds_noise.std_ref_event,
                # std_ref_library=ds_noise.std_ref_library,
                snr=snr_dB,
                xs=self.simulation.event_ship_x,
                ys=self.simulation.event_ship_y,
                root_img=os.path.join(
                    self.simulation.root_img_from_sig, f"snr_{snr_dB:.1f}dB"
                ),
            )
        )

        ds_res_from_sig = xr.Dataset(
            data_vars=dict(
                rtf_event_real=(["idx_rcv_ref", "idx_rcv", "f_rtf"], rtf_event.real),
                rtf_event_imag=(["idx_rcv_ref", "idx_rcv", "f_rtf"], rtf_event.imag),
                gcc_event_real=(["idx_rcv_ref", "idx_rcv", "f_gcc"], gcc_event.real),
                gcc_event_imag=(["idx_rcv_ref", "idx_rcv", "f_gcc"], gcc_event.imag),
                rtf_real=(
                    ["idx_rcv_ref", "idx_rcv", "f_rtf", "y", "x"],
                    rtf_library.real,
                ),
                rtf_imag=(
                    ["idx_rcv_ref", "idx_rcv", "f_rtf", "y", "x"],
                    rtf_library.imag,
                ),
                gcc_real=(
                    ["idx_rcv_ref", "idx_rcv", "f_gcc", "y", "x"],
                    gcc_library.real,
                ),
                gcc_imag=(
                    ["idx_rcv_ref", "idx_rcv", "f_gcc", "y", "x"],
                    gcc_library.imag,
                ),
            ),
            coords=dict(
                x=ds_feature_estimate.x.values,
                y=ds_feature_estimate.y.values,
                idx_rcv=ds_feature_estimate.idx_rcv.values,
                idx_rcv_ref=ds_feature_estimate.idx_rcv.values,
                f_gcc=f_gcc,
                f_rtf=f_rtf,
            ),
            attrs=attrs,
        )

        # Derive and save weights to use when deriving the mean hermitian angle
        if self.simulation.use_weighted_rtf:
            # Derive weights
            fpsd, psd = sp.welch(
                ds_feature_estimate.x_l.values,
                fs=self.simulation.fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                axis=1,
            )
            psd_db = 10 * np.log10(psd)
            psd_db_shift = psd_db + np.abs(np.min(psd_db, axis=1, keepdims=True))
            psd_norm = psd_db_shift / np.max(psd_db_shift, axis=1, keepdims=True)
            rtf_weights = psd_norm

            # Save weights as netcdf
            ds_weights = xr.Dataset(
                data_vars=dict(
                    rtf_weights=(["idx_rcv", "f", "y", "x"], rtf_weights),
                ),
                coords=dict(
                    f=fpsd,
                    x=ds_sig.x,
                    y=ds_sig.y,
                    idx_rcv=ds_sig.idx_rcv,
                ),
            )
            ds_weights = ds_weights.sel(
                f=ds_res_from_sig.f_rtf.values, method="nearest"
            )
            ds_weights.to_netcdf(self.simulation.rtf_weights_dataset_fpath)

            # plt.figure(figsize=(12, 6))
            # # plt.plot(fpsd, psd_db[0, :, 0, 0], label="db")
            # # plt.plot(fpsd, psd_db_shift[0, :, 0, 0], label="psd_db_shift")
            # plt.plot(fpsd, psd_norm[0, :, 0, 0], label="psd_norm")
            # plt.savefig("test")
            # print()
        # Subsample frequency to save memory
        # subsample_idx = np.arange(0, ds_res_from_sig.sizes["f"])[::5]
        # ds_res_from_sig = ds_res_from_sig.isel(f=subsample_idx)

        if self.check:
            self.check_rtf_features(ds_res_from_sig)
            if self.simulation.use_weighted_rtf:
                self.check_rtf_weights(ds_res_from_sig, ds_weights)

            # check_gcc_features(ds_res_from_sig, folder=ds_sig_noise.attrs["root_img"])

        # Add snr to the feature dataset filepath
        fpath = (
            self.simulation.feature_dataset_fpath.split(".nc")[0]
            + f"_snr_{snr_dB:.1f}dB.nc"
        )
        # Save updated dataset
        ds_res_from_sig.to_netcdf(fpath)
        ds_res_from_sig.close()

        print(f"Features derived from time signal in {time() - t_start:.2f} s")

    def check_rtf_features(self, ds_rtf):

        # Define folder to store images
        root_img = os.path.join(ds_rtf.root_img, "check_rtf")
        if not os.path.exists(root_img):
            os.makedirs(root_img)

        # Load dataset with KRAKEN TF to derive reference RTF
        ds_tf = xr.open_dataset(self.simulation.tf_grid_dataset_fpath)
        # Build complex tf
        tf = ds_tf.tf_real + 1j * ds_tf.tf_imag
        # Extract tf between fmin and fmax from ds_rtf
        tf = tf.sel(f=slice(ds_rtf.f_rtf.min(), ds_rtf.f_rtf.max()))

        # Define reference receiver to use
        i_rcv_ref = 0
        ds_rtf = ds_rtf.sel(idx_rcv_ref=i_rcv_ref)
        rtf_cs = ds_rtf.rtf_real + 1j * ds_rtf.rtf_imag

        # Define tf_ref
        tf_ref = tf.sel(idx_rcv=i_rcv_ref)

        # List position where we want to compare estimated RTF to ref RTF (KRAKEN)
        # Source position + the 4 corners of the grid + one position inside the grid
        x_check = [
            self.simulation.event_ship_x,
            ds_tf.x.min().values,
            ds_tf.x.min().values,
            ds_tf.x.max().values,
            ds_tf.x.max().values,
            ds_tf.x.values[int(ds_tf.sizes["x"] * 2 / 3)],
        ]

        y_check = [
            self.simulation.event_ship_y,
            ds_tf.y.min().values,
            ds_tf.y.max().values,
            ds_tf.y.max().values,
            ds_tf.y.min().values,
            ds_tf.y.values[int(ds_tf.sizes["y"] * 1 / 3)],
        ]

        # Iterate over receivers
        for i_rcv in tf.idx_rcv.values:

            # Build "true" RTF
            rtf_true = tf.sel(idx_rcv=i_rcv) / tf_ref

            # Iterate over positions to check
            for i_check in range(len(x_check)):
                x_i = x_check[i_check]
                y_i = y_check[i_check]

                # Extract data at required position
                rtf_cs_pos = rtf_cs.sel(idx_rcv=i_rcv).sel(
                    y=y_i, x=x_i, method="nearest"
                )
                rtf_true_pos = rtf_true.sel(y=y_i, x=x_i, method="nearest")

                abs_cs = np.abs(rtf_cs_pos)
                abs_true = np.abs(rtf_true_pos)
                # Compare rtf_true to estimated rtf

                plt.figure(figsize=(12, 6))
                abs_true.plot(
                    label=r"$\Pi_{" + str(i_rcv) + r"}^{(Kraken)}$",
                    linestyle="-",
                    color="k",
                    linewidth=1.5,
                )
                abs_cs.plot(
                    # x="f",
                    linestyle="-",
                    label=r"$\Pi_{" + str(i_rcv) + r"}^{(CS)}$",
                    color="r",
                    marker="o",
                    linewidth=0.2,
                    markersize=3,
                )
                plt.legend()
                plt.yscale("log")
                plt.xlabel(r"$f$" + " [Hz]")
                plt.ylabel(r"$|\Pi(f)|$")

                if i_rcv == i_rcv_ref:
                    plt.ylim(1e-1, 1e1)

                # Save figure
                fname = f"check_rtf_rcv{i_rcv}_x{x_i}_y{y_i}.png"
                fpath = os.path.join(root_img, fname)
                plt.savefig(fpath)
                plt.close("all")

        ds_tf.close()

    def check_rtf_weights(self, ds_rtf, ds_weights):

        # Define folder to store images
        root_img = os.path.join(ds_rtf.root_img, "check_rtf_weights")
        if not os.path.exists(root_img):
            os.makedirs(root_img)

        # Load dataset with KRAKEN TF to derive reference RTF
        ds_tf = xr.open_dataset(self.simulation.tf_grid_dataset_fpath)
        # Build complex tf
        tf = ds_tf.tf_real + 1j * ds_tf.tf_imag
        # Extract tf between fmin and fmax from ds_rtf
        tf = tf.sel(f=slice(ds_rtf.f_rtf.min(), ds_rtf.f_rtf.max()))

        # Define reference receiver to use
        i_rcv_ref = 0
        ds_rtf = ds_rtf.sel(idx_rcv_ref=i_rcv_ref)
        rtf_cs = ds_rtf.rtf_real + 1j * ds_rtf.rtf_imag

        # Define tf_ref
        tf_ref = tf.sel(idx_rcv=i_rcv_ref)

        # List position where we want to compare estimated RTF to ref RTF (KRAKEN)
        # Source position + the 4 corners of the grid + one position inside the grid
        x_check = [
            self.simulation.event_ship_x,
            ds_tf.x.min().values,
            ds_tf.x.min().values,
            ds_tf.x.max().values,
            ds_tf.x.max().values,
            ds_tf.x.values[int(ds_tf.sizes["x"] * 2 / 3)],
        ]

        y_check = [
            self.simulation.event_ship_y,
            ds_tf.y.min().values,
            ds_tf.y.max().values,
            ds_tf.y.max().values,
            ds_tf.y.min().values,
            ds_tf.y.values[int(ds_tf.sizes["y"] * 1 / 3)],
        ]

        dist_func = D_hermitian_angle_fast

        # Iterate over receivers
        for i_rcv in tf.idx_rcv.values:

            # Build "true" RTF
            rtf_true = tf.sel(idx_rcv=i_rcv) / tf_ref

            # Iterate over positions to check
            for i_check in range(len(x_check)):
                x_i = x_check[i_check]
                y_i = y_check[i_check]

                # Extract data at required position
                rtf_cs_pos = rtf_cs.sel(idx_rcv=i_rcv).sel(
                    y=y_i, x=x_i, method="nearest"
                )
                rtf_true_pos = rtf_true.sel(y=y_i, x=x_i, method="nearest")
                rtf_weights_pos = ds_weights.rtf_weights.sel(idx_rcv=i_rcv).sel(
                    y=y_i, x=x_i, method="nearest"
                )

                # Derive theta
                # ax_order = self.get_axis_order(
                #     da=tf_ref, ax_names=["idx_rcv", "f_rtf", "y", "x"]
                # )
                # ax_rcv = ax_order["idx_rcv"]
                # ax_f = ax_order["f_rtf"]

                ax_rcv = 1
                ax_f = 0
                dist_kwargs = {
                    "ax_rcv": ax_rcv,
                    "unit": "deg",
                    "apply_mean": False,
                    "weights": None,
                    "ax_f": ax_f,
                }
                rtf_true_pos = np.atleast_2d(rtf_true_pos).T
                rtf_cs_pos = np.atleast_2d(rtf_cs_pos).T
                theta = dist_func(rtf_true_pos, rtf_cs_pos, **dist_kwargs)

                plt.figure(figsize=(12, 6))
                plt.plot(rtf_true.f.values, theta, label=r"$\theta_k$")
                rtf_weights_pos.plot(label=r"$w_k$")
                # abs_cs.plot(
                #     # x="f",
                #     linestyle="-",
                #     label=r"$\Pi_{" + str(i_rcv) + r"}^{(CS)}$",
                #     color="r",
                #     marker="o",
                #     linewidth=0.2,
                #     markersize=3,
                # )
                # plt.legend()
                # plt.yscale("log")
                # plt.xlabel(r"$f$" + " [Hz]")
                # plt.ylabel(r"$|\Pi(f)|$")

                if i_rcv == i_rcv_ref:
                    plt.ylim(1e-1, 1e1)

                # Save figure
                fname = f"check_rtf_weigths_rcv{i_rcv}_x{x_i}_y{y_i}.png"
                fpath = os.path.join(root_img, fname)
                plt.savefig(fpath)
                plt.close("all")

        ds_tf.close()

    def check_signal_noise(self, ds_sig_noise):
        """
        Plot library signal at source position and event signal as well as associated noise signals to check that the dataset is built as required.
        """
        s_l = ds_sig_noise.s_l.sel(
            x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest"
        )
        x_l = ds_sig_noise.x_l.sel(
            x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest"
        )
        n_l = ds_sig_noise.n_l.sel(
            x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest"
        )
        s_e = ds_sig_noise.s_e
        x_e = ds_sig_noise.x_e
        n_e = ds_sig_noise.n_e

        img_check_path = os.path.join(ds_sig_noise.root_img, "check")
        if not os.path.exists(img_check_path):
            os.makedirs(img_check_path)

        for i_rcv in ds_sig_noise.idx_rcv.values:

            f, axs = plt.subplots(3, 2, figsize=(20, 12), sharex=True, sharey=True)

            # First column -> library
            s_l.sel(idx_rcv=i_rcv).plot(ax=axs[0, 0])
            axs[0, 0].set_title("$z(t)$")

            n_l.sel(idx_rcv=i_rcv).plot(ax=axs[1, 0])
            axs[1, 0].set_title("$v(t)$")

            x_l.sel(idx_rcv=i_rcv).plot(ax=axs[2, 0])
            axs[2, 0].set_title("$x(t) = z(t) + v(t)$")

            # Second column -> event
            s_e.sel(idx_rcv=i_rcv).plot(ax=axs[0, 1])
            axs[0, 1].set_title("$z(t)$")

            n_e.sel(idx_rcv=i_rcv).plot(ax=axs[1, 1])
            axs[1, 1].set_title("$v(t)$")

            x_e.sel(idx_rcv=i_rcv).plot(ax=axs[2, 1])
            axs[2, 1].set_title("$x(t) = z(t) + v(t)$")

            # Remove xlabel for row 0 and 1
            for irow in [0, 1]:
                for icol in [0, 1]:
                    axs[irow, icol].set_xlabel("")

            plt.suptitle(f"SNR = {ds_sig_noise.snr} dB")
            fpath = os.path.join(img_check_path, f"sig_noise_ircv{i_rcv}.png")
            plt.savefig(fpath)

        plt.close("all")

    def check_signal_noise_stft(self, ds_sig_noise):
        """
        Plot library signal stft at source position and event signal stft as well as associated noise signals stfts to check that the dataset is built as required.
        """
        s_l = ds_sig_noise.s_l.sel(
            x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest"
        )
        x_l = ds_sig_noise.x_l.sel(
            x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest"
        )
        n_l = ds_sig_noise.n_l.sel(
            x=ds_sig_noise.xs, y=ds_sig_noise.ys, method="nearest"
        )
        s_e = ds_sig_noise.s_e
        x_e = ds_sig_noise.x_e
        n_e = ds_sig_noise.n_e

        # Set stft params
        fs = 1 / ds_sig_noise.t.diff("t").values[0]
        nperseg = 2**8
        noverlap = nperseg // 2
        # Derive stfts
        ff, tt, s_l_stft = sp.stft(
            s_l.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
        )
        _, _, x_l_stft = sp.stft(
            x_l.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
        )
        _, _, n_l_stft = sp.stft(
            n_l.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
        )
        _, _, s_e_stft = sp.stft(
            s_e.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
        )
        _, _, x_e_stft = sp.stft(
            x_e.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
        )
        _, _, n_e_stft = sp.stft(
            n_e.values, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
        )

        # Normalize all stfts
        # s_l_stft = s_l_stft / np.max(np.abs(s_l_stft))
        # x_l_stft = x_l_stft / np.max(np.abs(x_l_stft))
        # n_l_stft = n_l_stft / np.max(np.abs(n_l_stft))
        # s_e_stft = s_e_stft / np.max(np.abs(s_e_stft))
        # x_e_stft = x_e_stft / np.max(np.abs(x_e_stft))
        # n_e_stft = n_e_stft / np.max(np.abs(n_e_stft))

        # Store stfts in xarray for plot facilities
        stft_ds = xr.Dataset(
            {
                # "s_l_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(s_l_stft))),
                # "x_l_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(x_l_stft))),
                # "n_l_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(n_l_stft))),
                # "s_e_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(s_e_stft))),
                # "x_e_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(x_e_stft))),
                # "n_e_stft": (["idx_rcv", "f", "t"], 20 * np.log10(np.abs(n_e_stft))),
                "s_l_stft": (["idx_rcv", "f", "t"], np.abs(s_l_stft)),
                "x_l_stft": (["idx_rcv", "f", "t"], np.abs(x_l_stft)),
                "n_l_stft": (["idx_rcv", "f", "t"], np.abs(n_l_stft)),
                "s_e_stft": (["idx_rcv", "f", "t"], np.abs(s_e_stft)),
                "x_e_stft": (["idx_rcv", "f", "t"], np.abs(x_e_stft)),
                "n_e_stft": (["idx_rcv", "f", "t"], np.abs(n_e_stft)),
            },
            coords={"idx_rcv": ds_sig_noise.idx_rcv.values, "f": ff, "t": tt},
        )

        img_check_path = os.path.join(ds_sig_noise.root_img, "check")
        if not os.path.exists(img_check_path):
            os.makedirs(img_check_path)

        cmap = "jet"
        # vmin = np.min([stft_ds[v].min() for v in list(stft_ds.keys())])
        # vmax = 0
        vmin = 0
        vmax = np.max([stft_ds[v].max() for v in list(stft_ds.keys())])

        for i_rcv in ds_sig_noise.idx_rcv.values:

            f, axs = plt.subplots(3, 2, figsize=(20, 12), sharex=True, sharey=True)

            # First column -> library
            # s_l.sel(idx_rcv=i_rcv).plot(ax=axs[0, 0])
            stft_ds.s_l_stft.sel(idx_rcv=i_rcv).plot(
                ax=axs[0, 0], cmap=cmap, vmin=vmin, vmax=vmax
            )
            axs[0, 0].set_title("$z(t)$")

            # n_l.sel(idx_rcv=i_rcv).plot(ax=axs[1, 0])
            stft_ds.n_l_stft.sel(idx_rcv=i_rcv).plot(
                ax=axs[1, 0], cmap=cmap, vmin=vmin, vmax=vmax
            )
            axs[1, 0].set_title("$v(t)$")

            # x_l.sel(idx_rcv=i_rcv).plot(ax=axs[2, 0])
            stft_ds.x_l_stft.sel(idx_rcv=i_rcv).plot(
                ax=axs[2, 0], cmap=cmap, vmin=vmin, vmax=vmax
            )
            axs[2, 0].set_title("$x(t) = z(t) + v(t)$")

            # Second column -> event
            # s_e.sel(idx_rcv=i_rcv).plot(ax=axs[0, 1])
            stft_ds.s_e_stft.sel(idx_rcv=i_rcv).plot(
                ax=axs[0, 1], cmap=cmap, vmin=vmin, vmax=vmax
            )
            axs[0, 1].set_title("$z(t)$")

            # n_e.sel(idx_rcv=i_rcv).plot(ax=axs[1, 1])
            stft_ds.n_e_stft.sel(idx_rcv=i_rcv).plot(
                ax=axs[1, 1], cmap=cmap, vmin=vmin, vmax=vmax
            )
            axs[1, 1].set_title("$v(t)$")

            # x_e.sel(idx_rcv=i_rcv).plot(ax=axs[2, 1])
            stft_ds.x_e_stft.sel(idx_rcv=i_rcv).plot(
                ax=axs[2, 1], cmap=cmap, vmin=vmin, vmax=vmax
            )
            axs[2, 1].set_title("$x(t) = z(t) + v(t)$")

            # Remove xlabel for row 0 and 1
            for irow in [0, 1]:
                for icol in [0, 1]:
                    axs[irow, icol].set_xlabel("")

            plt.suptitle(f"SNR = {ds_sig_noise.snr} dB")
            fpath = os.path.join(img_check_path, f"stft_sig_noise_ircv{i_rcv}.png")
            plt.savefig(fpath)

        plt.close("all")

    def build_features_fullsimu(
        self,
    ):
        """
        Step 4.1 : build localisation features for DCF GCC and RTF methods.
        Full simulation approach : DCF and RTF are build directly from transfer functions
        """

        # Load params
        dx = self.simulation.dx
        dy = self.simulation.dy

        # fpath = os.path.join(p.root_data, fname)
        ds_tf = xr.open_dataset(self.simulation.tf_grid_dataset_fpath)

        # Limit max frequency to speed up
        ds_tf = ds_tf.sel(f=slice(0, self.simulation.fs / 2))

        # Load library spectrum
        S_f_library = self.simulation.library_ship.spectrum
        f_library = self.simulation.library_ship.freq

        # Load event spectrum
        S_f_event = self.simulation.event_ship.spectrum

        # Restrict ds_tf, S_flibrary and S_f_event to the signal band [100, 500]
        idx_band = (f_library >= self.simulation.fmin) & (
            f_library <= self.simulation.fmax
        )

        ds_tf = ds_tf.sel(f=slice(self.simulation.fmin, self.simulation.fmax))
        S_f_library = S_f_library[idx_band]
        S_f_event = S_f_event[idx_band]

        # Subsample frequency to save memory
        subsample_idx = np.arange(0, ds_tf.sizes["f"])[::5]
        S_f_library = S_f_library[subsample_idx]
        S_f_event = S_f_event[subsample_idx]
        ds_tf = ds_tf.isel(f=subsample_idx)

        ### 1) Full simulation approach : rtf and gcc are derived directly from tfs ###

        # Init lists to save results
        rtf_event = []  # RFT vector at the source position
        rtf_library = []  # RTF vector evaluated at each grid pixel
        gcc_event = []  # GCC vector evaluated at the source position
        gcc_library = []  # GCC-SCOT vector evaluated at each grid pixel

        for i_ref in self.simulation.antenna.rcv_idx:

            tf_ref = ds_tf.tf_real.sel(idx_rcv=i_ref) + 1j * ds_tf.tf_imag.sel(
                idx_rcv=i_ref
            )
            tf_src_ref = tf_ref.sel(
                x=self.simulation.event_ship_x,
                y=self.simulation.event_ship_y,
                method="nearest",
            )

            # Received spectrum -> reference receiver
            y_ref = mult_along_axis(tf_ref.values, S_f_library, axis=0)
            y_ref_src = mult_along_axis(tf_src_ref.values, S_f_event, axis=0)

            # Power spectral density at each grid pixel associated to the reference receiver -> library
            Sxx_library_ref = y_ref * np.conj(y_ref)
            # Power spectral density at the source position associated to the reference receiver -> event
            Sxx_event_ref = y_ref_src * np.conj(y_ref_src)

            for i_rcv in self.simulation.antenna.rcv_idx:

                ## Kraken RTF ##
                tf_i = ds_tf.tf_real.sel(idx_rcv=i_rcv) + 1j * ds_tf.tf_imag.sel(
                    idx_rcv=i_rcv
                )
                rtf_i = tf_i.values / tf_ref.values
                rtf_i = rtf_i.reshape(
                    (ds_tf.sizes["f"], ds_tf.sizes["x"], ds_tf.sizes["y"])
                )
                rtf_library.append(rtf_i)

                # Source
                tf_src_i = tf_i.sel(
                    x=self.simulation.event_ship_x,
                    y=self.simulation.event_ship_y,
                    method="nearest",
                )
                rtf_event_i = tf_src_i.values / tf_src_ref.values
                rtf_event.append(rtf_event_i)

                ## GCC SCOT ##

                ## Grid -> library ##
                # Add the signal spectrum information
                y_i = mult_along_axis(tf_i.values, S_f_library, axis=0)

                # Power spectral density at each grid point associated to the receiver i
                Syy = y_i * np.conj(y_i)

                # Cross power spectral density between the reference receiver and receiver i
                Sxy = y_ref * np.conj(y_i)

                # Compute weights for GCC-SCOT
                w = 1 / np.abs(np.sqrt(Sxx_library_ref * Syy))
                # Apply GCC-SCOT
                gcc_library_i = w * Sxy
                gcc_library_i = gcc_library_i.reshape(
                    (ds_tf.sizes["f"], ds_tf.sizes["x"], ds_tf.sizes["y"])
                )
                gcc_library.append(gcc_library_i)

                ## Event source -> event ##
                y_src_i = mult_along_axis(tf_src_i.values, S_f_event, axis=0)

                # Power spectral density at the source position associated to the receiver i
                Syy_src = y_src_i * np.conj(y_src_i)

                # Cross power spectral density between reference receiver and receiver i at source position$
                Sxy_src = y_ref_src * np.conj(y_src_i)

                # Compute weights for GCC-SCOT
                w_src = 1 / np.abs(np.sqrt(Sxx_event_ref * Syy_src))
                # Apply GCC-SCOT
                gcc_event_i = w_src * Sxy_src
                gcc_event.append(gcc_event_i)

        # Read arrays sizes
        nf = ds_tf.sizes["f"]
        nx = ds_tf.sizes["x"]
        ny = ds_tf.sizes["y"]
        n_rcv_ref = ds_tf.sizes["idx_rcv"]
        n_rcv = ds_tf.sizes["idx_rcv"]

        # Set target shapes
        shape_event = (n_rcv_ref, n_rcv, nf)
        shape_library = (n_rcv_ref, n_rcv, nf, ny, nx)

        # RTF
        rtf_event = np.array(rtf_event).reshape(shape_event)
        rtf_event = np.moveaxis(rtf_event, 1, -1)  # (idx_rcv_ref, f, idx_rcv)
        rtf_library = np.array(rtf_library).reshape(shape_library)
        rtf_library = np.moveaxis(rtf_library, 1, -1)  # (idx_rcv_ref, f, y, x, idx_rcv)
        # Reshape to order x, y
        rtf_library = np.moveaxis(rtf_library, 2, 3)  # (idx_rcv_ref, f, x, y, idx_rcv)

        # GCC SCOT (idx_rcv_ref, f, x, y, idx_rcv)
        gcc_event = np.array(gcc_event).reshape(
            shape_event
        )  # (idx_rcv_ref, f, idx_rcv)
        gcc_event = np.moveaxis(gcc_event, 1, -1)
        gcc_library = np.array(gcc_library).reshape(shape_library)
        gcc_library = np.moveaxis(gcc_library, 1, -1)  # (idx_rcv_ref, f, y, x, idx_rcv)
        # Reshape to order x, y
        gcc_library = np.moveaxis(gcc_library, 2, 3)  # (idx_rcv_ref, f, y, x, idx_rcv)

        # Create dataset to store results
        ds_res_full_simu = xr.Dataset(
            data_vars=dict(
                rtf_event_real=(
                    ["idx_rcv_ref", "f_rtf", "idx_rcv"],
                    rtf_event.real,
                ),
                rtf_event_imag=(
                    ["idx_rcv_ref", "f_rtf", "idx_rcv"],
                    rtf_event.imag,
                ),
                gcc_event_real=(
                    ["idx_rcv_ref", "f_gcc", "idx_rcv"],
                    gcc_event.real,
                ),
                gcc_event_imag=(
                    ["idx_rcv_ref", "f_gcc", "idx_rcv"],
                    gcc_event.imag,
                ),
                rtf_real=(
                    ["idx_rcv_ref", "f_rtf", "x", "y", "idx_rcv"],
                    rtf_library.real,
                ),
                rtf_imag=(
                    ["idx_rcv_ref", "f_rtf", "x", "y", "idx_rcv"],
                    rtf_library.imag,
                ),
                gcc_real=(
                    ["idx_rcv_ref", "f_gcc", "x", "y", "idx_rcv"],
                    gcc_library.real,
                ),
                gcc_imag=(
                    ["idx_rcv_ref", "f_gcc", "x", "y", "idx_rcv"],
                    gcc_library.imag,
                ),
            ),
            coords=dict(
                x=ds_tf.x.values,
                y=ds_tf.y.values,
                idx_rcv=ds_tf.idx_rcv.values,
                idx_rcv_ref=ds_tf.idx_rcv.values,
                f_gcc=ds_tf.f.values,
                f_rtf=ds_tf.f.values,
            ),
            attrs=dict(
                xs=self.simulation.event_ship_x,
                ys=self.simulation.event_ship_y,
                snr=np.nan,
                dx=dx,
                dy=dy,
                root_img=os.path.join(p.root_img, f"fullsimu_dx{dx}m_dy{dy}m"),
            ),
        )

        # Save updated dataset
        ds_res_full_simu.to_netcdf(self.simulation.kraken_feature_dataset_fpath)
        ds_res_full_simu.close()


if __name__ == "__main__":
    # Class test
    from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna

    debug = True
    antenna = SparseAntenna(
        name="Test_sparse_antenna", n_elements=3, random_radius=5e3, rng_seed=42
    )
    simu = Simulation(debug=debug, antenna=antenna)
    fb = FeatureBuilder(simulation=simu)
    fb.build_features_from_time_signal(snr_dB=50)
    fb.build_features_fullsimu()
