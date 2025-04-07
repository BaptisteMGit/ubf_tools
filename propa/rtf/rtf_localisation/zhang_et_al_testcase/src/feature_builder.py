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
from time import time

import propa.rtf.rtf_localisation.zhang_et_al_testcase.src.params as p
from source.feature_processor import FeatureProcessor
from propa.rtf.rtf_localisation.zhang_et_al_testcase.src.data_builder import DataBuilder

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
        root_name: str = "zhang_output_from_signal",
        root_data: str = p.root_data,
        root_tmp: str = p.root_tmp,
        fmin: float = p.fmin,
        fmax: float = p.fmax,
        fs: float = p.fs,
        signal_duration: float = p.duration,
        antenna_type: str = p.antenna_type,
        library_stype: str = p.library_stype,
        event_stype: str = p.event_stype,
        rtf_method: str = p.rtf_method,
        gcc_method: str = p.gcc_method,
        verbose: bool = False,
        debug: bool = False,
        check: bool = False,
        use_dask: bool = True,
    ):
        """
        Init the class.
        """
        self.root_name = root_name
        self.root_data = root_data
        self.root_tmp = root_tmp

        self.fmin = fmin
        self.fmax = fmax
        self.fs = fs
        self.signal_duration = signal_duration

        self.antenna_type = antenna_type
        self.library_stype = library_stype
        self.event_stype = event_stype
        self.rtf_method = rtf_method
        self.gcc_method = gcc_method

        self.verbose = verbose
        self.debug = debug
        self.check = check
        self.use_dask = use_dask

        self.data_builder = DataBuilder(
            root_tmp=self.root_tmp,
            root_data=self.root_data,
            fmin=self.fmin,
            fmax=self.fmax,
            fs=self.fs,
            signal_duration=self.signal_duration,
            antenna_type=self.antenna_type,
            library_stype=self.library_stype,
            event_stype=self.event_stype,
            debug=self.debug,
        )

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
        _, receivers, source, grid, _, _ = params(
            debug=self.debug, antenna_type=self.antenna_type
        )
        dx = grid["dx"]
        dy = grid["dy"]

        # Dataset with time signal (for realistic approach)
        if self.debug:
            fname = f"zhang_library_dx{dx}m_dy{dy}m_debug.nc"
        else:
            fname = f"zhang_library_dx{dx}m_dy{dy}m.nc"

        fpath = os.path.join(p.root_data, fname)
        ds_sig = xr.open_dataset(fpath)
        attrs = ds_sig.attrs

        # Derive noise dataset
        ds_noise = self.data_builder.derive_received_noise(
            s_library=ds_sig.s_l,
            s_event=ds_sig.s_e,
            event_source=source,
            snr_dB=snr_dB,
        )  # (nr, nt, ny, nx)

        # NOTE : different noise dataset to simulate the fact that in real life the noise CSDM is
        # estimated from a different segment of the signal than the signal + noise CSDM (assuming noise is stationnary)
        ds_noise_bis = self.data_builder.derive_received_noise(
            s_library=ds_sig.s_l,
            s_event=ds_sig.s_e,
            event_source=source,
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
                    xs=source["x"],
                    ys=source["y"],
                    dx=dx,
                    dy=dy,
                    root_img=os.path.join(
                        p.root_img,
                        f"from_signal_dx{dx}m_dy{dy}m",
                        f"snr_{snr_dB:.1f}dB",
                    ),
                ),
            )

            check_signal_noise(ds_sig_noise)
            check_signal_noise_stft(ds_sig_noise)

        # List of potential reference receivers to test
        idx_rcv_refs = range(
            len(receivers["x"])
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
            fs=self.fs, idx_rcv_ref=0, nperseg=nperseg, noverlap=noverlap, window="hann"
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
            idx_band = (ff >= self.fmin) & (ff <= self.fmax)
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
            idx_band = (ff >= self.fmin) & (ff <= self.fmax)
            gcc_library = gcc_library[:, :, idx_band, ...]
            gcc_event = gcc_event[:, :, idx_band]
            f_gcc = ff[idx_band]

            # Clean up memory
            del xl_4D, vl_4D, xe_4D, ve_4D, gcc_dask, rtf_dask
            gc.collect()

        else:
            ### RTF ###
            # t0 = time()
            xl_4D = ds_feature_estimate.x_l.values
            vl_4D = ds_feature_estimate.n_l_bis.values

            ff, rtf_library = (
                fp.rtf_estimator.covariance_subtraction_major_eigen_vector_4D(
                    x_4D=xl_4D,
                    v_4D=vl_4D,
                    idx_rcv_refs=idx_rcv_refs,
                )
            )

            # We will use the same 4D function for the event signal so we need to add dimensions (single positions x, y)
            xe_4D = ds_feature_estimate.x_e.values[..., np.newaxis, np.newaxis]
            ve_4D = ds_feature_estimate.n_e_bis.values[..., np.newaxis, np.newaxis]
            ff, rtf_event = (
                fp.rtf_estimator.covariance_subtraction_major_eigen_vector_4D(
                    x_4D=xe_4D,
                    v_4D=ve_4D,
                    idx_rcv_refs=idx_rcv_refs,
                )
            )

            # Squeeze to remove dummy dimensions
            rtf_event = np.squeeze(rtf_event)

            # Restict to the frequency band of interest
            idx_band = (ff >= self.fmin) & (ff <= self.fmax)
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
            idx_band = (ff >= self.fmin) & (ff <= self.fmax)
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
                xs=source["x"],
                ys=source["y"],
                root_img=os.path.join(
                    p.root_img, f"from_signal_dx{dx}m_dy{dy}m", f"snr_{snr_dB:.1f}dB"
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

        # Subsample frequency to save memory
        # subsample_idx = np.arange(0, ds_res_from_sig.sizes["f"])[::5]
        # ds_res_from_sig = ds_res_from_sig.isel(f=subsample_idx)

        if self.check:
            check_rtf_features(
                ds_res_from_sig, folder=ds_sig_noise.attrs["root_img"], debug=self.debug
            )
            check_gcc_features(ds_res_from_sig, folder=ds_sig_noise.attrs["root_img"])

        # Save updated dataset
        fpath = os.path.join(
            p.root_data,
            f"{self.root_name}_dx{grid['dx']}m_dy{grid['dy']}m_snr{snr_dB:.1f}dB.nc",
        )
        ds_res_from_sig.to_netcdf(fpath)
        ds_res_from_sig.close()

        print(f"Features derived from time signal in {time() - t_start:.2f} s")

    def build_features_fullsimu(
        self,
    ):
        """
        Step 4.1 : build localisation features for DCF GCC and RTF methods.
        Full simulation approach : DCF and RTF are build directly from transfer functions
        """

        # Load params
        depth, receivers, source, grid, frequency, _ = params(
            debug=self.debug, antenna_type=self.antenna_type
        )
        dx = grid["dx"]
        dy = grid["dy"]

        # Dataset with gridded tf (for full simulation)
        if self.debug:
            fname = f"tf_zhang_grid_dx{dx}m_dy{dy}m_debug.nc"
        else:
            fname = f"tf_zhang_grid_dx{dx}m_dy{dy}m.nc"

        fpath = os.path.join(p.root_data, fname)
        ds_tf = xr.open_dataset(fpath)

        # Limit max frequency to speed up
        ds_tf = ds_tf.sel(f=slice(0, self.fs / 2))

        # Load library spectrum
        library_props, S_f_library, f_library, _ = library_src_spectrum(
            stype=self.library_stype,
            fs=self.fs,
            fmin=self.fmin,
            fmax=self.fmax,
            T=self.signal_duration,
            plot=False,
        )

        # Load event spectrum
        _, S_f_event, _ = event_src_spectrum(
            stype=self.event_stype,
            fs=self.fs,
            fmin=self.fmin,
            fmax=self.fmax,
            T=self.signal_duration,
            plot=False,
        )

        # Restrict ds_tf, S_flibrary and S_f_event to the signal band [100, 500]
        idx_band = (f_library >= self.fmin) & (f_library <= self.fmax)

        ds_tf = ds_tf.sel(f=slice(self.fmin, self.fmax))
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

        for i_ref in range(len(receivers["x"])):

            tf_ref = ds_tf.tf_real.sel(idx_rcv=i_ref) + 1j * ds_tf.tf_imag.sel(
                idx_rcv=i_ref
            )
            tf_src_ref = tf_ref.sel(x=source["x"], y=source["y"], method="nearest")

            # Received spectrum -> reference receiver
            y_ref = mult_along_axis(tf_ref.values, S_f_library, axis=0)
            y_ref_src = mult_along_axis(tf_src_ref.values, S_f_event, axis=0)

            # Power spectral density at each grid pixel associated to the reference receiver -> library
            Sxx_library_ref = y_ref * np.conj(y_ref)
            # Power spectral density at the source position associated to the reference receiver -> event
            Sxx_event_ref = y_ref_src * np.conj(y_ref_src)

            for i_rcv in range(len(receivers["x"])):

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
                tf_src_i = tf_i.sel(x=source["x"], y=source["y"], method="nearest")
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
                xs=source["x"],
                ys=source["y"],
                snr=np.nan,
                dx=dx,
                dy=dy,
                root_img=os.path.join(p.root_img, f"fullsimu_dx{dx}m_dy{dy}m"),
            ),
        )

        # Save updated dataset
        fpath = os.path.join(
            p.root_data, f"zhang_output_fullsimu_dx{grid['dx']}m_dy{grid['dy']}m.nc"
        )
        ds_res_full_simu.to_netcdf(fpath)
        ds_res_full_simu.close()


if __name__ == "__main__":
    debug = True
    fb = FeatureBuilder(debug=debug)

    fb.build_features_from_time_signal(snr_dB=50)
