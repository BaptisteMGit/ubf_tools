#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_passive_feature_manager.py
@Time    :   2026/10/28 17:07:08
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import os
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt

from scipy import stats
from datetime import datetime, timedelta
from scipy.signal import butter, lfilter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import source.global_constants as gc
import real_data_analysis.fiberscope_groix.src.params as p
from misc import progression_bar
from source.cov_manager import CovManager
from source.feature_processor import FeatureProcessor
from propa.rtf.rtf_utils import D_hermitian_angle_fast
from real_data_analysis.deconvolution_utils import (
    crosscorr_deconvolution,
    wiener_deconvolution,
)
from publication.publication_figure import (
    PubFigure,
    LargeFigure,
    color,
    set_subfigures_abc_labels,
)


# ======================================================================================================================
# Band filtering class
# ======================================================================================================================
class BandFilter:
    """
    Wrapping class to apply Butterworth filtering using scipy.signal.butter and scipy.signal.lfilter
    """

    def __init__(
        self,
        order: int = p.bandfilter_order,
        lowcut: float = p.bandfilter_lowcut,
        highcut: float = p.bandfilter_highcut,
    ):

        self.order = order
        self.lowcut = lowcut
        self.highcut = highcut

    def apply_filter(self, signal, fs):
        b, a = butter(self.order, [self.lowcut, self.highcut], fs=fs, btype="band")
        signal_filter = lfilter(b, a, signal)

        return signal_filter


class AcouplaneManager:
    """
    Mother class to apply RTF-MFP (or DCF-MFP) to Acouplane data
    """

    def __init__(
        self,
        ds_wav: xr.Dataset = None,
        root_processed_data: str = "",
        root_img: str = p.root_img,
        bandfilter: BandFilter = None,
        tau_ir: float = p.tau_ir,
        alpha_overlap: float = p.alpha_overlap,
        h_index_ref: int = p.h_index_ref,
        plot_feature: bool = False,
        theta_statistics: str = "mean",
        process_pulse_one_by_one: bool = True,
        estimate_ir_duration: bool = True,
        rtf_estimator: str = "cs-evd",
        obs_ids: list = [1, 2, 3],
        verbose: bool = False,
        plot_signal: bool = False,
        limit_frequency_band: bool = False,
    ):
        """
        Constructor
        """
        self.root_img = root_img
        self.root_img_sequence = os.path.join(self.root_img, "sequences")
        self.root_processed_data = root_processed_data
        self.root_data_sequence = os.path.join(self.root_processed_data, "sequences")

        if bandfilter is not None:
            self.apply_bandfilter = True
            self.bandfilter = bandfilter
        else:
            self.apply_bandfilter = False

        # Impulse response duration used to derived appropriate stft params
        self.tau_ir = tau_ir

        # Overlap factor used to derived appropriate stft params
        self.alpha_overlap = alpha_overlap

        # Hydrophone use as reference to derive rtf
        self.h_index_ref = h_index_ref

        self.plot_feature = plot_feature
        self.plot_csdm_mask = plot_feature

        # Method to derive caracteristic angle representing the theta distribution
        self.theta_statistics = theta_statistics

        # Process each pulse within the same sequence independently (in case constant source position within a single sequence does not hold)
        self.process_pulse_one_by_one = process_pulse_one_by_one

        # Estimate the impulse response duration online (from deconvolved waveguide response)
        self.estimate_ir_duration = estimate_ir_duration

        # RTF estimator to use
        self.rtf_estimator = rtf_estimator

        # Link to  wav dataset
        self.ds_wav = ds_wav
        self.datetime_fmt = self.ds_wav.attrs["datetime_format"]

        # Define usefull objects
        self.cm = CovManager()

        # OBS ids
        self.obs_ids = obs_ids

        # Verbose flag
        self.verbose = verbose

        # Flag to limit the frequency band used for the analysis to the one of the emitted signal (instead of using the entire available band)
        self.limit_frequency_band = limit_frequency_band

        # Plot audio signal flag
        self.plot_signal = plot_signal

    def set_stft_params(self, ts):
        """
        Set STFT params as the power of 2 closest to the provided impulse response duration self.tau_ir
        The objective is to ensure that the multiplicative transfer function (MTF) model holds.

        Parameters
        ----------
        ts : float
            Sampling period (s).

        Returns
        -------
        None
        """

        n_ir = int(
            self.tau_ir / ts
        )  # Number of samples corresponding to the assumed impulse response duration
        # #
        # nperseg = n_ir  # Number of sample per snapshot to use = closest power of two
        # noverlap = int(nperseg * self.alpha_overlap)

        # Get closer power of 2
        nperseg = 2 ** int(
            np.log2(n_ir) + 1
        )  # Number of sample per snapshot to use = closest power of two
        noverlap = int(nperseg * self.alpha_overlap)

        self.nperseg = nperseg
        self.noverlap = noverlap

        if self.verbose:
            print(f"STFT params set using tau_ir = {self.tau_ir} s")
            print(f"nperseg = {self.nperseg}, noverlap = {self.noverlap}")

    def set_managers(self, fs, idx_rcv_ref):
        """
        Initialise CovManager and FeatureProcessor classes used to derive RTF.

        Parameters
        ----------
        fs : float
            Sampling frequency (Hz).
        idx_rcv_ref : int
            Index of the receiver to use as reference.

        Returns
        -------
        None
        """
        # Define covariance manager with the right stft params
        self.cm = CovManager(nperseg=self.nperseg, noverlap=self.noverlap)

        # Define feature processor with the right stft params
        self.fp = FeatureProcessor(
            fs=fs,
            idx_rcv_ref=idx_rcv_ref,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            window="hann",
        )  # Feature processor to handle rtf_estimator and gcc_estimator

    def estimate_global_csdm(self, xr_data):
        pass

    def get_signal_presence_mask(self, x, fs, nperseg, noverlap):
        pass

    def plot_audio_signal(xr_data):
        pass

    def process_analysis(self):
        pass

    def load_data(self):
        pass

    def derive_feature(self):
        pass

    def plot_estimated_feature(self):
        pass

    def get_rtf(self):
        pass


# ======================================================================================================================
#  Passive source localisation
# ======================================================================================================================
class PassiveAcouplaneManager(AcouplaneManager):
    """
    Class to handle passive source localisation. The source to localise is typically a ship of opportunity.
    """

    def __init__(
        self,
        ds_wav: xr.Dataset = None,
        root_processed_data: str = "",
        root_img: str = p.root_img,
        # bandfilter: BandFilter = None,
        # tau_ir: float = p.tau_ir,
        alpha_overlap: float = p.alpha_overlap,
        h_index_ref: int = p.h_index_ref,
        plot_feature: bool = False,
        theta_statistics: str = "mean",
        # process_pulse_one_by_one: bool = True,
        # estimate_ir_duration: bool = True,
        rtf_estimator: str = "cs-evd",
        obs_ids: list = [1, 2, 3],
        verbose: bool = False,
        plot_signal: bool = False,
        analysis_segment_duration: float = 10,
        analysis_segment_alpha_overlap: float = 0.5,
    ):
        """
        Class constructor

        """
        # Using super() to initialize the parent class
        super().__init__(
            ds_wav=ds_wav,
            root_processed_data=root_processed_data,
            root_img=root_img,
            bandfilter=None,
            tau_ir=None,
            alpha_overlap=alpha_overlap,
            h_index_ref=h_index_ref,
            plot_feature=plot_feature,
            theta_statistics=theta_statistics,
            process_pulse_one_by_one=False,
            estimate_ir_duration=False,
            rtf_estimator=rtf_estimator,
            obs_ids=obs_ids,
            verbose=verbose,
            plot_signal=plot_signal,
        )

        self.analysis_segment_duration = analysis_segment_duration
        self.analysis_segment_alpha_overlap = analysis_segment_alpha_overlap

    def process_analysis(
        self,
        t_start,
        t_end,
        set_stft_props=True,
        Rv_global: np.ndarray = None,
    ):
        """
        Run analysis on the reduired recording analysis window.
        """

        if self.verbose:
            print(f"\nRTF processing of passive recording")

        ### Step 1 - Load audio data for the required analysis window ###
        xr_data = self.load_recording(t_start, t_end)

        ### Step 2 - Init CovManager and FeatureProcessor ###
        # If stfts props are already set we dont need to do it
        if set_stft_props:
            self.set_stft_params(ts=xr_data.ts)
        # Index of hydrophone might not be sorted or not start at 0
        idx_rcv_ref = np.argmin(np.abs(xr_data.h_index.values - self.h_index_ref))
        # Init managers
        self.set_managers(fs=xr_data.fs, idx_rcv_ref=idx_rcv_ref)

        ### Step 3 - Derive features ###
        self.derive_feature(xr_data, Rv_global=Rv_global)

    def load_recording(self, t_start, t_end):
        """
        Load data for the required analysis window.

        Parameters
        ----------
        t_start : datetime.datetime
            Start of the analysis window.
        t_end datetime.datetime
            End of the analysis window.

        Returns
        -------
        xr_data : xr.Dataset
            Selected portion of wav data for the required analysis window (form t_start to t_end).
        """

        ds_wav = self.ds_wav.copy()  # To avoid modifying original dataset
        datetime_fmt = ds_wav.attrs["datetime_format"]

        for i, obs_id in enumerate(self.obs_ids):

            # Name of the time coords in ds_wav
            time_coordsname = f"time{obs_id}"

            # Select a window of the signal
            fs = ds_wav.attrs[f"fs_obs{obs_id}"]

            # Start of recording
            t0 = ds_wav.attrs[f"start_datetime_obs{obs_id}"]
            t0 = datetime.strptime(t0, datetime_fmt)

            # Select the required window
            t_from_t0_start_s = (t_start - t0).total_seconds()
            n_start = int(t_from_t0_start_s * fs)
            t_from_t0_end_s = (t_end - t0).total_seconds()
            n_end = int(t_from_t0_end_s * fs)

            # Slice signal for current OBS
            ds_wav = ds_wav.isel({time_coordsname: slice(n_start, n_end)})

        # Reshape
        signal_mat = np.vstack([ds_wav[f"signal_obs{i}"].values for i in self.obs_ids])
        # Set common time vector
        common_time_vector = np.arange(ds_wav.sizes["time1"]) * 1 / fs

        # Define a record_id to be used to save results
        record_id = f"passive_{datetime.strftime(t_start, datetime_fmt)}_to_{datetime.strftime(t_end, datetime_fmt)}"

        # Build dataset
        xr_data = xr.Dataset(
            data_vars=dict(
                signal=(["h_index", "time"], signal_mat.astype(np.float32)),
                start_dt=t_start,
                end_dt=t_end,
            ),
            coords=dict(
                h_index=self.obs_ids,
                time=common_time_vector.astype(np.float32),
            ),
            attrs=dict(
                fs=fs,
                ts=1 / fs,
                datetime_format=self.datetime_fmt,
                # t_start=t_start.strftime(self.datetime_fmt),
                # t_end=t_end.strftime(self.datetime_fmt),
                record_id=record_id,
                root_img=os.path.join(self.root_img_sequence, record_id),
                root_data=os.path.join(self.root_data_sequence, "passive"),
            ),
        )

        # Ensure folders exists
        if not os.path.exists(xr_data.root_img):
            os.makedirs(xr_data.root_img)
        if not os.path.exists(xr_data.root_data):
            os.makedirs(xr_data.root_data)

        if self.plot_signal:
            self.plot_audio_signal(xr_data)

        return xr_data

    def plot_audio_signal(xr_data, arrivals_datetimes, nperseg=256, noverlap=128):
        # TODO : define for passive
        pass

    def derive_feature(
        self,
        xr_data,
        Rv_global: np.ndarray = None,
        save=True,
    ):
        """
        Derive RTF for each segment of the selected analysis window.

        Parameters
        ----------
        xr_data : xr.Dataset
            Wav data for the selected period of recording to analyse as provided by the load_recording method.
        Rv_global : np.array
            Noise covariance matrix to use. If None (default), the noise is neglected and Rv is set to 0.
        save : bool
            Save data to netcdf.

        Returns
        -------
        xr_data : xr.Dataset
            RTF dataset for the selected portion of data.
        """

        if self.verbose:
            print(f"RTF feature estimation...")

        # Derive rtf from recordings
        xr_data = self.get_rtf(xr_data=xr_data, Rv_global=Rv_global)

        # Slice along frequency axis to ensure we never use information outside of the signal bandwidth
        # This also reduce the memory size required
        # xr_data = xr_data.sel(f_rtf=slice(xr_data.fmin, xr_data.fmax))
        # xr_data = xr_data.sel(f_ir=slice(xr_data.fmin, xr_data.fmax))
        # xr_data = xr_data.sel(f_csdm=slice(xr_data.fmin, xr_data.fmax))

        # Plot feature components for analysis if required
        if self.plot_feature:
            self.plot_estimated_feature(xr_data)

        # Save results
        if save:
            xr_data.to_netcdf(
                os.path.join(xr_data.root_data, f"sequence_{xr_data.record_id}_rtf.nc")
            )
            xr_data.close()
        else:
            return xr_data

    def plot_estimated_feature(self, xr_data):

        # Ensure img folder exists
        if not os.path.exists(xr_data.root_img):
            os.makedirs(xr_data.root_img)

        for segment_dt in xr_data.segment_dt.values:
            xr_data_seg = xr_data.sel(segment_dt=segment_dt)

            segment_dt_str = datetime.strftime(
                pd.to_datetime(segment_dt).to_pydatetime(), "%H%M%S"
            )

            # for segment_id in xr_data.segment_id.values:
            #     xr_data_seg = xr_data.sel(segment_id=segment_id)

            nrcv = xr_data.sizes["h_index"]
            f_amp, axs_amp = plt.subplots(nrows=nrcv, ncols=1, sharex=True)
            f_phase, axs_phase = plt.subplots(nrows=nrcv, ncols=1, sharex=True)
            i = 0
            for rcv_idx in xr_data.h_index.values:

                # Plot RTF amplitude
                max_amp = xr_data_seg.rtf_amp_hat.max() * 1.2
                min_amp = xr_data_seg.rtf_amp_hat.min() * 0.8
                # xr_data_seg.rtf_amp.sel(h_index=rcv_idx).plot(
                #     ax=axs_amp[i], color="k", label=f"Ref - {rcv_idx}"
                # )
                xr_data_seg.rtf_amp_hat.sel(h_index=rcv_idx).plot(
                    ax=axs_amp[i],
                    color="k",
                    marker="o",
                    markersize=1,
                    linewidth=1,
                    linestyle="-",
                    label=f"{self.rtf_estimator.upper()} - {rcv_idx}",
                )
                axs_amp[i].set_xlabel("")
                axs_amp[i].set_ylabel(r"$|\Pi|$")
                axs_amp[i].set_ylim(min_amp, max_amp)
                axs_amp[i].set_yscale("log")
                axs_amp[i].set_title("")
                axs_amp[i].legend(fontsize=8)

                # Plot RTF phase
                # xr_data_seg.rtf_phase.sel(h_index=rcv_idx).plot(
                #     ax=axs_phase[i], color="k", label=f"Ref - {rcv_idx}"
                # )
                xr_data_seg.rtf_phase_hat.sel(h_index=rcv_idx).plot(
                    ax=axs_phase[i],
                    color="k",
                    marker="o",
                    markersize=1,
                    linewidth=1,
                    linestyle="-",
                    label=f"{self.rtf_estimator.upper()} - {rcv_idx}",
                )
                axs_phase[i].set_xlabel("")
                axs_phase[i].set_ylabel(r"$\Phi$")
                axs_phase[i].set_title("")
                axs_phase[i].legend(fontsize=8)

                i += 1

            # Save figures
            fpath = os.path.join(xr_data.root_img, f"rtf_amp_{segment_dt_str}.png")
            f_amp.savefig(fpath)
            fpath = os.path.join(xr_data.root_img, f"rtf_phase_{segment_dt_str}.png")
            f_phase.savefig(fpath)

            plt.close("all")

            # Plot csdms (noise, noisy signal, signal)
            f_csdm, axs_csdm = plt.subplots(nrows=1, ncols=3, sharey=True)
            f_csdm.suptitle("CSDM")

            # Mean CSDMs
            mean_Rx = xr_data_seg.Rx.mean(dim="f_csdm")
            mean_Rv = xr_data_seg.Rv.mean(dim="f_csdm")
            Rs = xr_data_seg.Rx - xr_data_seg.Rv
            mean_Rs = Rs.mean(dim="f_csdm")

            # Derive a common vmax for comparison purpose
            vmax = max(mean_Rx.values.max(), mean_Rv.values.max())

            # Plot Rx
            mean_Rx.plot(ax=axs_csdm[0], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[0].set_title(r"$\hat{R}_x$")
            axs_csdm[0].set_xlabel("Index")
            axs_csdm[0].set_ylabel("Index")
            # Ticks
            axs_csdm[0].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[0].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rv
            mean_Rv.plot(ax=axs_csdm[1], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[1].set_title(r"$\hat{R}_v$")
            axs_csdm[1].set_xlabel("Index")
            axs_csdm[1].set_ylabel("Index")
            # Ticks
            axs_csdm[1].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[1].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rs
            mean_Rs.plot(ax=axs_csdm[2], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[2].set_title(r"$\hat{R}_s = \hat{R}_x - \hat{R}_v$")
            axs_csdm[2].set_xlabel("Index")
            axs_csdm[2].set_ylabel("Index")
            # Ticks
            axs_csdm[2].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[2].set_yticks(np.arange(1, nrcv + 1, 1))

            # Save figure
            fpath = os.path.join(
                xr_data.root_img, f"estimated_csdms_{segment_dt_str}.png"
            )
            f_csdm.savefig(fpath)

            plt.close("all")

            # Plot csdms (noise, noisy signal, signal)
            f_csdm, axs_csdm = plt.subplots(nrows=1, ncols=3, sharey=True)

            # CSDMs at a center freq
            fc = (xr_data.f_rtf.max().values - xr_data.f_rtf.min().values) / 2
            f_csdm.suptitle(f"CSDM (f = {fc} Hz)")

            Rx = xr_data_seg.Rx.sel(f_csdm=fc, method="nearest")
            Rv = xr_data_seg.Rv.sel(f_csdm=fc, method="nearest")
            Rs = Rx - Rv

            # Derive a common vmax for comparison purpose
            # vmax = max(mean_Rx.values.max(), mean_Rv.values.max())
            vmax = max(Rx.values.max(), Rv.values.max())

            # Plot Rx
            Rx.plot(ax=axs_csdm[0], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[0].set_title(r"$\hat{R}_x$")
            axs_csdm[0].set_xlabel("Index")
            axs_csdm[0].set_ylabel("Index")
            # Ticks
            axs_csdm[0].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[0].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rv
            Rv.plot(ax=axs_csdm[1], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[1].set_title(r"$\hat{R}_v$")
            axs_csdm[1].set_xlabel("Index")
            axs_csdm[1].set_ylabel("Index")
            # Ticks
            axs_csdm[1].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[1].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rs
            Rs.plot(ax=axs_csdm[2], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[2].set_title(r"$\hat{R}_s = \hat{R}_x - \hat{R}_v$")
            axs_csdm[2].set_xlabel("Index")
            axs_csdm[2].set_ylabel("Index")
            # Ticks
            axs_csdm[2].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[2].set_yticks(np.arange(1, nrcv + 1, 1))

            # Save figure
            fpath = os.path.join(
                xr_data.root_img,
                f"estimated_csdms_{segment_dt_str}_f_{Rx.f_csdm.values:.1f}Hz.png",
            )
            f_csdm.savefig(fpath)

            plt.close("all")

    def get_rtf(self, xr_data, Rv_global=None):
        """
        Derive RTF for each segment of the selected analysis window.

        Parameters
        ----------
        xr_data : xr.Dataset
            Wav data for the selected period of recording to analyse as provided by the load_recording method.
        Rv_global : np.array
            Noise covariance matrix to use. If None (default), the noise is neglected and Rv is set to 0.

        Returns
        -------
        xr_data : xr.Dataset
            RTF dataset for the selected portion of data.

        """

        ts = xr_data.ts
        init_arr = True

        window_shift = self.analysis_segment_duration * (
            1 - self.analysis_segment_alpha_overlap
        )
        n_window = int(
            np.floor(
                (
                    xr_data.time.max().values
                    - self.analysis_segment_duration
                    * self.analysis_segment_alpha_overlap
                )
                / window_shift
            )
        )

        tstart = 0
        tend = self.analysis_segment_duration

        prev_progress = 0

        # Process sucessive windows
        for i_window in range(n_window):

            prev_progress = progression_bar(
                index=i_window + 1,
                index0=0,
                indexf=n_window,
                prev_progress=prev_progress,
            )

            # Select the corresponding time window
            x = xr_data.sel(time=slice(tstart, tend))
            tstart += window_shift
            tend += window_shift
            # print(x.time.values)

            x = x.signal.T
            f, Rx = self.cm.get_signal_csdm(
                y=x,
                fs=1 / ts,
                add_identity=False,
                mask_tt=None,
                mask_stft=None,
            )

            if Rv_global is not None:
                Rv = Rv_global
            else:
                # For continous signal we assume the noise to be negligible
                Rv = np.zeros_like(Rx)

            if self.rtf_estimator == "cs":
                rtf = self.fp.rtf_estimator.estimate_rtf_covariance_subtraction(
                    Rx - Rv, use_first_column=True
                )
            elif self.rtf_estimator == "cs-evd":
                rtf = self.fp.rtf_estimator.estimate_rtf_covariance_subtraction(
                    Rx - Rv, use_first_column=False
                )
            # elif self.rtf_estimator == "cw":
            #     rtf = self.fp.rtf_estimator.estimate_rtf_covariance_whitening(
            #         Rx, Rv
            #     )
            else:
                print(f"{self.rtf_estimator} not implemented yet!")

            if init_arr:
                n_rcv = xr_data.sizes["h_index"]
                nf = f.size

                rtf_hat = np.empty(
                    (n_rcv, nf, n_window),
                    dtype=complex,
                )
                Rx_hat = np.empty(
                    (nf, n_rcv, n_rcv, n_window),
                    dtype=complex,
                )
                Rv_hat = np.empty(
                    (nf, n_rcv, n_rcv, n_window),
                    dtype=complex,
                )
                init_arr = False

            rtf_hat[..., i_window] = rtf.T
            Rx_hat[..., i_window] = Rx
            Rv_hat[..., i_window] = Rv

        # Set new coords
        xr_data.coords["f_rtf"] = f.astype(np.float32)
        xr_data.coords["f_csdm"] = f.astype(np.float32)
        # Create h_index bis to avoid duplicate coordinates
        xr_data.coords["h_index_bis"] = xr_data.h_index.values

        # Define time coordinate as the center of each segment analysed
        segment_ids = np.arange(n_window)
        segment_dt = []
        for segment_id in segment_ids:
            t_end_segment = self.analysis_segment_duration * (
                1 + (segment_id - 1) * (1 - self.analysis_segment_alpha_overlap)
            )
            t_centre_segment_s = t_end_segment - 0.5 * self.analysis_segment_duration
            # t_centre_segment = pd.to_datetime(
            #     xr_data.start_dt.values
            # ).to_pydatetime() + timedelta(seconds=t_centre_segment_s)

            # Round to ms to use np.timedelta64 and avoid convertion of start_dt
            t_centre_segment_ms = np.round(t_centre_segment_s, 3) * 1000
            t_centre_segment = xr_data.start_dt.values + np.timedelta64(
                int(t_centre_segment_ms), "ms"
            )

            segment_dt.append(t_centre_segment)

        segment_dt = np.array(segment_dt)

        # xr_data.coords["segment_id"] = np.arange(n_window)
        xr_data.coords["segment_dt"] = segment_dt
        xr_data["segment_dt"].attrs = {
            "long_name": "Time analysis segment (UTC)",
            "description": "Time corresponding to the center of each analysis segment",
        }

        # Add variables
        xr_data["rtf_amp_hat"] = (
            ["h_index", "f_rtf", "segment_dt"],
            np.abs(rtf_hat).astype(np.float32),
        )
        xr_data["rtf_phase_hat"] = (
            ["h_index", "f_rtf", "segment_dt"],
            np.angle(rtf_hat).astype(np.float32),
        )
        xr_data.attrs["h_index_ref"] = self.h_index_ref

        # Add Rx and R_v to the dataset
        xr_data["Rx"] = (
            ["f_csdm", "h_index", "h_index_bis", "segment_dt"],
            np.abs(Rx_hat).astype(np.float32),
        )
        xr_data["Rv"] = (
            ["f_csdm", "h_index", "h_index_bis", "segment_dt"],
            np.abs(Rv_hat).astype(np.float32),
        )

        # Add attributes to keep track of the analysis parameters
        xr_data.attrs["analysis_segment_duration"] = self.analysis_segment_duration
        xr_data.attrs["analysis_segment_alpha_overlap"] = (
            self.analysis_segment_alpha_overlap
        )

        return xr_data


def get_theta_c(val, apply_mean):
    # We dont have anything to do we can store the mean value directly
    if apply_mean:
        theta_c = val

    # We need to derive expectation
    else:
        # Step 1: estimate the probability density function associate to the observed distribution
        kde = stats.gaussian_kde(val)
        # Step 2: derive expectation    (note: kde.evaluate(x) is 10 times faster than kde.pdf(x))
        expectation = np.sum(val * kde.evaluate(val)) / np.sum(kde.evaluate(val))
        theta_c = expectation

    return theta_c
