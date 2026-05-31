#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   fiberscope_manager.py
@Time    :   2025/04/30 16:23:22
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Classs to manage Fiberscope data analysis
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
import matplotlib.dates as mdates

from scipy import stats
from datetime import datetime
from scipy.signal import butter, lfilter

from misc import progression_bar
from source.cov_manager import CovManager
from source.feature_processor import FeatureProcessor

import real_data_analysis.fiberscope_groix.src.params as p


# ======================================================================================================================
# Band filtering class
# ======================================================================================================================
class BandFilter:
    """
    Wrapping class to apply Butterworth filtering using scipy.signal.butter and scipy.signal.lfilter
    """

    def __init__(
        self,
        order: int = 4,
        lowcut: float = 1,
        highcut: float = 50,
    ):

        self.order = order
        self.lowcut = lowcut
        self.highcut = highcut

    def apply_filter(self, signal, fs):
        b, a = butter(self.order, [self.lowcut, self.highcut], fs=fs, btype="band")
        signal_filter = lfilter(b, a, signal)

        return signal_filter


class FeatureManager:
    """
    Mother class to apply RTF-MFP to Fiberscope data
    """

    def __init__(
        self,
        ds_wav: xr.Dataset = None,
        root_img: str = "",
        root_processed_data: str = "",
        receiver_ids: list = [1, 2, 3],
        reference_receiver_id: int = 1,
        analysis_segment_duration: float = 10,
        analysis_segment_alpha_overlap: float = 0.5,
        theta_statistics: str = "mean",
        rtf_estimator: str = "cs-evd",
        verbose: bool = False,
        plot_feature: bool = False,
    ):
        """
        Constructor
        """
        self.root_img = root_img
        self.root_img_sequence = os.path.join(self.root_img, "sequences")
        self.root_processed_data = root_processed_data
        self.root_data_sequence = os.path.join(self.root_processed_data, "sequences")

        # Hydrophone use as reference to derive rtf
        self.reference_receiver_id = reference_receiver_id

        self.plot_feature = plot_feature
        self.plot_csdm_mask = plot_feature

        # Method to derive caracteristic angle representing the theta distribution
        self.theta_statistics = theta_statistics

        # RTF estimator to use
        self.rtf_estimator = rtf_estimator

        self.analysis_segment_duration = analysis_segment_duration
        self.analysis_segment_alpha_overlap = analysis_segment_alpha_overlap

        # Link to  wav dataset
        self.ds_wav = ds_wav
        # self.datetime_fmt = self.ds_wav.attrs["datetime_format"]

        # Define usefull objects
        self.cm = CovManager()

        # OBS ids
        self.receiver_ids = receiver_ids

        # Verbose flag
        self.verbose = verbose

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
        x = xr_data.signal.T
        ts = xr_data.ts

        # Derive mask
        tt, mask_tt_x = self.get_signal_presence_mask(
            x, fs=1 / xr_data.ts, nperseg=self.nperseg, noverlap=self.noverlap
        )
        mask_tt_v = ~mask_tt_x

        # Derive global noise csdm
        ff, Rv_global = self.cm.get_signal_csdm(y=x, fs=1 / ts, mask_tt=mask_tt_v)

        return ff, tt, Rv_global

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
        idx_rcv_ref = np.argmin(
            np.abs(xr_data.h_index.values - self.reference_receiver_id)
        )
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
        fs = ds_wav.fs

        for i, obs_id in enumerate(self.receiver_ids):

            rcv_id = f"RCV{obs_id:02d}"
            # Name of the time coords in ds_wav
            time_coordsname = f"time_{rcv_id}"

            # Select a window of the signal

            # Start of recording
            t0 = ds_wav.start_datetimes.sel(receiver_id=rcv_id).values
            t0 = pd.to_datetime(t0).to_pydatetime()  # Convert to datetime

            # Select the required window
            t_from_t0_start_s = (t_start - t0).total_seconds()
            n_start = int(t_from_t0_start_s * fs)
            t_from_t0_end_s = (t_end - t0).total_seconds()
            n_end = int(t_from_t0_end_s * fs)

            # Slice signal for current OBS
            ds_wav = ds_wav.isel({time_coordsname: slice(n_start, n_end)})

        # Reshape
        signal_mat = np.vstack(
            [ds_wav[f"signal_RCV{obs_id:02d}"].values for obs_id in self.receiver_ids]
        )  # WARNING : this assumes same fs, otherwise it will throw an error
        # Set common time vector
        common_time_vector = (
            np.arange(ds_wav.sizes[time_coordsname]) * 1 / fs
        )  # WARNING: assumes same fs

        # Define a record_id to be used to save results
        datetime_fmt = ds_wav.attrs["datetime_format"]
        record_id = f"passive_{datetime.strftime(t_start, datetime_fmt)}_to_{datetime.strftime(t_end, datetime_fmt)}"

        # Build dataset
        xr_data = xr.Dataset(
            data_vars=dict(
                signal=(["h_index", "time"], signal_mat.astype(np.float32)),
                start_dt=t_start,
                end_dt=t_end,
            ),
            coords=dict(
                h_index=self.receiver_ids,
                time=common_time_vector.astype(np.float32),
            ),
            attrs=dict(
                fs=fs,
                ts=1 / fs,
                datetime_format=datetime_fmt,
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

        return xr_data

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

        plt.figure(figsize=(20, 16))
        g = xr_data.rtf_amp_hat.plot(
            x="segment_dt",
            y="f_rtf",
            col="h_index",
            col_wrap=5,
            cmap="magma",
            robust=True,
            cbar_kwargs={"label": r"$|\Pi|$"},
        )
        formatter = mdates.DateFormatter("%H:%M")
        for i, ax in enumerate(g.axs.flat):
            rcv_title = (
                f"Rcv {xr_data.h_index.values[i]}" if i < xr_data.h_index.size else ""
            )
            ax.set_title(rcv_title)

            ax.xaxis.set_major_formatter(formatter)
            formatter = mdates.DateFormatter("%H:%M")
            ax.xaxis.set_major_formatter(formatter)
            locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
            ax.xaxis.set_major_locator(locator)
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

        g.set_xlabels("")
        g.set_ylabels("")
        g.fig.supxlabel("Time")
        g.fig.supylabel("Frequency [Hz]")

        fpath = os.path.join(xr_data.root_img, f"rtf_amp.png")
        plt.savefig(fpath)

        plt.figure(figsize=(20, 16))
        g = xr_data.rtf_phase_hat.plot(
            x="segment_dt",
            y="f_rtf",
            col="h_index",
            col_wrap=5,
            cmap="magma",
            robust=True,
            cbar_kwargs={"label": r"$\Phi$"},
        )

        for i, ax in enumerate(g.axs.flat):
            rcv_title = (
                f"Rcv {xr_data.h_index.values[i]}" if i < xr_data.h_index.size else ""
            )
            ax.set_title(rcv_title)

            ax.xaxis.set_major_formatter(formatter)
            formatter = mdates.DateFormatter("%H:%M")
            ax.xaxis.set_major_formatter(formatter)
            locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
            ax.xaxis.set_major_locator(locator)
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

        g.set_xlabels("")
        g.set_ylabels("")
        g.fig.supxlabel("Time")
        g.fig.supylabel("Frequency [Hz]")

        fpath = os.path.join(xr_data.root_img, f"rtf_phase.png")
        plt.savefig(fpath)

        plt.close("all")

        for segment_dt in xr_data.segment_dt.values:
            xr_data_seg = xr_data.sel(segment_dt=segment_dt)

            segment_dt_str = datetime.strftime(
                pd.to_datetime(segment_dt).to_pydatetime(), "%H%M%S"
            )

            # for segment_id in xr_data.segment_id.values:
            #     xr_data_seg = xr_data.sel(segment_id=segment_id)

            # nrcv = xr_data.sizes["h_index"]
            # f_amp, axs_amp = plt.subplots(nrows=nrcv, ncols=1, sharex=True)
            # f_phase, axs_phase = plt.subplots(nrows=nrcv, ncols=1, sharex=True)
            # i = 0
            # for rcv_idx in xr_data.h_index.values:

            #     # Plot RTF amplitude
            #     max_amp = xr_data_seg.rtf_amp_hat.max() * 1.2
            #     min_amp = xr_data_seg.rtf_amp_hat.min() * 0.8
            #     # xr_data_seg.rtf_amp.sel(h_index=rcv_idx).plot(
            #     #     ax=axs_amp[i], color="k", label=f"Ref - {rcv_idx}"
            #     # )
            #     xr_data_seg.rtf_amp_hat.sel(h_index=rcv_idx).plot(
            #         ax=axs_amp[i],
            #         color="k",
            #         marker="o",
            #         markersize=1,
            #         linewidth=1,
            #         linestyle="-",
            #         label=f"{self.rtf_estimator.upper()} - {rcv_idx}",
            #     )
            #     axs_amp[i].set_xlabel("")
            #     axs_amp[i].set_ylabel(r"$|\Pi|$")
            #     axs_amp[i].set_ylim(min_amp, max_amp)
            #     axs_amp[i].set_yscale("log")
            #     axs_amp[i].set_title("")
            #     axs_amp[i].legend(fontsize=8)

            #     # Plot RTF phase
            #     # xr_data_seg.rtf_phase.sel(h_index=rcv_idx).plot(
            #     #     ax=axs_phase[i], color="k", label=f"Ref - {rcv_idx}"
            #     # )
            #     xr_data_seg.rtf_phase_hat.sel(h_index=rcv_idx).plot(
            #         ax=axs_phase[i],
            #         color="k",
            #         marker="o",
            #         markersize=1,
            #         linewidth=1,
            #         linestyle="-",
            #         label=f"{self.rtf_estimator.upper()} - {rcv_idx}",
            #     )
            #     axs_phase[i].set_xlabel("")
            #     axs_phase[i].set_ylabel(r"$\Phi$")
            #     axs_phase[i].set_title("")
            #     axs_phase[i].legend(fontsize=8)

            #     i += 1

            # # Save figures
            # fpath = os.path.join(xr_data.root_img, f"rtf_amp_{segment_dt_str}.png")
            # f_amp.savefig(fpath)
            # fpath = os.path.join(xr_data.root_img, f"rtf_phase_{segment_dt_str}.png")
            # f_phase.savefig(fpath)

            # plt.close("all")

            # Plot csdms (noise, noisy signal, signal)
            f_csdm, axs_csdm = plt.subplots(nrows=1, ncols=3, sharey=True)
            f_csdm.suptitle("Mean CSDM")
            f_csdm.supxlabel("Receiver index")
            f_csdm.supylabel("Receiver index")

            # Mean CSDMs
            mean_Rx = xr_data_seg.Rx.mean(dim="f_csdm")
            mean_Rv = xr_data_seg.Rv.mean(dim="f_csdm")
            Rs = xr_data_seg.Rx - xr_data_seg.Rv
            mean_Rs = Rs.mean(dim="f_csdm")

            # Derive a common vmax for comparison purpose
            vmax = max(mean_Rx.values.max(), mean_Rv.values.max())

            # Plot Rx
            im = axs_csdm[0].imshow(
                mean_Rx.values, cmap="jet", vmax=vmax, vmin=0, aspect="equal"
            )

            # mean_Rx.plot(
            #     ax=axs_csdm[0],
            #     cmap="jet",
            #     x="h_index",
            #     vmax=vmax,
            #     vmin=0,
            #     aspect="equal",
            # )
            axs_csdm[0].set_title(r"$\hat{R}_x$")
            # axs_csdm[0].set_xlabel("Index")
            # axs_csdm[0].set_ylabel("Index")
            # Ticks
            # axs_csdm[0].set_xticks(np.arange(1, nrcv + 1, 1))
            # axs_csdm[0].set_yticks(np.arange(1, nrcv + 1, 1))
            # axs_csdm[0].set_xticks(self.receiver_ids)
            # axs_csdm[0].set_yticks(self.receiver_ids)
            axs_csdm[0].set_xticks(np.arange(len(self.receiver_ids)))
            axs_csdm[0].set_yticks(np.arange(len(self.receiver_ids)))
            axs_csdm[0].set_xticklabels(self.receiver_ids)
            axs_csdm[0].set_yticklabels(self.receiver_ids)

            # Plot Rv
            im = axs_csdm[1].imshow(
                mean_Rv.values, cmap="jet", vmax=vmax, vmin=0, aspect="equal"
            )
            # mean_Rv.plot(ax=axs_csdm[1], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[1].set_title(r"$\hat{R}_v$")
            # axs_csdm[1].set_xlabel("Index")
            # axs_csdm[1].set_ylabel("Index")
            # Ticks
            # axs_csdm[1].set_xticks(self.receiver_ids)
            # axs_csdm[1].set_yticks(self.receiver_ids)
            axs_csdm[1].set_xticks(np.arange(len(self.receiver_ids)))
            axs_csdm[1].set_yticks(np.arange(len(self.receiver_ids)))
            axs_csdm[1].set_xticklabels(self.receiver_ids)
            axs_csdm[1].set_yticklabels(self.receiver_ids)

            # Plot Rs
            im = axs_csdm[2].imshow(
                mean_Rs.values, cmap="jet", vmax=vmax, vmin=0, aspect="equal"
            )
            # mean_Rs.plot(ax=axs_csdm[2], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[2].set_title(r"$\hat{R}_s = \hat{R}_x - \hat{R}_v$")
            # axs_csdm[2].set_xlabel("Index")
            # axs_csdm[2].set_ylabel("Index")
            # Ticks
            # axs_csdm[2].set_xticks(self.receiver_ids)
            # axs_csdm[2].set_yticks(self.receiver_ids)
            axs_csdm[2].set_xticks(np.arange(len(self.receiver_ids)))
            axs_csdm[2].set_yticks(np.arange(len(self.receiver_ids)))
            axs_csdm[2].set_xticklabels(self.receiver_ids)
            axs_csdm[2].set_yticklabels(self.receiver_ids)

            clabel = r"$\lvert \hat{R} \rvert$"
            f_csdm.colorbar(
                im,
                ax=axs_csdm.ravel().tolist(),
                label=clabel,
                orientation="vertical",
                fraction=0.015,
                pad=0.03,
            )

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
            f_csdm.supxlabel("Receiver index")
            f_csdm.supylabel("Receiver index")

            Rx = xr_data_seg.Rx.sel(f_csdm=fc, method="nearest")
            Rv = xr_data_seg.Rv.sel(f_csdm=fc, method="nearest")
            Rs = Rx - Rv

            # Derive a common vmax for comparison purpose
            # vmax = max(mean_Rx.values.max(), mean_Rv.values.max())
            vmax = max(Rx.values.max(), Rv.values.max())

            # Plot Rx
            im = axs_csdm[0].imshow(
                Rx.values, cmap="jet", vmax=vmax, vmin=0, aspect="equal"
            )
            # Rx.plot(ax=axs_csdm[0], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[0].set_title(r"$\hat{R}_x$")
            # axs_csdm[0].set_xlabel("Index")
            # axs_csdm[0].set_ylabel("Index")
            # Ticks
            axs_csdm[0].set_xticks(np.arange(len(self.receiver_ids)))
            axs_csdm[0].set_yticks(np.arange(len(self.receiver_ids)))
            axs_csdm[0].set_xticklabels(self.receiver_ids)
            axs_csdm[0].set_yticklabels(self.receiver_ids)
            # axs_csdm[0].set_xticks(np.arange(1, nrcv + 1, 1))
            # axs_csdm[0].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rv
            im = axs_csdm[1].imshow(
                Rv.values, cmap="jet", vmax=vmax, vmin=0, aspect="equal"
            )
            # Rv.plot(ax=axs_csdm[1], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[1].set_title(r"$\hat{R}_v$")
            # axs_csdm[1].set_xlabel("Index")
            # axs_csdm[1].set_ylabel("Index")
            # Ticks
            # axs_csdm[1].set_xticks(np.arange(1, nrcv + 1, 1))
            # axs_csdm[1].set_yticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[1].set_xticks(np.arange(len(self.receiver_ids)))
            axs_csdm[1].set_yticks(np.arange(len(self.receiver_ids)))
            axs_csdm[1].set_xticklabels(self.receiver_ids)
            axs_csdm[1].set_yticklabels(self.receiver_ids)

            # Plot Rs
            im = axs_csdm[2].imshow(
                Rs.values, cmap="jet", vmax=vmax, vmin=0, aspect="equal"
            )
            # Rs.plot(ax=axs_csdm[2], cmap="jet", x="h_index", vmax=vmax, vmin=0)
            axs_csdm[2].set_title(r"$\hat{R}_s = \hat{R}_x - \hat{R}_v$")
            # axs_csdm[2].set_xlabel("Index")
            # axs_csdm[2].set_ylabel("Index")
            # Ticks
            # axs_csdm[2].set_xticks(np.arange(1, nrcv + 1, 1))
            # axs_csdm[2].set_yticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[2].set_xticks(np.arange(len(self.receiver_ids)))
            axs_csdm[2].set_yticks(np.arange(len(self.receiver_ids)))
            axs_csdm[2].set_xticklabels(self.receiver_ids)
            axs_csdm[2].set_yticklabels(self.receiver_ids)

            clabel = r"$\lvert \hat{R} \rvert$"
            f_csdm.colorbar(
                im,
                ax=axs_csdm.ravel().tolist(),
                label=clabel,
                orientation="vertical",
                fraction=0.015,
                pad=0.03,
            )

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
        xr_data.attrs["h_index_ref"] = self.reference_receiver_id

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


if __name__ == "__main__":
    pass
