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

from scipy import stats
from datetime import datetime, timedelta
from scipy.signal import butter, lfilter

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
from real_data_analysis.fiberscope_20.src.read_tdms import load_fiberscope_data

# import referrers
from pympler import muppy, summary


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


class FiberscopeManager:
    """
    Mother class to apply RTF-MFP to Fiberscope data
    """

    def __init__(
        self,
        root_processed_data: str,
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
        self.ds_wav_fpath = os.path.join(self.root_processed_data, "channel_H_wav.nc")
        self.ds_wav = xr.open_dataset(self.ds_wav_fpath)
        self.datetime_fmt = self.ds_wav.attrs["datetime_format"]

        # Define usefull objects
        self.cm = CovManager()

        # OBS ids
        self.obs_ids = obs_ids

        # Verbose flag
        self.verbose = verbose

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

    def get_signal_presence_mask_ft(self, x, tstart, tend, nperseg, noverlap):

        # Dummy stft calculation to get ff and tt (should be replace by single stft calc)
        ff, tt, stft_x = self.cm.get_stft_array(
            x.signal.T,
            fs=x.fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        t_first_arr_in_slice = (tstart - x.time.min()).values
        t_last_arr_in_slice = (tend - x.time.min()).values

        # Add a little offset to ensure to include all the signal energy
        alpha = np.ceil(noverlap / nperseg * 4)
        dtt = tt[1] - tt[0]
        t_left = t_first_arr_in_slice - alpha * dtt
        t_right = t_last_arr_in_slice + alpha * dtt

        # Define bounds
        t = (x.time - x.time.min()).values
        f0 = x.fmin
        f1 = x.fmax
        left_bound = f0 + (f1 - f0) / x.pulse_duration * (t - t_left)
        right_bound = f0 + (f1 - f0) / x.pulse_duration * (t - t_right)

        # Interpolate on tt grid
        left_tt = np.interp(tt, t, left_bound)
        right_tt = np.interp(tt, t, right_bound)

        # Define mask
        TT, FF = np.meshgrid(tt, ff)
        mask_stft = np.logical_and(
            (FF <= left_tt[np.newaxis, :]), (FF >= right_tt[np.newaxis, :])
        )

        # Plot mask
        if self.plot_csdm_mask:
            for ircv in range(stft_x.shape[0]):
                stft_i = np.abs(stft_x[ircv, ...])
                stft_i /= np.max(stft_i)
                plt.figure()
                plt.pcolormesh(tt, ff, 10 * np.log10(stft_i), vmin=-30, cmap="jet")
                plt.colorbar(label="[dB]")
                plt.plot(
                    tt,
                    left_tt,
                    linewidth=4,
                    color="k",
                )
                plt.plot(
                    tt,
                    right_tt,
                    linewidth=4,
                    color="k",
                )

                # Mask overlay
                alpha_mask = mask_stft.astype(float) * 0.45
                plt.pcolormesh(
                    tt,
                    ff,
                    np.ones_like(mask_stft),
                    cmap="gray",
                    alpha=alpha_mask,
                    shading="auto",
                )

                plt.xlabel("Time [s]")
                plt.ylabel("Frequency [Hz]")
                plt.ylim([0, 1000])
                fpath = os.path.join(x.root_img, f"csdm_mask_definition_rcv{ircv}.png")
                plt.savefig(fpath)

            self.plot_csdm_mask = False  # Plot only one time

        return mask_stft

    def get_signal_presence_mask(self, x, fs, nperseg, noverlap):

        # Energy detector based
        ff, tt, stft_x = self.cm.get_stft_array(
            x, fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        dtt = tt[1] - tt[0]

        duration = x.time.max() - x.time.min()
        sig_dist_samples = int(x.inter_pulse_period * 1 / dtt * 2 / 3)
        n_roll_avg = int(x.pulse_duration * 1 / dtt * 2 / 3)
        n_em = x.n_emissions
        # Define the signal presence mask
        mask_tt_x = np.zeros_like(tt, dtype=int)
        for ircv in range(stft_x.shape[0]):
            energy = np.sum(np.abs(stft_x[ircv, ...]) ** 2, axis=0)

            # Smooth with rolling average
            energy = np.convolve(energy, np.ones(n_roll_avg) / n_roll_avg, mode="same")

            if duration > 1:
                # Old method before 27/10/2025
                min_height = 0.2 * np.max(energy)
                # min_height = np.median(energy)
                idx_peaks = sp.find_peaks(
                    energy, height=min_height, distance=sig_dist_samples
                )[0]

                peaks_energy = energy[idx_peaks]
                peaks_energy_sorted = np.sort(peaks_energy)[::-1]
                peaks_energy_sorted_em = peaks_energy_sorted[:n_em]
                try:
                    min_peaks = np.min(peaks_energy_sorted_em)
                except:
                    default_mask = np.ones_like(energy)
                    print("debug")

                # threshold = 0.9 * min_peaks
                threshold = 0.5 * min_peaks
            else:
                # Other simple method
                threshold = np.median(energy)

            # Simpler method 27/10/2025 not working well
            # threshold = np.max(1.1 * np.min(energy), 0.1 * np.max(energy))

            # Define signal presence mask
            mask_tt_i = energy > threshold
            mask_tt_x = np.logical_or(mask_tt_x, mask_tt_i)

        # # # For debug purpose
        # plt.figure()
        # plt.plot(energy)
        # plt.scatter(idx_peaks, energy[idx_peaks], color="r")
        # plt.axhline(threshold, linestyle="--", color="r")
        # plt.savefig(f"debug_energy_rcv{ircv}")

        # plt.figure()
        # plt.plot(mask_tt_i)
        # plt.savefig(f"debug_masktt_rcv{ircv}")
        # plt.close("all")

        # plt.figure()
        # plt.plot(mask_tt_x)
        # plt.savefig(f"debug_masktt")
        # plt.close("all")

        # # # Un-comment for debug
        # for ircv in range(stft_x.shape[0]):
        #     plt.figure()
        #     plt.pcolormesh(tt, ff, np.abs(stft_x[ircv, ...]))
        #     plt.plot(tt, mask_tt_x.astype(int) * np.max(ff))
        #     plt.savefig(f"debug_stft_rcv{ircv}")

        return tt, mask_tt_x

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

    def localize_dyn_recording(
        self, static_signal, static_records_names, fs_dynamic_recording
    ):
        # TODO : recode this for Groix
        pass

        # # Localizing static records
        # d = []

        # # Order static records names by position order
        # position_ids = [int(name.split("_")[1][1]) for name in static_records_names]
        # # Sort the static records names by position
        # sorted_indices = np.argsort(position_ids)
        # static_records_names = [static_records_names[i] for i in sorted_indices]

        # # Init progress bar
        # i_test = 0
        # prev_progress = 0
        # n_test = len(static_records_names)
        # print("\nCompute distance map")

        # for recording_name in static_records_names:

        #     i_test += 1
        #     prev_progress = progression_bar(
        #         index=i_test,
        #         index0=0,
        #         indexf=n_test,
        #         prev_progress=prev_progress,
        #     )

        #     fpath = os.path.join(
        #         static_signal.records_folder, recording_name + "_rtf.nc"
        #     )
        #     xr_data_event = xr.open_dataset(fpath)

        #     # Localize using rtf
        #     d_rtf = self.localize_dyn_recording_rtf(xr_data_event, fs_dynamic_recording)
        #     d.append(d_rtf)

        #     # Derive constrast q
        #     # Eq 1.106 rapport RTF
        #     # q_rtf = (np.max(d_rtf) - d_rtf) / (np.max(d_rtf) - np.min(d_rtf))
        #     # q.append(q_rtf)

        # # q = np.array(q)
        # d = np.array(d)

        # return d

    def localize_dyn_recording_rtf(self, xr_data_event, fs_dynamic_recording):

        # TODO : recode this for Groix
        pass

        # # Reference rtf vector = rtf vector at the event position
        # sig = fs_dynamic_recording.signal
        # xr_data_event = xr_data_event.sel(f_rtf=slice(sig.fmin, sig.fmax))
        # rtf_event = xr_data_event.rtf_amp_hat * np.exp(1j * xr_data_event.rtf_phase_hat)
        # # rtf_event_true = xr_data_event.rtf_amp * np.exp(1j * xr_data_event.rtf_phase)
        # # List to store distance with each successive position
        # dist = []

        # # Sort the dynamic recordings by position
        # dist_from_P1 = [
        #     float(r_name.split("_")[-2][1:-1])
        #     for r_name in fs_dynamic_recording.splitted_records_names
        # ]
        # # Sort the dynamic recordings by position
        # sorted_indices = np.argsort(dist_from_P1)
        # splitted_records_names = [
        #     fs_dynamic_recording.splitted_records_names[i] for i in sorted_indices
        # ]

        # # Set distance args to use
        # if self.theta_statistics == "mean":
        #     apply_mean = True
        # elif self.theta_statistics == "expectation":
        #     apply_mean = False
        # dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": apply_mean}

        # # Iterate over dynamic recordings
        # for recording_name in splitted_records_names:
        #     # Load data
        #     fpath = os.path.join(
        #         fs_dynamic_recording.splitted_records_folder,
        #         recording_name + "_rtf.nc",
        #     )
        #     # Assert fpath exist
        #     if not os.path.exists(fpath):
        #         raise FileNotFoundError(
        #             f"File {fpath} does not exist. Please check the file path."
        #         )
        #     xr_data_library_i = xr.open_dataset(fpath)
        #     xr_data_library_i = xr_data_library_i.sel(f_rtf=slice(sig.fmin, sig.fmax))
        #     rtf_library_i = xr_data_library_i.rtf_amp_hat * np.exp(
        #         1j * xr_data_library_i.rtf_phase_hat
        #     )

        #     # Interpolate rtf_event at rtf_library freq (dynamic recording uses smaller window to
        #     # match the number of segment L )
        #     rtf_event = rtf_event.sel(
        #         f_rtf=rtf_library_i.f_rtf.values, method="nearest"
        #     )

        #     # Derive distance using hermitian angle
        #     theta = D_hermitian_angle_fast(
        #         rtf_event.values, rtf_library_i.values, **dist_kwargs
        #     )
        #     theta_c = get_theta_c(val=theta, apply_mean=apply_mean)
        #     dist.append(theta_c)

        # dist = np.array(dist)

        # return dist

    def plot_dyn_loc(
        self, d_rtf, time_step, axis_norm=1, fname=None, vmin=-5, save_eps=False
    ):

        # TODO : recode this for Groix
        pass

        # d = -d_rtf

        # if axis_norm is None:
        #     d_max = np.nanmax(d, axis=axis_norm) * np.ones_like(d)
        #     d_min = np.nanmin(d, axis=axis_norm) * np.ones_like(d)
        #     norm_label = "norm_over_entire_surface"
        # else:
        #     d_max = np.tile(
        #         np.nanmax(d, axis=axis_norm), (d.shape[axis_norm], 1)
        #     )  # Cast to d shape
        #     d_min = np.tile(np.nanmin(d, axis=axis_norm), (d.shape[axis_norm], 1))
        #     if axis_norm == 1:
        #         norm_label = f"norm_along_time_axis"
        #     elif axis_norm == 0:
        #         norm_label = f"norm_along_position_axis"

        # if axis_norm == 1:
        #     d_max = d_max.T
        #     d_min = d_min.T

        # # Normalize
        # q = (d - d_min) / (d_max - d_min)

        # # In dB
        # q[q == 0] = 1e-6
        # q_dB = 10 * np.log10(q)

        # t = np.arange(0, d.shape[1]) * time_step
        # ordered_pos = [f"$P_{i}$" for i in range(1, 7)]
        # truepos_order = [0, 5, 1, 4, 2, 3]
        # q_dB = q_dB[truepos_order, :]
        # plt.figure()
        # plt.imshow(q_dB, cmap="jet", aspect="auto", vmin=vmin, rasterized=False)
        # plt.xticks(np.arange(0, q_dB.shape[1], 10), np.round(t[::10], 2))
        # plt.yticks(np.arange(0, q.shape[0]), ordered_pos)
        # plt.xlabel("Time [s]")
        # plt.ylabel("Position")
        # plt.colorbar(label=r"$q\, \textrm{[dB]}$")
        # plt.gca().invert_yaxis()

        # folder = os.path.join(self.root_img, "localization", norm_label)
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # if fname is None:
        #     fname = f"dyn_qdB_href{self.h_index_ref}.png"

        # fpath = os.path.join(folder, fname)
        # plt.savefig(fpath)

        # if save_eps:
        #     fname = fname.split(".")[0] + ".eps"
        #     fpath = os.path.join(folder, fname)
        #     plt.savefig(fpath, format="eps")


# ======================================================================================================================
#  Active source localisation
# ======================================================================================================================
class ActiveFiberscopeManager(FiberscopeManager):
    """
    Class to handle active source localisation. The source to localise is the Lubell emiting series of LFM signals.
    """

    def __init__(
        self,
        root_processed_data: str,
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
        deconvolution_method: str = "wiener",
    ):
        """
        Class constructor

        """
        # Using super() to initialize the parent class
        super().__init__(
            root_processed_data=root_processed_data,
            root_img=root_img,
            bandfilter=bandfilter,
            tau_ir=tau_ir,
            alpha_overlap=alpha_overlap,
            h_index_ref=h_index_ref,
            plot_feature=plot_feature,
            theta_statistics=theta_statistics,
            process_pulse_one_by_one=process_pulse_one_by_one,
            estimate_ir_duration=estimate_ir_duration,
            rtf_estimator=rtf_estimator,
            obs_ids=obs_ids,
            verbose=verbose,
            plot_signal=plot_signal,
        )

        self.deconvolution_method = deconvolution_method

    def load_sequence_data(
        self,
        df_seq,
        pre_reception_time=p.pre_reception_time,
        post_reception_time=p.post_reception_time,
    ):

        # nperseg = 256 * 2
        # noverlap = int(nperseg * 0.5)

        fs = self.ds_wav.attrs[f"fs_obs{1}"]
        datetime_fmt = self.ds_wav.attrs["datetime_format"]

        emission_duration = df_seq["duration_s"].iloc[0]

        # Datetime bounds for the studied sequence
        arr_dt_obs1 = df_seq[f"arrival_datetime_obs1"]
        arr_dt_obs2 = df_seq[f"arrival_datetime_obs2"]
        arr_dt_obs3 = df_seq[f"arrival_datetime_obs3"]
        arr_dt_obs = [arr_dt_obs1, arr_dt_obs2, arr_dt_obs3]

        first_arr_dt = np.min([arr_dt.iloc[0] for arr_dt in arr_dt_obs])
        last_arr_dt = np.max([arr_dt.iloc[-1] for arr_dt in arr_dt_obs])

        data_obs = {}

        # Extract the corresponding signal portion
        for obs_id in self.obs_ids:
            # fs = ds_wav.attrs[f"fs_obs{obs_id}"]
            # print(fs)

            # Start of recording
            t0 = self.ds_wav.attrs[f"start_datetime_obs{obs_id}"]
            wav_start_dt = datetime.strptime(t0, datetime_fmt)

            # Signal slice to extract
            t_start = (first_arr_dt - wav_start_dt).total_seconds() - pre_reception_time
            n_start = int(np.floor(np.round(t_start * fs, 4)))
            t_end = (
                (last_arr_dt - wav_start_dt).total_seconds()
                + emission_duration
                + post_reception_time
            )  # Add emission duration to get the entire signal
            n_end = int(np.ceil(np.round(t_end * fs, 4)))

            # print(obs_id)
            # print(t_end, t_start)
            # print((t_end - t_start)*fs)
            # print(int((t_end - t_start)*fs))
            # print(t_start * fs, t_end * fs)
            # print(n_start, n_end)
            # print(n_end - n_start)

            # Slice signal
            sig_varname = f"signal_obs{obs_id}"
            time_varname = f"time{obs_id}"
            signal = self.ds_wav[sig_varname]
            sig_win = signal.isel({time_varname: slice(n_start, n_end)}).values

            t_win_sec = np.arange(sig_win.shape[0]) / fs + n_start * 1 / fs

            # Arrivals dt in seconds from start of slice
            arr_time_in_sec_from_wavstart = (
                arr_dt_obs[obs_id - 1] - wav_start_dt
            ).dt.total_seconds()
            arr_time_in_sec_from_slicestart = arr_time_in_sec_from_wavstart - t_start

            # Apply filter if required
            if self.apply_bandfilter:
                sig_win = self.bandfilter.apply_filter(sig_win, fs)

            # Datetime corresponding to the first instant in the slice
            t0_slice = wav_start_dt + timedelta(seconds=n_start * 1 / fs)
            # t1_slice = wav_start_dt + timedelta(seconds=n_end * 1 / fs)

            # Elapsed time from t0_slice to first arrival at current obs
            first_arr_dt_obs = df_seq[f"arrival_datetime_obs{obs_id}"].iloc[0]
            t0_first_arr = (first_arr_dt_obs - t0_slice).total_seconds()

            # Store data
            data_obs[obs_id] = dict(
                signal=sig_win,
                time=t_win_sec,
                t0_first_arr=t0_first_arr,
                arr_time_in_sec_from_start=arr_time_in_sec_from_slicestart,
            )

        arr_time_in_sec_from_start_mat = np.vstack(
            [data_obs[i]["arr_time_in_sec_from_start"].values for i in self.obs_ids]
        )
        signal_mat = np.vstack([data_obs[i]["signal"] for i in self.obs_ids])
        common_time_vector = np.arange(data_obs[1]["signal"].size) * 1 / fs

        first_arrival = np.min([data_obs[i]["t0_first_arr"] for i in self.obs_ids])
        last_arrival = np.max([data_obs[i]["t0_first_arr"] for i in self.obs_ids])

        xr_data = xr.Dataset(
            data_vars=dict(
                signal=(["h_index", "time"], signal_mat),
                arr_time_in_sec_from_start=(
                    ["h_index", "pulse_id"],
                    arr_time_in_sec_from_start_mat,
                ),
            ),
            coords=dict(
                h_index=self.obs_ids,
                time=common_time_vector,
                pulse_id=df_seq["pulse_id"],  # To be used later
            ),
            attrs=dict(
                fs=fs,
                ts=1 / fs,
                t0=first_arrival,
                t1=last_arrival,
                datetime_format=self.datetime_fmt,
                start_datetime=t0_slice.strftime(self.datetime_fmt),
                sequence_id=df_seq["sequence_id"].iloc[0],
                # n_emissions=df_seq["Nrepeat"].iloc[
                #     0
                # ],  # TODO : we might need to change that to account for non detected arrivals ?
                n_emissions=df_seq.shape[
                    0
                ],  # Assuming df_seq contains a limited number of emission/reception
                fmin=df_seq["frequency_min_hz"].iloc[0],
                fmax=df_seq["frequency_max_hz"].iloc[0],
                pulse_duration=df_seq["duration_s"].iloc[0],
                inter_pulse_period=df_seq["repeat_period_s"].iloc[0],
                process_pulse_one_by_one=int(self.process_pulse_one_by_one),
                root_img=os.path.join(
                    self.root_img_sequence, str(df_seq["sequence_id"].iloc[0])
                ),
                root_data=self.root_data_sequence,
            ),
        )

        if not os.path.exists(xr_data.root_img):
            os.makedirs(xr_data.root_img)
        if not os.path.exists(xr_data.root_data):
            os.makedirs(xr_data.root_data)

        if self.plot_signal:
            self.plot_audio_signal(xr_data)

        return xr_data

    def plot_audio_signal(xr_data, arrivals_datetimes=None, nperseg=256, noverlap=128):

        fs = xr_data.fs
        seq_id = xr_data.sequence_id
        t0_slice = xr_data.start_datetime
        t_win_sec = xr_data.time.values
        t_win = np.array([t0_slice + timedelta(seconds=t) for t in t_win_sec])

        for obs_id in xr_data.h_index.values:
            sig_win = xr_data.signal.sel(h_index=obs_id)
            # Derive stft
            ff, tt, stft = sp.stft(
                sig_win.values,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="psd",
            )
            sxx_0 = 1  # 1uPa**2 / Hz
            sxx = 10 * np.log10(np.abs(stft) / sxx_0)
            # Associated datetime vector
            tt_datetime = pd.date_range(
                t0_slice,
                t0_slice + timedelta(seconds=tt[-1]),
                freq=f"{tt[1]-tt[0]}s",
                inclusive="both",
            )

            # Plot
            cmap = "Greys"
            t_arrivals = arrivals_datetimes[obs_id - 1]

            fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            # Plot raw signal
            sig_win = sig_win / np.max(np.abs(sig_win))
            axs[0].plot(t_win, sig_win, color="k")
            axs[0].set_ylim([-1, 1])
            im = axs[1].pcolormesh(tt_datetime, ff, sxx, cmap=cmap)
            axs[1].set_ylabel("Fréquence [Hz]")

            # Add arrivals
            if len(t_arrivals) > 0:
                # Plot arrows for arrivals
                for iarr, t_arrival in enumerate(t_arrivals):
                    axs[0].annotate(
                        f"{iarr}",
                        xy=(t_arrival, 0.5),
                        xytext=(t_arrival, 0.9),
                        arrowprops=dict(arrowstyle="->", color="red"),
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                    axs[1].annotate(
                        f"{iarr}",
                        xy=(t_arrival, np.max(ff) * 0.75),
                        xytext=(t_arrival, np.max(ff) * 0.95),
                        arrowprops=dict(arrowstyle="->", color="red"),
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

            fig.supxlabel("Temps UTC")
            fig.supylabel("Signal")
            fig.suptitle(f"Sequence ID {seq_id} - OBS{obs_id}")

    def preprocess_data(self, xr_data, df_seq):

        # Create the source pulse signal
        n_hydro = xr_data.sizes["h_index"]
        t_ir = self.tau_ir  # TODO check this
        n_em = xr_data.n_emissions
        t_pulse = xr_data.pulse_duration
        t_interp_pulse = xr_data.inter_pulse_period
        f0 = xr_data.fmin
        f1 = xr_data.fmax
        ts = xr_data.ts

        # Time to add to ensure we englobe entire signal including last reflexions
        tau_plus = min(
            t_ir, t_interp_pulse - t_pulse
        )  # Avoid to include following pulse

        # Reference chirp
        t = xr_data.signal.sel(time=slice(0, t_pulse)).time.values
        x = sp.chirp(t, f0=f0, f1=f1, t1=t_pulse, method="linear")

        # Recording start datetime
        start_dt = datetime.strptime(xr_data.start_datetime, xr_data.datetime_format)
        # Time to first arrival
        t0 = xr_data.t0

        init_ri_hat = True
        # Loop over each hydrophone to process
        i_hydro = 0
        for hydro_idx in xr_data.h_index.values:

            # Process each pulse
            # for i_em in range(n_em):
            for i_em, pulse_id in enumerate(xr_data.pulse_id.values):

                # Extract the current pulse
                df_pulse = df_seq.loc[df_seq["pulse_id"] == pulse_id]
                arr_dt_pulse = df_pulse[f"arrival_datetime_obs{hydro_idx}"].iloc[0]
                t0 = (arr_dt_pulse - start_dt).total_seconds()
                y = xr_data.signal.sel(
                    time=slice(
                        t0 - ts / 2,
                        t0 + t_pulse + tau_plus + ts / 2,
                    ),
                    h_index=hydro_idx,
                )
                # Note: +/- ts/2 to ensure we include boundary samples

                # # Extract the emission
                # y = xr_data.signal.sel(
                #     time=slice(
                #         t0 + i_em * t_interp_pulse - ts / 2,
                #         t0 + i_em * t_interp_pulse + t_pulse + tau_plus + ts / 2,
                #     ),
                #     h_index=hydro_idx,
                # )
                # # Note: +/- ts/2 to ensure we include boundary samples

                # Estimate the impulse response
                if self.deconvolution_method == "crosscorr":
                    h_hat = crosscorr_deconvolution(x=x, y=y.values)
                elif self.deconvolution_method == "wiener":
                    h_hat = wiener_deconvolution(x=x, y=y.values)
                else:
                    raise ValueError(
                        f"Deconvolution method {self.deconvolution_method} not recognized. Please choose 'crosscorr' or 'wiener'."
                    )

                if init_ri_hat:
                    ri_hat = np.zeros((n_hydro, n_em, y.sizes["time"]))
                    time = y.time.values
                    init_ri_hat = False

                ri_hat[i_hydro, i_em, :] = h_hat

            i_hydro += 1

        # Time vector for impulse response
        xr_data["t_ir"] = np.arange(time.size) * xr_data.ts
        # nstft = self.nperseg
        nstft = time.size

        if not self.process_pulse_one_by_one:
            # Take the mean impulse response over all sweeps analysed
            # (assuming source position is constant over the analysis window)
            ri_hat_mean = np.mean(ri_hat, axis=1)

            xr_data["ri_hat"] = (
                ["h_index", "t_ir"],
                ri_hat_mean,
            )

            # Derive the corresponding frequency response
            tf_hat_mean = np.fft.rfft(ri_hat_mean, n=nstft, axis=1)
            f_ir = np.fft.rfftfreq(nstft, d=ts)
            xr_data["f_ir"] = f_ir

            # Store amplitude and phase in two separate variables to avoid issues with complex in netcdf
            xr_data["tf_hat_amp"] = (
                ["h_index", "f_ir"],
                np.abs(tf_hat_mean),
            )
            xr_data["tf_hat_phase"] = (
                ["h_index", "f_ir"],
                np.angle(tf_hat_mean),
            )

        else:
            # Pulse are processed independently
            xr_data["ri_hat"] = (
                ["h_index", "t_ir", "pulse_id"],
                np.swapaxes(ri_hat, 1, 2),
            )

            # Derive the corresponding frequency response
            tf_hat = np.fft.rfft(ri_hat, n=nstft, axis=-1)
            f_ir = np.fft.rfftfreq(nstft, d=ts)
            xr_data["f_ir"] = f_ir

            # Store amplitude and phase in two separate variables to avoid issues with complex in netcdf
            xr_data["tf_hat_amp"] = (
                ["h_index", "f_ir", "pulse_id"],
                np.swapaxes(np.abs(tf_hat), 1, 2),
            )
            xr_data["tf_hat_phase"] = (
                ["h_index", "f_ir", "pulse_id"],
                np.swapaxes(np.angle(tf_hat), 1, 2),
            )

        if self.estimate_ir_duration:
            xr_data = self.get_ir_duration(xr_data)

        # Save results
        data_fpath = os.path.join(
            self.root_data_sequence, f"sequence_{xr_data.sequence_id}.nc"
        )
        xr_data.attrs["fpath"] = data_fpath
        xr_data.to_netcdf(data_fpath)
        xr_data.close()
        del xr_data

    def get_ir_duration(self, xr_data):
        n_window_rms = int(0.2 * xr_data.fs)

        impulse_response = xr_data.ri_hat
        p2_roll = (impulse_response**2).rolling(t_ir=n_window_rms, center=True).mean()
        # RMS impulse response
        p_rms = np.sqrt(p2_roll)

        # Impulse response SPL
        spl = 20 * np.log10(p_rms / gc.p0)

        # Derive threshold for ir duration
        background_lvl_threshold = spl.quantile(0.97, dim="t_ir")
        ir_duration_estimation_th = -(spl.max(dim="t_ir") - background_lvl_threshold)

        ir_duration = np.empty((xr_data.sizes["h_index"], xr_data.sizes["pulse_id"]))

        for ih, h_index in enumerate(xr_data.h_index.values):
            for ip, pulse_id in enumerate(xr_data.pulse_id.values):

                spl_ = spl.sel(pulse_id=pulse_id, h_index=h_index)
                th = background_lvl_threshold.sel(pulse_id=pulse_id, h_index=h_index)
                above_th = spl_.t_ir.where(spl_ >= th).dropna("t_ir")

                # Derive tau_th
                arr_start = above_th.values[0]
                arr_end = above_th.values[-1]
                tau_ir = arr_end - arr_start
                # print(f"Reverberation time ({threshold:.1f} dB) : {tau_ir:.2f} s")

                # Store
                ir_duration[ih, ip] = tau_ir

                # # Plot for debug

                # # Plot p_rms
                # plt.figure()
                # p_rms.sel(h_index=1).plot(color="k")
                # plt.ylabel("RMS pressure")
                # plt.title("")
                # # Plot ir
                # plt.figure()
                # impulse_response.sel(h_index=h_index, pulse_id=pulse_id).plot(color="k")
                # plt.savefig("test_ir.png")

                # # Plot spl for anormal values
                # if tau_ir >= 1:
                #     img_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\src\localisation\rtf\debug"
                #     plt.figure()
                #     spl.sel(h_index=h_index, pulse_id=pulse_id).plot(color="k")
                #     plt.axhline(
                #         y=th,
                #         color="r",
                #         linestyle="--",
                #         linewidth=1,
                #         # label=f"{threshold:.1f} dB",
                #     )
                #     plt.ylabel(r"$L_p$ [dB re 1$\mu$Pa$^2$]")
                #     # plt.legend()
                #     plt.title("")
                #     plt.savefig(
                #         os.path.join(img_path, f"test_ir_{h_index}_{pulse_id}.png")
                #     )

                #     plt.close("all")

        # Save impulse response duration
        xr_data["ir_duration"] = (["h_index", "pulse_id"], np.array(ir_duration))
        xr_data.ir_duration.attrs["units"] = "s"
        xr_data.ir_duration.attrs["long_name"] = "Impulse response duration"

        # Save estimation threshold
        xr_data["ir_duration_estimation_th"] = ir_duration_estimation_th

        # Store usefull duration
        xr_data["eff_ir_duration"] = np.median(
            xr_data.ir_duration.values[xr_data.ir_duration_estimation_th <= -5]
        )

        return xr_data

    def process_analysis(
        self,
        df_arrivals,
        set_stft_props=True,
    ):

        i_test = 0
        prev_progress = 0

        # Ensure we process only valid arrivals
        valid = (
            df_arrivals["valid_detection_obs1"]
            * df_arrivals["valid_detection_obs2"]
            * df_arrivals["valid_detection_obs3"]
        )
        df_arrivals = df_arrivals.loc[valid]

        # Number of individual sequences to process
        sequence_ids = df_arrivals["sequence_id"].unique()
        n_seq = sequence_ids.size

        if self.verbose:
            print(f"RTF processing of sequences {sequence_ids} ({n_seq})")

        for seq_id in sequence_ids:

            i_test += 1
            prev_progress = progression_bar(
                index=i_test,
                index0=0,
                indexf=n_seq,
                prev_progress=prev_progress,
            )

            ### Step 1 - Load audio data correponding to current sequence ###
            df_seq = df_arrivals.loc[df_arrivals["sequence_id"] == seq_id]
            xr_data = self.load_sequence_data(df_seq=df_seq)  # TODO : complete args

            # If stfts props are already set we dont need to do it
            if set_stft_props:
                self.set_stft_params(ts=xr_data.ts)
                # print(f"nperseg = {self.nperseg}, noverlap = {self.noverlap}")

            idx_rcv_ref = np.argmin(
                np.abs(xr_data.h_index.values - self.h_index_ref)
            )  # Index of hydrophone might not be sorted or not start at 0
            self.set_managers(fs=xr_data.fs, idx_rcv_ref=idx_rcv_ref)

            ### Step 2 - Preprocess data ###
            self.preprocess_data(xr_data=xr_data, df_seq=df_seq)

            ### Step 3 - Derive features ###
            self.derive_feature(sequence_id=seq_id)

    def estimate_global_csdm(self, xr_data):
        # TODO : set dedicated fct for active loc ?
        pass

    def derive_feature(
        self,
        sequence_id,
        Rv_global=None,
        save=True,
    ):

        if self.verbose:
            print(f"Processing sequence {sequence_id} - RTF estimation")

        # Load data
        fpath = os.path.join(self.root_data_sequence, f"sequence_{sequence_id}.nc")

        # Assert fpath exist
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"File {fpath} does not exist. Please check the file path."
            )

        xr_data = xr.open_dataset(fpath)

        # Derive rtf from recordings
        xr_data = self.get_rtf(xr_data=xr_data, Rv_global=Rv_global)

        # # Derive rtf from tf estimated by deconvolution
        xr_data = self.derive_rtf_from_tf(xr_data=xr_data)

        # Slice along frequency axis to ensure we never use information outside of the signal bandwidth
        # This also reduce the memory size required
        xr_data = xr_data.sel(f_rtf=slice(xr_data.fmin, xr_data.fmax))
        xr_data = xr_data.sel(f_ir=slice(xr_data.fmin, xr_data.fmax))
        xr_data = xr_data.sel(f_csdm=slice(xr_data.fmin, xr_data.fmax))

        # Plot feature components for analysis if required
        if self.plot_feature:
            self.plot_estimated_feature(xr_data)
        # # Derive GCC for comparison
        # xr_data = derive_gcc(xr_data=xr_data, gcc_methods=gcc_methods)

        # Save results
        if save:
            xr_data.to_netcdf(
                os.path.join(
                    xr_data.root_data, f"sequence_{xr_data.sequence_id}_rtf.nc"
                )
            )
            xr_data.close()
        else:
            return xr_data

    def plot_estimated_feature(self, xr_data):

        # Ensure img folder exists
        if not os.path.exists(xr_data.root_img):
            os.makedirs(xr_data.root_img)

        if self.process_pulse_one_by_one:

            for pulse_id in xr_data.pulse_id.values:
                xr_data_pulse = xr_data.sel(pulse_id=pulse_id)

                nrcv = xr_data.sizes["h_index"]
                f_amp, axs_amp = plt.subplots(nrows=nrcv, ncols=1, sharex=True)
                f_phase, axs_phase = plt.subplots(nrows=nrcv, ncols=1, sharex=True)
                i = 0
                for rcv_idx in xr_data.h_index.values:

                    # Plot RTF amplitude
                    max_amp = xr_data_pulse.rtf_amp.max() * 1.2
                    min_amp = xr_data_pulse.rtf_amp.min() * 0.8
                    xr_data_pulse.rtf_amp.sel(h_index=rcv_idx).plot(
                        ax=axs_amp[i], color="k", label=f"Ref - {rcv_idx}"
                    )
                    xr_data_pulse.rtf_amp_hat.sel(h_index=rcv_idx).plot(
                        ax=axs_amp[i],
                        color="b",
                        marker="o",
                        markersize=1,
                        linewidth=1,
                        linestyle="--",
                        label=f"{self.rtf_estimator.upper()} - {rcv_idx}",
                    )
                    axs_amp[i].set_xlabel("")
                    axs_amp[i].set_ylabel(r"$|\Pi|$")
                    axs_amp[i].set_ylim(min_amp, max_amp)
                    axs_amp[i].set_yscale("log")
                    axs_amp[i].set_title("")
                    axs_amp[i].legend(fontsize=8)

                    # Plot RTF phase
                    xr_data_pulse.rtf_phase.sel(h_index=rcv_idx).plot(
                        ax=axs_phase[i], color="k", label=f"Ref - {rcv_idx}"
                    )
                    xr_data_pulse.rtf_phase_hat.sel(h_index=rcv_idx).plot(
                        ax=axs_phase[i],
                        color="b",
                        marker="o",
                        markersize=1,
                        linewidth=1,
                        linestyle="--",
                        label=f"{self.rtf_estimator.upper()} - {rcv_idx}",
                    )
                    axs_phase[i].set_xlabel("")
                    axs_phase[i].set_ylabel(r"$\Phi$")
                    axs_phase[i].set_title("")
                    axs_phase[i].legend(fontsize=8)

                    i += 1

                # Save figures
                fpath = os.path.join(xr_data.root_img, f"rtf_amp_pulseID{pulse_id}.png")
                f_amp.savefig(fpath)
                fpath = os.path.join(
                    xr_data.root_img, f"rtf_phase_pulseID{pulse_id}.png"
                )
                f_phase.savefig(fpath)

                plt.close("all")

                # Plot csdms (noise, noisy signal, signal)
                f_csdm, axs_csdm = plt.subplots(nrows=1, ncols=3, sharey=True)
                f_csdm.suptitle("CSDM")

                # Mean CSDMs
                mean_Rx = xr_data_pulse.Rx.mean(dim="f_csdm")
                mean_Rv = xr_data_pulse.Rv.mean(dim="f_csdm")
                Rs = xr_data_pulse.Rx - xr_data_pulse.Rv
                mean_Rs = Rs.mean(dim="f_csdm")

                # Derive a common vmax for comparison purpose
                vmax = max(mean_Rx.values.max(), mean_Rv.values.max())

                # Plot Rx
                mean_Rx.plot(ax=axs_csdm[0], cmap="jet", x="h_index", vmax=vmax)
                axs_csdm[0].set_title(r"$\hat{R}_x$")
                axs_csdm[0].set_xlabel("Index")
                axs_csdm[0].set_ylabel("Index")
                # Ticks
                axs_csdm[0].set_xticks(np.arange(1, nrcv + 1, 1))
                axs_csdm[0].set_yticks(np.arange(1, nrcv + 1, 1))

                # Plot Rv
                mean_Rv.plot(ax=axs_csdm[1], cmap="jet", x="h_index", vmax=vmax)
                axs_csdm[1].set_title(r"$\hat{R}_v$")
                axs_csdm[1].set_xlabel("Index")
                axs_csdm[1].set_ylabel("Index")
                # Ticks
                axs_csdm[1].set_xticks(np.arange(1, nrcv + 1, 1))
                axs_csdm[1].set_yticks(np.arange(1, nrcv + 1, 1))

                # Plot Rs
                mean_Rs.plot(ax=axs_csdm[2], cmap="jet", x="h_index", vmax=vmax)
                axs_csdm[2].set_title(r"$\hat{R}_s = \hat{R}_x - \hat{R}_v$")
                axs_csdm[2].set_xlabel("Index")
                axs_csdm[2].set_ylabel("Index")
                # Ticks
                axs_csdm[2].set_xticks(np.arange(1, nrcv + 1, 1))
                axs_csdm[2].set_yticks(np.arange(1, nrcv + 1, 1))

                # Save figure
                fpath = os.path.join(
                    xr_data.root_img, f"estimated_csdms_pulseID{pulse_id}.png"
                )
                f_csdm.savefig(fpath)

                plt.close("all")

                # Plot csdms (noise, noisy signal, signal)
                f_csdm, axs_csdm = plt.subplots(nrows=1, ncols=3, sharey=True)

                # CSDMs at a center freq
                fc = (xr_data.fmax - xr_data.fmin) / 2
                f_csdm.suptitle(f"CSDM (f = {fc} Hz)")

                Rx = xr_data_pulse.Rx.sel(f_csdm=fc, method="nearest")
                Rv = xr_data_pulse.Rv.sel(f_csdm=fc, method="nearest")
                Rs = Rx - Rv

                # Derive a common vmax for comparison purpose
                vmax = max(mean_Rx.values.max(), mean_Rv.values.max())

                # Plot Rx
                Rx.plot(ax=axs_csdm[0], cmap="jet", x="h_index", vmax=vmax)
                axs_csdm[0].set_title(r"$\hat{R}_x$")
                axs_csdm[0].set_xlabel("Index")
                axs_csdm[0].set_ylabel("Index")
                # Ticks
                axs_csdm[0].set_xticks(np.arange(1, nrcv + 1, 1))
                axs_csdm[0].set_yticks(np.arange(1, nrcv + 1, 1))

                # Plot Rv
                Rv.plot(ax=axs_csdm[1], cmap="jet", x="h_index", vmax=vmax)
                axs_csdm[1].set_title(r"$\hat{R}_v$")
                axs_csdm[1].set_xlabel("Index")
                axs_csdm[1].set_ylabel("Index")
                # Ticks
                axs_csdm[1].set_xticks(np.arange(1, nrcv + 1, 1))
                axs_csdm[1].set_yticks(np.arange(1, nrcv + 1, 1))

                # Plot Rs
                Rs.plot(ax=axs_csdm[2], cmap="jet", x="h_index", vmax=vmax)
                axs_csdm[2].set_title(r"$\hat{R}_s = \hat{R}_x - \hat{R}_v$")
                axs_csdm[2].set_xlabel("Index")
                axs_csdm[2].set_ylabel("Index")
                # Ticks
                axs_csdm[2].set_xticks(np.arange(1, nrcv + 1, 1))
                axs_csdm[2].set_yticks(np.arange(1, nrcv + 1, 1))

                # Save figure
                fpath = os.path.join(
                    xr_data.root_img,
                    f"estimated_csdms_pulseID{pulse_id}_f_{Rx.f_csdm.values:.1f}Hz.png",
                )
                f_csdm.savefig(fpath)

                plt.close("all")

                # Memory leaks
                # top_10_objects = (muppy.sort(muppy.get_objects()))[-10:]
                # top_10_objects.reverse()

                # for obj in top_10_objects:
                #     print(
                #         referrers.get_referrer_graph(
                #             obj,
                #             exclude_object_ids=[id(top_10_objects)],
                #         )
                #     )

                # all_objects = muppy.get_objects()
                # # all_objects = (muppy.sort(muppy.get_objects()))[-10:]
                # # all_objects.reverse()
                # summary.print_(summary.summarize(all_objects))

                # del xr_data_pulse

                # all_objects = muppy.get_objects()
                # # all_objects = (muppy.sort(muppy.get_objects()))[-10:]
                # # all_objects.reverse()
                # summary.print_(summary.summarize(all_objects))

        else:
            nrcv = xr_data.sizes["h_index"]
            f_amp, axs_amp = plt.subplots(nrows=nrcv, ncols=1, sharex=True)
            f_phase, axs_phase = plt.subplots(nrows=nrcv, ncols=1, sharex=True)
            i = 0
            for rcv_idx in xr_data.h_index.values:

                # Plot RTF amplitude
                xr_data.rtf_amp.sel(h_index=rcv_idx).plot(
                    ax=axs_amp[i], color="k", label=f"Ref - {rcv_idx}"
                )
                xr_data.rtf_amp_hat.sel(h_index=rcv_idx).plot(
                    ax=axs_amp[i],
                    color="b",
                    marker="o",
                    markersize=1,
                    linewidth=1,
                    linestyle="--",
                    label=f"CS - {rcv_idx}",
                )
                axs_amp[i].set_xlabel("")
                axs_amp[i].set_ylabel(r"$|\Pi|$")
                axs_amp[i].set_yscale("log")
                axs_amp[i].set_title("")
                axs_amp[i].legend(fontsize=8)

                # Plot RTF phase
                xr_data.rtf_phase.sel(h_index=rcv_idx).plot(
                    ax=axs_phase[i], color="k", label=f"Ref - {rcv_idx}"
                )
                xr_data.rtf_phase_hat.sel(h_index=rcv_idx).plot(
                    ax=axs_phase[i],
                    color="b",
                    marker="o",
                    markersize=1,
                    linewidth=1,
                    linestyle="--",
                    label=f"CS - {rcv_idx}",
                )
                axs_phase[i].set_xlabel("")
                axs_phase[i].set_ylabel(r"$\Phi$")
                axs_phase[i].set_title("")
                axs_phase[i].legend()
                axs_phase[i].legend(fontsize=8)

                i += 1

            # Ensure img folder exists
            if not os.path.exists(xr_data.root_img):
                os.makedirs(xr_data.root_img)

            # Save figures
            fpath = os.path.join(xr_data.root_img, "rtf_amp.png")
            f_amp.savefig(fpath)
            fpath = os.path.join(xr_data.root_img, "rtf_phase.png")
            f_phase.savefig(fpath)

            # Plot csdms (noise, noisy signal, signal)
            f_csdm, axs_csdm = plt.subplots(nrows=1, ncols=3, sharey=True)
            f_csdm.suptitle("CSDM")

            # Mean CSDMs
            mean_Rx = xr_data.Rx.mean(dim="f_csdm")
            mean_Rv = xr_data.Rv.mean(dim="f_csdm")
            Rs = xr_data.Rx - xr_data.Rv
            mean_Rs = Rs.mean(dim="f_csdm")

            # Derive a common vmax for comparison purpose
            vmax = max(mean_Rx.values.max(), mean_Rv.values.max())

            # Plot Rx
            mean_Rx.plot(ax=axs_csdm[0], cmap="jet", x="h_index", vmax=vmax)
            axs_csdm[0].set_title(r"$\hat{R}_x$")
            axs_csdm[0].set_xlabel("Index")
            axs_csdm[0].set_ylabel("Index")
            # Ticks
            axs_csdm[0].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[0].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rv
            mean_Rv.plot(ax=axs_csdm[1], cmap="jet", x="h_index", vmax=vmax)
            axs_csdm[1].set_title(r"$\hat{R}_v$")
            axs_csdm[1].set_xlabel("Index")
            axs_csdm[1].set_ylabel("Index")
            # Ticks
            axs_csdm[1].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[1].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rs
            mean_Rs.plot(ax=axs_csdm[2], cmap="jet", x="h_index", vmax=vmax)
            axs_csdm[2].set_title(r"$\hat{R}_s = \hat{R}_x - \hat{R}_v$")
            axs_csdm[2].set_xlabel("Index")
            axs_csdm[2].set_ylabel("Index")
            # Ticks
            axs_csdm[2].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[2].set_yticks(np.arange(1, nrcv + 1, 1))

            # Save figure
            fpath = os.path.join(xr_data.root_img, "estimated_csdms.png")
            f_csdm.savefig(fpath)

            plt.close("all")

    def get_rtf(self, xr_data, Rv_global=None):
        ts = xr_data.ts

        if self.process_pulse_one_by_one:

            t_pulse = xr_data.pulse_duration
            t_interp_pulse = xr_data.inter_pulse_period

            # Time to add to ensure we englobe entire signal including last reflexions
            t_silence = t_interp_pulse - t_pulse
            tau_plus = 0.9 * t_silence  # Avoid to include following pulse
            tau_minus = 0.9 * (
                t_silence - self.tau_ir
            )  # Avoid to include previous pulse
            tau_minus = np.max(tau_minus, 0)  # In case tau_ir > t_silence

            # Time to first arrival
            t0 = xr_data.t0

            init_arr = True

            # Process each emission
            for i_pulse, pulse_id in enumerate(xr_data.pulse_id.values):

                # Extract the pulse of interest
                # x = xr_data.signal.sel(
                #     time=slice(
                #         t0 + pulse_id * t_interp_pulse - tau_minus - ts / 2,
                #         t0 + pulse_id * t_interp_pulse + t_pulse + tau_plus + ts / 2,
                #     )
                # )

                # Smallest arrival time in seconds from start (ie corresponding to closest OBS)
                tstart = xr_data.arr_time_in_sec_from_start.sel(pulse_id=pulse_id).min()
                # Longest arrival time in seconds from start (ie corresponding to furthest OBS)
                tend = xr_data.arr_time_in_sec_from_start.sel(pulse_id=pulse_id).max()
                # Select the corresponding time window
                x = xr_data.sel(
                    time=slice(
                        tstart - tau_minus - ts / 2,
                        tstart + t_pulse + tau_plus + ts / 2,
                    )
                )
                # Note: +/- ts/2 to ensure we include boundary samples

                # x = x.T  # Transpose to fit required format

                # # # Copy usefull attrs
                # x.attrs["inter_pulse_period"] = xr_data.inter_pulse_period
                # x.attrs["pulse_duration"] = xr_data.pulse_duration
                # x.attrs["n_emissions"] = xr_data.n_emissions
                # x.attrs["t_start"] = t0 + pulse_id * t_interp_pulse
                # x.attrs["t_end"] = (
                #     xr_data.t1 + pulse_id * t_interp_pulse + t_pulse + self.tau_ir
                # )

                # # Get mask defining signal+noise period
                # tt, mask_tt_x = self.get_signal_presence_mask(
                #     x, fs=1 / ts, nperseg=self.nperseg, noverlap=self.noverlap
                # )
                # mask_tt_v = ~mask_tt_x
                # x = x.signal.T
                # f, Rx = self.cm.get_signal_csdm(
                #     y=x, fs=1 / ts, add_identity=False, mask_tt=mask_tt_x
                # )

                mask_stft_x = self.get_signal_presence_mask_ft(
                    x,
                    tstart=tstart,
                    tend=tend + self.tau_ir,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                )
                mask_stft_v = ~mask_stft_x

                x = x.signal.T
                f, Rx = self.cm.get_signal_csdm(
                    y=x,
                    fs=1 / ts,
                    add_identity=False,
                    mask_tt=None,
                    mask_stft=mask_stft_x,
                )

                if Rv_global is not None:
                    Rv = Rv_global
                else:
                    # f, Rv = self.cm.get_signal_csdm(
                    #     y=x, fs=1 / ts, add_identity=False, mask_tt=mask_tt_v
                    # )
                    f, Rv = self.cm.get_signal_csdm(
                        y=x,
                        fs=1 / ts,
                        add_identity=False,
                        mask_tt=None,
                        mask_stft=mask_stft_v,
                    )

                # Rv[...] = 0  # TODO REMOVE

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
                    npulse = xr_data.sizes["pulse_id"]
                    nf = f.size

                    rtf_hat = np.empty(
                        (n_rcv, nf, npulse),
                        dtype=complex,
                    )
                    Rx_hat = np.empty(
                        (nf, n_rcv, n_rcv, npulse),
                        dtype=complex,
                    )
                    Rv_hat = np.empty(
                        (nf, n_rcv, n_rcv, npulse),
                        dtype=complex,
                    )
                    init_arr = False

                rtf_hat[..., i_pulse] = rtf.T
                Rx_hat[..., i_pulse] = Rx
                Rv_hat[..., i_pulse] = Rv

            xr_data.coords["f_rtf"] = f
            xr_data["rtf_amp_hat"] = (
                ["h_index", "f_rtf", "pulse_id"],
                np.abs(rtf_hat),
            )
            xr_data["rtf_phase_hat"] = (
                ["h_index", "f_rtf", "pulse_id"],
                np.angle(rtf_hat),
            )
            xr_data.attrs["h_index_ref"] = self.h_index_ref

            # Add Rx and R_v to the dataset
            xr_data.coords["f_csdm"] = f

            # Create h_index bis to avoid duplicate coordinates
            xr_data.coords["h_index_bis"] = xr_data.h_index.values
            xr_data["Rx"] = (
                ["f_csdm", "h_index", "h_index_bis", "pulse_id"],
                np.abs(Rx_hat),
            )
            xr_data["Rv"] = (
                ["f_csdm", "h_index", "h_index_bis", "pulse_id"],
                np.abs(Rv_hat),
            )

        else:
            # Covariance substraction
            x = xr_data.signal.T
            # Copy usefull attrs
            x.attrs["inter_pulse_period"] = xr_data.inter_pulse_period
            x.attrs["pulse_duration"] = xr_data.pulse_duration
            x.attrs["n_emissions"] = xr_data.n_emissions

            # Get mask defining signal+noise period
            tt, mask_tt_x = self.get_signal_presence_mask(
                x, fs=1 / ts, nperseg=self.nperseg, noverlap=self.noverlap
            )
            mask_tt_v = ~mask_tt_x

            # xr_data.coords["tt"] = tt
            # xr_data["mask_tt_x"] = (["tt"], mask_tt_x)

            # # Plot stfts
            # fig, axs = plt.subplots(nrows=stft_y.shape[0], ncols=1, sharex=True)
            # for ircv in range(stft_y.shape[0]):
            #     sg.plot_spectrogram(t=tt, f=ff, S_tf=stft_y[ircv, ...], ax=axs[ircv])
            #     axs[ircv].set_title(f"Rcv n°{ircv}")
            #     axs[ircv].set_ylim([0, 20000])
            # plt.suptitle("X")

            f, Rx = self.cm.get_signal_csdm(
                y=x, fs=1 / ts, add_identity=False, mask_tt=mask_tt_x
            )
            if Rv_global is not None:
                Rv = Rv_global
            else:
                f, Rv = self.cm.get_signal_csdm(
                    y=x, fs=1 / ts, add_identity=False, mask_tt=mask_tt_v
                )
            Rv[...] = 0  # TODO REMOVE

            if self.rtf_estimator == "cs":
                rtf = self.fp.rtf_estimator.estimate_rtf_covariance_subtraction(
                    Rx - Rv, use_first_column=True
                )
            elif self.rtf_estimator == "cs-evd":
                rtf = self.fp.rtf_estimator.estimate_rtf_covariance_subtraction(
                    Rx - Rv, use_first_column=False
                )
            else:
                print(f"{self.rtf_estimator} not implemented yet!")

            xr_data.coords["f_rtf"] = f
            xr_data["rtf_amp_hat"] = (
                ["h_index", "f_rtf"],
                np.abs(rtf).T,
            )
            xr_data["rtf_phase_hat"] = (
                ["h_index", "f_rtf"],
                np.angle(rtf).T,
            )
            xr_data.attrs["h_index_ref"] = self.h_index_ref

            # Add Rx and R_v to the dataset
            xr_data.coords["f_csdm"] = f

            # Create h_index bis to avoid duplicate coordinates
            xr_data.coords["h_index_bis"] = xr_data.h_index.values
            xr_data["Rx"] = (
                ["f_csdm", "h_index", "h_index_bis"],
                np.abs(Rx),
            )
            xr_data["Rv"] = (
                ["f_csdm", "h_index", "h_index_bis"],
                np.abs(Rv),
            )

        return xr_data

    # TODO : add wrapping function to set the required signal shape fct (LFM) ?
    # def get_signal_presence_mask_ft(self, x, tstart, tend, nperseg, noverlap):
    #     pass

    def derive_rtf_from_tf(self, xr_data):

        # Unpack usefull properties
        tf_ref = xr_data.tf_hat_amp.sel(h_index=self.h_index_ref) * np.exp(
            1j * xr_data.tf_hat_phase.sel(h_index=self.h_index_ref)
        )

        if self.process_pulse_one_by_one:
            rtf = np.zeros(
                (
                    xr_data.sizes["h_index"],
                    xr_data.sizes["f_ir"],
                    xr_data.sizes["pulse_id"],
                ),
                dtype=complex,
            )
            for i_hydro in range(xr_data.sizes["h_index"]):
                tf = xr_data.tf_hat_amp.isel(h_index=i_hydro) * np.exp(
                    1j * xr_data.tf_hat_phase.isel(h_index=i_hydro)
                )
                rtf[i_hydro, :] = tf / tf_ref

            xr_data["rtf_amp"] = (
                ["h_index", "f_ir", "pulse_id"],
                np.abs(rtf),
            )
            xr_data["rtf_phase"] = (
                ["h_index", "f_ir", "pulse_id"],
                np.angle(rtf),
            )
        else:

            rtf = np.zeros(
                (xr_data.sizes["h_index"], xr_data.sizes["f_ir"]), dtype=complex
            )
            for i_hydro in range(xr_data.sizes["h_index"]):
                tf = xr_data.tf_hat_amp.isel(h_index=i_hydro) * np.exp(
                    1j * xr_data.tf_hat_phase.isel(h_index=i_hydro)
                )
                rtf[i_hydro, :] = tf / tf_ref

            xr_data["rtf_amp"] = (
                ["h_index", "f_ir"],
                np.abs(rtf),
            )
            xr_data["rtf_phase"] = (
                ["h_index", "f_ir"],
                np.angle(rtf),
            )

        return xr_data


# ======================================================================================================================
#  Passive source localisation
# ======================================================================================================================
class PassiveFiberscopeManager(FiberscopeManager):
    """
    Class to handle passive source localisation. The source to localise is typically a ship of opportunity.
    """

    def __init__(
        self,
        root_processed_data: str,
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
        analysis_segment_duration: float = 10,
        analysis_segment_alpha_overlap: float = 0.5,
    ):
        """
        Class constructor

        """
        # Using super() to initialize the parent class
        super().__init__(
            root_processed_data=root_processed_data,
            root_img=root_img,
            bandfilter=bandfilter,
            tau_ir=tau_ir,
            alpha_overlap=alpha_overlap,
            h_index_ref=h_index_ref,
            plot_feature=plot_feature,
            theta_statistics=theta_statistics,
            process_pulse_one_by_one=process_pulse_one_by_one,
            estimate_ir_duration=estimate_ir_duration,
            rtf_estimator=rtf_estimator,
            obs_ids=obs_ids,
            verbose=verbose,
            plot_signal=plot_signal,
        )

        self.analysis_segment_duration = analysis_segment_duration
        self.analysis_segment_alpha_overlap = analysis_segment_alpha_overlap

    def process_analysis(
        self,
        ds_wav,
        t_start,
        t_end,
        set_stft_props=True,
    ):
        """
        Run analysis on the reduired recording analysis window.
        """

        if self.verbose:
            print(f"RTF processing of passive recording.")

        ### Step 1 - Load audio data for the required analysis window ###
        xr_data = self.load_recording(ds_wav, t_start, t_end)

        ### Step 2 - Init CovManager and FeatureProcessor
        # If stfts props are already set we dont need to do it
        if set_stft_props:
            self.set_stft_params(ts=xr_data.ts)
        # Index of hydrophone might not be sorted or not start at 0
        idx_rcv_ref = np.argmin(np.abs(xr_data.h_index.values - self.h_index_ref))
        # Init managers
        self.set_managers(fs=xr_data.fs, idx_rcv_ref=idx_rcv_ref)

        ### Step 3 - Derive features ###
        self.derive_feature(xr_data)

    def load_recording(self, ds_wav, t_start, t_end):
        """
        Load data for the required analysis window.

        Parameters
        ----------
        ds_wav : xr.Dataset
            Wav dataset (containing the entire recordings)
        t_start : datetime.datetime
            Start of the analysis window.
        t_end datetime.datetime
            End of the analysis window.

        Returns
        -------
        xr_data : xr.Dataset
            Selected portion of wav data for the required analysis window (form t_start to t_end).
        """

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
                signal=(["h_index", "time"], signal_mat), start_dt=t_start, end_dt=t_end
            ),
            coords=dict(
                h_index=self.obs_ids,
                time=common_time_vector,
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
        Rv_global=None,
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
            pass

        # Derive rtf from recordings
        xr_data = self.get_rtf(xr_data=xr_data, Rv_global=Rv_global)

        # Slice along frequency axis to ensure we never use information outside of the signal bandwidth
        # This also reduce the memory size required
        # xr_data = xr_data.sel(f_rtf=slice(xr_data.fmin, xr_data.fmax))
        # xr_data = xr_data.sel(f_ir=slice(xr_data.fmin, xr_data.fmax))
        # xr_data = xr_data.sel(f_csdm=slice(xr_data.fmin, xr_data.fmax))

        # Plot feature components for analysis if required
        if self.plot_feature:
            self.plot_estimated_feature_passive(xr_data)

        # Save results
        if save:
            xr_data.to_netcdf(
                os.path.join(xr_data.root_data, f"sequence_{xr_data.record_id}_rtf.nc")
            )
            xr_data.close()
        else:
            return xr_data

    def plot_estimated_feature_passive(self, xr_data):

        # Ensure img folder exists
        if not os.path.exists(xr_data.root_img):
            os.makedirs(xr_data.root_img)

        for segment_id in xr_data.segment_id.values:
            xr_data_seg = xr_data.sel(segment_id=segment_id)

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
            fpath = os.path.join(xr_data.root_img, f"rtf_amp_segmentID{segment_id}.png")
            f_amp.savefig(fpath)
            fpath = os.path.join(
                xr_data.root_img, f"rtf_phase_segmentID{segment_id}.png"
            )
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
            mean_Rx.plot(ax=axs_csdm[0], cmap="jet", x="h_index", vmax=vmax)
            axs_csdm[0].set_title(r"$\hat{R}_x$")
            axs_csdm[0].set_xlabel("Index")
            axs_csdm[0].set_ylabel("Index")
            # Ticks
            axs_csdm[0].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[0].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rv
            mean_Rv.plot(ax=axs_csdm[1], cmap="jet", x="h_index", vmax=vmax)
            axs_csdm[1].set_title(r"$\hat{R}_v$")
            axs_csdm[1].set_xlabel("Index")
            axs_csdm[1].set_ylabel("Index")
            # Ticks
            axs_csdm[1].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[1].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rs
            mean_Rs.plot(ax=axs_csdm[2], cmap="jet", x="h_index", vmax=vmax)
            axs_csdm[2].set_title(r"$\hat{R}_s = \hat{R}_x - \hat{R}_v$")
            axs_csdm[2].set_xlabel("Index")
            axs_csdm[2].set_ylabel("Index")
            # Ticks
            axs_csdm[2].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[2].set_yticks(np.arange(1, nrcv + 1, 1))

            # Save figure
            fpath = os.path.join(
                xr_data.root_img, f"estimated_csdms_segmentID{segment_id}.png"
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
            vmax = max(mean_Rx.values.max(), mean_Rv.values.max())

            # Plot Rx
            Rx.plot(ax=axs_csdm[0], cmap="jet", x="h_index", vmax=vmax)
            axs_csdm[0].set_title(r"$\hat{R}_x$")
            axs_csdm[0].set_xlabel("Index")
            axs_csdm[0].set_ylabel("Index")
            # Ticks
            axs_csdm[0].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[0].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rv
            Rv.plot(ax=axs_csdm[1], cmap="jet", x="h_index", vmax=vmax)
            axs_csdm[1].set_title(r"$\hat{R}_v$")
            axs_csdm[1].set_xlabel("Index")
            axs_csdm[1].set_ylabel("Index")
            # Ticks
            axs_csdm[1].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[1].set_yticks(np.arange(1, nrcv + 1, 1))

            # Plot Rs
            Rs.plot(ax=axs_csdm[2], cmap="jet", x="h_index", vmax=vmax)
            axs_csdm[2].set_title(r"$\hat{R}_s = \hat{R}_x - \hat{R}_v$")
            axs_csdm[2].set_xlabel("Index")
            axs_csdm[2].set_ylabel("Index")
            # Ticks
            axs_csdm[2].set_xticks(np.arange(1, nrcv + 1, 1))
            axs_csdm[2].set_yticks(np.arange(1, nrcv + 1, 1))

            # Save figure
            fpath = os.path.join(
                xr_data.root_img,
                f"estimated_csdms_segmentID{segment_id}_f_{Rx.f_csdm.values:.1f}Hz.png",
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

        # Process sucessive windows
        for i_window in range(n_window):

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
        xr_data.coords["f_rtf"] = f
        xr_data.coords["f_csdm"] = f
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
            t_centre_segment = xr_data.t_start + timedelta(seconds=t_centre_segment_s)
            segment_dt.append(t_centre_segment)

        segment_dt = np.array(segment_dt)

        # xr_data.coords["segment_id"] = np.arange(n_window)
        xr_data.coords["time"] = segment_dt

        # Add variables
        xr_data["rtf_amp_hat"] = (
            ["h_index", "f_rtf", "time"],
            np.abs(rtf_hat),
        )
        xr_data["rtf_phase_hat"] = (
            ["h_index", "f_rtf", "time"],
            np.angle(rtf_hat),
        )
        xr_data.attrs["h_index_ref"] = self.h_index_ref

        # Add Rx and R_v to the dataset
        xr_data["Rx"] = (
            ["f_csdm", "h_index", "h_index_bis", "time"],
            np.abs(Rx_hat),
        )
        xr_data["Rv"] = (
            ["f_csdm", "h_index", "h_index_bis", "time"],
            np.abs(Rv_hat),
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

    # Create an instance of FiberscopeManager
    # root_processed_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\data"
    # fsm = FiberscopeManager(
    #     root_processed_data=root_processed_data,
    #     h_index_ref=h_index_ref,
    #     plot_feature=plot_feature,
    # )

    # fsm.process_static_analysis(
    #     static_signal=fs_sweep1,
    #     static_records_names=fs_sweep1.records_N5,
    # )

    # from real_data_analysis.fiberscope_20.src.fiberscope_recording import (
    #     FiberscopeDynamicRecording,
    #     FiberscopeSweep1,
    # )

    # h_index_ref = 5
    # plot_feature = False

    # fs_dr = FiberscopeDynamicRecording()

    # # Create an instance of FiberscopeManager
    # root_processed_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\data"
    # fsm = FiberscopeManager(
    #     root_processed_data=root_processed_data,
    #     h_index_ref=h_index_ref,
    #     plot_feature=plot_feature,
    # )

    # # # preSplit the dynamic recording
    # fsm.presplit_dynamic_record(
    #     fs_dynamic_recording=fs_dr,
    #     n_sweep=8,
    #     n_records=95,
    # )

    # # # Split dynamic records and save as nc
    # fsm.split_dynamic_record(fs_dynamic_recording=fs_dr)

    # # Derive features
    # fsm.process_dyn_analysis(
    #     fs_dynamic_recording=fs_dr,
    #     use_global_noise_csdm=False,
    # )

    # # Load and preprocess static record
    # fs_sweep1 = FiberscopeSweep1()
    # fs_sweep1.records_folder = os.path.join(root_processed_data, "static")
    # if not os.path.exists(fs_sweep1.records_folder):
    #     os.makedirs(fs_sweep1.records_folder)

    # fsm.process_static_analysis(
    #     static_signal=fs_sweep1,
    #     static_records_names=fs_sweep1.records_N5,
    # )
    # # # Run localization process
    # d = fsm.localize_dyn_recording(
    #     static_signal=fs_sweep1,
    #     static_records_names=fs_sweep1.records_N5,
    #     fs_dynamic_recording=fs_dr,
    # )

    # fsm.plot_dyn_loc(
    #     d_rtf=d, axis_norm=1, time_step=fs_dr.time_step, vmin=-5, save_eps=True
    # )
