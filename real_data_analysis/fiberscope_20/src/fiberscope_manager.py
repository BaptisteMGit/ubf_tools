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
import scipy.signal as sp
import matplotlib.pyplot as plt

from scipy import stats
import real_data_analysis.fiberscope_20.src.params as p

from propa.rtf.rtf_utils import D_hermitian_angle_fast
from real_data_analysis.deconvolution_utils import crosscorr_deconvolution
from real_data_analysis.fiberscope_20.src.read_tdms import load_fiberscope_data
from propa.rtf.rtf_localisation.zhang_et_al_testcase.src.feature_builder import (
    FeatureProcessor,
)
from source.cov_manager import CovManager
from misc import progression_bar


class FiberscopeManager:
    """
    Class to manage Fiberscope data analysis
    """

    def __init__(
        self,
        root_processed_data: str,
        root_tdms_data: str = p.root_tdms_data,
        root_img: str = p.root_img,
        tau_ir: float = p.tau_ir,
        alpha_overlap: float = p.alpha_overlap,
        subsampling_factor: float = p.subsampling_factor,
        h_index_ref: int = p.h_index_ref,
        plot_feature: bool = False,
        theta_statistics: str = "mean",
    ):
        """
        Constructor
        :param root_processed_data: Root data folder to store processed data
        :param root_tdms_data: Root data folder where the tdms files are stored
        :param root_img: Root data folder to store images
        """
        self.root_processed_data = root_processed_data
        self.root_tdms_data = root_tdms_data
        self.root_img = root_img

        # Impulse response duration used to derived appropriate stft params
        self.tau_ir = tau_ir

        # Overlap factor used to derived appropriate stft params
        self.alpha_overlap = alpha_overlap

        # Subsampling factor used to reduce data size
        self.subsampling_factor = subsampling_factor

        # Hydrophone use as reference to derive rtf
        self.h_index_ref = h_index_ref

        self.plot_feature = plot_feature

        # Method to derive caracteristic angle representing the theta distribution
        self.theta_statistics = theta_statistics

        # Define usefull objects
        self.cm = CovManager()

    def presplit_dynamic_record(self, fs_dynamic_recording, n_sweep=3, t_max=274.6):
        # Derive time step
        time_step = n_sweep * fs_dynamic_recording.signal.interp_pulse_period
        fs_dynamic_recording.time_step = time_step
        n_records = int(t_max / time_step)

        displacement_from_start_pos = [
            ((i + 1) * time_step - time_step / 2) * fs_dynamic_recording.src_speed
            for i in range(n_records)
        ]
        recording_names_dynamic = [
            f"{fs_dynamic_recording.recording_name}_{fs_dynamic_recording.src_start_pos}_r{np.round(dr, 2)}m_{fs_dynamic_recording.src_end_pos}"
            for dr in displacement_from_start_pos
        ]

        # Ensure folder exists
        dynamic_folder = os.path.join(
            self.root_processed_data, f"dynamic_{n_sweep}pulses"
        )
        if not os.path.exists(dynamic_folder):
            os.makedirs(dynamic_folder)

        # Store folder and recording names in the dynamic recording object
        fs_dynamic_recording.splitted_records_folder = dynamic_folder
        fs_dynamic_recording.splitted_records_names = recording_names_dynamic

        # Update signal object with select number of pulse
        fs_dynamic_recording.signal.n_sweep = n_sweep

    def split_dynamic_record(
        self,
        fs_dynamic_recording,
        hydro_to_process: int = None,
        force_reload: bool = False,
    ):

        # Assert that the dynamic recording has been presplit
        if not fs_dynamic_recording.splitted_records_names:
            self.presplit_dynamic_record(fs_dynamic_recording)
            warnings.warn(
                "Dynamic recording has not been presplit yet. presplit_dynamic_record was applied with default parameters."
            )

        ### Step 1 - load data ###
        full_data_fpath = os.path.join(
            self.root_processed_data, fs_dynamic_recording.recording_name + ".nc"
        )
        if not os.path.exists(full_data_fpath):
            full_data = self.tdms_to_xr(
                recording_name=fs_dynamic_recording.recording_name,
            )
            full_data.to_netcdf(full_data_fpath)
        else:
            full_data = xr.open_dataset(full_data_fpath)

        # List files in the target folder
        split_files = os.listdir(fs_dynamic_recording.splitted_records_folder)
        if not split_files or force_reload:
            # Drop stfts to drastically reduce the size of the dataset
            for var in ["stft_amp", "stft_phase", "ff", "tt"]:
                if var in full_data:
                    full_data = full_data.drop_vars(var)

            start_of_current_period = 0
            end_of_current_period = fs_dynamic_recording.time_step
            i_name = 0
            while end_of_current_period < full_data.time.max().values:
                # Extract the data corresponding to the current period
                split_data = full_data.sel(
                    time=slice(
                        start_of_current_period - full_data.ts, end_of_current_period
                    )
                )

                # Update time vector to start at 0
                split_data["time"] = split_data.time - start_of_current_period

                ### Step 2 - Preprocess data ###
                split_data.attrs["recording_name"] = (
                    fs_dynamic_recording.splitted_records_names[i_name]
                )
                split_data.attrs["root_data"] = (
                    fs_dynamic_recording.splitted_records_folder
                )

                self.preprocess_data(
                    xr_data=split_data,
                    signal=fs_dynamic_recording.signal,
                    hydro_to_process=hydro_to_process,
                )

                start_of_current_period += fs_dynamic_recording.time_step
                end_of_current_period += fs_dynamic_recording.time_step
                i_name += 1

            if i_name < len(fs_dynamic_recording.splitted_records_names):
                warnings.warn(
                    f"Number of splitted recordings ({i_name}) does not match the expected number ({len(fs_dynamic_recording.splitted_records_names)})"
                )
                # Restrict the number of splitted recordings to the number of processed ones
                fs_dynamic_recording.splitted_records_names = (
                    fs_dynamic_recording.splitted_records_names[:i_name]
                )
                # print(i_name)
                # print(len(fs_dynamic_recording.splitted_records_names))
        else:
            fs_dynamic_recording.splitted_records_names = [
                ".".join(sf.split(".")[:-1]) for sf in split_files if "rtf" not in sf
            ]

    def tdms_to_xr(self, recording_name):

        # Build filepath
        date = recording_name.split("T")[0]
        data_path = os.path.join(self.root_tdms_data, f"Campagne_{date}")
        file_name = f"{recording_name}.tdms"
        file_path = os.path.join(data_path, file_name)

        # Assert file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File {file_path} does not exist. Please check the file path."
            )

        # Load data into xr dataset
        xr_data = load_fiberscope_data(file_path, self.subsampling_factor)
        xr_data.attrs["recording_name"] = recording_name

        return xr_data

    def preprocess_data(self, xr_data, signal, hydro_to_process: int = None):

        # Unpack signal properties
        t_interp_pulse = signal.interp_pulse_period
        t_pulse = signal.t_pulse
        t_ir = signal.ir_duration
        n_em = signal.n_sweep
        f0 = signal.fmin
        f1 = signal.fmax

        # Add attrs
        img_path = os.path.join(self.root_img, xr_data.recording_name)
        xr_data.attrs["img_path"] = img_path

        # Restrict to desired hydrophone if specified
        if hydro_to_process is not None:
            xr_data = xr_data.sel(
                h_index=slice(
                    hydro_to_process,
                )
            )  # Select hydrophone while keeping the dimension

        n_hydro = xr_data.sizes["h_index"]

        # Create the source pulse signal
        ts = xr_data.ts
        # t = xr_data.signal.sel(time=slice(0, t_ir)).time.values
        # x = sp.chirp(t, f0=f0, f1=f1, t1=t.max(), method="linear")
        t = xr_data.signal.sel(time=slice(0, t_pulse)).time.values
        x = sp.chirp(t, f0=f0, f1=f1, t1=t_pulse, method="linear")

        # sweep_hat = np.zeros((n_em, len(t)))
        # ri_hat = np.zeros((n_hydro, n_em, len(t)))
        init_ri_hat = True
        # Loop over each hydrophone to process
        # for i_hydro in range(n_hydro):
        i_hydro = 0
        for hydro_idx in xr_data.h_index.values:

            # Process each emission
            for i_em in range(n_em):
                # Extract the emission
                # hydro_idx = xr_data.h_index.isel(h_index=i_hydro)
                y = xr_data.signal.sel(
                    time=slice(
                        i_em * t_interp_pulse - ts, i_em * t_interp_pulse + t_ir
                    ),
                    h_index=hydro_idx,
                )

                if init_ri_hat:
                    ri_hat = np.zeros((n_hydro, n_em, y.sizes["time"]))
                    time = y.time.values
                    init_ri_hat = False

                # Estimate the impulse response
                h_hat = crosscorr_deconvolution(x=x, y=y.values)
                ri_hat[i_hydro, i_em, :] = h_hat

            i_hydro += 1

        # Take the mean impulse response over all sweeps analysed
        ri_hat_mean = np.mean(ri_hat, axis=1)

        xr_data["t_ir"] = time
        xr_data["ri_hat"] = (
            ["h_index", "t_ir"],
            ri_hat_mean,
        )

        # Derive the corresponding frequency response
        # nstft = self.nperseg
        nstft = time.size
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

        # Save results
        xr_data.to_netcdf(
            os.path.join(xr_data.root_data, f"{xr_data.recording_name}.nc")
        )
        xr_data.close()
        del xr_data

    def process_dyn_analysis(
        self,
        fs_dynamic_recording,
        use_global_noise_csdm=False,
        set_stft_props=True,
        rtf_estimator="cs",
    ):
        # Assert that the dynamic recording has been presplit
        if not fs_dynamic_recording.splitted_records_names:
            self.presplit_dynamic_record(fs_dynamic_recording)
            warnings.warn(
                "Dynamic recording has not been split yet. presplit_dynamic_record was applied with default parameters."
            )

        # Assert splitted recordings exist
        if not np.all(
            [
                os.path.exists(
                    os.path.join(
                        fs_dynamic_recording.splitted_records_folder, name + ".nc"
                    )
                )
                for name in fs_dynamic_recording.splitted_records_names
            ]
        ):
            raise FileNotFoundError(
                "Some splitted recordings do not exist. Please check the folder and recording names."
            )

        # Load entire signal
        fpath = os.path.join(
            self.root_processed_data, fs_dynamic_recording.recording_name + ".nc"
        )
        xr_data = xr.open_dataset(fpath)
        ts = xr_data.ts

        # If stfts props are already set we dont need to do it
        if set_stft_props:
            self.set_stft_params(ts=ts)

        idx_rcv_ref = np.argmin(
            np.abs(xr_data.h_index.values - self.h_index_ref)
        )  # Index of hydrophone might not be sorted or not start at 0
        self.set_managers(fs=1 / ts, idx_rcv_ref=idx_rcv_ref)

        if use_global_noise_csdm:
            ff, tt, Rv_global = self.estimate_global_csdm(xr_data)
        else:
            Rv_global = None

        i_test = 0
        prev_progress = 0
        n_test = len(fs_dynamic_recording.splitted_records_names)
        print("\nDerive RTF for each segment of the dynamic recording")

        for recording_name in fs_dynamic_recording.splitted_records_names:
            i_test += 1
            prev_progress = progression_bar(
                index=i_test,
                index0=0,
                indexf=n_test,
                prev_progress=prev_progress,
            )

            self.derive_feature(
                recording_name=recording_name,
                records_folder=fs_dynamic_recording.splitted_records_folder,
                signal=fs_dynamic_recording.signal,
                Rv_global=Rv_global,
                # signal=fs_dynamic_recording.signal,
                rtf_estimator=rtf_estimator,
            )

    def set_stft_params(self, ts):

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

    def set_managers(self, fs, idx_rcv_ref):
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

    def process_static_analysis(
        self,
        static_signal,
        static_records_names,
        set_stft_props=True,
        rtf_estimator="cs",
    ):

        i_test = 0
        prev_progress = 0
        n_test = len(static_records_names)
        print("\nDerive RTF for each static recording")

        for recording_name in static_records_names:
            i_test += 1
            prev_progress = progression_bar(
                index=i_test,
                index0=0,
                indexf=n_test,
                prev_progress=prev_progress,
            )

            ### Step 1 - Load data from tdms ###
            xr_data = self.tdms_to_xr(
                recording_name=recording_name,
            )
            xr_data.attrs["recording_name"] = recording_name
            xr_data.attrs["root_data"] = static_signal.records_folder

            ts = xr_data.ts
            # If stfts props are already set we dont need to do it
            if set_stft_props:
                self.set_stft_params(ts=ts)

            idx_rcv_ref = np.argmin(
                np.abs(xr_data.h_index.values - self.h_index_ref)
            )  # Index of hydrophone might not be sorted or not start at 0
            self.set_managers(fs=1 / ts, idx_rcv_ref=idx_rcv_ref)

            ### Step 2 - Preprocess data ###
            self.preprocess_data(
                xr_data=xr_data,
                signal=static_signal,
                # hydro_to_process=hydro_to_process,
            )

            ### Step 3 - Derive features ###
            self.derive_feature(
                recording_name=recording_name,
                records_folder=static_signal.records_folder,
                signal=static_signal,
                rtf_estimator=rtf_estimator,
            )

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

    def derive_feature(
        self,
        recording_name,
        records_folder,
        signal,
        Rv_global=None,
        # gcc_methods=["blank", "scot", "phat", "ml"],
        verbose=False,
        save=True,
        rtf_estimator="cs",
    ):

        if verbose:
            print(f"Processing recording {recording_name} - RTF estimation")

        # Load data
        name_parts = recording_name.split(".")
        if "nc" not in name_parts:  # No extension
            recording_name += ".nc"

        fpath = os.path.join(records_folder, recording_name)
        # Assert fpath exist
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"File {fpath} does not exist. Please check the file path."
            )

        xr_data = xr.open_dataset(fpath)

        # Derive rtf from recordings
        xr_data = self.get_rtf(
            xr_data=xr_data, Rv_global=Rv_global, rtf_estimator=rtf_estimator
        )

        # # Derive rtf from tf estimated by deconvolution
        xr_data = self.derive_rtf_from_tf(xr_data=xr_data)

        # Slice along frequency axis to ensure we never use information outside of the signal bandwidth
        # This also reduce the memory size required
        xr_data = xr_data.sel(f_rtf=slice(signal.fmin, signal.fmax))
        xr_data = xr_data.sel(f_ir=slice(signal.fmin, signal.fmax))
        xr_data = xr_data.sel(f_csdm=slice(signal.fmin, signal.fmax))

        # Plot feature components for analysis if required
        if self.plot_feature:
            self.plot_estimated_feature(xr_data)
        # # Derive GCC for comparison
        # xr_data = derive_gcc(xr_data=xr_data, gcc_methods=gcc_methods)

        # Save results
        if save:
            xr_data.to_netcdf(
                os.path.join(xr_data.root_data, f"{xr_data.recording_name}_rtf.nc")
            )
            xr_data.close()
        else:
            return xr_data

    def plot_estimated_feature(self, xr_data):

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
        if not os.path.exists(xr_data.img_path):
            os.makedirs(xr_data.img_path)

        # Save figures
        fpath = os.path.join(xr_data.img_path, "rtf_amp.png")
        f_amp.savefig(fpath)
        fpath = os.path.join(xr_data.img_path, "rtf_phase.png")
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
        fpath = os.path.join(xr_data.img_path, "estimated_csdms.png")
        f_csdm.savefig(fpath)

        plt.close("all")

    def get_rtf(self, xr_data, Rv_global=None, rtf_estimator="cs"):
        ts = xr_data.ts

        # Covariance substraction
        x = xr_data.signal.T

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

        if rtf_estimator == "cs":
            rtf = self.fp.rtf_estimator.estimate_rtf_covariance_subtraction(
                Rx - Rv, use_first_column=True
            )
        elif rtf_estimator == "cs-evd":
            rtf = self.fp.rtf_estimator.estimate_rtf_covariance_subtraction(
                Rx - Rv, use_first_column=False
            )
        else:
            print(f"{rtf_estimator} not implemented yet!")

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

    def get_signal_presence_mask(self, x, fs, nperseg, noverlap):
        ff, tt, stft_x = self.cm.get_stft_array(
            x, fs=fs, nperseg=nperseg, noverlap=noverlap
        )

        duration = x.time.max() - x.time.min()
        # Define the signal presence mask
        mask_tt_x = np.zeros_like(tt, dtype=int)
        for ircv in range(stft_x.shape[0]):
            energy = np.sum(np.abs(stft_x[ircv, ...]) ** 2, axis=0)

            if duration > 1:
                # Old method before 27/10/2025
                min_height = 0.2 * np.max(energy)
                # min_height = np.median(energy)
                idx_peaks = sp.find_peaks(energy, height=min_height)[0]
                min_peaks = np.min(energy[idx_peaks])
                # threshold = 0.3 * min_peaks
                threshold = 0.005 * min_peaks
            else:
                # Other simple method
                threshold = np.median(energy)

            # Simpler method 27/10/2025 not working well
            # threshold = np.max(1.1 * np.min(energy), 0.1 * np.max(energy))

            # Define signal presence mask
            mask_tt_i = energy > threshold
            mask_tt_x = np.logical_or(mask_tt_x, mask_tt_i)

        # # For debug purpose
        # plt.figure()
        # plt.pcolormesh(tt, ff, np.abs(stft_x[0, ...]))
        # plt.plot(tt, mask_tt_x.astype(int) * np.max(ff))
        # plt.savefig("test.png")

        return tt, mask_tt_x

    def derive_rtf_from_tf(self, xr_data):

        # Unpack usefull properties
        tf_ref = xr_data.tf_hat_amp.sel(h_index=self.h_index_ref) * np.exp(
            1j * xr_data.tf_hat_phase.sel(h_index=self.h_index_ref)
        )

        rtf = np.zeros((xr_data.sizes["h_index"], xr_data.sizes["f_ir"]), dtype=complex)
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

    def localize_dyn_recording(
        self, static_signal, static_records_names, fs_dynamic_recording
    ):
        # Localizing static records
        d = []

        # Order static records names by position order
        position_ids = [int(name.split("_")[1][1]) for name in static_records_names]
        # Sort the static records names by position
        sorted_indices = np.argsort(position_ids)
        static_records_names = [static_records_names[i] for i in sorted_indices]

        # Init progress bar
        i_test = 0
        prev_progress = 0
        n_test = len(static_records_names)
        print("\nCompute distance map")

        for recording_name in static_records_names:

            i_test += 1
            prev_progress = progression_bar(
                index=i_test,
                index0=0,
                indexf=n_test,
                prev_progress=prev_progress,
            )

            fpath = os.path.join(
                static_signal.records_folder, recording_name + "_rtf.nc"
            )
            xr_data_event = xr.open_dataset(fpath)

            # Localize using rtf
            d_rtf = self.localize_dyn_recording_rtf(xr_data_event, fs_dynamic_recording)
            d.append(d_rtf)

            # Derive constrast q
            # Eq 1.106 rapport RTF
            # q_rtf = (np.max(d_rtf) - d_rtf) / (np.max(d_rtf) - np.min(d_rtf))
            # q.append(q_rtf)

        # q = np.array(q)
        d = np.array(d)

        return d

    def localize_dyn_recording_rtf(self, xr_data_event, fs_dynamic_recording):

        # TODO add assertion to check if previous steps have been covered

        # Reference rtf vector = rtf vector at the event position
        sig = fs_dynamic_recording.signal
        xr_data_event = xr_data_event.sel(f_rtf=slice(sig.fmin, sig.fmax))
        rtf_event = xr_data_event.rtf_amp_hat * np.exp(1j * xr_data_event.rtf_phase_hat)
        # rtf_event_true = xr_data_event.rtf_amp * np.exp(1j * xr_data_event.rtf_phase)
        # List to store distance with each successive position
        dist = []

        # Sort the dynamic recordings by position
        dist_from_P1 = [
            float(r_name.split("_")[-2][1:-1])
            for r_name in fs_dynamic_recording.splitted_records_names
        ]
        # Sort the dynamic recordings by position
        sorted_indices = np.argsort(dist_from_P1)
        splitted_records_names = [
            fs_dynamic_recording.splitted_records_names[i] for i in sorted_indices
        ]

        # Set distance args to use
        if self.theta_statistics == "mean":
            apply_mean = True
        elif self.theta_statistics == "expectation":
            apply_mean = False
        dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": apply_mean}

        # Iterate over dynamic recordings
        for recording_name in splitted_records_names:
            # Load data
            fpath = os.path.join(
                fs_dynamic_recording.splitted_records_folder,
                recording_name + "_rtf.nc",
            )
            # Assert fpath exist
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"File {fpath} does not exist. Please check the file path."
                )
            xr_data_library_i = xr.open_dataset(fpath)
            xr_data_library_i = xr_data_library_i.sel(f_rtf=slice(sig.fmin, sig.fmax))
            rtf_library_i = xr_data_library_i.rtf_amp_hat * np.exp(
                1j * xr_data_library_i.rtf_phase_hat
            )

            # Interpolate rtf_event at rtf_library freq (dynamic recording uses smaller window to
            # match the number of segment L )
            rtf_event = rtf_event.sel(
                f_rtf=rtf_library_i.f_rtf.values, method="nearest"
            )

            # Derive distance using hermitian angle
            theta = D_hermitian_angle_fast(
                rtf_event.values, rtf_library_i.values, **dist_kwargs
            )
            theta_c = get_theta_c(val=theta, apply_mean=apply_mean)
            dist.append(theta_c)

        dist = np.array(dist)

        return dist

    def plot_dyn_loc(
        self, d_rtf, time_step, axis_norm=1, fname=None, vmin=-5, save_eps=False
    ):
        d = -d_rtf

        if axis_norm is None:
            d_max = np.nanmax(d, axis=axis_norm) * np.ones_like(d)
            d_min = np.nanmin(d, axis=axis_norm) * np.ones_like(d)
            norm_label = "norm_over_entire_surface"
        else:
            d_max = np.tile(
                np.nanmax(d, axis=axis_norm), (d.shape[axis_norm], 1)
            )  # Cast to d shape
            d_min = np.tile(np.nanmin(d, axis=axis_norm), (d.shape[axis_norm], 1))
            if axis_norm == 1:
                norm_label = f"norm_along_time_axis"
            elif axis_norm == 0:
                norm_label = f"norm_along_position_axis"

        if axis_norm == 1:
            d_max = d_max.T
            d_min = d_min.T

        # Normalize
        q = (d - d_min) / (d_max - d_min)

        # In dB
        q[q == 0] = 1e-6
        q_dB = 10 * np.log10(q)

        t = np.arange(0, d.shape[1]) * time_step
        ordered_pos = [f"$P_{i}$" for i in range(1, 7)]
        truepos_order = [0, 5, 1, 4, 2, 3]
        q_dB = q_dB[truepos_order, :]
        plt.figure()
        plt.imshow(q_dB, cmap="jet", aspect="auto", vmin=vmin, rasterized=False)
        plt.xticks(np.arange(0, q_dB.shape[1], 10), np.round(t[::10], 2))
        plt.yticks(np.arange(0, q.shape[0]), ordered_pos)
        plt.xlabel("Time [s]")
        plt.ylabel("Position")
        plt.colorbar(label=r"$q\, \textrm{[dB]}$")
        plt.gca().invert_yaxis()

        folder = os.path.join(self.root_img, "localization", norm_label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if fname is None:
            fname = f"dyn_qdB_href{self.h_index_ref}.png"

        fpath = os.path.join(folder, fname)
        plt.savefig(fpath)

        if save_eps:
            fname = fname.split(".")[0] + ".eps"
            fpath = os.path.join(folder, fname)
            plt.savefig(fpath, format="eps")


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
    from real_data_analysis.fiberscope_20.src.fiberscope_recording import (
        FiberscopeDynamicRecording,
        FiberscopeSweep1,
    )

    h_index_ref = 5
    plot_feature = False

    fs_dr = FiberscopeDynamicRecording()

    # Create an instance of FiberscopeManager
    root_processed_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_20\data"
    fsm = FiberscopeManager(
        root_processed_data=root_processed_data,
        h_index_ref=h_index_ref,
        plot_feature=plot_feature,
    )

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

    # Load and preprocess static record
    fs_sweep1 = FiberscopeSweep1()
    fs_sweep1.records_folder = os.path.join(root_processed_data, "static")
    if not os.path.exists(fs_sweep1.records_folder):
        os.makedirs(fs_sweep1.records_folder)

    fsm.process_static_analysis(
        static_signal=fs_sweep1,
        static_records_names=fs_sweep1.records_N5,
    )

    # # Run localization process
    # d = fsm.localize_dyn_recording(
    #     static_signal=fs_sweep1,
    #     static_records_names=fs_sweep1.records_N5,
    #     fs_dynamic_recording=fs_dr,
    # )

    # fsm.plot_dyn_loc(
    #     d_rtf=d, axis_norm=1, time_step=fs_dr.time_step, vmin=-5, save_eps=True
    # )
