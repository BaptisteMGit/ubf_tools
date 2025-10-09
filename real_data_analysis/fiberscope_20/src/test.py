#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test.py
@Time    :   2025/05/03 17:29:25
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to test other module functionalities
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from real_data_analysis.fiberscope_20.src.fiberscope_manager import FiberscopeManager
from real_data_analysis.fiberscope_20.src.fiberscope_recording import (
    FiberscopeDynamicRecording,
    FiberscopeSweep1,
)


class TestFiberscopeManager:
    """
    Class to test FiberscopeManager functionalities
    """

    def __init__(self):
        """
        Constructor
        """

        self.fs_dynamic_recording = FiberscopeDynamicRecording()

        # Instance of FiberscopeManager
        root_processed_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_20\data"
        self.manager = FiberscopeManager(root_processed_data=root_processed_data)

        self.img_folder = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_20\img\tests"

    def test_signal_subsampling(self):
        # Load original data
        data_o = self.manager.tdms_to_xr(
            recording_name=self.fs_dynamic_recording.recording_name,
            subsampling_factor=None,
        )

        # Load subsampled data
        data_ss = self.manager.tdms_to_xr(
            recording_name=self.fs_dynamic_recording.recording_name,
            subsampling_factor=4,
        )

        # Plot both signal
        plt.figure()
        data_o.signal.isel(h_index=0).sel(time=slice(0.04, 0.05)).plot(
            label=f"fs = {data_o.fs:.1f} Hz"
        )
        data_ss.signal.isel(h_index=0).sel(time=slice(0.04, 0.05)).plot(
            label=f"fs = {data_ss.fs:.1f} Hz", color="r", linestyle="--"
        )
        fpath = os.path.join(self.img_folder, "subsampling.png")
        plt.legend()
        plt.savefig(fpath)

    def test_rtf_vs_old_code_static(self):
        # recording = r"09-10-2024T10-39-11-308093_P1_N5_Sweep_38_rtf.nc"
        recording = r"09-10-2024T16-55-08-243011_P2_N5_Sweep_97_rtf.nc"

        # Load processed data with fiberscope manager
        root_fsm = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\data\static"
        fpath = os.path.join(root_fsm, recording)
        data_fsm = xr.open_dataset(fpath)
        rtf_fsm = data_fsm.rtf_amp_hat * np.exp(1j * data_fsm.rtf_phase_hat)

        # Load processed data with fiberscope_publi
        root_publi = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\data_processed"
        fpath = os.path.join(root_publi, recording)
        data_publi = xr.open_dataset(fpath)
        rtf_publi = data_publi.rtf_amp_cs * np.exp(1j * data_publi.rtf_phase_cs)

        f, axs = plt.subplots(nrows=rtf_fsm.sizes["h_index"], sharex=True)
        for i, hydro_idx in enumerate(rtf_fsm.h_index.values):
            np.abs(rtf_fsm).sel(h_index=hydro_idx).plot(
                label=f"FiberscopeManager - {hydro_idx}", ax=axs[i]
            )
            np.abs(rtf_publi).sel(h_index=hydro_idx).plot(
                label=f"Publi - {hydro_idx}", ax=axs[i]
            )
            axs[i].set_yscale("log")
            axs[i].set_xlim([1e4, 1.3 * 1e4])
            axs[i].set_title("")
            axs[i].set_xlabel("")
            axs[i].set_ylim([0.1, 10])
            axs[i].legend(fontsize=8)
        # plt.legend()
        plt.suptitle(recording)
        fpath = os.path.join(self.img_folder, "fiberscope_vs_publi_static.png")
        plt.savefig(fpath)

    def test_rtf_vs_old_code_dyn(self):
        recording = r"10-10-2024T16-53-43-200271_PR_N1_346_P1_r12.15m_P4_rtf.nc"
        # recording = r"10-10-2024T16-53-43-200271_PR_N1_346_P1_r0.15m_P4_rtf.nc"
        # Load processed data with fiberscope manager
        root_fsm = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\data\dynamic_3pulses"
        recordings = [file for file in os.listdir(root_fsm) if "rtf" in file]
        for recording in recordings:
            fpath = os.path.join(root_fsm, recording)
            data_fsm = xr.open_dataset(fpath)
            rtf_fsm = data_fsm.rtf_amp_hat * np.exp(1j * data_fsm.rtf_phase_hat)

            # Load processed data with fiberscope_publi
            root_publi = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\data_processed"
            fpath = os.path.join(root_publi, recording)
            data_publi = xr.open_dataset(fpath)
            rtf_publi = data_publi.rtf_amp_cs * np.exp(1j * data_publi.rtf_phase_cs)

            f, axs = plt.subplots(nrows=rtf_fsm.sizes["h_index"], sharex=True)
            for i, hydro_idx in enumerate(rtf_fsm.h_index.values):
                np.abs(rtf_fsm).sel(h_index=hydro_idx).plot(
                    label=f"FiberscopeManager - {hydro_idx}", ax=axs[i]
                )
                np.abs(rtf_publi).sel(h_index=hydro_idx).plot(
                    label=f"Publi - {hydro_idx}", ax=axs[i]
                )
                axs[i].set_yscale("log")
                axs[i].set_xlim([1e4, 1.3 * 1e4])
                axs[i].set_title("")
                axs[i].set_xlabel("")
                axs[i].set_ylim([0.1, 10])
                axs[i].legend(fontsize=8)
            plt.suptitle(recording)
            pos_lab = recording.split("_")[-3]
            fpath = os.path.join(
                self.img_folder, "fiberscope_vs_publi_dyn", f"{pos_lab}.png"
            )
            plt.savefig(fpath)

    def test_ir_duration(self):
        "Test the influence of the ir_duration on the localisation performance"
        t_ir = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        for tau_ir in t_ir:
            self.run_dyn_loc(h_index_ref=5, n_sweep=3, tau_ir=tau_ir)

    def test_h_index_ref(self):
        "Test the influence of the h_index_ref on the localisation performance"
        h_index_ref = [1, 2, 3, 4, 5]
        for idx in h_index_ref:
            self.run_dyn_loc(h_index_ref=idx, n_sweep=3, tau_ir=0.5)

    def test_nsweep(self):
        "Test the influence of the n_sweep on the localisation performance"
        n_sweep = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        for n_s in n_sweep:
            self.run_dyn_loc(h_index_ref=5, n_sweep=n_s, tau_ir=0.5)

    def run_dyn_loc(self, h_index_ref, n_sweep, tau_ir):

        fs_dr = FiberscopeDynamicRecording()

        # Create an instance of FiberscopeManager
        root_processed_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\data"
        fsm = FiberscopeManager(
            root_processed_data=root_processed_data,
            h_index_ref=h_index_ref,
            tau_ir=tau_ir,
            plot_feature=False,
            # root_img=self.img_folder,
        )

        # # preSplit the dynamic recording
        n_sweep_tot = 290
        fsm.presplit_dynamic_record(
            fs_dynamic_recording=fs_dr,
            n_sweep=n_sweep,
            n_records=int(np.ceil(n_sweep_tot / n_sweep)),
        )

        # # Split dynamic records and save as nc
        fsm.split_dynamic_record(fs_dynamic_recording=fs_dr)

        # # Derive features
        fsm.process_dyn_analysis(
            fs_dynamic_recording=fs_dr,
            use_global_noise_csdm=False,
        )

        # Load and preprocess static record
        fs_sweep1 = FiberscopeSweep1()
        fs_sweep1.records_folder = os.path.join(root_processed_data, "static")
        if not os.path.exists(fs_sweep1.records_folder):
            os.makedirs(fs_sweep1.records_folder)

        fsm.process_static_analysis(
            static_signal=fs_sweep1,
            static_records_names=fs_sweep1.records_N5,
        )
        # Run localization process
        d = fsm.localize_dyn_recording(
            static_signal=fs_sweep1,
            static_records_names=fs_sweep1.records_N5,
            fs_dynamic_recording=fs_dr,
        )

        fname = f"nsweep_{n_sweep}_ir_{tau_ir}_hindex_{h_index_ref}.png"
        fsm.plot_dyn_loc(d_rtf=d, axis_norm=1, time_step=fs_dr.time_step, fname=fname)


if __name__ == "__main__":
    test = TestFiberscopeManager()

    # test.test_signal_subsampling()

    # test.test_rtf_vs_old_code_static()
    # test.test_rtf_vs_old_code_dyn()

    # test.run_dyn_loc(h_index_ref=5, n_sweep=3, tau_ir=0.5)
    # test.test_h_index_ref()
    test.test_ir_duration()
    test.test_nsweep()
