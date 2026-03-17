#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_mfp.py
@Time    :   2026/03/16 16:26:31
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to handle RTF-MFP
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
from datetime import datetime
from real_data_analysis.fiberscope_groix.src.fiberscope_groix_manager import (
    ActiveFiberscopeManager,
    PassiveFiberscopeManager,
    BandFilter,
)


# DEFAUTS # TODO move this in a dedicated file ?
ROOT_DATA = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data"
REF_RCV_ID = 1
RTF_ESTIMATOR = "cs-evd"
WAV_DATASET_FILEPATH = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data\sequences\wav_dataset.csv"


# ======================================================================================================================
# Class definition
# ======================================================================================================================


class RTF_MFP_Library:
    """
    Class to handle RTF-MFP library
    """

    def __init__(
        self,
        root_data: str = ROOT_DATA,
        reference_receiver_id: int = REF_RCV_ID,
        rtf_estimator: str = RTF_ESTIMATOR,
        fsm_active_kwargs: dict = {},
        fsm_passive_kwargs: dict = {},
        plot_library_replicas_features: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Constructor of the class
        """

        # Path to data folder
        self.root_data = root_data
        # Paths
        self.set_paths()
        # Load datasets
        self.load_datasets()

        # Index of the receiver to use as reference
        self.reference_receiver_id = reference_receiver_id

        # RTF estimator
        self.rtf_estimator = rtf_estimator
        # Flags
        self.plot_library_replicas_features = plot_library_replicas_features
        self.verbose = verbose

        # Define Fiberscope Managers
        self.fsm_active = None
        self.fsm_active_bandfilter = fsm_active_kwargs.get("bandfilter", None)
        self.fsm_active_tau_ir = fsm_active_kwargs.get("tau_ir", 3)
        self.fsm_active_process_pulse_one_by_one = fsm_active_kwargs.get(
            "pulse_one_by_one", True
        )
        self.fsm_active_estimate_ir_duration = fsm_active_kwargs.get(
            "estimate_ir_duration", False
        )

        self.fsm_passive = None
        self.fsm_passive_analysis_segment_duration = fsm_passive_kwargs.get(
            "analysis_segment_duration", 10
        )
        self.fsm_passive_analysis_segment_alpha_overlap = fsm_passive_kwargs.get(
            "analysis_segment_alpha_overlap", 0.5
        )

        self.set_fsm()

    def set_paths(self):
        """
        Set usefull paths
        """
        # Folder to store RTFs
        self.root_rtf_data = os.path.join(self.root_data, "sequences")
        # Dedicated folders for active and passive rtf data
        self.root_rtf_data_active = os.path.join(self.root_rtf_data, "active")
        self.root_rtf_data_passive = os.path.join(self.root_rtf_data, "passive")
        # Wav netcdf dataset filepath
        self.wav_dataset_filepath = os.path.join(self.root_data, "channel_H_wav.nc")
        # GPS netcdf dataset filepath
        self.gps_dataset_filepath = os.path.join(self.root_data, "gps.nc")
        # AIS netcdf dataset filepath
        self.ais_dataset_filepath = os.path.join(self.root_data, "ais.nc")
        # Bathy netcdf dataset filepath
        self.bathy_dataset_filepath = os.path.join(self.root_data, "bathy.nc")
        # Arrivals netcdf dataset filepath
        self.arrivals_dataset_filepath = os.path.join(
            self.root_data, f"processed_arrivals.nc"
        )

    def load_datasets(self):
        """
        Load usefull datasets
        """
        # Load wav dataset
        self.ds_wav = xr.open_dataset(self.wav_dataset_filepath)
        # Load GPS dataset
        self.ds_gps = xr.open_dataset(self.gps_dataset_filepath)
        # Load AIS dataset
        self.ds_ais = xr.open_dataset(self.ais_dataset_filepath)
        # Load bathy dataset
        self.ds_bathy = xr.open_dataset(self.bathy_dataset_filepath)
        # Load arrivals dataset
        self.ds_arrivals = xr.open_dataset(self.arrivals_dataset_filepath)
        self.df_arrivals = (
            self.ds_arrivals.to_dataframe()
        )  # Convert to dataframe for easier handling

    def set_fsm(self):
        """
        Set Fiberscope Managers
        """
        self.set_fsm_active()
        self.set_fsm_passive()
        self.set_fsm_props()

    def set_fsm_props(self):
        """
        Set Fiberscope Managers properties
        """
        # TODO : update this to avoid hardcoding any params
        fs = 2000
        tau_rtf_analysis = 3

        # Number of samples corresponding to the assumed impulse response duration
        n_rtf_analysis = int(tau_rtf_analysis * fs)
        # Get closer power of 2
        nperseg = 2 ** int(
            np.log2(n_rtf_analysis) + 1
        )  # Number of sample per snapshot to use = closest power of two
        alpha_overlap = 0.75
        noverlap = int(nperseg * alpha_overlap)

        self.fsm_nperseg = nperseg
        self.fsm_noverlap = noverlap

        if self.fsm_active is not None and self.fsm_passive is not None:
            # Active manager
            self.fsm_active.nperseg = self.fsm_nperseg
            self.fsm_active.noverlap = self.fsm_noverlap

            # Passive manager
            self.fsm_passive.nperseg = self.fsm_nperseg
            self.fsm_passive.noverlap = self.fsm_noverlap
        else:
            self.set_fsm()

    def set_fsm_active(
        self,
    ):
        """
        Set active Fiberscope Manager
        """
        self.fsm_active = ActiveFiberscopeManager(
            ds_wav=self.ds_wav,
            root_processed_data=self.root_rtf_data,
            h_index_ref=self.reference_receiver_id,
            plot_feature=self.plot_library_replicas_features,
            bandfilter=self.fsm_active_bandfilter,
            tau_ir=self.fsm_active_tau_ir,
            process_pulse_one_by_one=self.fsm_active_process_pulse_one_by_one,
            estimate_ir_duration=self.fsm_active_estimate_ir_duration,
            rtf_estimator=self.rtf_estimator,
            verbose=self.verbose,
        )

    def set_fsm_passive(self):
        """
        Set passive Fiberscope Manager
        """
        self.fsm_passive = PassiveFiberscopeManager(
            ds_wav=self.ds_wav,
            root_processed_data=self.root_rtf_data,
            h_index_ref=self.reference_receiver_id,
            plot_feature=self.plot_library_replicas_features,
            analysis_segment_duration=self.fsm_passive_analysis_segment_duration,
            analysis_segment_alpha_overlap=self.fsm_passive_analysis_segment_alpha_overlap,
            rtf_estimator=self.rtf_estimator,
            verbose=self.verbose,
        )

    def populate(
        self, active_replicas_args: dict = {}, passive_replicas_args: dict = {}
    ):
        """
        Method to populate the library
        """
        self.derive_replicas(
            active_replicas_args=active_replicas_args,
            passive_replicas_args=passive_replicas_args,
        )
        # self
        pass

    def derive_replicas(
        self, active_replicas_args: dict = {}, passive_replicas_args: dict = {}
    ):
        """
        Method to derive replicas
        """
        replica_sequence_ids = self.derive_replicas_active(
            active_replicas_args=active_replicas_args
        )
        self.derive_replicas_passive(passive_replicas_args=passive_replicas_args)

    def derive_replicas_active(self, active_replicas_args: dict = {}):
        """
        Method to derive replicas (active source)
        """
        # Unpack arguments
        replica_sequence_ids = active_replicas_args.get(
            "replica_sequence_ids", [144]
        )  # Default -> use trailed sequence 144
        replica_sequence_ids = np.atleast_1d(
            replica_sequence_ids
        )  # Ensure it's an array

        df_arrivals_selected = self.df_arrivals.loc[
            self.df_arrivals["sequence_id"].isin(replica_sequence_ids)
        ]

        # # Remove pulse along the transect (quicker)
        # pulse_max = 250
        # df_library = df_library.loc[df_library["pulse_id"] <= pulse_max]

        # Reset index
        df_arrivals_selected = df_arrivals_selected.reset_index(drop=True)

        # Derive replicas for selected sequences
        self.fsm_active.process_analysis(
            df_arrivals=df_arrivals_selected,
            set_stft_props=False,
        )

        return replica_sequence_ids

    def derive_replicas_passive(self, passive_replicas_args: dict = {}):
        """
        Method to derive replicas (passive source)
        """
        # Unpack arguments
        start_datetimes = passive_replicas_args.get(
            "start_datetimes",
            [datetime(year=2025, month=10, day=15, hour=1, minute=30, second=00)],
        )
        self.fsm_passive.process_analysis(
            t_start,
            t_end,
            set_stft_props=False,
        )


# ======================================================================================================================
# Test
# =====================================================================================================================


def test():
    """
    Test function
    """
    root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data"
    ref_rcv_id = 1
    rtf_estimator = "cs-evd"

    tau_ir_hat = 0.2  # estimated impulse response duration from sequence 144
    tau_ir_hat *= 2  # To ensure we include the entire response
    fsm_active_kwargs = {
        "bandfilter": BandFilter(order=4, lowcut=100, highcut=900),
        "tau_ir": tau_ir_hat,
        "process_pulse_one_by_one": True,
        "estimate_ir_duration": False,
    }
    fsm_passive_kwargs = {
        "analysis_segment_duration": 10,
        "analysis_segment_alpha_overlap": 0.5,
    }

    # Instantiate RTF-MFP library
    rtf_mfp_library = RTF_MFP_Library(
        root_data=root_data,
        reference_receiver_id=ref_rcv_id,
        rtf_estimator=rtf_estimator,
        fsm_active_kwargs=fsm_active_kwargs,
        fsm_passive_kwargs=fsm_passive_kwargs,
        plot_library_replicas_features=False,
        verbose=True,
    )

    # Populate library
    rtf_mfp_library.populate(
        active_replicas_args={"replica_sequence_ids": [144]},
        passive_replicas_args={},
    )


if __name__ == "__main__":
    test()
