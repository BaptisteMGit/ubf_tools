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
            root_processed_data=self.root_data,
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
            root_processed_data=self.root_data,
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
        # Derive replicas
        active_replicas_info, passive_replicas_info = self.derive_replicas(
            active_replicas_args=active_replicas_args,
            passive_replicas_args=passive_replicas_args,
        )

        # Load derived replicas
        self.load_replicas(
            active_replicas_info=active_replicas_info,
            passive_replicas_info=passive_replicas_info,
        )

    def derive_replicas(
        self, active_replicas_args: dict = {}, passive_replicas_args: dict = {}
    ):
        """
        Method to derive replicas
        """
        active_replicas_info = self.derive_replicas_active(
            active_replicas_args=active_replicas_args
        )
        passive_replicas_info = self.derive_replicas_passive(
            passive_replicas_args=passive_replicas_args
        )

        return active_replicas_info, passive_replicas_info

    def derive_replicas_active(self, active_replicas_args: dict = {}) -> dict:
        """
        Method to derive replicas (active source)
        """
        if self.verbose:
            print("\nDeriving replicas for active source...")

        # Unpack arguments
        replica_sequence_ids = active_replicas_args.get(
            "replica_sequence_ids", [144]
        )  # Default -> use trailed sequence 144
        replica_sequence_ids = np.atleast_1d(
            replica_sequence_ids
        )  # Ensure it's an array
        load_precomputed_replicas = active_replicas_args.get(
            "load_precomputed_replicas", True
        )  # If True, will load precomputed replicas if they exist, otherwise will compute them

        replica_sequence_ids_to_compute = (
            replica_sequence_ids.copy()
        )  # Initialize list of sequence ids to compute
        if load_precomputed_replicas:
            for rep_seq_id in replica_sequence_ids:
                rep_filepath = os.path.join(
                    self.root_rtf_data_active, f"sequence_{rep_seq_id}_rtf.nc"
                )
                if os.path.exists(rep_filepath):
                    # Check file validity
                    ds_replica = xr.open_dataset(rep_filepath)
                    valid = True
                    if ds_replica.h_index_ref != self.reference_receiver_id:
                        print(
                            f"Replica file for sequence {rep_seq_id} found at {rep_filepath} is not valid: reference receiver id mismatch (expected {self.reference_receiver_id}, found {ds_replica.h_index_ref}). This sequence will be recomputed."
                        )
                        valid = False
                        # Avoid permission error when trying to open the file again in the future by closing and deleting the dataset before recomputing
                        ds_replica.close()  # Close the dataset to avoid memory leak
                        del ds_replica  # Delete dataset

                    if valid:
                        print(
                            f"Precomputed replica for sequence {rep_seq_id} found at {rep_filepath}. This sequence will be loaded instead of being computed."
                        )
                        # Remove this sequence id from the list of sequence ids to compute
                        replica_sequence_ids_to_compute = (
                            replica_sequence_ids_to_compute[
                                replica_sequence_ids_to_compute != rep_seq_id
                            ]
                        )

        if len(replica_sequence_ids_to_compute) > 0:
            df_arrivals_selected = self.df_arrivals.loc[
                self.df_arrivals["sequence_id"].isin(replica_sequence_ids_to_compute)
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

        active_replicas_info = {
            "replica_sequence_ids": replica_sequence_ids,
        }
        return active_replicas_info

    def derive_replicas_passive(self, passive_replicas_args: dict = {}) -> dict:
        """
        Method to derive replicas (passive source)
        """
        if self.verbose:
            print("\nDeriving replicas for passive source...")

        # Unpack arguments
        start_datetimes = passive_replicas_args.get(
            "start_datetimes",
            [datetime(year=2025, month=10, day=15, hour=00, minute=15, second=00)],
        )
        start_datetimes = np.atleast_1d(start_datetimes)
        end_datetimes = passive_replicas_args.get(
            "end_datetimes",
            [datetime(year=2025, month=10, day=15, hour=00, minute=30, second=00)],
        )
        end_datetimes = np.atleast_1d(end_datetimes)
        load_precomputed_replicas = passive_replicas_args.get(
            "load_precomputed_replicas", True
        )  # If True, will load precomputed replicas if they exist, otherwise will compute them

        for start_dt, end_dt in zip(start_datetimes, end_datetimes):

            if load_precomputed_replicas:
                record_id = f"passive_{datetime.strftime(start_dt, self.fsm_active.datetime_fmt)}_to_{datetime.strftime(end_dt, self.fsm_active.datetime_fmt)}"
                rep_filepath = os.path.join(
                    self.root_rtf_data_passive, f"sequence_{record_id}_rtf.nc"
                )
                if os.path.exists(rep_filepath):
                    # Check file validity
                    ds_replica = xr.open_dataset(rep_filepath)
                    valid = True
                    if ds_replica.h_index_ref != self.reference_receiver_id:
                        print(
                            f"Replica file for passive segment {start_dt} - {end_dt} found at {rep_filepath} is not valid: reference receiver id mismatch (expected {self.reference_receiver_id}, found {ds_replica.h_index_ref}). This segment will be recomputed."
                        )
                        valid = False
                        #  Avoid permission error when trying to open the file again in the future by closing and deleting the dataset before recomputing
                        ds_replica.close()  # Close the dataset to avoid memory leak
                        del ds_replica  # Delete dataset

                    if valid:
                        print(
                            f"Precomputed replica for passive segment {start_dt} - {end_dt} found at {rep_filepath}. This segment will be loaded instead of being computed."
                        )
                        continue

            self.fsm_passive.process_analysis(
                t_start=start_dt,
                t_end=end_dt,
                set_stft_props=False,
            )

        passive_replicas_info = {
            "start_datetimes": start_datetimes,
            "end_datetimes": end_datetimes,
        }

        return passive_replicas_info

    def load_replicas(
        self, active_replicas_info: dict = {}, passive_replicas_info: dict = {}
    ):
        """
        Method to load replicas
        """
        self.load_active_replicas(active_replicas_info=active_replicas_info)
        self.load_passive_replicas(passive_replicas_info=passive_replicas_info)

    def load_active_replicas(self, active_replicas_info: dict = {}):
        """
        Method to load replicas (active source)
        """
        rep_seq_ids = active_replicas_info.get("replica_sequence_ids", [])

        for i, rep_seq_id in enumerate(rep_seq_ids):
            # Load replica data
            rep_filepath = os.path.join(
                self.root_rtf_data_active, f"sequence_{rep_seq_id}_rtf.nc"
            )
            if os.path.exists(rep_filepath):
                ds_replica = xr.open_dataset(rep_filepath)
                # Extract usefull data to store to the library
                if i == 0:
                    rtf_amp = ds_replica["rtf_amp_hat"].values
                    rtf_phase = ds_replica["rtf_phase_hat"].values
                else:
                    # Concatenate along the pulse axis : (n_rcv, n_freq, n_pulse) -> (n_rcv, n_freq, n_replicas)
                    rtf_amp = np.concatenate(
                        [rtf_amp, ds_replica["rtf_amp_hat"].values], axis=-1
                    )
                    rtf_phase = np.concatenate(
                        [rtf_phase, ds_replica["rtf_phase_hat"].values], axis=-1
                    )

            else:
                print(
                    f"Replica file for sequence {rep_seq_id} not found at {rep_filepath}."
                )
                # Remove this sequence id from the list of sequence ids
                rep_seq_ids = rep_seq_ids[rep_seq_ids != rep_seq_id]

        # Store replicas in active_library
        ds_active_library = xr.Dataset(
            data_vars={
                "rtf_amp": (("h_index", "f_rtf", "replica_id"), rtf_amp),
                "rtf_phase": (("h_index", "f_rtf", "replica_id"), rtf_phase),
            },
            coords={
                "h_index": ds_replica["h_index"].values,
                "f_rtf": ds_replica["f_rtf"].values,
                "replica_id": np.arange(rtf_amp.shape[-1]),
            },
        )

        return ds_active_library

    def load_passive_replicas(self, passive_replicas_info: dict = {}):
        """
        Method to load replicas (passive source)
        """
        start_datetimes = passive_replicas_info.get("start_datetimes", [])
        end_datetimes = passive_replicas_info.get("end_datetimes", [])

        i = 0
        for start_dt, end_dt in zip(start_datetimes, end_datetimes):
            # Load replica data
            record_id = f"passive_{datetime.strftime(start_dt, self.fsm_active.datetime_fmt)}_to_{datetime.strftime(end_dt, self.fsm_active.datetime_fmt)}"
            rep_filepath = os.path.join(
                self.root_rtf_data_passive, f"sequence_{record_id}_rtf.nc"
            )

            if os.path.exists(rep_filepath):
                ds_replica = xr.open_dataset(rep_filepath)

                if i == 0:
                    rtf_amp = ds_replica["rtf_amp_hat"].values
                    rtf_phase = ds_replica["rtf_phase_hat"].values
                else:
                    # Concatenate along the time axis : (n_rcv, n_freq, n_segment_dt) -> (n_rcv, n_freq, n_replicas)
                    rtf_amp = np.concatenate(
                        [rtf_amp, ds_replica["rtf_amp_hat"].values], axis=-1
                    )
                    rtf_phase = np.concatenate(
                        [rtf_phase, ds_replica["rtf_phase_hat"].values], axis=-1
                    )
            else:
                print(
                    f"Replica file for passive segment {start_dt} - {end_dt} not found at {rep_filepath}."
                )

        # Store replicas in passive_library
        ds_passive_library = xr.Dataset(
            data_vars={
                "rtf_amp": (("h_index", "f_rtf", "replica_id"), rtf_amp),
                "rtf_phase": (("h_index", "f_rtf", "replica_id"), rtf_phase),
            },
            coords={
                "h_index": ds_replica["h_index"].values,
                "f_rtf": ds_replica["f_rtf"].values,
                "replica_id": np.arange(rtf_amp.shape[-1]),
            },
        )
        return ds_passive_library


# ======================================================================================================================
# Test
# =====================================================================================================================


def test():
    """
    Test function
    """
    root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data"
    ref_rcv_id = 2
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
    active_replicas_args = {
        "replica_sequence_ids": [147],
        "load_precomputed_replicas": True,
    }
    passive_replicas_args = {
        "start_datetimes": [
            datetime(year=2025, month=10, day=15, hour=00, minute=15, second=00)
        ],
        "end_datetimes": [
            datetime(year=2025, month=10, day=15, hour=00, minute=30, second=00)
        ],
        "load_precomputed_replicas": True,
    }
    rtf_mfp_library.populate(
        active_replicas_args=active_replicas_args,
        passive_replicas_args=passive_replicas_args,
    )


if __name__ == "__main__":
    test()
