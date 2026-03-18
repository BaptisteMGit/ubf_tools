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

from real_data_analysis.fiberscope_groix.src.localisation.rtf.rtf_mfp_utils import (
    filter_ais,
)

# DEFAUTS # TODO move this in a dedicated file ?
ROOT_DATA = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data"
REF_RCV_ID = 1
RTF_ESTIMATOR = "cs-evd"
WAV_DATASET_FILEPATH = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data\sequences\wav_dataset.csv"


# ======================================================================================================================
# Mother class
# ======================================================================================================================


class RTF_MFP:
    """
    Mother class to handle RTF-MFP
    """

    def __init__(
        self,
        root_data: str = ROOT_DATA,
        reference_receiver_id: int = REF_RCV_ID,
        rtf_estimator: str = RTF_ESTIMATOR,
        fsm_active_kwargs: dict = {},
        fsm_passive_kwargs: dict = {},
        mode="overwrite",
        library_id: str = None,
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

        self.mode = mode
        self.library_id = library_id

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
        # Folder to store the library
        self.root_library_data = os.path.join(self.root_data, "library")
        self.root_event_data = os.path.join(self.root_data, "event")

        for path in [
            self.root_rtf_data,
            self.root_rtf_data_active,
            self.root_rtf_data_passive,
            self.root_library_data,
            self.root_event_data,
        ]:
            os.makedirs(path, exist_ok=True)

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
            limit_frequency_band=False,
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

    def compute(self, active_feature_args: dict = {}, passive_feature_args: dict = {}):
        """
        Method to populate the library
        """
        # Derive replicas
        active_feature_info, passive_feature_info = self.derive_feature(
            active_feature_args=active_feature_args,
            passive_feature_args=passive_feature_args,
        )

        # Load derived replicas
        ds_active, ds_passive = self.load_feature(
            active_feature_info=active_feature_info,
            passive_feature_info=passive_feature_info,
        )

        # Concatenate active and passive features
        ds = self.fusion(ds_active=ds_active, ds_passive=ds_passive)

        # Save
        self.save(ds)

        # Write associated metadata file with the same ID as the library to keep track of the library content and properties
        # TODO ?
        # self.save_library_metadata(ds_library=ds_library, active_feature_info=active_feature_info, passive_feature_info=passive_feature_info)

    def fusion(self, ds_active: xr.Dataset, ds_passive: xr.Dataset) -> xr.Dataset:

        # Fusion active and passive replicas
        ds = xr.concat([ds_active, ds_passive], dim="replica_id")
        ds.attrs = {
            "reference_receiver_id": self.reference_receiver_id,
            "rtf_estimator": self.rtf_estimator,
            "replica_type_1": "active",
            "replica_type_2": "passive",
        }

        # Add attributes to each variables
        ds["rtf_amp"].attrs = {
            "description": "Estimated RTF amplitude",
            "units": "",
            "long_name": r"$\lvert \Pi \rvert$",
        }
        ds["rtf_phase"].attrs = {
            "description": "Estimated RTF phase",
            "units": "rad",
            "long_name": r"$\angle \Pi$",
        }

        for coord in ["e", "n", "u"]:
            ds[f"{coord}_replica"].attrs = {
                "description": f"{coord.upper()} coordinate of the replica position (local ENU frame)",
                "units": "m",
                "long_name": coord.upper(),
            }
        ds["h_index"].attrs = {
            "description": "ID of the receiver in the dataset",
            "units": "",
            "long_name": "Receiver ID",
        }
        ds["f_rtf"].attrs = {
            "description": "Frequency of the RTF estimation",
            "units": "Hz",
            "long_name": "Frequency",
        }

        return ds

    def save(self, ds: xr.Dataset, ds_type: str = "library", id: int = None) -> None:
        """
        Method to save dataset
        """

        if ds_type == "library":
            root_data = self.root_library_data
        elif ds_type == "event":
            root_data = self.root_event_data
        else:
            print(
                f"Error: unknown dataset type {ds_type}, expected 'library' or 'event."
            )
            return

        # Check which ID already exists in the dataset folder to avoid overwriting an existing dataset with a new one. If a dataset already exists, a warning is printed and the new library is not saved.
        existing_datasets = [
            f
            for f in os.listdir(root_data)
            if f.endswith(".nc") and f.startswith(ds_type)
        ]
        existing_datasets_ids = [
            int(f.split(".nc")[0].split("_")[1]) for f in existing_datasets
        ]

        if len(existing_datasets_ids) > 0:
            new_id = max(existing_datasets_ids) + 1
        else:
            new_id = 0

        # Define a unique ID for the library
        if self.mode == "overwrite" and self.library_id is not None:
            if self.library_id in existing_libraries_ids:
                print(
                    f"A {ds_type} with ID {self.library_id} already exists in {root_data}. This {ds_type} will be overwritten."
                )
            else:
                print(
                    f"Warning: no existing library with ID {self.library_id} found in {self.root_library_data} using mode = overwrite. The new library will be saved with this ID."
                )

        elif self.mode == "overwrite" and self.library_id is None:
            self.library_id = (
                new_id  # Define a new ID based on the number of existing libraries
            )
            print(
                f"Warning: no library ID provided while using mode = overwrite. The new library will be saved with new ID = {self.library_id}."
            )

        elif self.mode == "new":
            if self.library_id is None:
                self.library_id = new_id
                print(
                    f"Warning: no library ID provided while using mode = new. The new library will be saved with new ID = {self.library_id}."
                )
            elif self.library_id in existing_libraries_ids:
                print(
                    f"Warning: a library with ID {self.library_id} already exists in {self.root_library_data} using mode = new. The new library will not be saved to avoid overwriting the existing one."
                )
                return

        library_filepath = os.path.join(
            self.root_library_data, f"library_{self.library_id}.nc"
        )
        ds_library.to_netcdf(library_filepath)

        if self.verbose:
            print(f"Library saved at {library_filepath}.")

    def derive_feature(
        self, active_feature_args: dict = {}, passive_feature_args: dict = {}
    ):
        """
        Method to derive feature
        """
        active_feature_info = self.derive_feature_active(
            active_feature_args=active_feature_args
        )
        passive_feature_info = self.derive_feature_passive(
            passive_feature_args=passive_feature_args
        )

        return active_feature_info, passive_feature_info

    def derive_feature_active(self, active_feature_args: dict = {}) -> dict:
        """
        Method to derive replicas (active source)
        """
        if self.verbose:
            print("\nDeriving replicas for active source...")

        # Unpack arguments
        replica_sequence_ids = active_feature_args.get(
            "replica_sequence_ids", [144]
        )  # Default -> use trailed sequence 144
        replica_sequence_ids = np.atleast_1d(
            replica_sequence_ids
        )  # Ensure it's an array
        load_precomputed_feature = active_feature_args.get(
            "load_precomputed_feature", True
        )  # If True, will load precomputed replicas if they exist, otherwise will compute them

        replica_sequence_ids_to_compute = (
            replica_sequence_ids.copy()
        )  # Initialize list of sequence ids to compute
        if load_precomputed_feature:
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

        active_feature_info = {
            "replica_sequence_ids": replica_sequence_ids,
            # "df_arrivals_processed": df_arrivals_processed,
        }
        return active_feature_info

    def derive_feature_passive(self, passive_feature_args: dict = {}) -> dict:
        """
        Method to derive replicas (passive source)
        """
        if self.verbose:
            print("\nDeriving replicas for passive source...")

        # Unpack arguments
        start_datetimes = passive_feature_args.get(
            "start_datetimes",
            [datetime(year=2025, month=10, day=15, hour=00, minute=15, second=00)],
        )
        start_datetimes = np.atleast_1d(start_datetimes)
        end_datetimes = passive_feature_args.get(
            "end_datetimes",
            [datetime(year=2025, month=10, day=15, hour=00, minute=30, second=00)],
        )
        end_datetimes = np.atleast_1d(end_datetimes)
        load_precomputed_feature = passive_feature_args.get(
            "load_precomputed_feature", True
        )  # If True, will load precomputed replicas if they exist, otherwise will compute them

        for start_dt, end_dt in zip(start_datetimes, end_datetimes):

            if load_precomputed_feature:
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

        passive_feature_info = {
            "start_datetimes": start_datetimes,
            "end_datetimes": end_datetimes,
        }

        return passive_feature_info

    def load_feature(
        self, active_feature_info: dict = {}, passive_feature_info: dict = {}
    ):
        """
        Method to load features
        """
        ds_active_library = self.load_active_feature(
            active_feature_info=active_feature_info
        )
        ds_passive_library = self.load_passive_feature(
            passive_feature_info=passive_feature_info
        )

        return ds_active_library, ds_passive_library

    def load_active_feature(self, active_feature_info: dict = {}):
        """
        Method to load features (active source)
        """
        rep_seq_ids = active_feature_info.get("replica_sequence_ids", [])
        # df_arrivals_processed = active_feature_info.get("df_arrivals_processed", None)

        df_arrivals_processed = (
            self.df_arrivals.copy()
        )  # Start from the original arrivals dataframe

        for i, rep_seq_id in enumerate(rep_seq_ids):
            # Load replica data
            rep_filepath = os.path.join(
                self.root_rtf_data_active, f"sequence_{rep_seq_id}_rtf.nc"
            )
            if os.path.exists(rep_filepath):
                ds_replica = xr.open_dataset(rep_filepath)

                # Get replica position
                rep_pulse_ids = ds_replica["pulse_id"].values
                # Slice current sequence
                df_arrivals_processed_seq = df_arrivals_processed.loc[
                    df_arrivals_processed["sequence_id"] == rep_seq_id
                ]
                df_arrivals_processed_seq = df_arrivals_processed_seq.loc[
                    df_arrivals_processed_seq["pulse_id"].isin(rep_pulse_ids)
                ]  # keep only detected pulse

                # Get replica position from arrivals
                e_replica = df_arrivals_processed_seq["emission_interp_e_gps"]
                n_replica = df_arrivals_processed_seq["emission_interp_n_gps"]
                u_replica = df_arrivals_processed_seq["emission_interp_u_gps"]

                # Extract usefull data to store to the library
                if i == 0:
                    rtf_amp = ds_replica["rtf_amp_hat"].values
                    rtf_phase = ds_replica["rtf_phase_hat"].values
                    e_replica_all = e_replica.values
                    n_replica_all = n_replica.values
                    u_replica_all = u_replica.values
                else:
                    # Concatenate along the pulse axis : (n_rcv, n_freq, n_pulse) -> (n_rcv, n_freq, n_feature)
                    rtf_amp = np.concatenate(
                        [rtf_amp, ds_replica["rtf_amp_hat"].values], axis=-1
                    )
                    rtf_phase = np.concatenate(
                        [rtf_phase, ds_replica["rtf_phase_hat"].values], axis=-1
                    )
                    e_replica_all = np.concatenate([e_replica_all, e_replica.values])
                    n_replica_all = np.concatenate([n_replica_all, n_replica.values])
                    u_replica_all = np.concatenate([u_replica_all, u_replica.values])

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
                "e_replica": (("replica_id"), e_replica_all.astype(np.float32)),
                "n_replica": (("replica_id"), n_replica_all.astype(np.float32)),
                "u_replica": (("replica_id"), u_replica_all.astype(np.float32)),
                "replica_type": (("replica_id"), [1] * rtf_amp.shape[-1]),
            },
            coords={
                "h_index": ds_replica["h_index"].values,
                "f_rtf": ds_replica["f_rtf"].values,
                "replica_id": np.arange(rtf_amp.shape[-1]),
            },
        )

        return ds_active_library

    def load_passive_feature(self, passive_feature_info: dict = {}):
        """
        Method to load replicas (passive source)
        """
        start_datetimes = passive_feature_info.get("start_datetimes", [])
        end_datetimes = passive_feature_info.get("end_datetimes", [])

        i = 0
        for start_dt, end_dt in zip(start_datetimes, end_datetimes):
            # Load replica data
            record_id = f"passive_{datetime.strftime(start_dt, self.fsm_active.datetime_fmt)}_to_{datetime.strftime(end_dt, self.fsm_active.datetime_fmt)}"
            rep_filepath = os.path.join(
                self.root_rtf_data_passive, f"sequence_{record_id}_rtf.nc"
            )

            if os.path.exists(rep_filepath):
                ds_replica = xr.open_dataset(rep_filepath)

                # Extract source position
                ais_replica = self.ds_ais.sel(time=slice(start_dt, end_dt))
                ais_replica = filter_ais(ais_event=ais_replica)

                # Ensure only one vessel in the segment
                if len(ais_replica["mmsi"].values) > 1:
                    print(
                        f"Warning: more than one vessel detected in the AIS data for passive segment {start_dt} - {end_dt}. This segment will be skipped."
                    )
                    continue
                else:
                    ais_replica = ais_replica.isel(mmsi=0)  # Keep only the first vessel

                e_replica = ais_replica.e.interp(
                    time=ds_replica.segment_dt.values
                ).values
                n_replica = ais_replica.n.interp(
                    time=ds_replica.segment_dt.values
                ).values
                u_replica = ais_replica.u.interp(
                    time=ds_replica.segment_dt.values
                ).values

                if i == 0:
                    rtf_amp = ds_replica["rtf_amp_hat"].values
                    rtf_phase = ds_replica["rtf_phase_hat"].values
                    e_replica_all = e_replica
                    n_replica_all = n_replica
                    u_replica_all = u_replica
                else:
                    # Concatenate along the time axis : (n_rcv, n_freq, n_segment_dt) -> (n_rcv, n_freq, n_feature)
                    rtf_amp = np.concatenate(
                        [rtf_amp, ds_replica["rtf_amp_hat"].values], axis=-1
                    )
                    rtf_phase = np.concatenate(
                        [rtf_phase, ds_replica["rtf_phase_hat"].values], axis=-1
                    )
                    e_replica_all = np.concatenate([e_replica_all, e_replica])
                    n_replica_all = np.concatenate([n_replica_all, n_replica])
                    u_replica_all = np.concatenate([u_replica_all, u_replica])
            else:
                print(
                    f"Replica file for passive segment {start_dt} - {end_dt} not found at {rep_filepath}."
                )

        # Store replicas in passive_library
        ds_passive_library = xr.Dataset(
            data_vars={
                "rtf_amp": (("h_index", "f_rtf", "replica_id"), rtf_amp),
                "rtf_phase": (("h_index", "f_rtf", "replica_id"), rtf_phase),
                "e_replica": (("replica_id"), e_replica_all.astype(np.float32)),
                "n_replica": (("replica_id"), n_replica_all.astype(np.float32)),
                "u_replica": (("replica_id"), u_replica_all.astype(np.float32)),
                "replica_type": (("replica_id"), [2] * rtf_amp.shape[-1]),
            },
            coords={
                "h_index": ds_replica["h_index"].values,
                "f_rtf": ds_replica["f_rtf"].values,
                "replica_id": np.arange(rtf_amp.shape[-1]),
            },
        )
        return ds_passive_library


# ======================================================================================================================
# Library
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
        mode="overwrite",
        library_id: str = None,
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

        self.mode = mode
        self.library_id = library_id

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
        # Folder to store the library
        self.root_library_data = os.path.join(self.root_data, "library")

        for path in [
            self.root_rtf_data,
            self.root_rtf_data_active,
            self.root_rtf_data_passive,
            self.root_library_data,
        ]:
            os.makedirs(path, exist_ok=True)

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
            limit_frequency_band=False,
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
        ds_active_library, ds_passive_library = self.load_replicas(
            active_replicas_info=active_replicas_info,
            passive_replicas_info=passive_replicas_info,
        )

        # Concatenate active and passive replicas to create the library
        ds_library = self.fusion_library(
            ds_active_library=ds_active_library, ds_passive_library=ds_passive_library
        )

        # Save
        self.save_library(ds_library=ds_library)

        # Write associated metadata file with the same ID as the library to keep track of the library content and properties
        # TODO ?
        # self.save_library_metadata(ds_library=ds_library, active_replicas_info=active_replicas_info, passive_replicas_info=passive_replicas_info)

    def fusion_library(
        self, ds_active_library: xr.Dataset, ds_passive_library: xr.Dataset
    ) -> xr.Dataset:

        # Fusion active and passive replicas
        ds_library = xr.concat(
            [ds_active_library, ds_passive_library], dim="replica_id"
        )
        ds_library.attrs = {
            "reference_receiver_id": self.reference_receiver_id,
            "rtf_estimator": self.rtf_estimator,
            "replica_type_1": "active",
            "replica_type_2": "passive",
        }

        # Add attributes to each variables
        ds_library["rtf_amp"].attrs = {
            "description": "Estimated RTF amplitude",
            "units": "",
            "long_name": r"$\lvert \Pi \rvert$",
        }
        ds_library["rtf_phase"].attrs = {
            "description": "Estimated RTF phase",
            "units": "rad",
            "long_name": r"$\angle \Pi$",
        }

        for coord in ["e", "n", "u"]:
            ds_library[f"{coord}_replica"].attrs = {
                "description": f"{coord.upper()} coordinate of the replica position (local ENU frame)",
                "units": "m",
                "long_name": coord.upper(),
            }
        ds_library["h_index"].attrs = {
            "description": "ID of the receiver in the dataset",
            "units": "",
            "long_name": "Receiver ID",
        }
        ds_library["f_rtf"].attrs = {
            "description": "Frequency of the RTF estimation",
            "units": "Hz",
            "long_name": "Frequency",
        }

        return ds_library

    def save_library(self, ds_library: xr.Dataset) -> None:
        """
        Method to save the library
        """

        # Check which ID already exists in the library folder to avoid overwriting an existing library with a new one. If a library already exists, a warning is printed and the new library is not saved.
        existing_libraries = [
            f
            for f in os.listdir(self.root_library_data)
            if f.endswith(".nc") and f.startswith("library")
        ]
        existing_libraries_ids = [
            int(f.split(".nc")[0].split("_")[1]) for f in existing_libraries
        ]

        if len(existing_libraries_ids) > 0:
            new_id = max(existing_libraries_ids) + 1
        else:
            new_id = 0

        # Define a unique ID for the library
        if self.mode == "overwrite" and self.library_id is not None:
            if self.library_id in existing_libraries_ids:
                print(
                    f"A library with ID {self.library_id} already exists in {self.root_library_data}. This library will be overwritten."
                )
            else:
                print(
                    f"Warning: no existing library with ID {self.library_id} found in {self.root_library_data} using mode = overwrite. The new library will be saved with this ID."
                )

        elif self.mode == "overwrite" and self.library_id is None:
            self.library_id = (
                new_id  # Define a new ID based on the number of existing libraries
            )
            print(
                f"Warning: no library ID provided while using mode = overwrite. The new library will be saved with new ID = {self.library_id}."
            )

        elif self.mode == "new":
            if self.library_id is None:
                self.library_id = new_id
                print(
                    f"Warning: no library ID provided while using mode = new. The new library will be saved with new ID = {self.library_id}."
                )
            elif self.library_id in existing_libraries_ids:
                print(
                    f"Warning: a library with ID {self.library_id} already exists in {self.root_library_data} using mode = new. The new library will not be saved to avoid overwriting the existing one."
                )
                return

        library_filepath = os.path.join(
            self.root_library_data, f"library_{self.library_id}.nc"
        )
        ds_library.to_netcdf(library_filepath)

        if self.verbose:
            print(f"Library saved at {library_filepath}.")

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
            # "df_arrivals_processed": df_arrivals_processed,
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
        ds_active_library = self.load_active_replicas(
            active_replicas_info=active_replicas_info
        )
        ds_passive_library = self.load_passive_replicas(
            passive_replicas_info=passive_replicas_info
        )

        return ds_active_library, ds_passive_library

    def load_active_replicas(self, active_replicas_info: dict = {}):
        """
        Method to load replicas (active source)
        """
        rep_seq_ids = active_replicas_info.get("replica_sequence_ids", [])
        # df_arrivals_processed = active_replicas_info.get("df_arrivals_processed", None)

        df_arrivals_processed = (
            self.df_arrivals.copy()
        )  # Start from the original arrivals dataframe

        for i, rep_seq_id in enumerate(rep_seq_ids):
            # Load replica data
            rep_filepath = os.path.join(
                self.root_rtf_data_active, f"sequence_{rep_seq_id}_rtf.nc"
            )
            if os.path.exists(rep_filepath):
                ds_replica = xr.open_dataset(rep_filepath)

                # Get replica position
                rep_pulse_ids = ds_replica["pulse_id"].values
                # Slice current sequence
                df_arrivals_processed_seq = df_arrivals_processed.loc[
                    df_arrivals_processed["sequence_id"] == rep_seq_id
                ]
                df_arrivals_processed_seq = df_arrivals_processed_seq.loc[
                    df_arrivals_processed_seq["pulse_id"].isin(rep_pulse_ids)
                ]  # keep only detected pulse

                # Get replica position from arrivals
                e_replica = df_arrivals_processed_seq["emission_interp_e_gps"]
                n_replica = df_arrivals_processed_seq["emission_interp_n_gps"]
                u_replica = df_arrivals_processed_seq["emission_interp_u_gps"]

                # Extract usefull data to store to the library
                if i == 0:
                    rtf_amp = ds_replica["rtf_amp_hat"].values
                    rtf_phase = ds_replica["rtf_phase_hat"].values
                    e_replica_all = e_replica.values
                    n_replica_all = n_replica.values
                    u_replica_all = u_replica.values
                else:
                    # Concatenate along the pulse axis : (n_rcv, n_freq, n_pulse) -> (n_rcv, n_freq, n_replicas)
                    rtf_amp = np.concatenate(
                        [rtf_amp, ds_replica["rtf_amp_hat"].values], axis=-1
                    )
                    rtf_phase = np.concatenate(
                        [rtf_phase, ds_replica["rtf_phase_hat"].values], axis=-1
                    )
                    e_replica_all = np.concatenate([e_replica_all, e_replica.values])
                    n_replica_all = np.concatenate([n_replica_all, n_replica.values])
                    u_replica_all = np.concatenate([u_replica_all, u_replica.values])

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
                "e_replica": (("replica_id"), e_replica_all.astype(np.float32)),
                "n_replica": (("replica_id"), n_replica_all.astype(np.float32)),
                "u_replica": (("replica_id"), u_replica_all.astype(np.float32)),
                "replica_type": (("replica_id"), [1] * rtf_amp.shape[-1]),
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

                # Extract source position
                ais_replica = self.ds_ais.sel(time=slice(start_dt, end_dt))
                ais_replica = filter_ais(ais_event=ais_replica)

                # Ensure only one vessel in the segment
                if len(ais_replica["mmsi"].values) > 1:
                    print(
                        f"Warning: more than one vessel detected in the AIS data for passive segment {start_dt} - {end_dt}. This segment will be skipped."
                    )
                    continue
                else:
                    ais_replica = ais_replica.isel(mmsi=0)  # Keep only the first vessel

                e_replica = ais_replica.e.interp(
                    time=ds_replica.segment_dt.values
                ).values
                n_replica = ais_replica.n.interp(
                    time=ds_replica.segment_dt.values
                ).values
                u_replica = ais_replica.u.interp(
                    time=ds_replica.segment_dt.values
                ).values

                if i == 0:
                    rtf_amp = ds_replica["rtf_amp_hat"].values
                    rtf_phase = ds_replica["rtf_phase_hat"].values
                    e_replica_all = e_replica
                    n_replica_all = n_replica
                    u_replica_all = u_replica
                else:
                    # Concatenate along the time axis : (n_rcv, n_freq, n_segment_dt) -> (n_rcv, n_freq, n_replicas)
                    rtf_amp = np.concatenate(
                        [rtf_amp, ds_replica["rtf_amp_hat"].values], axis=-1
                    )
                    rtf_phase = np.concatenate(
                        [rtf_phase, ds_replica["rtf_phase_hat"].values], axis=-1
                    )
                    e_replica_all = np.concatenate([e_replica_all, e_replica])
                    n_replica_all = np.concatenate([n_replica_all, n_replica])
                    u_replica_all = np.concatenate([u_replica_all, u_replica])
            else:
                print(
                    f"Replica file for passive segment {start_dt} - {end_dt} not found at {rep_filepath}."
                )

        # Store replicas in passive_library
        ds_passive_library = xr.Dataset(
            data_vars={
                "rtf_amp": (("h_index", "f_rtf", "replica_id"), rtf_amp),
                "rtf_phase": (("h_index", "f_rtf", "replica_id"), rtf_phase),
                "e_replica": (("replica_id"), e_replica_all.astype(np.float32)),
                "n_replica": (("replica_id"), n_replica_all.astype(np.float32)),
                "u_replica": (("replica_id"), u_replica_all.astype(np.float32)),
                "replica_type": (("replica_id"), [2] * rtf_amp.shape[-1]),
            },
            coords={
                "h_index": ds_replica["h_index"].values,
                "f_rtf": ds_replica["f_rtf"].values,
                "replica_id": np.arange(rtf_amp.shape[-1]),
            },
        )
        return ds_passive_library


# =====================================================================================================================
# Event
# =====================================================================================================================


class RTF_MFP_Event:
    """
    Class to handle RTF-MFP events
    """

    def __init__(
        self,
        root_data: str = ROOT_DATA,
        reference_receiver_id: int = REF_RCV_ID,
        rtf_estimator: str = RTF_ESTIMATOR,
        fsm_active_kwargs: dict = {},
        fsm_passive_kwargs: dict = {},
        mode="overwrite",
        library_id: str = None,
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

        self.mode = mode
        self.library_id = library_id

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
        # Folder to store the event
        self.root_event_data = os.path.join(self.root_data, "event")

        for path in [
            self.root_rtf_data,
            self.root_rtf_data_active,
            self.root_rtf_data_passive,
            self.root_event_data,
        ]:
            os.makedirs(path, exist_ok=True)

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
            limit_frequency_band=False,
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

    def get_event(
        self, active_replicas_args: dict = {}, passive_replicas_args: dict = {}
    ):
        """
        Method to derive event feature
        """
        # Derive replicas
        active_replicas_info, passive_replicas_info = self.derive_replicas(
            active_replicas_args=active_replicas_args,
            passive_replicas_args=passive_replicas_args,
        )

        # Load derived replicas
        ds_active_library, ds_passive_library = self.load_replicas(
            active_replicas_info=active_replicas_info,
            passive_replicas_info=passive_replicas_info,
        )

        # Concatenate active and passive replicas to create the library
        ds_library = self.fusion_library(
            ds_active_library=ds_active_library, ds_passive_library=ds_passive_library
        )

        # Save
        self.save(ds_library=ds_library)

        # Write associated metadata file with the same ID as the library to keep track of the library content and properties
        # TODO ?
        # self.save_library_metadata(ds_library=ds_library, active_replicas_info=active_replicas_info, passive_replicas_info=passive_replicas_info)

    def fusion_library(
        self, ds_active_library: xr.Dataset, ds_passive_library: xr.Dataset
    ) -> xr.Dataset:

        # Fusion active and passive replicas
        ds_library = xr.concat(
            [ds_active_library, ds_passive_library], dim="replica_id"
        )
        ds_library.attrs = {
            "reference_receiver_id": self.reference_receiver_id,
            "rtf_estimator": self.rtf_estimator,
            "replica_type_1": "active",
            "replica_type_2": "passive",
        }

        # Add attributes to each variables
        ds_library["rtf_amp"].attrs = {
            "description": "Estimated RTF amplitude",
            "units": "",
            "long_name": r"$\lvert \Pi \rvert$",
        }
        ds_library["rtf_phase"].attrs = {
            "description": "Estimated RTF phase",
            "units": "rad",
            "long_name": r"$\angle \Pi$",
        }

        for coord in ["e", "n", "u"]:
            ds_library[f"{coord}_replica"].attrs = {
                "description": f"{coord.upper()} coordinate of the replica position (local ENU frame)",
                "units": "m",
                "long_name": coord.upper(),
            }
        ds_library["h_index"].attrs = {
            "description": "ID of the receiver in the dataset",
            "units": "",
            "long_name": "Receiver ID",
        }
        ds_library["f_rtf"].attrs = {
            "description": "Frequency of the RTF estimation",
            "units": "Hz",
            "long_name": "Frequency",
        }

        return ds_library

    def save_library(self, ds_library: xr.Dataset) -> None:
        """
        Method to save the library
        """

        # Check which ID already exists in the library folder to avoid overwriting an existing library with a new one. If a library already exists, a warning is printed and the new library is not saved.
        existing_libraries = [
            f
            for f in os.listdir(self.root_library_data)
            if f.endswith(".nc") and f.startswith("library")
        ]
        existing_libraries_ids = [
            int(f.split(".nc")[0].split("_")[1]) for f in existing_libraries
        ]

        if len(existing_libraries_ids) > 0:
            new_id = max(existing_libraries_ids) + 1
        else:
            new_id = 0

        # Define a unique ID for the library
        if self.mode == "overwrite" and self.library_id is not None:
            if self.library_id in existing_libraries_ids:
                print(
                    f"A library with ID {self.library_id} already exists in {self.root_library_data}. This library will be overwritten."
                )
            else:
                print(
                    f"Warning: no existing library with ID {self.library_id} found in {self.root_library_data} using mode = overwrite. The new library will be saved with this ID."
                )

        elif self.mode == "overwrite" and self.library_id is None:
            self.library_id = (
                new_id  # Define a new ID based on the number of existing libraries
            )
            print(
                f"Warning: no library ID provided while using mode = overwrite. The new library will be saved with new ID = {self.library_id}."
            )

        elif self.mode == "new":
            if self.library_id is None:
                self.library_id = new_id
                print(
                    f"Warning: no library ID provided while using mode = new. The new library will be saved with new ID = {self.library_id}."
                )
            elif self.library_id in existing_libraries_ids:
                print(
                    f"Warning: a library with ID {self.library_id} already exists in {self.root_library_data} using mode = new. The new library will not be saved to avoid overwriting the existing one."
                )
                return

        library_filepath = os.path.join(
            self.root_library_data, f"library_{self.library_id}.nc"
        )
        ds_library.to_netcdf(library_filepath)

        if self.verbose:
            print(f"Library saved at {library_filepath}.")

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
            # "df_arrivals_processed": df_arrivals_processed,
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
        ds_active_library = self.load_active_replicas(
            active_replicas_info=active_replicas_info
        )
        ds_passive_library = self.load_passive_replicas(
            passive_replicas_info=passive_replicas_info
        )

        return ds_active_library, ds_passive_library

    def load_active_replicas(self, active_replicas_info: dict = {}):
        """
        Method to load replicas (active source)
        """
        rep_seq_ids = active_replicas_info.get("replica_sequence_ids", [])
        # df_arrivals_processed = active_replicas_info.get("df_arrivals_processed", None)

        df_arrivals_processed = (
            self.df_arrivals.copy()
        )  # Start from the original arrivals dataframe

        for i, rep_seq_id in enumerate(rep_seq_ids):
            # Load replica data
            rep_filepath = os.path.join(
                self.root_rtf_data_active, f"sequence_{rep_seq_id}_rtf.nc"
            )
            if os.path.exists(rep_filepath):
                ds_replica = xr.open_dataset(rep_filepath)

                # Get replica position
                rep_pulse_ids = ds_replica["pulse_id"].values
                # Slice current sequence
                df_arrivals_processed_seq = df_arrivals_processed.loc[
                    df_arrivals_processed["sequence_id"] == rep_seq_id
                ]
                df_arrivals_processed_seq = df_arrivals_processed_seq.loc[
                    df_arrivals_processed_seq["pulse_id"].isin(rep_pulse_ids)
                ]  # keep only detected pulse

                # Get replica position from arrivals
                e_replica = df_arrivals_processed_seq["emission_interp_e_gps"]
                n_replica = df_arrivals_processed_seq["emission_interp_n_gps"]
                u_replica = df_arrivals_processed_seq["emission_interp_u_gps"]

                # Extract usefull data to store to the library
                if i == 0:
                    rtf_amp = ds_replica["rtf_amp_hat"].values
                    rtf_phase = ds_replica["rtf_phase_hat"].values
                    e_replica_all = e_replica.values
                    n_replica_all = n_replica.values
                    u_replica_all = u_replica.values
                else:
                    # Concatenate along the pulse axis : (n_rcv, n_freq, n_pulse) -> (n_rcv, n_freq, n_replicas)
                    rtf_amp = np.concatenate(
                        [rtf_amp, ds_replica["rtf_amp_hat"].values], axis=-1
                    )
                    rtf_phase = np.concatenate(
                        [rtf_phase, ds_replica["rtf_phase_hat"].values], axis=-1
                    )
                    e_replica_all = np.concatenate([e_replica_all, e_replica.values])
                    n_replica_all = np.concatenate([n_replica_all, n_replica.values])
                    u_replica_all = np.concatenate([u_replica_all, u_replica.values])

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
                "e_replica": (("replica_id"), e_replica_all.astype(np.float32)),
                "n_replica": (("replica_id"), n_replica_all.astype(np.float32)),
                "u_replica": (("replica_id"), u_replica_all.astype(np.float32)),
                "replica_type": (("replica_id"), [1] * rtf_amp.shape[-1]),
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

                # Extract source position
                ais_replica = self.ds_ais.sel(time=slice(start_dt, end_dt))
                ais_replica = filter_ais(ais_event=ais_replica)

                # Ensure only one vessel in the segment
                if len(ais_replica["mmsi"].values) > 1:
                    print(
                        f"Warning: more than one vessel detected in the AIS data for passive segment {start_dt} - {end_dt}. This segment will be skipped."
                    )
                    continue
                else:
                    ais_replica = ais_replica.isel(mmsi=0)  # Keep only the first vessel

                e_replica = ais_replica.e.interp(
                    time=ds_replica.segment_dt.values
                ).values
                n_replica = ais_replica.n.interp(
                    time=ds_replica.segment_dt.values
                ).values
                u_replica = ais_replica.u.interp(
                    time=ds_replica.segment_dt.values
                ).values

                if i == 0:
                    rtf_amp = ds_replica["rtf_amp_hat"].values
                    rtf_phase = ds_replica["rtf_phase_hat"].values
                    e_replica_all = e_replica
                    n_replica_all = n_replica
                    u_replica_all = u_replica
                else:
                    # Concatenate along the time axis : (n_rcv, n_freq, n_segment_dt) -> (n_rcv, n_freq, n_replicas)
                    rtf_amp = np.concatenate(
                        [rtf_amp, ds_replica["rtf_amp_hat"].values], axis=-1
                    )
                    rtf_phase = np.concatenate(
                        [rtf_phase, ds_replica["rtf_phase_hat"].values], axis=-1
                    )
                    e_replica_all = np.concatenate([e_replica_all, e_replica])
                    n_replica_all = np.concatenate([n_replica_all, n_replica])
                    u_replica_all = np.concatenate([u_replica_all, u_replica])
            else:
                print(
                    f"Replica file for passive segment {start_dt} - {end_dt} not found at {rep_filepath}."
                )

        # Store replicas in passive_library
        ds_passive_library = xr.Dataset(
            data_vars={
                "rtf_amp": (("h_index", "f_rtf", "replica_id"), rtf_amp),
                "rtf_phase": (("h_index", "f_rtf", "replica_id"), rtf_phase),
                "e_replica": (("replica_id"), e_replica_all.astype(np.float32)),
                "n_replica": (("replica_id"), n_replica_all.astype(np.float32)),
                "u_replica": (("replica_id"), u_replica_all.astype(np.float32)),
                "replica_type": (("replica_id"), [2] * rtf_amp.shape[-1]),
            },
            coords={
                "h_index": ds_replica["h_index"].values,
                "f_rtf": ds_replica["f_rtf"].values,
                "replica_id": np.arange(rtf_amp.shape[-1]),
            },
        )
        return ds_passive_library


# =====================================================================================================================
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
        mode="overwrite",
        library_id=0,
        plot_library_replicas_features=False,
        verbose=True,
    )

    # Populate library
    active_replicas_args = {
        "replica_sequence_ids": [144, 146],
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
