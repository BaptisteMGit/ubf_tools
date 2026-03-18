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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime
from scipy.spatial.distance import cdist
from propa.rtf.rtf_utils import D_hermitian_angle_fast
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
ROOT_IMG = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\img\rtf_mfp"
REF_RCV_ID = 1
RTF_ESTIMATOR = "cs-evd"
WAV_DATASET_FILEPATH = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data\sequences\wav_dataset.csv"


# ======================================================================================================================
# Mother class
# ======================================================================================================================


class RTF_MFP_Processor:
    """
    Mother class to handle RTF-MFP
    """

    def __init__(
        self,
        root_data: str = ROOT_DATA,
        root_img: str = ROOT_IMG,
        reference_receiver_id: int = REF_RCV_ID,
        rtf_estimator: str = RTF_ESTIMATOR,
        fsm_active_kwargs: dict = {},
        fsm_passive_kwargs: dict = {},
        mode="overwrite",
        plot_replicas_features: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Constructor of the class
        """

        # Path to data folder
        self.root_data = root_data
        # Path to img folder
        self.root_img = root_img
        # Paths
        self.set_paths()
        # Load datasets
        self.load_datasets()

        # Index of the receiver to use as reference
        self.reference_receiver_id = reference_receiver_id

        # RTF estimator
        self.rtf_estimator = rtf_estimator
        self.mode = mode

        # Flags
        self.plot_replicas_features = plot_replicas_features
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

        # Folder to save results figures
        self.root_results_fig = os.path.join(self.root_img, "results")

        for path in [
            self.root_rtf_data,
            self.root_rtf_data_active,
            self.root_rtf_data_passive,
            self.root_library_data,
            self.root_event_data,
            self.root_results_fig,
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
            plot_feature=self.plot_replicas_features,
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
            plot_feature=self.plot_replicas_features,
            analysis_segment_duration=self.fsm_passive_analysis_segment_duration,
            analysis_segment_alpha_overlap=self.fsm_passive_analysis_segment_alpha_overlap,
            rtf_estimator=self.rtf_estimator,
            verbose=self.verbose,
        )

    ###########################
    # Compute
    ###########################
    def compute(
        self,
        active_feature_args: dict = {},
        passive_feature_args: dict = {},
        ds_type: str = "library",
        id: int = None,
        single_vessel_per_segment: bool = True,
    ):
        """
        Method to compute dataset
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
            single_vessel_per_segment=single_vessel_per_segment,
        )

        # Concatenate active and passive features
        ds = self.fusion(ds_active=ds_active, ds_passive=ds_passive, ds_type=ds_type)

        # Save
        self.save(ds=ds, ds_type=ds_type, id=id)

        # Write associated metadata file with the same ID as the library to keep track of the library content and properties
        # TODO ?
        # self.save_library_metadata(ds_library=ds_library, active_feature_info=active_feature_info, passive_feature_info=passive_feature_info)

    def compute_library(
        self,
        active_feature_args: dict = {},
        passive_feature_args: dict = {},
        id: int = None,
    ):
        """Method to compute library dataset"""

        self.compute(
            active_feature_args=active_feature_args,
            passive_feature_args=passive_feature_args,
            ds_type="library",
            id=id,
            single_vessel_per_segment=True,  # For the library we wish to use segment containing only one vessel to ensure the quality of the passive replicas
        )

    def compute_event(
        self,
        active_feature_args: dict = {},
        passive_feature_args: dict = {},
        id: int = None,
    ):
        """Method to compute event dataset"""

        self.compute(
            active_feature_args=active_feature_args,
            passive_feature_args=passive_feature_args,
            ds_type="event",
            id=id,
            single_vessel_per_segment=False,  # For the event we want to be able to use segments containing multiple vessels to be able to test the method in more complex scenarios
        )

    def fusion(
        self, ds_active: xr.Dataset, ds_passive: xr.Dataset, ds_type: str = "library"
    ) -> xr.Dataset:

        # Fusion active and passive replicas
        if ds_active is not None and ds_passive is not None:
            ds = xr.concat([ds_active, ds_passive], dim="replica_id")
        elif ds_active is not None and ds_passive is None:
            ds = ds_active
        elif ds_active is None and ds_passive is not None:
            ds = ds_passive
        else:
            print("Error: no active or passive replica to fuse.")
            return None

        ds.attrs = {
            "reference_receiver_id": self.reference_receiver_id,
            "rtf_estimator": self.rtf_estimator,
            "replica_type_1": "active",
            "replica_type_2": "passive",
            "description": f"{ds_type.capitalize()} dataset containing RTF-MFP features for active and passive sources",
            "type": ds_type,
        }

        # Add receiver positions as attributes to the dataset
        for k in ["obs1", "obs2", "obs3"]:
            ds.attrs[f"{k}_e_apriori"] = self.ds_gps.attrs[f"{k}_e_apriori"]
            ds.attrs[f"{k}_n_apriori"] = self.ds_gps.attrs[f"{k}_n_apriori"]
            ds.attrs[f"{k}_u_apriori"] = self.ds_gps.attrs[f"{k}_u_apriori"]

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
        if ds is None:
            print("Error: no dataset to save.")
            return

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
        if self.mode == "overwrite" and id is not None:
            if id in existing_datasets_ids:
                print(
                    f"A {ds_type} dataset with ID {id} already exists in {root_data}. This {ds_type} dataset will be overwritten."
                )
            else:
                print(
                    f"Warning: no existing {ds_type} dataset with ID {id} found in {root_data} using mode = overwrite. The new {ds_type} dataset will be saved with this ID."
                )

        elif self.mode == "overwrite" and id is None:
            id = new_id  # Define a new ID based on the number of existing libraries
            print(
                f"Warning: no {ds_type} ID provided while using mode = overwrite. The new {ds_type} dataset will be saved with new ID = {id}."
            )

        elif self.mode == "new":
            if id is None:
                id = new_id
                print(
                    f"Warning: no {ds_type} ID provided while using mode = new. The new {ds_type} dataset will be saved with new ID = {id}."
                )
            elif id in existing_datasets_ids:
                print(
                    f"Warning: a {ds_type} dataset with ID {id} already exists in {root_data} using mode = new. The new {ds_type} dataset will not be saved to avoid overwriting the existing one."
                )
                return

        # Add ID as attribute to the dataset
        ds.attrs["id"] = id

        ds_filepath = os.path.join(root_data, f"{ds_type}_{id}.nc")
        ds.to_netcdf(ds_filepath)

        if self.verbose:
            print(f"Dataset ({ds_type}) saved at {ds_filepath}.")

    ###########################
    # Derive features
    ###########################
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
            print("\nDeriving features for active source...")

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
        Method to derive features (passive source)
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

    ###########################
    # Load features
    ###########################
    def load_feature(
        self,
        active_feature_info: dict = {},
        passive_feature_info: dict = {},
        single_vessel_per_segment: bool = True,
    ):
        """
        Method to load features
        """
        ds_active = self.load_active_feature(active_feature_info=active_feature_info)
        ds_passive = self.load_passive_feature(
            passive_feature_info=passive_feature_info,
            single_vessel_per_segment=single_vessel_per_segment,
        )

        return ds_active, ds_passive

    def load_active_feature(self, active_feature_info: dict = {}):
        """
        Method to load features (active source)
        """
        rep_seq_ids = active_feature_info.get("replica_sequence_ids", [])
        # df_arrivals_processed = active_feature_info.get("df_arrivals_processed", None)

        if len(rep_seq_ids) == 0:
            return None  # No active replica to load

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

        # Store replicas
        ds_active = xr.Dataset(
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

        return ds_active

    def load_passive_feature(
        self, passive_feature_info: dict = {}, single_vessel_per_segment: bool = True
    ):
        """
        Method to load replicas (passive source)
        """
        start_datetimes = passive_feature_info.get("start_datetimes", [])
        end_datetimes = passive_feature_info.get("end_datetimes", [])

        if len(start_datetimes) == 0 or len(end_datetimes) == 0:
            return None  # No passive replica to load

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
                if single_vessel_per_segment and (len(ais_replica["mmsi"].values) > 1):
                    print(
                        f"Warning: more than one vessel detected in the AIS data for passive segment {start_dt} - {end_dt} while using single_vessel_per_segment = True. This segment will be skipped."
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

        # Store replicas
        ds_passive = xr.Dataset(
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
        return ds_passive

    ###########################
    # Matrching library and event features
    ###########################

    def match(self, id_library: int = None, id_event: int = None):
        """
        Method to match library and event features
        """
        # Load data to match
        ds_library = xr.open_dataset(
            os.path.join(self.root_library_data, f"library_{id_library}.nc")
        )
        ds_event = xr.open_dataset(
            os.path.join(self.root_event_data, f"event_{id_event}.nc")
        )

        root_img = os.path.join(
            self.root_results_fig, f"library_{id_library}_event_{id_event}"
        )
        os.makedirs(root_img, exist_ok=True)

        # Plot positions fo the library replicas
        plot_mfp_dataset(ds=ds_library, cmap="managua", root_img=root_img)
        # Plot positions of the event replicas
        plot_mfp_dataset(ds=ds_event, cmap="vanimo", root_img=root_img)

        # Plot library and event replicas positions together
        plot_mfp_datasets(ds_library, ds_event, root_img=root_img)

        # Compute distance matrix between library and event replicas
        ds_results = ambiguity(
            ds_library=ds_library,
            ds_event=ds_event,
            fmin=100,
            fmax=900,
            dist_type="hermitian_angle",
            use_weighted_mean=False,
            verbose=True,
        )

        # Plot distances
        plot_results_dist(ds_results=ds_results, root_img=root_img)

        plot_results_sorted_dist(ds_results=ds_results, root_img=root_img)

        plt.show()


# =======================================================================================================================
# Utils
# =======================================================================================================================


def plot_mfp_datasets(ds_library, ds_event, root_img: str = None):

    if root_img is not None:
        os.makedirs(root_img, exist_ok=True)
        save_fig = True

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Plot receiver positions
    keys = ["obs1", "obs2", "obs3"]
    for k in keys:
        e = ds_library.attrs[f"{k}_e_apriori"]
        n = ds_library.attrs[f"{k}_n_apriori"]
        ax.scatter(
            e,
            n,
            marker="D",
            label=k,
            zorder=1,
            s=150,
        )

    # Plot library replicas positions
    e_library = ds_library["e_replica"].values
    n_library = ds_library["n_replica"].values
    im_lib = ax.scatter(
        e_library,
        n_library,
        marker="+",
        label=f"{ds_library.type.capitalize()} ({ds_library.id})",
        c=np.arange(e_library.size),
        cmap="managua",
    )

    # Plot event replicas positions
    e_event = ds_event["e_replica"].values
    n_event = ds_event["n_replica"].values
    im_event = ax.scatter(
        e_event,
        n_event,
        marker="x",
        label=f"{ds_event.type.capitalize()} ({ds_event.id})",
        c=np.arange(e_event.size),
        cmap="vanimo",
    )

    # Add colorbars
    plt.colorbar(im_lib, label="Library replica index")
    plt.colorbar(im_event, label="Event replica index")

    plt.legend(fontsize=12)
    plt.xlabel("E [m]")
    plt.ylabel("N [m]")

    if save_fig:
        fpath = os.path.join(
            root_img,
            f"{ds_library.type}_{ds_library.id}_and_{ds_event.type}_{ds_event.id}_positions.png",
        )
        plt.savefig(fpath, bbox_inches="tight")


def plot_mfp_dataset(ds, cmap="jet", root_img: str = None):

    if root_img is not None:
        os.makedirs(root_img, exist_ok=True)
        save_fig = True

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Plot receiver positions
    keys = ["obs1", "obs2", "obs3"]
    for k in keys:
        e = ds.attrs[f"{k}_e_apriori"]
        n = ds.attrs[f"{k}_n_apriori"]
        ax.scatter(
            e,
            n,
            marker="D",
            label=k,
            zorder=1,
            s=150,
        )

    # Plot replicas positions
    e_library = ds["e_replica"].values
    n_library = ds["n_replica"].values
    im = ax.scatter(
        e_library,
        n_library,
        marker="+",
        label=f"{ds.type.capitalize()} ({ds.id})",
        c=np.arange(e_library.size),
        cmap=cmap,
    )
    plt.colorbar(im, label="Replica index")

    plt.legend(fontsize=12)
    plt.xlabel("E [m]")
    plt.ylabel("N [m]")

    if save_fig:
        fpath = os.path.join(root_img, f"{ds.type}_{ds.id}_positions.png")
        plt.savefig(fpath, bbox_inches="tight")


def ambiguity(
    ds_library,
    ds_event,
    fmin=100,
    fmax=900,
    dist_type="hermitian_angle",
    use_weighted_mean=False,
    verbose=False,
):

    if dist_type == "hermitian_angle":
        dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}
        comment = "Compute distance using hermitian angle distance (in C^N)"
    elif dist_type == "euclidean":
        comment = "Compute distance using euclidean distance (in C^N)"
    elif dist_type == "euclidean_module":
        comment = "Compute distance using euclidean distance on the module of RTF vectors (in R^N)"
    elif dist_type == "euclidean_phase":
        comment = "Compute distance using euclidean distance on the phase of RTF vectors (in R^N)"
    else:
        print("Warning : unknown distance, set to default -> hermitian angle")
        dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}
        comment = "Compute distance using hermitian angle distance (in C^N)"

    if verbose:
        print(comment)

    # Define comon frequency band to use
    fmin_common_band = max(fmin, max(ds_library.f_rtf.min(), ds_event.f_rtf.min()))
    fmax_common_band = min(fmax, min(ds_library.f_rtf.max(), ds_event.f_rtf.max()))

    # Slice common band
    ds_library = ds_library.sel(f_rtf=slice(fmin_common_band, fmax_common_band))
    ds_event = ds_event.sel(f_rtf=slice(fmin_common_band, fmax_common_band))

    # Library RTFs
    library_replicas = ds_library.rtf_amp * np.exp(1j * ds_library.rtf_phase)

    # Event RTFs
    event_feature = ds_event.rtf_amp * np.exp(1j * ds_event.rtf_phase)
    # Reshape to 4D array to be able to apply distance function : (n_rcv, n_freq, n_segment_dt) -> (n_rcv, n_freq, n_segment_dt, 1)
    event_feature_4d = event_feature.values[..., np.newaxis]

    rtf_distances = []

    # TODO : add option to use weighted mean

    if dist_type == "hermitian_angle":
        # Iterate of each replica of the library
        for rep_id in library_replicas.replica_id.values:

            replica_i = library_replicas.sel(replica_id=rep_id)
            dist = D_hermitian_angle_fast(
                rtf_ref=replica_i.values,
                rtf=event_feature_4d,
                **dist_kwargs,
            )

            rtf_distances.append(dist)
    rtf_dist = np.array(rtf_distances)
    rtf_dist = (
        rtf_dist.T
    )  # Transpose to have shape (n_event_feature, n_library_replica)

    # Spatial distance between library and event replicas
    event_e = ds_event["e_replica"].values
    event_n = ds_event["n_replica"].values
    libray_e = ds_library["e_replica"].values
    libray_n = ds_library["n_replica"].values
    event_coords = np.column_stack((event_e, event_n))
    library_coords = np.column_stack((libray_e, libray_n))

    spatial_dist = cdist(event_coords, library_coords, metric="euclidean")

    # Build results dataset
    ds_results = xr.Dataset(
        data_vars={
            "rtf_dist": (("event_replica_id", "library_replica_id"), rtf_dist),
            "spatial_dist": (("event_replica_id", "library_replica_id"), spatial_dist),
        },
        coords={
            "event_replica_id": ds_event.replica_id.values,
            "library_replica_id": ds_library.replica_id.values,
        },
    )

    # Add attributes to the dataset
    ds_results.attrs = {
        "description": f"Distance matrix between library and event replicas computed using {dist_type} distance for the RTF features and euclidean distance for the spatial coordinates.",
        "rtf_dist_type": dist_type,
        "spatial_dist_type": "euclidean",
        "library_id": ds_library.id,
        "event_id": ds_event.id,
    }
    # Add attributes to variables
    ds_results["rtf_dist"].attrs = {
        "description": f"Distance between library and event replicas computed using {dist_type} distance on the RTF features.",
        "units": "°",
        "long_name": r"$\theta$",
    }
    ds_results["spatial_dist"].attrs = {
        "description": "Euclidean distance between library and event replicas in the spatial domain.",
        "units": "m",
        "long_name": "Spatial distance",
    }

    # Add attributes to coordinates
    ds_results["event_replica_id"].attrs = {
        "description": "ID of the event replica",
        "long_name": "Event replica ID",
    }
    ds_results["library_replica_id"].attrs = {
        "description": "ID of the library replica",
        "long_name": "Library replica ID",
    }

    return ds_results


def plot_results_dist(ds_results, root_img: str = None):

    print(f"\tPlotting RTF distance vs spatial distance")

    if root_img is not None:
        os.makedirs(root_img, exist_ok=True)
        save_fig = True

    # Find crossing point between library replicas and event replicas in the spatial domain
    min_dist_ids = ds_results.spatial_dist.argmin(...)
    min_dist_event_replica_id = min_dist_ids["event_replica_id"].values
    min_dist_library_replica_id = min_dist_ids["library_replica_id"].values
    min_dist_results = ds_results.isel(
        event_replica_id=min_dist_event_replica_id,
        library_replica_id=min_dist_library_replica_id,
    )

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16, 12))

    # Define colorbar limits
    vmin = np.percentile(ds_results.rtf_dist.values, 0.1)
    vmax = np.percentile(ds_results.rtf_dist.values, 50)

    # Theta distance
    ds_results.rtf_dist.plot(
        x="event_replica_id",
        y="library_replica_id",
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
        ax=axs[0],
    )
    axs[0].set_xlabel("")

    # Add marker at minimum spatial distance point
    axs[0].scatter(
        min_dist_results.event_replica_id,
        min_dist_results.library_replica_id,
        marker="X",
        s=80,
        color="cyan",
        label="Minimum spatial distance",
        zorder=5,
    )

    # Spatial distance
    ds_results.spatial_dist.plot(
        x="event_replica_id",
        y="library_replica_id",
        cmap="magma",
        vmin=0,
        vmax=1000,
        ax=axs[1],
    )

    # Add marker at minimum spatial distance point
    axs[1].scatter(
        min_dist_results.event_replica_id,
        min_dist_results.library_replica_id,
        marker="X",
        s=80,
        color="cyan",
        label="Minimum spatial distance",
        zorder=5,
    )

    axs[0].legend(fontsize=12)
    axs[1].legend(fontsize=12)

    if save_fig:
        fpath = os.path.join(
            root_img,
            f"res_library_{ds_results.library_id}_event_{ds_results.event_id}_distances.png",
        )
        plt.savefig(fpath, bbox_inches="tight")


def plot_results_sorted_dist(
    ds_results: xr.Dataset, offset_around_min_dist: int = 2, root_img: str = None
):

    print(f"\tPlotting RTF distance vs spatial distance (sorted by distance)")

    if root_img is not None:
        os.makedirs(root_img, exist_ok=True)
        save_fig = True

    # Find crossing point between library replicas and event replicas in the spatial domain
    min_dist_ids = ds_results.spatial_dist.argmin(...)
    min_dist_event_replica_id = min_dist_ids["event_replica_id"].values
    min_dist_library_replica_id = min_dist_ids["library_replica_id"].values
    min_dist_results = ds_results.isel(
        event_replica_id=min_dist_event_replica_id,
        library_replica_id=min_dist_library_replica_id,
    )

    # Compare RTF distance and spatial distance for a few replicas around the minimum spatial distance point
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16, 12))

    # Extract rtf distance for a few replicas around the minimum spatial distance point
    min_rep = max(0, min_dist_results.library_replica_id - offset_around_min_dist)
    max_rep = min(
        ds_results.library_replica_id.max().values,
        min_dist_results.library_replica_id + offset_around_min_dist,
    )
    ds_results_around_min_dist = ds_results.sel(
        library_replica_id=slice(min_rep, max_rep)
    )

    ds_results_around_min_dist.rtf_dist.plot(ax=axs[0], hue="library_replica_id")
    ds_results_around_min_dist.spatial_dist.plot(ax=axs[1], hue="library_replica_id")

    if save_fig:
        fpath = os.path.join(
            root_img,
            f"res_library_{ds_results.library_id}_event_{ds_results.event_id}_var_around_min_dist.png",
        )
        plt.savefig(fpath, bbox_inches="tight")

    # Extract theta and dist variation for the selected replica
    ds_results_min_dist = ds_results.sel(
        library_replica_id=min_dist_results.library_replica_id.values
    )
    dist_to_cpa_argsort = np.argsort(ds_results_min_dist.spatial_dist.values)
    sorted_spatial_dist = ds_results_min_dist.spatial_dist.values[dist_to_cpa_argsort]
    sorted_rtf_dist = ds_results_min_dist.rtf_dist.values[dist_to_cpa_argsort]

    plt.figure()
    plt.scatter(
        sorted_spatial_dist,
        sorted_rtf_dist,
        # label=f"{pos} ({theta_dist_obs[pos]['id']})",
    )
    plt.xlabel("Spatial distance to closest replica [m]")
    plt.ylabel(r"$\theta$ [°]")

    if save_fig:
        fpath = os.path.join(
            root_img,
            f"res_library_{ds_results.library_id}_event_{ds_results.event_id}_sorted_var_at_min_dist.png",
        )
        plt.savefig(fpath, bbox_inches="tight")


# =====================================================================================================================
# Test
# =====================================================================================================================


def test():
    """
    Test function
    """
    root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data"
    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\img\rtf_mfp"
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

    rtf_mfp_processor = RTF_MFP_Processor(
        root_data=root_data,
        root_img=root_img,
        reference_receiver_id=ref_rcv_id,
        rtf_estimator=rtf_estimator,
        fsm_active_kwargs=fsm_active_kwargs,
        fsm_passive_kwargs=fsm_passive_kwargs,
        mode="overwrite",
        plot_replicas_features=False,
        verbose=True,
    )

    ###########################
    # Library computation
    ###########################

    # Populate library
    active_replicas_args = {
        "replica_sequence_ids": [],
        "load_precomputed_feature": True,
    }
    passive_replicas_args = {
        "start_datetimes": [
            datetime(year=2025, month=10, day=15, hour=00, minute=15, second=00)
        ],
        "end_datetimes": [
            datetime(year=2025, month=10, day=15, hour=00, minute=30, second=00)
        ],
        "load_precomputed_feature": True,
    }
    # passive_replicas_args = {
    #     "start_datetimes": [],
    #     "end_datetimes": [],
    #     "load_precomputed_feature": True,
    # }
    rtf_mfp_processor.compute_library(
        active_feature_args=active_replicas_args,
        passive_feature_args=passive_replicas_args,
        id=0,
    )

    ###########################
    # Event computation
    ###########################

    # Derive event
    active_feature_args = {
        "replica_sequence_ids": [],
        "load_precomputed_feature": True,
    }
    passive_feature_args = {
        "start_datetimes": [
            datetime(year=2025, month=10, day=15, hour=1, minute=40, second=00)
        ],
        "end_datetimes": [
            datetime(year=2025, month=10, day=15, hour=1, minute=50, second=00)
        ],
        "load_precomputed_feature": True,
    }
    rtf_mfp_processor.compute_event(
        active_feature_args=active_feature_args,
        passive_feature_args=passive_feature_args,
        id=0,
    )

    ###########################
    # Matching library and event features
    ###########################

    rtf_mfp_processor.match(id_library=0, id_event=0)


if __name__ == "__main__":
    test()
