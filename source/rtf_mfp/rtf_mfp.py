#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_mfp.py
@Time    :   2026/05/18 16:26:31
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

from publication.publication_figure import PubFigure
from source.rtf_mfp.rtf_mfp_feature_manager import FeatureManager
from source.rtf_mfp.rtf_mfp_misc import get_psd
from source.rtf_mfp.rtf_mfp_dist_utils import ambiguity
from source.rtf_mfp.rtf_mfp_plot_utils import (
    plot_results_dist,
    plot_results_sorted_dist,
    plot_features,
    plot_mfp_dataset,
    plot_mfp_datasets,
)

# DEFAUTS # TODO move this in a dedicated file ?
ROOT_DATA = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\reseau_9R\data"
)
ROOT_IMG = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\reseau_9R\img\rtf_mfp"
REF_RCV_ID = 1
RTF_ESTIMATOR = "cs-evd"

PubFigure()
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
        receiver_ids: list = [1, 2, 3],
        reference_receiver_id: int = REF_RCV_ID,
        rtf_estimator: str = RTF_ESTIMATOR,
        fm_props: dict = {},
        fm_passive_kwargs: dict = {},
        mode="overwrite",
        plot_replicas_features: bool = False,
        wav_dataset_filename: str = "channel_H_wav.nc",
        ais_dataset_filename: str = "ais.nc",
        bathy_dataset_filename: str = "bathy.nc",
        verbose: bool = False,
    ) -> None:
        """
        Constructor of the class
        """

        # Path to data folder
        self.root_data = root_data
        # Path to img folder
        self.root_img = root_img

        # Filenames
        self.wav_dataset_filename = wav_dataset_filename
        self.ais_dataset_filename = ais_dataset_filename
        self.bathy_dataset_filename = bathy_dataset_filename

        self.datetime_fmt = "%Y-%m-%d_%H-%M-%S"

        # Paths
        self.set_paths()
        # Load datasets
        self.load_datasets()

        # List of receiver index
        self.receiveir_ids = receiver_ids
        # Index of the receiver to use as reference
        self.reference_receiver_id = reference_receiver_id

        # RTF estimator
        self.rtf_estimator = rtf_estimator
        self.mode = mode

        # Flags
        self.plot_replicas_features = plot_replicas_features
        self.verbose = verbose

        # Define Fiberscope Managers
        self.fs = fm_props.get("fs", 100)
        self.tau_rtf_analysis = fm_props.get("tau_rtf_analysis", 3)
        self.alpha_overlap_rtf_analysis = fm_props.get("alpha_overlap", 0.5)

        self.fm_passive = None
        self.fm_passive_analysis_segment_duration = fm_passive_kwargs.get(
            "analysis_segment_duration", 10
        )
        self.fm_passive_analysis_segment_alpha_overlap = fm_passive_kwargs.get(
            "analysis_segment_alpha_overlap", 0.5
        )

        self.set_fm()

    def set_paths(self):
        """
        Set usefull paths
        """
        # Folder to store RTFs
        self.root_rtf_data = os.path.join(self.root_data, "sequences")
        # Dedicated folders for active and passive rtf data
        self.root_rtf_data_passive = os.path.join(self.root_rtf_data, "passive")
        # Folder to store the library
        self.root_library_data = os.path.join(self.root_data, "library")
        self.root_event_data = os.path.join(self.root_data, "event")

        # Folder to save results figures
        self.root_results_fig = os.path.join(self.root_img, "results")
        self.root_features_fig = os.path.join(self.root_img, "features")

        for path in [
            self.root_rtf_data,
            self.root_rtf_data_passive,
            self.root_library_data,
            self.root_event_data,
            self.root_results_fig,
            self.root_features_fig,
        ]:
            os.makedirs(path, exist_ok=True)

        # Wav netcdf dataset filepath
        self.wav_dataset_filepath = os.path.join(
            self.root_data, self.wav_dataset_filename
        )
        # AIS netcdf dataset filepath
        self.ais_dataset_filepath = os.path.join(
            self.root_data, self.ais_dataset_filename
        )
        # Bathy netcdf dataset filepath
        self.bathy_dataset_filepath = os.path.join(
            self.root_data, self.bathy_dataset_filename
        )

    def load_datasets(self):
        """
        Load usefull datasets
        """
        # Load wav dataset
        self.ds_wav = xr.open_dataset(self.wav_dataset_filepath)
        # Load AIS dataset
        self.ds_ais = xr.open_dataset(self.ais_dataset_filepath)
        # Load bathy dataset
        self.ds_bathy = xr.open_dataset(self.bathy_dataset_filepath)

    def set_fm(self):
        """
        Set FeatureManager
        """
        self.set_fm_()
        self.set_fm_props()

    def set_fm_props(self):
        """
        Set FeatureManager properties
        """

        # Number of samples corresponding to the assumed impulse response duration
        n_rtf_analysis = int(self.tau_rtf_analysis * self.fs)
        # Get closer power of 2
        nperseg = 2 ** int(
            np.log2(n_rtf_analysis) + 1
        )  # Number of sample per snapshot to use = closest power of two
        noverlap = int(nperseg * self.alpha_overlap_rtf_analysis)

        self.fm_nperseg = nperseg
        self.fm_noverlap = noverlap

        if self.fm_passive is not None:

            # Passive manager
            self.fm_passive.nperseg = self.fm_nperseg
            self.fm_passive.noverlap = self.fm_noverlap
        else:
            self.set_fm()

    def set_fm_(self):
        """
        Set passive Fiberscope Manager
        """

        self.fm_passive = FeatureManager(
            ds_wav=self.ds_wav,
            root_img=self.root_features_fig,
            root_processed_data=self.root_data,
            receiver_ids=self.receiveir_ids,
            reference_receiver_id=self.reference_receiver_id,
            plot_feature=self.plot_replicas_features,
            analysis_segment_duration=self.fm_passive_analysis_segment_duration,
            analysis_segment_alpha_overlap=self.fm_passive_analysis_segment_alpha_overlap,
            rtf_estimator=self.rtf_estimator,
            verbose=self.verbose,
        )

    ###########################
    # Compute
    ###########################
    def compute(
        self,
        passive_feature_args: dict = {},
        ds_type: str = "library",
        id: int = None,
        single_vessel_per_segment: bool = True,
        target_mmsi: int = None,
        Rv_global: np.ndarray = None,
    ):
        """
        Method to compute dataset
        """
        # Derive replicas
        passive_feature_info = self.derive_feature(
            passive_feature_args=passive_feature_args,
            Rv_global=Rv_global,
        )

        # Load derived replicas
        ds_passive = self.load_feature(
            passive_feature_info=passive_feature_info,
            single_vessel_per_segment=single_vessel_per_segment,
            target_mmsi=target_mmsi,
        )

        #
        # Concatenate active and passive features
        ds = self.add_metadata(ds=ds_passive, ds_type=ds_type)

        # Save
        self.save(ds=ds, ds_type=ds_type, id=id)

        # Write associated metadata file with the same ID as the library to keep track of the library content and properties
        # TODO ?
        # self.save_library_metadata(ds_library=ds_library, active_feature_info=active_feature_info, passive_feature_info=passive_feature_info)

    def compute_library(
        self,
        passive_feature_args: dict = {},
        id: int = None,
        target_mmsi: int = None,
        Rv_global: np.ndarray = None,
    ):
        """Method to compute library dataset"""

        self.compute(
            passive_feature_args=passive_feature_args,
            ds_type="library",
            id=id,
            single_vessel_per_segment=True,  # For the library we wish to use segment containing only one vessel to ensure the quality of the passive replicas
            target_mmsi=target_mmsi,
            Rv_global=Rv_global,
        )

    def compute_event(
        self,
        passive_feature_args: dict = {},
        target_mmsi: int = None,
        id: int = None,
        Rv_global: np.ndarray = None,
    ):
        """Method to compute event dataset"""

        self.compute(
            passive_feature_args=passive_feature_args,
            ds_type="event",
            id=id,
            single_vessel_per_segment=False,  # For the event we want to be able to use segments containing multiple vessels to be able to test the method in more complex scenarios
            target_mmsi=target_mmsi,
            Rv_global=Rv_global,
        )

    def add_metadata(self, ds: xr.Dataset, ds_type: str = "library") -> xr.Dataset:

        ds.attrs = {
            "reference_receiver_id": self.reference_receiver_id,
            "rtf_estimator": self.rtf_estimator,
            "description": f"{ds_type.capitalize()} dataset containing RTF-MFP features.",
            "type": ds_type,
        }

        # # Add receiver positions as attributes to the dataset
        # for k in ["obs1", "obs2", "obs3"]:
        #     ds.attrs[f"{k}_e_apriori"] = self.ds_gps.attrs[f"{k}_e_apriori"]
        #     ds.attrs[f"{k}_n_apriori"] = self.ds_gps.attrs[f"{k}_n_apriori"]
        #     ds.attrs[f"{k}_u_apriori"] = self.ds_gps.attrs[f"{k}_u_apriori"]

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

        ds["feature_psd"].attrs = {
            "description": "PSD of the signal associated to each feature.",
            "long_name": r"$S_{xx}$",
        }

        # for coord in ["e", "n", "u"]:
        #     ds[f"{coord}_replica"].attrs = {
        #         "description": f"{coord.upper()} coordinate of the replica position (local ENU frame)",
        #         "units": "m",
        #         "long_name": coord.upper(),
        #     }

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
        self,
        passive_feature_args: dict = {},
        Rv_global: np.ndarray = None,
    ):
        """
        Method to derive feature
        """
        passive_feature_info = self.derive_feature_passive(
            passive_feature_args=passive_feature_args, Rv_global=Rv_global
        )

        return passive_feature_info

    def derive_feature_passive(
        self, passive_feature_args: dict = {}, Rv_global: np.ndarray = None
    ) -> dict:
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
                record_id = f"passive_{datetime.strftime(start_dt, self.fm_passive.datetime_fmt)}_to_{datetime.strftime(end_dt, self.fm_passive.datetime_fmt)}"
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

            self.fm_passive.process_analysis(
                t_start=start_dt,
                t_end=end_dt,
                set_stft_props=False,
                Rv_global=Rv_global,
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
        passive_feature_info: dict = {},
        single_vessel_per_segment: bool = True,
        target_mmsi: int = None,
    ):
        """
        Method to load features
        """
        ds_passive = self.load_passive_feature(
            passive_feature_info=passive_feature_info,
            single_vessel_per_segment=single_vessel_per_segment,
            target_mmsi=target_mmsi,
        )

        return ds_passive

    def load_passive_feature(
        self,
        passive_feature_info: dict = {},
        single_vessel_per_segment: bool = True,
        target_mmsi: int = None,
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
            record_id = f"passive_{datetime.strftime(start_dt, self.datetime_fmt)}_to_{datetime.strftime(end_dt, self.datetime_fmt)}"
            rep_filepath = os.path.join(
                self.root_rtf_data_passive, f"sequence_{record_id}_rtf.nc"
            )

            if os.path.exists(rep_filepath):
                ds_feature = xr.open_dataset(rep_filepath)

                # # Extract source position
                # time_offset = timedelta(seconds=60)
                # ais_feature = self.ds_ais.sel(
                #     time=slice(start_dt - time_offset, end_dt + time_offset)
                # )
                # ais_feature = filter_ais(ais_event=ais_feature)

                # # Ensure only one vessel in the segment
                # if single_vessel_per_segment and (len(ais_feature["mmsi"].values) > 1):
                #     print(
                #         f"Warning: more than one vessel detected in the AIS data for passive segment {start_dt} - {end_dt} while using single_vessel_per_segment = True. This segment will be skipped."
                #     )
                #     continue
                # elif len(ais_feature["mmsi"].values) == 0:
                #     print(
                #         f"Warning: no vessel detected in the AIS data for passive segment {start_dt} - {end_dt}. This segment will be skipped."
                #     )
                #     continue
                # else:
                #     if (
                #         target_mmsi is not None
                #         and target_mmsi in ais_feature.mmsi.values
                #     ):
                #         ais_feature = ais_feature.sel(mmsi=target_mmsi)
                #         print(f"Using target vessel : mmsi={target_mmsi}")
                #     else:
                #         ais_feature = ais_feature.isel(
                #             mmsi=0
                #         )  # Keep only the first vessel
                #         print(
                #             f"Using first vessel in list (default behavior) : mmsi={ais_feature.mmsi.values}"
                #         )

                feature_psd = self.get_feature_psd(
                    ds_feature=ds_feature, src_type="passive"
                )

                #     e_feature = ais_feature.e.interp(
                #         time=ds_feature.segment_dt.values
                #     ).values
                #     n_feature = ais_feature.n.interp(
                #         time=ds_feature.segment_dt.values
                #     ).values
                #     u_feature = ais_feature.u.interp(
                #         time=ds_feature.segment_dt.values
                #     ).values

                if i == 0:
                    rtf_amp = ds_feature["rtf_amp_hat"].values
                    rtf_phase = ds_feature["rtf_phase_hat"].values
                    # e_feature_all = e_feature
                    # n_feature_all = n_feature
                    # u_feature_all = u_feature
                    # feature_weights_all = feature_weights
                    feature_psd_all = feature_psd

                    i += 1
                else:
                    # Concatenate along the time axis : (n_rcv, n_freq, n_segment_dt) -> (n_rcv, n_freq, n_feature)
                    rtf_amp = np.concatenate(
                        [rtf_amp, ds_feature["rtf_amp_hat"].values], axis=-1
                    )
                    rtf_phase = np.concatenate(
                        [rtf_phase, ds_feature["rtf_phase_hat"].values], axis=-1
                    )
                    # e_feature_all = np.concatenate([e_feature_all, e_feature])
                    # n_feature_all = np.concatenate([n_feature_all, n_feature])
                    # u_feature_all = np.concatenate([u_feature_all, u_feature])
                    # feature_weights_all = np.concatenate(
                    #     [feature_weights_all, feature_weights], axis=0
                    # )
                    feature_psd_all = np.concatenate(
                        [feature_psd_all, feature_psd], axis=0
                    )
            # else:
            #     print(
            #         f"Replica file for passive segment {start_dt} - {end_dt} not found at {rep_filepath}."
            #     )

        # Store replicas
        ds_passive = xr.Dataset(
            data_vars={
                "rtf_amp": (("h_index", "f_rtf", "replica_id"), rtf_amp),
                "rtf_phase": (("h_index", "f_rtf", "replica_id"), rtf_phase),
                "feature_psd": (
                    ("f_rtf", "replica_id"),
                    feature_psd_all.T.astype(np.float32),
                ),
                # "e_replica": (("replica_id"), e_feature_all.astype(np.float32)),
                # "n_replica": (("replica_id"), n_feature_all.astype(np.float32)),
                # "u_replica": (("replica_id"), u_feature_all.astype(np.float32)),
                "replica_type": (("replica_id"), [2] * rtf_amp.shape[-1]),
            },
            coords={
                "h_index": ds_feature["h_index"].values,
                "f_rtf": ds_feature["f_rtf"].values,
                "replica_id": np.arange(rtf_amp.shape[-1]),
            },
        )
        return ds_passive

    def get_feature_psd(self, ds_feature: xr.Dataset, src_type: str = "active"):

        replica_psd = []

        if src_type == "passive":

            start_seg = 0
            end_seg = ds_feature.analysis_segment_duration
            segment_shift = ds_feature.analysis_segment_duration * (
                1 - ds_feature.analysis_segment_alpha_overlap
            )

            for i_seg in range(ds_feature.sizes["segment_dt"]):
                # Get signal corresponding to the current segment
                passive_sig_seg = ds_feature.signal.sel(time=slice(start_seg, end_seg))
                # Compute weights using signal from the reference receiver
                passive_sig_seg_rcv_ref = passive_sig_seg.sel(
                    h_index=ds_feature.h_index_ref
                )

                psd_kwargs = {
                    "fs": ds_feature.fs,
                    "nperseg": self.fm_passive.nperseg,
                    "noverlap": self.fm_passive.noverlap,
                    "fmin": ds_feature.f_rtf.min().values,
                    "fmax": ds_feature.f_rtf.max().values,
                }

                pxx = get_psd(signal=passive_sig_seg_rcv_ref.values, **psd_kwargs)

                replica_psd.append(pxx)

                end_seg += segment_shift
                start_seg += segment_shift

        replica_psd = np.array(replica_psd)  # Shape (n_seg, nf)

        return replica_psd

    ###########################
    # Matrching library and event features
    ###########################

    def match(
        self,
        id_library: int = None,
        id_event: int = None,
        ds_library: xr.Dataset = None,
        ds_event: xr.Dataset = None,
        dist_args: dict = {},
        plot_args: dict = {},
        root_img: str = None,
    ):
        """
        Method to match library and event features
        """

        # Load data to match
        if ds_library is None:
            ds_library = xr.open_dataset(
                os.path.join(self.root_library_data, f"library_{id_library}.nc")
            )
        else:
            id_library = ds_library.id

        if ds_event is None:
            ds_event = xr.open_dataset(
                os.path.join(self.root_event_data, f"event_{id_event}.nc")
            )
        else:
            id_event = ds_event.id

        if root_img is None:
            root_img = os.path.join(
                self.root_results_fig, f"library_{id_library}_event_{id_event}"
            )
            os.makedirs(root_img, exist_ok=True)

        plot_lib_pos = plot_args.get("plot_lib_pos", True)
        plot_event_pos = plot_args.get("plot_event_pos", True)
        plot_lib_and_event_pos = plot_args.get("plot_lib_and_event_pos", True)

        if plot_lib_pos:
            # Plot positions fo the library replicas
            plot_mfp_dataset(ds=ds_library, cmap="managua", root_img=root_img)

        if plot_event_pos:
            # Plot positions of the event replicas
            plot_mfp_dataset(ds=ds_event, cmap="vanimo", root_img=root_img)

        if plot_lib_and_event_pos:
            # Plot library and event replicas positions together
            plot_mfp_datasets(ds_library, ds_event, root_img=root_img)

        # Compute distance matrix between library and event replicas
        fmin = dist_args.get("fmin", 100)
        fmax = dist_args.get("fmax", 400)
        dist_type = dist_args.get("dist_type", "hermitian_angle")
        use_weighted_mean = dist_args.get("use_weighted_mean", False)

        ds_results = ambiguity(
            ds_library=ds_library,
            ds_event=ds_event,
            fmin=fmin,
            fmax=fmax,
            dist_type=dist_type,
            use_weighted_mean=use_weighted_mean,
            verbose=True,
        )

        plot_dist_matrices = plot_args.get("plot_dist_matrices", True)
        plot_dist_around_sptial_min_dist = plot_args.get(
            "plot_dist_around_sptial_min_dist", True
        )
        plot_feature_details = plot_args.get("plot_feature_details", True)
        plot_feature_details_module = plot_args.get("plot_feature_details_module", True)
        plot_feature_details_phase = plot_args.get("plot_feature_details_phase", False)
        plot_feature_details_theta = plot_args.get("plot_feature_details_theta", False)

        if plot_dist_matrices:
            # Plot distances
            plot_results_dist(
                ds_results=ds_results,
                ds_event=ds_event,
                ds_library=ds_library,
                root_img=root_img,
            )

        if plot_dist_around_sptial_min_dist:
            plot_results_sorted_dist(ds_results=ds_results, root_img=root_img)

        if plot_feature_details:
            # Plot feature details
            plot_features(
                ds_results=ds_results,
                ds_library=ds_library,
                ds_event=ds_event,
                root_img=root_img,
                plot_module=plot_feature_details_module,
                plot_phase=plot_feature_details_phase,
                plot_theta=plot_feature_details_theta,
            )

        return ds_results


if __name__ == "__main__":
    pass
