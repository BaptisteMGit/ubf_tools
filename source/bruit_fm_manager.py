#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   bruit_fm_manager.py
@Time    :   2025/09/25 11:32:47
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import pandas as pd
import xarray as xr
import source.global_constants as g

from pyproj import Geod
from obspy import UTCDateTime
from datetime import datetime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import URL_MAPPINGS


class BruitfmManager:
    """
    Class to get information about available dataset on the bruit-fm platform.

    https://www.bruit-fm.org/
    """

    def __init__(
        self,
        root_station_storage: str,
        available_clients: list = None,
        bathy_fpath: str = None,
    ):

        # Root folder to store stations information
        self.root_station_storage = root_station_storage
        if not os.path.exists(self.root_station_storage):
            os.makedirs(self.root_station_storage)

        # Folder to store scraped stations
        self.scraped_stations_folder = os.path.join(
            self.root_station_storage, "scraped_stations"
        )
        # Folder to store filtered stations
        self.filtered_stations_folder = os.path.join(
            self.root_station_storage, "filtered_stations"
        )
        if not os.path.exists(self.filtered_stations_folder):
            os.makedirs(self.filtered_stations_folder)

        # All networks available
        self.all_networks_fpath = os.path.join(
            self.scraped_stations_folder, "bruit_fm_networks.csv"
        )
        # All stations available
        self.all_stations_fpath = os.path.join(
            self.scraped_stations_folder, "bruit_fm_stations.csv"
        )

        # if client_address is None:
        #     self.client_address = "IRIS"  # Default client, other option is "RESIF"
        # else:
        #     self.client_address = client_address
        # self.client = Client(self.client_address)
        # self.available_clients = list(URL_MAPPINGS.keys())

        if available_clients is None:
            self.available_clients = ["IRIS", "RESIF"]
        else:
            self.available_clients = available_clients

        # Bathymetry file path
        if bathy_fpath is None:
            self.bathy_fpath = os.path.join(
                g.project_root,
                "data",
                "bathy",
                "GEBCO_2021_sub_ice_topo.nc",
            )

    def run_full_scraping_procedure(self):
        # # Step 1: Parse stations from html
        # networks = self.parse_stations_from_html(save=True)
        # print(f"Found {len(networks)} networks.")

        # Step 2: Scrap stations for each network
        self.scrap_networks_stations()
        print("Scraping of stations completed.")

        # Step 3: Merge all networks into a single csv file
        self.merge_scraped_networks()
        print("Merging of networks completed.")

        # Step 4: Compute elevation difference with GEBCO bathymetry
        self.compute_elevation_diff_gebco()
        print("Computation of elevation difference completed.")

    def parse_stations_from_html(self, save: bool = True):
        # Parse stations name from html ugly sequence
        html_balise_fpath = os.path.join(
            self.root_station_storage, "stations_html_balises.txt"
        )
        with open(html_balise_fpath, "r") as f:
            lines = f.readlines()

        # Merge all lines if several
        lines = "".join(lines)

        # Find network balise
        net_balise = 'target="_blank">'
        split_lines = lines.split(net_balise)[1:]  # First one is not needed
        # Extract networks names
        networks = np.atleast_1d(np.array([net.split("<")[0] for net in split_lines]))

        # Save networks
        if save:
            networks_df = pd.DataFrame({"netwrok_id": networks})
            networks_df.to_csv(self.all_networks_fpath, index=False)

        return networks

    def scrap_networks_stations(self, networks: list = None):
        # level1 = "station"

        for client_name in self.available_clients:
            try:
                cli = Client(client_name)
            except:
                continue

            print(f"Using client {client_name} for scraping.")

            # Init folder for this client
            client_folder = os.path.join(self.scraped_stations_folder, client_name)
            if not os.path.exists(client_folder):
                os.makedirs(client_folder)

            # Get list networks
            try:
                inv_cli_at_net_lvl = cli.get_stations(level="network")
                networks = inv_cli_at_net_lvl.get_contents()["networks"]
            except:
                continue

            # Avoid loading networks already scraped
            client_existing_files = os.listdir(client_folder)
            already_scraped_networks = [
                f.split("_")[-1].split(".")[0]
                for f in client_existing_files
                if f.startswith("stations_") and f.endswith(".csv")
            ]
            networks = [net for net in networks if net not in already_scraped_networks]

            # Walk through networks
            for net in networks:
                # print(f"Scraping network {net.code}")

                # Get list of stations
                inv_net_at_ch_lvl = cli.get_stations(network=net, level="channel")
                # stations = inv_sta.get_contents()["stations"]

                # Walk through stations
                station_to_save = False
                init_ds_current_network = True
                for station_grp in inv_net_at_ch_lvl.networks:

                    if not station_grp.stations:
                        print(f"No stations found for network {station_grp.code}")

                    # Walk through stations
                    for sta in station_grp.stations:

                        channels = [channel.code for channel in sta.channels]

                        end_date = (
                            sta.end_date if sta.end_date else UTCDateTime()
                        )  # Handle stations that are still active
                        start_date = sta.start_date
                        duration = end_date - start_date

                        channel_tag = "_".join(channels)
                        # Store relevant information about the station
                        station_info = {
                            "client": client_name,
                            "network": station_grp.code,
                            "station": sta.code,
                            "restricted_status": sta.restricted_status,
                            "latitude": sta.latitude,
                            "longitude": sta.longitude,
                            "elevation": sta.elevation,
                            "channels": channel_tag,
                            "start_date": start_date,
                            "end_date": end_date,
                            "duration": duration,
                        }

                        # Add information line to dedicated dataframe
                        if init_ds_current_network:
                            ds_current_network = pd.DataFrame([station_info])
                            init_ds_current_network = False
                            station_to_save = True  # To avoid< the case where no station are available
                        else:
                            ds_current_network = pd.concat(
                                [ds_current_network, pd.DataFrame([station_info])],
                                ignore_index=True,
                            )

                # Save data
                if station_to_save:
                    fpath = os.path.join(
                        client_folder, f"stations_{station_grp.code}.csv"
                    )
                    # Drop duplicates
                    ds_current_network_clean = ds_current_network.copy()
                    ds_current_network_clean["start_date"] = pd.to_datetime(
                        [x.datetime for x in ds_current_network_clean["start_date"]]
                    )

                    # By the default, with the RESIF client, if the recording is still active the end_date is set to year 2500 which cant be handle by Timestamp in nanoseconds
                    # To handle this we set the end_date to pd.Timestamp.max if the end_date year is > pd.Timestamp.max
                    max_end_date = datetime(2030, 1, 1)
                    recording_2500_mask = np.array(
                        [
                            end_date > max_end_date
                            for end_date in ds_current_network_clean["end_date"]
                        ]
                    )
                    corrected_date = ds_current_network_clean["end_date"].copy()
                    corrected_date[recording_2500_mask] = max_end_date
                    corrected_date[~recording_2500_mask] = pd.to_datetime(
                        [
                            x.datetime
                            for x in ds_current_network_clean["end_date"][
                                ~recording_2500_mask
                            ]
                        ]
                    )
                    ds_current_network_clean["end_date"] = corrected_date

                    # ds_current_network_clean["end_date"] = pd.to_datetime(
                    #     [x.datetime for x in ds_current_network_clean["end_date"]]
                    # )
                    ds_current_network_clean = (
                        ds_current_network_clean.drop_duplicates()
                    )
                    ds_current_network_clean.to_csv(fpath, index=False)

            # if networks is None:
            #     # Load networks from csv
            #     networks_df = pd.read_csv(self.all_networks_fpath)
            #     networks = np.atleast_1d(networks_df["network_id"].values)

            # for network in networks:

            #     try:
            #         network_inventory = self.client.get_stations(
            #             network=network, level=level1
            #         )
            #         network_stations = network_inventory.networks[0].stations
            #         # print(network_inventory)
            #         init_ds_current_network = True

            #         for station in network_stations:
            #             level2 = "channel"
            #             station_inventory = self.client.get_stations(
            #                 network=network, station=station.code, level=level2
            #             )
            #             channels = [
            #                 channel.code
            #                 for channel in station_inventory.networks[0]
            #                 .stations[0]
            #                 .channels
            #             ]

            #             end_date = (
            #                 station.end_date if station.end_date else UTCDateTime()
            #             )  # Handle stations that are still active
            #             start_date = station.start_date
            #             duration = end_date - start_date

            #             channel_tag = "_".join(channels)
            #             # Store relevant information about the station
            #             station_info = {
            #                 "network": network,
            #                 "station": station.code,
            #                 "latitude": station.latitude,
            #                 "longitude": station.longitude,
            #                 "elevation": station.elevation,
            #                 "channels": channel_tag,
            #                 "start_date": start_date,
            #                 "end_date": end_date,
            #                 "duration": duration,
            #             }

            #             # Add information line to dedicated dataframe
            #             if init_ds_current_network:
            #                 ds_current_network = pd.DataFrame([station_info])
            #                 init_ds_current_network = False
            #             else:
            #                 ds_current_network = pd.concat(
            #                     [ds_current_network, pd.DataFrame([station_info])],
            #                     ignore_index=True,
            #                 )
            #     except:
            #         pass

            # # Save data
            # fpath = os.path.join(
            #     self.scraped_stations_folder, f"stations_{network}.csv"
            # )
            # # Drop duplicates
            # ds_current_network_clean = ds_current_network.copy()
            # ds_current_network_clean["start_date"] = pd.to_datetime(
            #     [x.datetime for x in ds_current_network_clean["start_date"]]
            # )
            # ds_current_network_clean["end_date"] = pd.to_datetime(
            #     [x.datetime for x in ds_current_network_clean["end_date"]]
            # )
            # ds_current_network_clean = ds_current_network_clean.drop_duplicates()
            # ds_current_network_clean.to_csv(fpath, index=False)

    def merge_scraped_networks(self):

        init_ds_global = True
        for root, dirs, file in os.walk(self.scraped_stations_folder):
            for f in file:
                if f.startswith("stations_") and f.endswith(".csv"):
                    fpath = os.path.join(root, f)
                    ds_current_network = pd.read_csv(fpath)

                    if init_ds_global:
                        ds_all_stations = ds_current_network
                        init_ds_global = False
                    else:
                        ds_all_stations = pd.concat(
                            [ds_all_stations, ds_current_network]
                        )

        # Remove duplicates
        ds_all_stations = ds_all_stations.drop_duplicates()
        # Drop index
        ds_all_stations = ds_all_stations.reset_index(drop=True)
        # Save
        ds_all_stations.to_csv(self.all_stations_fpath, index=False)

    def compute_elevation_diff_gebco(self, stations_df: pd.DataFrame = None):
        if stations_df is None:
            stations_df = pd.read_csv(self.all_stations_fpath)  # Load all stations
            save = True
            process_all_stations = True
        else:
            save = False
            process_all_stations = False

        # Load bathymetry data (# Height above mean sea level (in meters))
        xr_bathy = xr.open_dataset(self.bathy_fpath)

        if process_all_stations:
            # Init folder for batch
            batch_folder = os.path.join(
                os.path.dirname(self.all_stations_fpath), "batch_tmp"
            )
            if not os.path.exists(batch_folder):
                os.makedirs(batch_folder)

            # Size of each batch
            batch_size = 50
            # Split processing in severals batches to reduce memory load and interpolation effort
            n_stations = stations_df.shape[0]
            i_batch = 0
            i_end = 0
            while i_end <= n_stations:
                ## Extract batch ##
                batch = stations_df.iloc[i_end : i_end + batch_size].copy()
                # Update batch index
                i_end += batch_size
                ## Process current batch ##
                # Extract bathymetry at station locations
                lons = batch["longitude"].values
                lats = batch["latitude"].values
                bathy_at_stations = xr_bathy.elevation.sel(
                    lon=xr.DataArray(lons, dims="points"),
                    lat=xr.DataArray(lats, dims="points"),
                    method="nearest",  # ou "linear" si tu veux interpolation bilinéaire
                )
                # Convert to numpy array
                bathy_at_stations = bathy_at_stations.values
                batch["height_above_mean_sea_level"] = bathy_at_stations

                # Compute difference between station elevation and bathymetry
                diff = batch["elevation"] - batch["height_above_mean_sea_level"]
                batch["elevation_diff"] = diff

                # Save batch
                fpath_batch = os.path.join(batch_folder, f"batch_{i_batch}.csv")
                batch.to_csv(fpath_batch, index=False)

                # Update batch number
                i_batch += 1

            # Merge all batch
            init_global_df = True
            for root, dirs, file in os.walk(batch_folder):
                for f in file:
                    if f.startswith("batch_") and f.endswith(".csv"):
                        fpath = os.path.join(root, f)
                        batch = pd.read_csv(fpath)

                        if init_global_df:
                            all_station_df = batch
                            init_global_df = False
                        else:
                            all_station_df = pd.concat([all_station_df, batch])

                        # Delete batch file
                        os.remove(fpath)

            # Drop index
            all_station_df = all_station_df.reset_index(drop=True)
            # Save
            all_station_df.to_csv(self.all_stations_fpath, index=False)

            # Delete batch folder
            os.rmdir(batch_folder)

        else:
            # Extract bathymetry at station locations
            lons = stations_df["longitude"].values
            lats = stations_df["latitude"].values
            bathy_at_stations = xr_bathy.elevation.sel(
                lon=xr.DataArray(lons, dims="points"),
                lat=xr.DataArray(lats, dims="points"),
                method="nearest",  # ou "linear" si tu veux interpolation bilinéaire
            )
            # Convert to numpy array
            bathy_at_stations = bathy_at_stations.values
            stations_df["height_above_mean_sea_level"] = bathy_at_stations

            # Compute difference between station elevation and bathymetry
            stations_df["elevation_diff"] = (
                stations_df["elevation"] - stations_df["height_above_mean_sea_level"]
            )
            # Save
            if save:
                stations_df.to_csv(self.all_stations_fpath, index=False)
            else:
                return stations_df

    def filter_stations(
        self,
        filters: dict = None,
        save: bool = False,
        filtered_stations_fname: str = None,
        filtered_stations_folder: str = None,
        verbose: bool = True,
    ):
        # Load data
        ds_all_stations = pd.read_csv(self.all_stations_fpath)
        ds_all_stations = self.pre_process_stations(ds_all_stations)

        # Backup
        ds_all_stations_backup = ds_all_stations.copy()

        # Apply filters
        if filters is None:
            raise ValueError("Filters dictionary must be provided.")

        # Extract filters
        # Restricted status (open ?)
        restricted_status = filters.get("restricted_status", "open")
        # Year that must be included in the recording period
        selected_year = filters.get("selected_year", None)
        # Minimum recording duration (in seconds)
        min_recording_duration = filters.get("min_recording_duration", None)
        # Required channels
        required_channels = filters.get("required_channels", None)
        # Minimum depth (in meters)
        min_depth = filters.get("min_depth", None)
        # Maximum distance between stations considered as part of the same array (in km) (defined by the median distance between stations)
        max_distance_between_stations = filters.get(
            "max_distance_between_stations", None
        )
        # Minimum number of receivers by array
        min_nb_rcv_by_array = filters.get("min_nb_rcv_by_array", None)
        # Wheter to check if the receiver lies on the seafloor
        receiver_on_seafloor = filters.get("receiver_on_seafloor", False)
        # Tolerance to consider that the receiver lies on the seafloor (in meters)  (elevation might not be very accurate because
        # of different global bathy model or different datum)
        receiver_on_seafloor_tolerance = filters.get(
            "receiver_on_seafloor_tolerance", 50
        )
        # Init mask to True
        mask = pd.Series([True] * ds_all_stations.shape[0])
        # Init filter description
        filter_description = []

        if restricted_status == "open":
            # Keep only stations with open access
            mask_restricted = self.filter_restricted_status(
                ds_all_stations, restricted_status
            )
            mask = mask & mask_restricted
            # Add line to filter description
            filter_description.append(f"Restricted status: {restricted_status}")

        if required_channels is not None:
            # Check if station contains the required channel
            mask_channels = self.filter_channels(ds_all_stations, required_channels)
            mask = mask & mask_channels
            # Add line to filter description
            filter_description.append(
                f"Required channels: {', '.join(required_channels)}"
            )

        if min_recording_duration is not None:
            # Check if recording match duration requirement
            mask_duration = self.filter_duration(
                ds_all_stations, min_recording_duration
            )
            mask = mask & mask_duration
            # Add line to filter description
            filter_description.append(
                f"Minimum recording duration {min_recording_duration/3600:.1f} hours"
            )

        if selected_year is not None:
            # Check if station was recording during the selected year
            mask_date = self.filter_date(ds_all_stations, selected_year)
            mask = mask & mask_date
            # Add line to filter description
            filter_description.append(f"Station active in {selected_year}")

        if min_depth is not None:
            # Check if station is below minimum depth
            mask_depth = self.filter_depth(ds_all_stations, min_depth)
            mask = mask & mask_depth
            # Add line to filter description
            filter_description.append(f"Minimum depth {min_depth:.1f} m")

        if receiver_on_seafloor:
            mask_seafloor = self.filter_receiver_on_seafloor(
                ds_all_stations, receiver_on_seafloor_tolerance
            )
            mask = mask & mask_seafloor
            # Add line to filter description
            filter_description.append(
                f"Receiver lying on seafloor (tolerance = +/-{receiver_on_seafloor_tolerance} m)"
            )

        # Apply mask
        filtered_stations = ds_all_stations[mask]
        filtered_stations = filtered_stations.reset_index(drop=True)

        if verbose:
            print(f"Stations selected after initial filtering : \n {filtered_stations}")

        if max_distance_between_stations is not None:
            # Add distance filtering between stations
            filtered_stations = self.filter_distance(
                filtered_stations, max_distance_between_stations
            )
            # Add line to filter description
            filter_description.append(
                f"Maximum distance between stations {max_distance_between_stations:.1f} km"
            )

            if verbose:
                print(
                    f"Stations selected after distance filtering : \n {filtered_stations}"
                )

            if min_nb_rcv_by_array is not None:
                filtered_stations = self.filter_number_of_receivers_by_array(
                    filtered_stations, min_nb_rcv_by_array
                )

        # Print selections stats
        self.print_filtering_stats(filtered_stations)

        # Save selected stations
        if save:

            if filtered_stations_folder is not None:
                if not os.path.exists(filtered_stations_folder):
                    os.makedirs(filtered_stations_folder)
            else:
                filtered_stations_folder = self.filtered_stations_folder

            # Define a unique id to identify the filtered stations
            unique_id = datetime.now().microsecond
            if filtered_stations_fname is None:
                filtered_stations_fname = f"filtered_stations_{unique_id}.csv"
                fpath_filters = os.path.join(
                    filtered_stations_folder, f"filters_{unique_id}.txt"
                )
            else:
                # Ensure extension is inlcuded
                filtered_stations_fname_root = filtered_stations_fname.split(".")[0]
                filtered_stations_fname = filtered_stations_fname_root + ".csv"
                filtered_stations_fname = f"filtered_stations_{filtered_stations_fname}"
                fpath_filters = os.path.join(
                    filtered_stations_folder,
                    f"filters_{filtered_stations_fname_root}.txt",
                )

            # Save filters used for this selection
            with open(fpath_filters, "w") as f:
                f.write("\n".join(filter_description))

            fpath_filtered_stations = os.path.join(
                filtered_stations_folder, filtered_stations_fname
            )
            filtered_stations.to_csv(fpath_filtered_stations, index=False)

        return filtered_stations, ds_all_stations_backup

    def filter_receiver_on_seafloor(
        self, stations_df: pd.DataFrame, tolerance
    ) -> pd.Series:
        # Check if station is close enough to the seafloor
        if "elevation_diff" not in stations_df.columns:
            stations_df = self.compute_elevation_diff_gebco(stations_df)
        mask_seafloor = np.abs(stations_df["elevation_diff"]) <= tolerance

        return mask_seafloor

    @staticmethod
    def pre_process_stations(stations_df: pd.DataFrame) -> pd.DataFrame:
        # Convert date columns to datetime
        stations_df["end_date"] = pd.to_datetime(
            stations_df["end_date"], format="mixed"
        )
        stations_df["start_date"] = pd.to_datetime(
            stations_df["start_date"], format="mixed"
        )
        # Drop stations with no channels information
        stations_df = stations_df.dropna(subset=["channels"])
        stations_df = stations_df.reset_index(drop=True)

        return stations_df

    @staticmethod
    def filter_restricted_status(
        stations_df: pd.DataFrame, restricted_status: str = "open"
    ) -> pd.Series:
        # Keep only stations with required restricted_status access
        mask_restricted = stations_df["restricted_status"] == restricted_status
        return mask_restricted

    @staticmethod
    def filter_channels(
        stations_df: pd.DataFrame, required_channels: list
    ) -> pd.Series:

        # Check if station contains the required channel
        required_channel_present = np.array(
            [stations_df["channels"].str.contains(ch) for ch in required_channels],
            dtype=bool,
        )
        n_ch_present = np.sum(required_channel_present, axis=0)
        mask_channels = n_ch_present >= 1

        return mask_channels

    @staticmethod
    def filter_duration(
        stations_df: pd.DataFrame, min_recording_duration: int
    ) -> pd.Series:
        # Check if recording match duration requirement
        duration = stations_df["end_date"] - stations_df["start_date"]
        mask_duration = duration.dt.total_seconds() >= min_recording_duration
        return mask_duration

    @staticmethod
    def filter_date(stations_df: pd.DataFrame, selected_year: int) -> pd.Series:
        # Check if station was recording during the selected year
        start_of_year = pd.to_datetime(f"{selected_year}-01-01")
        end_of_year = pd.to_datetime(f"{selected_year}-12-31")
        mask_date = (stations_df["start_date"] <= end_of_year) & (
            stations_df["end_date"] >= start_of_year
        )
        return mask_date

    @staticmethod
    def filter_depth(stations_df: pd.DataFrame, min_depth: float) -> pd.Series:
        # Check if station is below minimum depth
        mask_depth = (
            stations_df["elevation"] <= -min_depth
        )  # Elevation is in meters above sea level, so negative elevation means below sea level
        return mask_depth

    @staticmethod
    def filter_distance(
        stations_df: pd.DataFrame, max_distance_between_stations: float
    ) -> pd.DataFrame:
        # Filter stations based on distance criterion
        geod = Geod(ellps="WGS84")

        # Initialize to empty dataset with same columns as stations_df in case no stations are selected
        output_columns = stations_df.columns.tolist() + ["median_rcv_dist", "group_id"]
        selected_stations = pd.DataFrame(columns=output_columns)
        init_pd_selected_stations = True

        unique_id = 0  # Unique id to identify group of stations

        # Find stations that share common names (ex: PFO and PFO1, PFO2, etc)
        while len(stations_df) > 1:
            # Get index position of the last character that is a letter
            test_sta_name = stations_df["station"].iloc[0]
            alpha_ch_in_test_sta_name = [ch.isalpha() for ch in test_sta_name]
            index_char = np.where(alpha_ch_in_test_sta_name)[0]

            # Sometimes we might uncounter station name with only figures, so far we discart those stations
            if index_char.size > 0:
                i_sta_last_char = np.max(index_char)
            else:
                stations_df = stations_df.drop(index=stations_df.index.start)
                continue

            alpha_test_sta_name = test_sta_name[: i_sta_last_char + 1]
            # Find other stations that share the same root name
            slice_sta = stations_df[
                stations_df["station"].str.startswith(alpha_test_sta_name)
            ]

            if alpha_test_sta_name == "H01W":
                pass
            # Some station can share the name root name but belong to different networks e.g CY10x and CY20x we need to distinguish them$
            # Check number of digits after the root name
            n_digits_after_root_name = np.array(
                [
                    len(sta_name) - (i_sta_last_char + 1)
                    for sta_name in slice_sta["station"]
                ]
            )
            max_n_digits = np.max(n_digits_after_root_name)

            if max_n_digits > 1:
                if np.unique(n_digits_after_root_name).size > 1:
                    # More than one type of station found, we need to separate them
                    # Update test_name to root including the first digit after the root name
                    # Get first station with max number of digits
                    sta_with_max_n_digits = slice_sta[
                        n_digits_after_root_name == max_n_digits
                    ]["station"].iloc[0]
                    # Update test_name
                    alpha_test_sta_name = sta_with_max_n_digits[: i_sta_last_char + 2]

                else:
                    # All stations have the same number of digits after the root name

                    # Further split slice_sta based on the first digit after the root name (which is the highest order)
                    alpha_test_sta_name = test_sta_name[: i_sta_last_char + 2]

                # Find other stations that share the same root name
                slice_sta = stations_df[
                    stations_df["station"].str.startswith(alpha_test_sta_name)
                ]

            index_slice_sta = (
                slice_sta.index
            )  # Keep track of index to drop processed stations later
            slice_sta = slice_sta.reset_index(drop=True)

            # We need at least 2 stations to derive a distance
            if len(slice_sta) < 2:
                stations_df = stations_df.drop(index=index_slice_sta)
                continue

            # Compute distance between stations
            # print(
            #     f"Processing stations with root name {alpha_test_sta_name} ({len(slice_sta)} stations)"
            # )
            dist_slice_sta = []
            for i in range(len(slice_sta)):
                sta_i = slice_sta.iloc[i]
                lon_i = sta_i["longitude"]
                lat_i = sta_i["latitude"]
                for j in range(i + 1, len(slice_sta)):
                    sta_j = slice_sta.iloc[j]
                    lon_j = sta_j["longitude"]
                    lat_j = sta_j["latitude"]
                    _, _, dij = geod.inv(
                        lons1=lon_i,
                        lats1=lat_i,
                        lons2=lon_j,
                        lats2=lat_j,
                    )
                    dist_slice_sta.append(dij)

            # Decide whether to keep or remove stations based on distance criterion
            median_dist_slice_sta = np.median(dist_slice_sta)

            if median_dist_slice_sta < max_distance_between_stations * 1000:
                # Add median distance info to df
                slice_sta["median_rcv_dist"] = np.array(
                    [median_dist_slice_sta] * slice_sta.shape[0]
                )

                # Add unique id to identify the group of stations
                slice_sta["group_id"] = np.array([unique_id] * slice_sta.shape[0])
                unique_id += 1

                if init_pd_selected_stations:
                    selected_stations = slice_sta
                    init_pd_selected_stations = False
                else:
                    selected_stations = pd.concat(
                        [selected_stations, slice_sta], ignore_index=True
                    )
            # else:
            #     print(
            #         f"Stations with root name {alpha_test_sta_name} are too far apart ({median_dist_slice_sta/1000:.1f} km)"
            #     )

            # Drop all stations that were just processed
            stations_df = stations_df.drop(index=index_slice_sta)
            # Reset index of remaining stations
            stations_df = stations_df.reset_index(drop=True)

        return selected_stations

    @staticmethod
    def filter_number_of_receivers_by_array(
        stations_df: pd.DataFrame, min_nb_rcv_by_array: int
    ) -> pd.DataFrame:
        stations_df = stations_df.groupby("group_id").filter(
            lambda x: len(x) >= min_nb_rcv_by_array
        )
        return stations_df

    @staticmethod
    def print_filtering_stats(stations_df: pd.DataFrame):

        print("================ Selected stations stats ================")
        n_stations = stations_df.shape[0]
        print(f"Number of selected stations : {n_stations}")

        # Number of unique networks
        n_networks = stations_df["network"].nunique()
        print(f"Number of networks : {n_networks}")

        # Number of grouped stations
        if "group_id" in stations_df.columns:
            n_grouped_stations = stations_df["group_id"].nunique()
            print(f"Number of arrays (grouped stations) : {n_grouped_stations}")

        # Number of stations per network
        for network in stations_df["network"].unique():
            n_stations_in_network = stations_df[
                stations_df["network"] == network
            ].shape[0]
            print(f"Number of stations in network {network} : {n_stations_in_network}")
