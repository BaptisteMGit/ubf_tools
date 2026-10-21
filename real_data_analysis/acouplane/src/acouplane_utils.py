#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2026/09/28 21:54:27
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Useful functions for the acouplane project
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
# import os
# import sys

import re
import numpy as np
import xarray as xr
import pandas as pd

# import soundfile as sf
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta

# from scipy.ndimage import uniform_filter1d
from publication.publication_figure import set_subfigures_abc_labels, color
from real_data_analysis.acouplane.src.acouplane_cst import *
from get_data.obs_bin.obs_reader_acouplane import convert_data_raw_to_physical

# ======================================================================================================================
# Data preprocessing
# ======================================================================================================================


def build_obs_datetime_vector(ds_pressure):

    datetime_obs_dict = {}

    for obs_id in ds_pressure.obs_id.values:
        start_datetime_obs = pd.to_datetime(
            ds_pressure.start_datetime.sel(obs_id=obs_id).values
        ).to_pydatetime()

        # # OLD (before 12/10/2026)
        # signal_duration_obs_s = (
        #     ds_pressure[f"time_{obs_id}"].values[-1] * 1 / ds_pressure.fs
        # )
        # end_datetime_obs = start_datetime_obs + timedelta(seconds=signal_duration_obs_s)
        # datetime_obs = pd.date_range(
        #     start=start_datetime_obs,
        #     end=end_datetime_obs,
        #     freq=f"{1/ds_pressure.fs}s",
        #     inclusive="both",
        # )

        # NEW (after 12/10/2026)  -> exactly the same but cleaner (no need to explicitly compute end_datetime)
        datetime_obs = pd.date_range(
            start=start_datetime_obs,
            freq=f"{1/ds_pressure.fs}s",
            inclusive="left",
            periods=ds_pressure.sizes[f"time_{obs_id}"],
        )

        datetime_obs_dict[obs_id] = datetime_obs

    return datetime_obs_dict


def convert_raw_pressure_to_physical_units(ds_pressure):
    for obs_id in ds_pressure.obs_id.values:
        raw_pressure = ds_pressure[f"pressure_{obs_id}"]
        pressure = convert_data_raw_to_physical(
            raw_data=raw_pressure, fullScale=ds_pressure.fullscale
        )
        # time_attrs = ds_pressure[f"time_{obs_id}"].attrs
        ds_pressure[f"pressure_{obs_id}"] = pressure
        # ds_pressure[f"time_{obs_id}"].attrs = time_attrs

        # Update unit
        ds_pressure[f"pressure_{obs_id}"].attrs["units"] = "V"

    return ds_pressure


def build_section_orientation(sections_orientation_code):
    sections_orientation = [
        "South/North" if code == "sn" else "West/East"
        for code in sections_orientation_code
    ]
    sections_orientation_label = [
        f"{code}{i}" for i, code in enumerate(sections_orientation_code)
    ]
    return sections_orientation, sections_orientation_label


# ======================================================================================================================
# Celerity data preprocessing
# ======================================================================================================================


def edf_parser_convert_value(value):
    """Convert a string to a suitable Python type."""
    value = value.strip()

    # Empty values
    if not value:
        return None

    # Explicit missing values
    if value.lower() in {"none", "n/a", "null"}:
        return None

    # Integer or float, including scientific notation
    try:
        number = float(value)

        if number.is_integer() and not any(
            char in value.lower() for char in (".", "e")
        ):
            return int(number)

        return number

    except ValueError:
        return value


def parse_edf_header(lines):
    """Parse a SIPPICANS *.edf file header into a clean nested dictionary."""
    header = {}
    section = "general"

    header_length = None

    for line in lines:
        line_ = line  # Unstripped line
        line = line.strip()

        if not line:
            continue

        # Detect section headings
        if line.startswith("//"):
            section_name = line.lstrip("/ ").strip()

            if section_name:
                if section_name != "Data":
                    section = section_name.lower().replace(" ", "_")
                    header.setdefault(section, {})
                    # print(section_name)
                else:
                    header_length = lines.index(line_)

            continue

        # Parse key-value pairs
        match = re.match(r"^([^:]+?)\s*:\s*(.*)$", line)

        if not match:
            continue

        key, value = match.groups()
        key = key.strip()
        value = edf_parser_convert_value(value)

        header.setdefault(section, {})[key] = value

    return header, header_length


def dmm_to_decimal(coord):
    """
    Convert degrees and decimal minutes to decimal degrees.
    Example: '43 9.88329N' -> 43.1647215
    """
    match = re.fullmatch(r"\s*(\d+)\s+(\d+(?:\.\d+)?)\s*([NSEW])\s*", coord.upper())

    if not match:
        raise ValueError(f"Invalid coordinate format: {coord}")

    degrees = float(match.group(1))
    minutes = float(match.group(2))
    hemisphere = match.group(3)

    decimal = degrees + minutes / 60

    if hemisphere in ("S", "W"):
        decimal = -decimal

    return decimal


def read_sippicans(data_folder=ACOUPLANE_DATA_CELERITY_SIPPICANS_DIR, verbose=True):

    from charset_normalizer import from_bytes

    # List files in SIPPICANS folder
    sip_files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if os.path.isfile(os.path.join(data_folder, f))
    ]

    celerity_data = {}

    default_value = -99
    for fpath in sip_files:
        # Step 1: read data in binary format and guess encoding
        with open(fpath, "rb") as f:
            data = f.read()
        result = from_bytes(data).best()
        encoding = result.encoding
        # encoding = "cp1250"

        if verbose:
            print(
                f"Reading file {fpath} with encoding {encoding}"
                if result
                else "Encoding not detected"
            )

        with open(fpath, "r", encoding=result.encoding) as f:
            lines = f.readlines()

        # Read metadata
        metadata, header_length = parse_edf_header(lines)
        # metadata

        # Read data
        names = [
            metadata["data_fields"][f"Field{i}"]
            for i in range(1, metadata["data_fields"]["Num Data Fields"] + 1)
        ]
        df_celerity = pd.read_csv(
            fpath, header=header_length, delimiter="\t", encoding=encoding, names=names
        )
        # df_celerity

        # Replace invalid values by nan
        df_celerity.loc[
            df_celerity["Temperature (°C)"] == default_value, "Temperature (°C)"
        ] = np.nan
        df_celerity.loc[
            df_celerity["Sound Velocity (m/s)"] == default_value, "Sound Velocity (m/s)"
        ] = np.nan
        # print(df_celerity)

        # Store
        # print(metadata['launch_information']["Longitude"], metadata['launch_information']["Latitude"])
        f_id = metadata["file_information"]["Raw Data Filename"]
        celerity_data[f_id] = {}  # Init
        celerity_data[f_id]["longitude"] = np.round(
            dmm_to_decimal(metadata["launch_information"]["Longitude"]), 7
        )
        celerity_data[f_id]["latitude"] = np.round(
            dmm_to_decimal(metadata["launch_information"]["Latitude"]), 7
        )
        celerity_data[f_id]["surface_temperatude_degC"] = metadata[
            "launch_information"
        ]["T°surf_(dd.d)"]
        celerity_data[f_id]["pression_atmo_hPa"] = metadata["launch_information"][
            "P_atm(pppp.p)"
        ]
        celerity_data[f_id]["salinity_ppt"] = float(
            metadata["information_-_xbt"]["Salinity"].split(" ")[0]
        )
        celerity_data[f_id]["data"] = df_celerity

    return celerity_data


# ======================================================================================================================
# Data processing
# ======================================================================================================================


def extract_pressure_section(section_start, section_end, ds_pressure, obs_datetimes):

    # Build slice for each rcv
    obs_slice_idx = []
    for id in ds_pressure.obs_id.values:
        obsi_slice_idx = (
            np.where(obs_datetimes[id] >= section_start)[0][0],
            np.where(obs_datetimes[id] <= section_end)[0][-1],
        )
        obs_slice_idx.append(obsi_slice_idx)

    # Build slicing dict
    isel_dict = {
        f"time_{id}": slice(obs_slice_idx[i][0], obs_slice_idx[i][1] + 1)
        for i, id in enumerate(ds_pressure.obs_id.values)
    }
    # Slice pressure dataset
    pressure_slice = ds_pressure.isel(isel_dict)

    # Convert time to UTC
    for i, id in enumerate(ds_pressure.obs_id.values):
        pressure_slice[f"time_{id}"] = obs_datetimes[id][
            obs_slice_idx[i][0] : obs_slice_idx[i][1] + 1
        ]
        # Update attributes accordingly
        pressure_slice[f"time_{id}"].attrs = {"unit": "UTC", "long_name": "Time"}

    return pressure_slice


def extract_and_plot_pressure_section(
    section_start, section_end, ds_pressure, obs_datetimes, single_fig=False
):

    # Extract section of interest
    ds_pressure_section = extract_pressure_section(
        section_start, section_end, ds_pressure, obs_datetimes
    )
    # Convert to volts
    ds_pressure_section = convert_raw_pressure_to_physical_units(
        ds_pressure=ds_pressure_section
    )

    # Plot
    plot_pressure_section(
        ds_pressure_section=ds_pressure_section, single_fig=single_fig
    )

    return ds_pressure_section


def select_pos_section(section_start, section_end, ds_pos):
    return ds_pos.sel(
        atl_datetime=slice(section_start, section_end),
        ais_datetime=slice(section_start, section_end),
        src_datetime=slice(section_start, section_end),
    )


def select_and_plot_section(section_start, section_end, ds_pos):

    ds_pos_section = select_pos_section(section_start, section_end, ds_pos)

    fig, ax = plt.subplots()

    dates = ds_pos_section.src_datetime.values
    date_num = mdates.date2num(dates)

    sc = ax.scatter(
        ds_pos_section.tir_src_lon,
        ds_pos_section.tir_src_lat,
        c=date_num,
        cmap="jet",
    )

    plot_obs_pos(ds_pos_section, ax=ax)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Source datetime")
    cbar.ax.yaxis.set_major_locator(mdates.DayLocator())
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    return ds_pos_section


def compute_src_rcv_geodesic_dist(ds_pos_sections):
    from pyproj import Geod

    geod = Geod(ellps="WGS84")

    for ds_pos_section in ds_pos_sections:

        horizontal_distance = []

        slon, slat = (
            ds_pos_section.tir_src_lon.values,
            ds_pos_section.tir_src_lat.values,
        )
        for rcv_id in ds_pos_section.rcv_id.values:
            rlon, rlat = (
                ds_pos_section.sel(rcv_id=rcv_id).rcv_lon.values,
                ds_pos_section.sel(rcv_id=rcv_id).rcv_lat.values,
            )

            # Cast to src pos shape
            rlon_ = np.ones(slon.size) * rlon
            rlat_ = np.ones(slon.size) * rlat
            # Compute geodesic distance
            fwd_az, back_az, horiz_d = geod.inv(
                lons1=slon, lats1=slat, lons2=rlon_, lats2=rlat_
            )
            # Store
            # NOTE : for now we only store distance
            horizontal_distance.append(horiz_d)

        horizontal_distance = np.array(horizontal_distance)
        # print(horizontal_distance.shape)

        # Add to dataset
        ds_pos_section["src_rcv_geodesic_dist"] = (
            ["rcv_id", "src_datetime"],
            horizontal_distance,
        )

    return ds_pos_sections


def build_and_save_portion_tree(
    ds_pos_sections,
    ds_pressure_sections,
    sections_orientation,
    sections_label,
    portion_id,
    root_fpath=ACOUPLANE_PORTION_NC_ROOT_FPATH,
):
    # Copy to avoid muting the same object
    ds_pos_sections = [ds.copy() for ds in ds_pos_sections]
    ds_pressure_sections = [ds.copy() for ds in ds_pressure_sections]

    # Add horizontal distance info to dataset
    ds_pos_sections = compute_src_rcv_geodesic_dist(ds_pos_sections=ds_pos_sections)

    # Add metadata to improve traceability
    for i in range(len(ds_pos_sections)):
        ds_pos_sections[i].attrs["portion_id"] = portion_id
        ds_pos_sections[i].attrs["section_orientation"] = sections_orientation[i]
        ds_pos_sections[i].attrs["section_id"] = i

        ds_pressure_sections[i].attrs["portion_id"] = portion_id
        ds_pressure_sections[i].attrs["section_orientation"] = sections_orientation[i]
        ds_pressure_sections[i].attrs["section_id"] = i

        # print(
        #     sections_label[i],
        #     ds_pos_sections[i].attrs["section_id"],
        #     ds_pos_sections[i].attrs["section_orientation"],
        # )

    # Build tree dict
    tree_dict_positions = {
        f"/position/{sections_label[i]}": ds_pos_sections[i]
        for i in range(len(sections_label))
    }
    tree_dict_pressure = {
        f"/pressure/{sections_label[i]}": ds_pressure_sections[i]
        for i in range(len(sections_label))
    }
    tree_dict = tree_dict_positions | tree_dict_pressure

    # Build tree
    portion_tree = xr.DataTree.from_dict(tree_dict)

    # Save as netcdf
    fpath = f"{root_fpath}_{portion_id}.nc"
    portion_tree.to_netcdf(fpath)

    return portion_tree


def get_portion_tir_mean_pos(portion_tree):
    mean_lon = []
    mean_lat = []
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        mean_lon.append(ds_pos_section.tir_src_lon.mean().values)
        mean_lat.append(ds_pos_section.tir_src_lat.mean().values)

    return np.mean(mean_lon), np.mean(mean_lat)


def get_portion_tir_arr_dt(portion_tree, offset1=200, offset2=1500, obs_ids=None):
    # if obs_ids is None:
    #     obs_ids = ds_pressure_section.obs_id.values

    # tps_propa_measured_s_ = []
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        # Associated ds_pos
        ds_pressure_section = portion_tree.pressure[name].to_dataset()

        tps_propa_measured_s = []
        # iterate over tirs
        for i_tir in range(ds_pos_section.sizes["src_datetime"]):
            # Theoretical arrival datetime
            arr_dt_th = ds_pos_section["tir_reception_datetime"].isel(
                src_datetime=slice(i_tir, i_tir + 1)
            )
            # Emission datetime
            emission_dt = ds_pos_section["src_datetime"].isel(
                src_datetime=slice(i_tir, i_tir + 1)
            )

            tps_propa_measured_s_rcv = []
            # fig, axs = plt.subplots(nrows=obs_ids.size, ncols=1, figsize=(16, 14), sharex=False)
            # axs = axs.flatten()
            for i, rcv_id in enumerate(ds_pos_section.rcv_id.values):

                if not rcv_id in obs_ids:
                    propagation_time_measured_s = np.nan

                else:

                    arr_dt_th_rcv = arr_dt_th.sel(rcv_id=rcv_id)

                    # Select associated data window
                    press_rcv = ds_pressure_section[f"pressure_{rcv_id}"].sel(
                        {
                            f"time_{rcv_id}": slice(
                                arr_dt_th_rcv.values[0] - np.timedelta64(offset1, "ms"),
                                arr_dt_th_rcv.values[0] + np.timedelta64(offset2, "ms"),
                            )
                        }
                    )

                    # We might fall at the border -> skip to avoid issues
                    if press_rcv.size == 0:
                        propagation_time_measured_s = np.nan

                    else:

                        # Normalize and center
                        press_rcv = press_rcv / press_rcv.max()  # Useless here
                        press_rcv = (
                            press_rcv - press_rcv.mean()
                        )  # Should be done globally

                        # press_rcv.plot(label=rcv_id.upper(), color=color(i), ax=axs[i])
                        # Plot arrivals time
                        # axs[i].vlines(x=tir_dt_rcv, ymin=-1, ymax=1, color="k", linestyle="--")

                        # Find peak
                        peak_height = 5 * np.std(
                            press_rcv.values[-100:]
                        )  # 5 times noise std seems like a reasonable choice
                        # axs[i].axhline(peak_height, color="k", linestyle="--")
                        peaks_idx, peaks_prop = sp.find_peaks(
                            press_rcv, height=peak_height
                        )
                        if peaks_idx.size > 0:
                            # Get arrival datetime
                            arr_dt_measured = press_rcv[f"time_{rcv_id}"].values[
                                peaks_idx[0]
                            ]
                            # Associated propagation time
                            propagation_time_measured_ns = arr_dt_measured - emission_dt
                            propagation_time_measured_s = (
                                propagation_time_measured_ns.values[0]
                                / np.timedelta64(1, "s")
                            )
                        else:
                            propagation_time_measured_s = np.nan

                tps_propa_measured_s_rcv.append(propagation_time_measured_s)

            tps_propa_measured_s.append(np.array(tps_propa_measured_s_rcv))

        # Add to portion
        ds_pos_section["tps_propagation_measured_s"] = (
            ["rcv_id", "src_datetime"],
            np.vstack(tps_propa_measured_s).T,
        )

        # Update attribute
        ds_pos_section.tps_propagation_measured_s.attrs = {
            "units": "s",
            "long_name": "Estimated propagation time",
        }

        # Update tree
        node.ds = ds_pos_section

        # tps_propa_measured_s_.append(tps_propa_measured_s)

    # return np.vstack(tps_propa_measured_s_)


def apply_rcv_position_correction_to_portion(
    portion_tree, rcv_pos_corr_fname="rcv_position_corrected.csv"
):
    rcv_pos_fpath = os.path.join(ROOT_ACOUPLANE_PROCESSED_DATA_DIR, rcv_pos_corr_fname)
    df_rcv_pos_corr = pd.read_csv(rcv_pos_fpath)

    # Update tree
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()

        # Update ds_pos_section
        ds_pos_section.update(
            {
                "rcv_lon": ("rcv_id", df_rcv_pos_corr["lon"]),
                "rcv_lat": ("rcv_id", df_rcv_pos_corr["lat"]),
            }
        )
        # No need to update immersion as it is not changed (2D LMS estimation)

        # Update geodetic distance between source and receiver
        ds_pos_section = compute_src_rcv_geodesic_dist(
            ds_pos_sections=[ds_pos_section]
        )[0]

        # Update tree
        node.ds = ds_pos_section


# ======================================================================================================================
# ENU <-> WGS84 conversion
# ======================================================================================================================
from source.utils.utils_geo import geodetic_to_ecef, ecef_to_enu


def transform_pos_wgs84_enu(
    lon, lat, local_frame_origin, N_geoid_undulation, zeta=0, sensor_h_above_sea_level=5
):
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    # GPS h above WGS84 ellipsoid
    h = N_geoid_undulation + zeta + sensor_h_above_sea_level
    # WGS84 -> ECEF
    x, y, z = geodetic_to_ecef(lat, lon, h)
    # ECEF -> ENU
    lat0, lon0, h0 = (
        np.deg2rad(local_frame_origin["lat"]),
        np.deg2rad(local_frame_origin["lon"]),
        local_frame_origin["h"],
    )
    e, n, u = ecef_to_enu(x, y, z, lat0, lon0, h0)

    return e, n, u


def _enu_tir_pos(ds, local_frame_origin, N_geoid_undulation=48.5):
    # Convert source position from WGS84 to ecef
    tir_src_e, tir_src_n, tir_src_u = transform_pos_wgs84_enu(
        lon=ds.tir_src_lon,
        lat=ds.tir_src_lat,
        local_frame_origin=local_frame_origin,
        N_geoid_undulation=N_geoid_undulation,
        zeta=0,
        sensor_h_above_sea_level=-5,  # Source is approximately 5 meter below sea surface = MSL
    )
    # Add to portion
    ds["tir_src_e"] = (["src_datetime"], tir_src_e)
    ds["tir_src_n"] = (["src_datetime"], tir_src_n)
    ds["tir_src_u"] = (["src_datetime"], tir_src_u)
    # Update attribute
    ds.tir_src_e.attrs = {
        "units": "m",
        "long_name": "E",
    }
    ds.tir_src_n.attrs = {
        "units": "m",
        "long_name": "N",
    }
    ds.tir_src_u.attrs = {
        "units": "m",
        "long_name": "U",
    }


def get_enu_tir_pos(portion_tree, local_frame_origin, N_geoid_undulation=48.5):

    for group in list(portion_tree.children):
        if group == "position":
            for name, node in portion_tree.position.children.items():
                ds_pos_section = node.to_dataset()
                _enu_tir_pos(
                    ds_pos_section, local_frame_origin, N_geoid_undulation=48.5
                )

                # Update tree
                node.ds = ds_pos_section

        elif group == "feature":
            for name, node in portion_tree.feature.children.items():
                ds_feature_section = node.to_dataset()
                _enu_tir_pos(
                    ds_feature_section, local_frame_origin, N_geoid_undulation=48.5
                )

                # Update tree
                node.ds = ds_feature_section

        else:
            pass


def _enu_rcv_pos(ds, local_frame_origin, N_geoid_undulation=48.5):
    # Convert OBS position from WGS84 to ecef
    rcv_e, rcv_n, rcv_u = transform_pos_wgs84_enu(
        lon=ds.rcv_lon,
        lat=ds.rcv_lat,
        local_frame_origin=local_frame_origin,
        N_geoid_undulation=N_geoid_undulation,
        zeta=0,
        sensor_h_above_sea_level=-ds.rcv_immersion,
    )
    # Add to portion
    ds["rcv_e"] = (["rcv_id"], rcv_e)
    ds["rcv_n"] = (["rcv_id"], rcv_n)
    ds["rcv_u"] = (["rcv_id"], rcv_u)
    # Update attribute
    ds.rcv_e.attrs = {
        "units": "m",
        "long_name": "E",
    }
    ds.rcv_n.attrs = {
        "units": "m",
        "long_name": "N",
    }
    ds.rcv_u.attrs = {
        "units": "m",
        "long_name": "U",
    }


def get_enu_rcv_pos(portion_tree, local_frame_origin, N_geoid_undulation=48.5):

    for group in list(portion_tree.children):
        if group == "position":
            for name, node in portion_tree.position.children.items():
                ds_pos_section = node.to_dataset()
                _enu_rcv_pos(
                    ds_pos_section, local_frame_origin, N_geoid_undulation=48.5
                )

                # Update tree
                node.ds = ds_pos_section

        elif group == "feature":
            for name, node in portion_tree.feature.children.items():
                ds_feature_section = node.to_dataset()
                _enu_rcv_pos(
                    ds_feature_section, local_frame_origin, N_geoid_undulation=48.5
                )

                # Update tree
                node.ds = ds_feature_section

        else:
            pass

    # for name, node in portion_tree.position.children.items():
    #     ds_pos_section = node.to_dataset()
    #     # Convert OBS position from WGS84 to ecef
    #     rcv_e, rcv_n, rcv_u = transform_pos_wgs84_enu(
    #         lon=ds_pos_section.rcv_lon,
    #         lat=ds_pos_section.rcv_lat,
    #         local_frame_origin=local_frame_origin,
    #         N_geoid_undulation=N_geoid_undulation,
    #         zeta=0,
    #         sensor_h_above_sea_level=-ds_pos_section.rcv_immersion,
    #     )
    #     # Add to portion
    #     ds_pos_section["rcv_e"] = (["rcv_id"], rcv_e)
    #     ds_pos_section["rcv_n"] = (["rcv_id"], rcv_n)
    #     ds_pos_section["rcv_u"] = (["rcv_id"], rcv_u)
    #     # Update attribute
    #     ds_pos_section.rcv_e.attrs = {
    #         "units": "m",
    #         "long_name": "E",
    #     }
    #     ds_pos_section.rcv_n.attrs = {
    #         "units": "m",
    #         "long_name": "N",
    #     }
    #     ds_pos_section.rcv_u.attrs = {
    #         "units": "m",
    #         "long_name": "U",
    #     }

    #     # Update tree
    #     node.ds = ds_pos_section


def get_vertical_dist(portion_tree, z_src=5):
    # Derive vertical distance between source and receiver
    v_dist_wgs84 = []
    v_dist_enu = []
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        source_z_below_surface = np.ones_like(ds_pos_section.tir_src_u) * z_src
        vd_wgs84 = (
            ds_pos_section.rcv_immersion.values[:, np.newaxis]
            - source_z_below_surface[np.newaxis, :]
        )  # (nrcv, npos)
        v_dist_wgs84.append(vd_wgs84)
        vd_enu = (
            ds_pos_section.rcv_u.values[:, np.newaxis]
            - ds_pos_section.tir_src_u.values[np.newaxis, :]
        )  # (nrcv, npos)
        v_dist_enu.append(vd_enu)

    v_dist_wgs84 = np.hstack(v_dist_wgs84)
    v_dist_enu = -np.hstack(v_dist_enu)

    return v_dist_wgs84, v_dist_enu


# ======================================================================================================================
# Plot functions
# ======================================================================================================================


### Positions ###
def plot_obs_pos(ds_pos, ax=None, add_legend=True, add_title=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for id in ds_pos.rcv_id.values:
        rcv = ds_pos.sel(rcv_id=id)
        ax.scatter(rcv.rcv_lon, rcv.rcv_lat, marker="d", label=f"{id.upper()}", s=200)

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")

    if add_title:
        ax.set_title("OBS positions")
    if add_legend:
        _, labels = ax.get_legend_handles_labels()
        ncols = min(len(labels), 4)
        ax.legend(ncols=ncols)

    return ax


def plot_obs_pos_enu(ds_pos, ax=None, add_legend=True, add_title=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for id in ds_pos.rcv_id.values:
        rcv = ds_pos.sel(rcv_id=id)
        ax.scatter(rcv.rcv_e, rcv.rcv_n, marker="d", label=f"{id.upper()}", s=200)

    ax.set_xlabel("E [m]")
    ax.set_ylabel("N [m]")

    if add_title:
        ax.set_title("OBS positions")
    if add_legend:
        _, labels = ax.get_legend_handles_labels()
        ncols = min(len(labels), 4)
        ax.legend(ncols=ncols)

    return ax


def plot_portion_tir_pos_enu(portion_tree, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    # i = 0
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        sc = ax.scatter(
            ds_pos_section.tir_src_e,
            ds_pos_section.tir_src_n,
            color=color(ds_pos_section.section_id),
            label=f"{ds_pos_section.section_orientation} {ds_pos_section.section_id}",
        )

    plot_obs_pos_enu(ds_pos_section, ax=ax)

    return fig, ax


def plot_pressure_section(ds_pressure_section, single_fig=False):
    if single_fig:
        plt.figure()
        for i, id in enumerate(ds_pressure_section.obs_id.values):
            ds_pressure_section[f"pressure_{id}"].plot(label=id.upper(), color=color(i))
        plt.legend(loc="upper right")
        return plt.gcf(), plt.gca()
    else:
        fig, axs = plt.subplots(
            nrows=ds_pressure_section.obs_id.size,
            ncols=1,
            figsize=(16, 14),
            sharey=True,
            sharex=True,
        )
        axs = axs.flatten()
        for i, id in enumerate(ds_pressure_section.obs_id.values):
            ds_pressure_section[f"pressure_{id}"].plot(
                label=id.upper(), color=color(i), ax=axs[i]
            )
            axs[i].legend(loc="upper right")

        xlab = axs[0].get_xlabel()
        ylab = axs[0].get_ylabel()

        for ax in axs:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        fig.supxlabel(xlab)
        fig.supylabel(ylab)

        return fig, axs


def plot_spectrogram_section(
    ds_pressure_section, nperseg=2**8, noverlap=2**7, cmap="magma"
):

    # Compute sxx
    sxx_plot = []
    # tt_dt = []

    for i, id in enumerate(ds_pressure_section.obs_id.values):
        pressure = ds_pressure_section[f"pressure_{id}"]
        # Derive stft
        ff, tt, stft = sp.stft(
            pressure.values,
            fs=ds_pressure_section.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",  # V^2 / Hz
        )
        sxx = 10 * np.log10(np.abs(stft) + 1e-6)  # dB re V**2 / Hz
        sxx_plot.append(sxx)

        # Associated datetime vector
        t0 = pd.to_datetime(ds_pressure_section[f"time_{id}"].values[0]).to_pydatetime()
        tt_datetime = pd.date_range(
            t0,
            t0 + timedelta(seconds=tt[-1]),
            freq=f"{tt[1]-tt[0]}s",
            inclusive="both",
        )
        # tt_dt.append(tt_datetime)

    # Plot
    sxx_plot = np.array(sxx_plot)
    vmin = np.percentile(sxx_plot, 10)
    vmax = np.percentile(sxx_plot, 99)

    fig, axs = plt.subplots(
        nrows=ds_pressure_section.obs_id.size,
        ncols=1,
        figsize=(16, 14),
        sharey=True,
        sharex=True,
    )
    axs = axs.flatten()
    for i, id in enumerate(ds_pressure_section.obs_id.values):
        im = axs[i].pcolormesh(
            tt_datetime, ff, sxx_plot[i, ...], cmap=cmap, vmin=vmin, vmax=vmax
        )
    clabel = r"dB re 1V$^2$ / Hz"
    fig.colorbar(
        im,
        ax=axs.ravel().tolist(),
        label=clabel,
        orientation="vertical",
        fraction=1.0,
        pad=0.03,
    )
    # xlab = axs[0].get_xlabel()
    # ylab = axs[0].get_ylabel()

    for ax in axs:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
    fig.supxlabel("Time")
    fig.supylabel("Frequency [Hz]")

    formatter = mdates.DateFormatter("%H:%M:%S")
    axs[-1].xaxis.set_major_formatter(formatter)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axs[-1].xaxis.set_major_locator(locator)

    set_subfigures_abc_labels(
        axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom"
    )

    return fig, axs


def plot_portion_tir_pos(portion_tree, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    # i = 0
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        sc = ax.scatter(
            ds_pos_section.tir_src_lon,
            ds_pos_section.tir_src_lat,
            color=color(ds_pos_section.section_id),
            label=f"{ds_pos_section.section_orientation} {ds_pos_section.section_id}",
        )

    plot_obs_pos(ds_pos_section, ax=ax)

    return fig, ax


# ======================================================================================================================
# Feature estimation (for RTF / DCF estimation)
# ======================================================================================================================


def derive_portion_cpsd(
    portion_tree, offset_ms=100, signal_duration_s=3, nfft=None, obs_ids=None
):

    signal_duration_ms = int(signal_duration_s * 1000)

    if obs_ids is None:
        obs_ids = (
            list(portion_tree.pressure.children.items())[0][1]
            .to_dataset()
            .obs_id.values
        )

    # 1) Iterate over portion's sections
    for name, node in portion_tree.pressure.children.items():
        ds_pressure_section = node.to_dataset()
        # Associated ds_pos
        ds_pos_section = portion_tree.position[name].to_dataset()

        Sxy_data = []
        valid_emission_dt = []
        # 2) Iterate over emissions
        for i_tir, tir_emission_dt in enumerate(ds_pos_section.src_datetime.values):
            ds_pos_section_current_tir = ds_pos_section.sel(
                src_datetime=tir_emission_dt
            )

            # Reception datetime
            tir_reception_dt = ds_pos_section_current_tir["tir_reception_datetime"]

            # 3) Build list of signal_i
            signal_arr = []
            for i in range(obs_ids.size):
                rcv_id_i = obs_ids[i]
                # Reception datetime for current rcv i
                tir_reception_dt_rcvi = tir_reception_dt.sel(rcv_id=rcv_id_i)

                # Select associated data window
                signal_i = ds_pressure_section[f"pressure_{rcv_id_i}"].sel(
                    {
                        f"time_{rcv_id_i}": slice(
                            tir_reception_dt_rcvi.data
                            - np.timedelta64(offset_ms, "ms"),
                            tir_reception_dt_rcvi.data
                            + np.timedelta64(signal_duration_ms - offset_ms, "ms"),
                        )
                    }
                )

                # Ensure signal is centered
                signal_i = signal_i - np.nanmean(signal_i)

                # Store in array
                signal_arr.append(signal_i.values)

            # Check all rcv have complete signal
            signal_is_complete = np.all(
                [
                    signal_arr[k].size
                    == int(signal_duration_s * ds_pressure_section.fs)
                    for k in range(len(signal_arr))
                ]
            )
            # print(signal_is_complete)
            if signal_is_complete:
                valid_emission_dt.append(tir_emission_dt)
                signal_arr = np.asarray(signal_arr)  # (nrcv, nt)
            else:
                continue

            # a) Compute PSD Sxx
            rfft = np.fft.rfft(signal_arr, n=nfft, axis=1)  # (nrcv, nfft)
            Sxx_arr = rfft * np.conj(rfft)  # (nrcv, nfft)

            # a) Compute CPSD Sxy
            # Ugly double loop to vectorized
            nr, nf = rfft.shape
            Sxy_arr = np.empty(shape=(nr, nr, nf), dtype=np.complex64) * np.nan
            for i in range(obs_ids.size):
                Sxy_arr[i, i, :] = Sxx_arr[i, :]  # Diagonal = PSD
                x_fft_i = rfft[i, :]
                for j in range(i, obs_ids.size):
                    rcv_id_j = obs_ids[j]
                    y_fft_j = rfft[j, :]
                    Sxy = x_fft_i * np.conj(y_fft_j)
                    # Upper triangle Sij
                    Sxy_arr[i, j] = Sxy
                    # Lower triangle Sji = conj(Sij)
                    Sxy_arr[j, i] = np.conj(Sxy)

            # Store for current tir
            Sxy_data.append(Sxy_arr)
        Sxy_data = np.array(Sxy_data)  # (ntir, nr, nr, nf)

        # Store in ds_pressure_section
        ds_feature = ds_pos_section.copy()
        ds_feature = ds_feature.drop_vars(
            [
                "atl_lon",
                "atl_lat",
                "atl_datetime",
                "ais_lon",
                "ais_lat",
                "ais_datetime",
                "tir_SEGD",
                "tir_ECOS",
                "tir_vessel_lon",
                "tir_vessel_lat",
            ]
        )
        attrs = ds_feature.attrs
        attrs.pop("interpolation_time_step")
        attrs["description"] = "Localisation features"

        # Sel only valid tir
        ds_feature = ds_feature.sel(src_datetime=valid_emission_dt)

        # Add coords rcv_i, rcv_j and freq
        freq = np.fft.rfftfreq(nfft, d=1 / ds_pressure_section.fs)

        ds_feature = ds_feature.assign_coords(
            dict(rcv_i=obs_ids, rcv_j=obs_ids, freq=freq)
        )  # NOTE we need to declare new coords because we can use duplicated coords

        ds_feature["Sxy"] = (["src_datetime", "rcv_i", "rcv_j", "freq"], Sxy_data)

        # Update tree
        portion_tree[f"/feature/{name}"] = ds_feature


def get_distance_between_sections(ds_feature_section1, ds_feature_section2):
    from pyproj import Geod

    geod = Geod(ellps="WGS84")

    src_lon1, src_lat1 = (
        ds_feature_section1.tir_src_lon,
        ds_feature_section1.tir_src_lat,
    )
    src_lon2, srcv_lat2 = (
        ds_feature_section2.tir_src_lon,
        ds_feature_section2.tir_src_lat,
    )

    dist_12 = []
    for i in range(src_lon2.sizes["src_datetime"]):

        # Cast to src pos shape
        lon2 = np.ones(src_lon1.size) * src_lon2.isel(src_datetime=i).values
        lat2 = np.ones(src_lon1.size) * srcv_lat2.isel(src_datetime=i).values

        # Compute geodesic distance
        fwd_az, back_az, horiz_d = geod.inv(
            lons1=src_lon1.values, lats1=src_lat1.values, lons2=lon2, lats2=lat2
        )

        dist_12.append(horiz_d)

    dist_12 = np.asarray(dist_12)

    return dist_12


def derive_feature_distance_between_sections(features_section1, features_section2):
    # Extract features
    pi_ij_a, gamma_a, sij_scot_a = (
        features_section1["pi_ij"],
        features_section1["gamma_ij"],
        features_section1["sij_scot"],
    )

    pi_ij_b, gamma_b, sij_scot_b = (
        features_section2["pi_ij"],
        features_section2["gamma_ij"],
        features_section2["sij_scot"],
    )

    # Reshape for broadcasting
    pi_a = pi_ij_a[:, np.newaxis, :]  # (nt1, 1, nf)
    gamma_a = gamma_a[:, np.newaxis, :]  # (nt1, 1, nf)
    sij_scot_a = sij_scot_a[:, np.newaxis, :]  # (nt1, 1, nf)

    pi_b = pi_ij_b[np.newaxis, :, :]  # (1, nt2, nf)
    gamma_b = gamma_b[np.newaxis, :, :]  # (1, nt2, nf)
    sij_scot_b = sij_scot_b[np.newaxis, :, :]  # (1, nt2, nf)

    # Compute theta with pi
    inner_prod = np.nansum(pi_a * np.conj(pi_b), axis=-1)
    norm_a = np.sqrt(np.nansum(pi_a * np.conj(pi_a), axis=-1))
    norm_b = np.sqrt(np.nansum(pi_b * np.conj(pi_b), axis=-1))
    # Clip to [0, 1] for stability
    cos_angle = np.clip(np.abs(inner_prod) / np.abs(norm_a * norm_b), 0, 1.0)
    dist_theta = 1 - cos_angle  #  to [0, 1]

    # Compute L2 with gamma
    dist_L2_gamma = np.sqrt(np.nansum(np.abs(gamma_a - gamma_b) ** 2, axis=-1))
    dist_L2_gamma = dist_L2_gamma / np.nanmax(dist_L2_gamma)  # Normalize to [0, 1]

    # Compute theta with gamma
    inner_prod = np.nansum(gamma_a * gamma_b, axis=-1)
    norm_a = np.sqrt(np.nansum(gamma_a**2, axis=-1))
    norm_b = np.sqrt(np.nansum(gamma_b**2, axis=-1))
    # Clip to [-1, 1] for stability
    cos_angle = np.clip(inner_prod / (norm_a * norm_b), -1.0, 1.0)
    dist_theta_gamma = (1 - cos_angle) / 2  #  [-1, 1] to [0, 1]

    # DCF
    dist_dcf = np.abs(
        np.nansum(
            sij_scot_a * np.conj(sij_scot_b),
            axis=-1,
        )
    )
    dist_dcf = 1 - dist_dcf / (
        np.sqrt(np.nansum(np.abs(sij_scot_a) ** 2, axis=-1))
        * np.sqrt(np.nansum(np.abs(sij_scot_b) ** 2, axis=-1))
    )

    return dist_theta, dist_theta_gamma, dist_L2_gamma, dist_dcf


def derive_section_features(
    ds_feature_section, fmin=5, fmax=50, rcv_ref="obs1", rcv_i="obs2"
):

    ref_rcv_id = rcv_ref
    rcv_id_i = rcv_i
    ########################################
    # RTF
    ########################################

    sjj = ds_feature_section.sel(
        rcv_i=ref_rcv_id, rcv_j=ref_rcv_id, freq=slice(fmin, fmax)
    ).Sxy
    # Sij
    sij = ds_feature_section.sel(
        rcv_i=rcv_id_i, rcv_j=ref_rcv_id, freq=slice(fmin, fmax)
    ).Sxy
    pi_ij_a = sij.values / sjj.values
    gamma_ij_a = 20 * np.log10(np.abs(pi_ij_a))  # (nt1, nf)

    ########################################
    # DCF
    ########################################

    ### Section 1 ###
    # Sii
    sii = ds_feature_section.sel(
        rcv_i=rcv_id_i, rcv_j=rcv_id_i, freq=slice(fmin, fmax)
    ).Sxy
    # Sjj
    sjj = ds_feature_section.sel(
        rcv_i=ref_rcv_id, rcv_j=ref_rcv_id, freq=slice(fmin, fmax)
    ).Sxy
    # Sij
    sij = ds_feature_section.sel(
        rcv_i=rcv_id_i, rcv_j=ref_rcv_id, freq=slice(fmin, fmax)
    ).Sxy
    sij_scot_a = sij.values / np.sqrt(sii.values * sjj.values)

    return pi_ij_a, gamma_ij_a, sij_scot_a


def find_min_distance_position(dt1, dt2, dist):
    ind_min_dist = np.argmin(dist)
    xx, yy = np.meshgrid(dt1, dt2)
    xmin, ymin = xx.flatten()[ind_min_dist], yy.flatten()[ind_min_dist]
    return xmin, ymin


def derive_features_array(portion, fmin=5, fmax=50, rcv_ref="obs1", rcv_i="obs2"):
    pi_ij_arr = []
    gamma_arr = []
    sij_scot_arr = []
    for _, feature_node in portion.feature.children.items():
        ds_feature_section = feature_node.to_dataset()

        # Derive feature for current section
        pi_ij_a, gamma_a, sij_scot_a = derive_section_features(
            ds_feature_section=ds_feature_section,
            fmin=fmin,
            fmax=fmax,
            rcv_ref=ref_rcv_id,
            rcv_i=rcv_id_i,
        )
        pi_ij_arr.append(pi_ij_a)
        gamma_arr.append(gamma_a)
        sij_scot_arr.append(sij_scot_a)

    # Build data
    data_pi = np.concatenate(
        pi_ij_arr, axis=0
    )  # (n_sample, n_features) = (sum(nti), nfreq)
    data_gamma = np.concatenate(
        gamma_arr, axis=0
    )  # (n_sample, n_features) = (sum(nti), nfreq)
    data_sij_scot = np.concatenate(
        sij_scot_arr, axis=0
    )  # (n_sample, n_features) = (sum(nti), nfreq)

    return data_pi, data_gamma, data_sij_scot


if __name__ == "__main__":
    pass
