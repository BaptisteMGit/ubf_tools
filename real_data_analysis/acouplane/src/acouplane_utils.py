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
from get_data.obs_bin.obs_reader_acouplane import convert_raw_data

# ======================================================================================================================
# Data preprocessing
# ======================================================================================================================


def build_obs_datetime_vector(ds_pressure):

    datetime_obs_dict = {}

    for obs_id in ds_pressure.obs_id.values:
        start_datetime_obs = pd.to_datetime(
            ds_pressure.start_datetime.sel(obs_id=obs_id).values
        ).to_pydatetime()
        signal_duration_obs_s = (
            ds_pressure[f"time_{obs_id}"].values[-1] * 1 / ds_pressure.fs
        )
        end_datetime_obs = start_datetime_obs + timedelta(seconds=signal_duration_obs_s)
        datetime_obs = pd.date_range(
            start=start_datetime_obs,
            end=end_datetime_obs,
            freq=f"{1/ds_pressure.fs}s",
            inclusive="both",
        )

        datetime_obs_dict[obs_id] = datetime_obs

    return datetime_obs_dict


def convert_raw_pressure_to_physical_units(ds_pressure):
    for obs_id in ds_pressure.obs_id.values:
        raw_pressure = ds_pressure[f"pressure_{obs_id}"]
        pressure = convert_raw_data(
            raw_data=raw_pressure, fullScale=ds_pressure.fullscale
        )
        # time_attrs = ds_pressure[f"time_{obs_id}"].attrs
        ds_pressure[f"pressure_{obs_id}"] = pressure
        # ds_pressure[f"time_{obs_id}"].attrs = time_attrs

        # Update unit
        ds_pressure[f"pressure_{obs_id}"].attrs["units"] = "V"

    return ds_pressure

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
    isel_dict = {f"time_{id}": slice(obs_slice_idx[i][0], obs_slice_idx[i][1] + 1) for i, id in enumerate(ds_pressure.obs_id.values)}
    # Slice pressure dataset 
    pressure_slice = ds_pressure.isel(isel_dict)

    # Convert time to UTC 
    for i, id in enumerate(ds_pressure.obs_id.values):
        pressure_slice[f"time_{id}"] = obs_datetimes[id][obs_slice_idx[i][0] : obs_slice_idx[i][1] + 1]
        # Update attributes accordingly 
        pressure_slice[f"time_{id}"].attrs = {"unit": "UTC", "long_name": "Time"}

    return pressure_slice


def extract_and_plot_pressure_section(section_start, section_end, ds_pressure, obs_datetimes, single_fig=False):

    # Extract section of interest 
    ds_pressure_section = extract_pressure_section(section_start, section_end, ds_pressure, obs_datetimes)
    # Convert to volts
    ds_pressure_section = convert_raw_pressure_to_physical_units(ds_pressure=ds_pressure_section)

    # Plot 
    plot_pressure_section(ds_pressure_section=ds_pressure_section, single_fig=single_fig)

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

        slon, slat = ds_pos_section.tir_src_lon.values, ds_pos_section.tir_src_lat.values
        for rcv_id in ds_pos_section.rcv_id.values:
            rlon, rlat = ds_pos_section.sel(rcv_id=rcv_id).rcv_lon.values, ds_pos_section.sel(rcv_id=rcv_id).rcv_lat.values

            # Cast to src pos shape 
            rlon_ = np.ones(slon.size) * rlon
            rlat_ = np.ones(slon.size) * rlat
            # Compute geodesic distance 
            fwd_az, back_az, horiz_d = geod.inv(lons1=slon, lats1=slat, lons2=rlon_, lats2=rlat_)
            # Store
            # NOTE : for now we only store distance 
            horizontal_distance.append(horiz_d)

        horizontal_distance = np.array(horizontal_distance)
        # print(horizontal_distance.shape)

        # Add to dataset 
        ds_pos_section["src_rcv_geodesic_dist"] = (["rcv_id", "src_datetime"], horizontal_distance)

    return ds_pos_sections

def build_and_save_portion_tree(ds_pos_sections, ds_pressure_sections, sections_orientation, sections_label, portion_id, root_fpath=ACOUPLANE_PORTION_NC_ROOT_FPATH):
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
    tree_dict_positions = {f"/position/{sections_label[i]}": ds_pos_sections[i] for i in range(len(sections_label))}
    tree_dict_pressure = {f"/pressure/{sections_label[i]}": ds_pressure_sections[i] for i in range(len(sections_label))}
    tree_dict = tree_dict_positions | tree_dict_pressure

    # Build tree
    portion_tree = xr.DataTree.from_dict(tree_dict)

    # Save as netcdf 
    fpath = f"{root_fpath}_{portion_id}.nc"
    portion_tree.to_netcdf(fpath)

    return portion_tree

# ======================================================================================================================
# Plot functions
# ======================================================================================================================


### Positions ###
def plot_obs_pos(ds_pos, ax=None, add_legend=True, add_title=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for id in ds_pos.rcv_id.values:
        rcv = ds_pos.sel(rcv_id=id)
        ax.scatter(rcv.rcv_lon, rcv.rcv_lat, marker="d", label=f"{id}", s=200)

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")

    if add_title:
        ax.set_title("OBS positions")
    if add_legend:
        ax.legend()

    return ax

def plot_pressure_section(ds_pressure_section, single_fig=False):
    if single_fig: 
        plt.figure()
        for i, id in enumerate(ds_pressure_section.obs_id.values):
            ds_pressure_section[f"pressure_{id}"].plot(label=id.upper(), color=color(i))
        plt.legend(loc="upper right")
        return plt.gcf(), plt.gca()
    else:
        fig, axs = plt.subplots(nrows=ds_pressure_section.obs_id.size, ncols=1, figsize=(16, 14), sharey=True, sharex=True)        
        axs = axs.flatten()
        for i, id in enumerate(ds_pressure_section.obs_id.values):
            ds_pressure_section[f"pressure_{id}"].plot(label=id.upper(), color=color(i), ax=axs[i])
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
    

def plot_spectrogram_section(ds_pressure_section, nperseg=2**8, noverlap=2**7, cmap="magma"):

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


    fig, axs = plt.subplots(nrows=ds_pressure_section.obs_id.size, ncols=1, figsize=(16, 14), sharey=True, sharex=True)        
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

    set_subfigures_abc_labels(axs, x_pos=0.5, y_pos=1.02, fontsize=20, ha="center", va="bottom")

    return fig, axs

def plot_portion_tir_pos(portion_tree):
    fig, ax = plt.subplots()
    # i = 0
    for name, node in portion_tree.position.children.items():
        ds_pos_section = node.to_dataset()
        sc = ax.scatter(
            ds_pos_section.tir_src_lon,
            ds_pos_section.tir_src_lat,
            color=color(ds_pos_section.section_id), 
            label=f"{ds_pos_section.section_orientation} {ds_pos_section.section_id}"
        )

    plot_obs_pos(ds_pos_section, ax=ax)

    return fig, ax 

if __name__ == "__main__":
    pass