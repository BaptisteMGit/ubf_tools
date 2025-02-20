#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   load_data_from_cmems.py
@Time    :   2024/06/25 18:02:52
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import xarray as xr
import copernicusmarine
import matplotlib.pyplot as plt

from localisation.verlinden.plateform.init_dataset import init_grid
from localisation.verlinden.misc.verlinden_utils import load_rhumrum_obs_pos


# Dataset ID
DATASET_ID = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"


def load_data(data_request):
    """
    Load data from the Copernicus Marine database.

    Parameters
    ----------
    data_request : dict
        Dictionary containing the request parameters.
    save : bool, optional
        Save the dataset to a netcdf file, by default False
    path : str, optional
        Path to save the dataset, by default "data.nc"

    Returns
    -------
    xr.Dataset
        Xarray dataset containing the requested data.
    """

    # Load xarray dataset
    ds = copernicusmarine.subset(
        dataset_id=data_request["dataset_id"],
        dataset_version=data_request["dataset_version"],
        minimum_longitude=data_request["minimum_longitude"],
        maximum_longitude=data_request["maximum_longitude"],
        minimum_latitude=data_request["minimum_latitude"],
        maximum_latitude=data_request["maximum_latitude"],
        start_datetime=data_request["start_datetime"],
        end_datetime=data_request["end_datetime"],
        variables=data_request["variables"],
        output_filename=data_request["output_filename"],
        output_directory=data_request["output_dir"],
        force_download=data_request["force_download"],
    )

    return ds


if __name__ == "__main__":

    rcv_id = ["RR44"]
    minimum_distance_around_rcv = 10 * 1e3
    rcv_info = {
        "id": rcv_id,
        "lons": [],
        "lats": [],
        "z": [],
    }
    for obs_id in rcv_info["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info["lons"].append(pos_obs.lon)
        rcv_info["lats"].append(pos_obs.lat)
        rcv_info["z"].append(pos_obs.elev)

    grid_info = init_grid(rcv_info, minimum_distance_around_rcv, 100, 100)
    ymin, ymax = (
        grid_info["min_lat"],
        grid_info["max_lat"],
    )
    xmin, xmax = (
        grid_info["min_lon"],
        grid_info["max_lon"],
    )

    start_datetime = "2013-01-01T00:00:00"
    end_datetime = "2013-12-30T23:00:00"

    # Root
    root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\cmems"
    fname = "data.nc"

    # Set parameters
    data_request = dict(
        dataset_id="cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m",
        dataset_version="202311",
        variables=[
            "so_mean",
            "so_std",
            "thetao_mean",
            "thetao_std",
        ],
        minimum_longitude=grid_info["min_lon"],
        maximum_longitude=grid_info["max_lon"],
        minimum_latitude=grid_info["min_lat"],
        maximum_latitude=grid_info["max_lat"],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_depth=0.5057600140571594,
        maximum_depth=5902.1,
        output_dir=root,
        output_filename=fname,
        force_download=True,
    )

    ds = load_data(data_request)

    # data_request["dataset_id"] = "cmems_mod_glo_phy-all_my_0.25deg_P1D-m"
    # data_request["dataset_version"] = "202311"
    # data_request["variables"] = [
    #     # "so_cglo",
    #     "so_glor",
    #     # "so_oras",
    #     # "thetao_cglo",
    #     "thetao_glor",
    #     # "thetao_oras",
    # ]

    # ds = load_data(data_request)

    # Plot
    f, axs = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
    for it in range(ds.sizes["time"]):
        if it % 15 == 0:
            ds.isel(time=it).so_glor.plot(
                y="depth", yincrease=False, alpha=0.25, color="b", ax=axs[0]
            )
            ds.isel(time=it).thetao_glor.plot(
                y="depth", yincrease=False, alpha=0.25, color="b", ax=axs[1]
            )
            axs[0].set_title("")
            axs[1].set_title("")
            axs[1].set_ylabel("")

    plt.tight_layout()
    plt.show()
