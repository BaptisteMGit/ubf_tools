#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   build_dataset.py
@Time    :   2024/04/23 10:35:21
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
from localisation.verlinden.plateform.init_dataset import init_dataset


# ======================================================================================================================
# Functions
# ======================================================================================================================
def build_dataset(
    rcv_info,
    testcase,
):
    """
    Build the dataset to be used by the localisation algorithm.

    Returns
    -------
    xr.Dataset
    """
    xr_dataset = init_dataset(
        rcv_info,
        testcase,
    )

    # Init arrays to store transfert functions

    print()

    # Init array to store field grid
    return xr_dataset


if __name__ == "__main__":
    from localisation.verlinden.testcases.testcase_envs import TestCase3_1
    from localisation.verlinden.verlinden_utils import load_rhumrum_obs_pos

    rcv_info_dw = {
        "id": ["RR45", "RR48", "RR44"],
        "lons": [],
        "lats": [],
    }

    for obs_id in rcv_info_dw["id"]:
        pos_obs = load_rhumrum_obs_pos(obs_id)
        rcv_info_dw["lons"].append(pos_obs.lon)
        rcv_info_dw["lats"].append(pos_obs.lat)

    initial_ship_pos_dw = {
        "lon": rcv_info_dw["lons"][0],
        "lat": rcv_info_dw["lats"][0] + 0.07,
        "crs": "WGS84",
    }

    tc = TestCase3_1()

    xr_dataset = build_dataset(rcv_info=rcv_info_dw, testcase=tc)
    print(xr_dataset)
