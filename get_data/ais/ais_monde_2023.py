#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_ais_monde_2023.py
@Time    :   2026/05/07 08:44:07
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
import matplotlib.pyplot as plt


def read_ais_123(fpath):
    # Read
    df_ais_123 = pd.read_parquet(fpath)

    # Selection columns of interest
    sel_col = [
        "Date",
        "Mmsi",
        "Latitude",
        "Longitude",
        "SpeedOverGround",
        "TrueHeadingDegrees",
    ]
    df_ais_123 = df_ais_123[sel_col]

    return df_ais_123


def aggregate_ais(root_data, type="123"):

    # List all files
    ais_123_fpath = []
    ais_5_fpath = []
    for root, dirs, file in os.walk(root_data):
        for f in file:
            if ".parquet" in f:
                fullpath = os.path.join(root, f)
                # print(fullpath)
                if "123" in f:
                    ais_123_fpath.append(fullpath)
                elif "5" in f:
                    ais_5_fpath.append(fullpath)

    # Aggregate
    if type == "123":
        df_ais_123 = pd.DataFrame()
        for fpath in ais_123_fpath:
            df_i = read_ais_123(fpath=fpath)
            df_ais_123 = pd.concat([df_ais_123, df_i], ignore_index=True)

        # Rename columns
        colname_map = {
            "Date": "datetime",
            "Mmsi": "mmsi",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "SpeedOverGround": "speed_over_ground",
            "TrueHeadingDegrees": "true_heading_degrees",
        }
        df_ais_123 = df_ais_123.rename(columns=colname_map)

        return df_ais_123

    elif type == "5":
        pass  # Not needed yet


if __name__ == "__main__":
    root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ais\ais_monde_2023"
    # Warning : it could require a lot of RAM
    ais_123 = aggregate_ais(root_data=root_data)

    # Save ais dataset
    fname = "ais_aggregated_pos.parquet"
    fpath = os.path.join(root_data, fname)
    ais_123.to_parquet(fpath)

    print(ais_123)
