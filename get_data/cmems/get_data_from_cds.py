#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   get_data_from_cds.py
@Time    :   2025/06/04 11:52:46
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import cdsapi

"""
Code pour télécharger les données depuis la plateforme Climate Data Store (CDS).
Prérequis (voir https://cds.climate.copernicus.eu/how-to-api) : 

1) Installer la librairie cdsapi 
   pip install cdsapi / pipenv install cdsapi

2) Se créer un compte sur la plateforme CDS 

3) Créer un fichier .cdsapirc dans le répertoire utilisateur (C:\Users\baptiste.menetrier) avec vos informations 
d'identification CDS disponibles, une fois connecté sur la plateforme CDS, à l'adresse https://cds.climate.copernicus.eu/how-to-api 

Une fois ces étapes réalisées, vous pouvez exécuter ce script pour télécharger les données souhaitées.

"""

# swir_area = [-27.9, 65.2, -27.4, 66.2]
swir_area = [-26, 64, -29, 67]  # [south, west, north, east]

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation",
    ],
    "year": ["2013"],
    # "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]",
    "month": ["01", "03", "05"],
    "day": [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
    ],
    "time": [
        "00:00",
        "01:00",
        "02:00",
        "03:00",
        "04:00",
        "05:00",
        "06:00",
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
        "19:00",
        "20:00",
        "21:00",
        "22:00",
        "23:00",
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": swir_area,
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
