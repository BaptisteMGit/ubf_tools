import xarray as xr

# Dataset ID
DATASET_ID = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"

# Subsetting parameters
TIME = '2022-01-01'
DEPTH = 0
LATITUDE = slice(35,60)
LONGITUDE = slice(-15,5)

# Read product via OPeNDAP
DS = xr.open_dataset(f"https://nrt.cmems-du.eu/thredds/dodsC/{DATASET_ID}")\
.sel(time=TIME, latitude=LATITUDE, longitude=LONGITUDE)\
.isel(depth=DEPTH)

# Save to netcdf
DS.to_netcdf('SST_data.nc')