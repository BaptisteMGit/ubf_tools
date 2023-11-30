import xarray as xr

swir_salinity_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp\cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m_1700737971954.nc"
swir_temp_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp\cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m_1700737952634.nc"

# Load temperatue and salinity
ds_sal = xr.open_dataset(swir_salinity_path)
ds_temp = xr.open_dataset(swir_temp_path)
