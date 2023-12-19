import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\test_case"
# env_fname = "verlinden_1_ssp"
env_fname = "verlinden_1_test_case"

nc_path = os.path.join(root, env_fname + ".nc")
ds = xr.open_dataset(nc_path)
