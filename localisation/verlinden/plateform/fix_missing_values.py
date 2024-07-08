import os
import xarray as xr

from localisation.verlinden.plateform.build_dataset import build_dataset

import numpy as np
from signals.signals import pulse, generate_ship_signal
from localisation.verlinden.misc.AcousticComponent import AcousticSource
from localisation.verlinden.testcases.testcase_envs import TestCase3_1


import xarray as xr

from signals.signals import pulse
from localisation.verlinden.plateform.utils import *
from localisation.verlinden.plateform.init_dataset import init_grid
from localisation.verlinden.misc.AcousticComponent import AcousticSource
from localisation.verlinden.plateform.init_dataset import init_dataset
from localisation.verlinden.plateform.build_dataset import build_dataset
from localisation.verlinden.plateform.populate_dataset import (
    populate_dataset,
    grid_synthesis,
)
from localisation.verlinden.testcases.testcase_envs import TestCase3_1
from localisation.verlinden.misc.verlinden_utils import load_rhumrum_obs_pos
from localisation.verlinden.misc.params import DATA_ROOT, ROOT_DATASET

testcase = "testcase3_1"
root_dir = os.path.join(
    ROOT_DATASET,
    testcase,
)
root_propa = os.path.join(root_dir, "propa")
root_propa_grid = os.path.join(root_dir, "propa_grid")

# fname = "propa_65.5523_65.9926_-27.7023_-27.4882_backup.zarr"
# p = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\data\localisation_dataset\testcase3_1\propa\propa_65.5523_65.9926_-27.7023_-27.4882_backup.zarr"
fname = "propa_65.5523_65.9926_-27.7023_-27.4882_backup.zarr"
path = os.path.join(root_propa, fname)

ds_master = xr.open_dataset(path, engine="zarr", chunks={})

rcv_info = {
    "id": ["RR45", "RR48", "RR44"],
    # "id": ["RRpftim0", "RRpftim1", "RRpftim2"],
    # "id": ["RRdebug0", "RRdebug1"],
    "lons": [],
    "lats": [],
}
testcase = TestCase3_1()
min_dist = 5 * 1e3
dx, dy = 100, 100

# Define source signal
min_waveguide_depth = 5000
dt = 10
fs = 100  # Sampling frequency
f0_lib = 1  # Fundamental frequency of the ship signal
src_info = {
    "sig_type": "ship",
    "f0": f0_lib,
    "std_fi": f0_lib * 1 / 100,
    "tau_corr_fi": 1 / f0_lib,
    "fs": fs,
}
src_sig, t_src_sig = generate_ship_signal(
    Ttot=dt,
    f0=src_info["f0"],
    std_fi=src_info["std_fi"],
    tau_corr_fi=src_info["tau_corr_fi"],
    fs=src_info["fs"],
)

src_sig *= np.hanning(len(src_sig))
nfft = None
# nfft = 2**3
src = AcousticSource(
    signal=src_sig,
    time=t_src_sig,
    name="ship",
    waveguide_depth=min_waveguide_depth,
    nfft=nfft,
)

for obs_id in rcv_info["id"]:
    pos_obs = load_rhumrum_obs_pos(obs_id)
    rcv_info["lons"].append(pos_obs.lon)
    rcv_info["lats"].append(pos_obs.lat)

grid_info = init_grid(rcv_info, min_dist, dx, dy)
boundaries_label = build_boundaries_label(grid_info)
fullpath_dataset_propa = build_propa_path(testcase.name, boundaries_label)

root_dir = build_root_dir(testcase.name)
grid_label = build_grid_label(dx, dy)
fullpath_dataset_propa_grid = build_propa_grid_path(
    root_dir, boundaries_label, grid_label
)

ds = init_dataset(
    rcv_info=rcv_info,
    testcase=testcase,
    minimum_distance_around_rcv=min_dist,
    dx=dx,
    dy=dy,
    nfft=src.nfft,
    fs=src.fs,
)

ds["tf"] = ds_master.tf
ds["propa_done"] = True
ds["propa_grid_done"] = False
ds["propa_grid_src_done"] = False
# ds.attrs["dataset_root_dir"] = root_dir
# ds.attrs["fullpath_dataset_propa"] = fullpath_dataset_propa
# ds.attrs["fullpath_dataset_propa_grid"] = fullpath_dataset_propa_grid
ds.to_zarr(ds.fullpath_dataset_propa, compute=True, mode="w")
