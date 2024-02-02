import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from arlpy import uwa
from cst import SAND_PROPERTIES, RHO_W
from cst import TICKS_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE
from propa.kraken_toolbox.kraken_env import (
    KrakenEnv,
    KrakenTopHalfspace,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
    KrakenFlp,
    Bathymetry,
)
from propa.kraken_toolbox.utils import default_nb_rcv_z

ENV_ROOT = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\testcase_working_directory"


def isotropic_ideal_env(
    env_root,
    env_filename,
    title="Isotropic ideal environment",
    max_depth=3000,
    src_depth=None,
    freq=[20],
):
    if src_depth is None:
        src_depth = max_depth - 2  # 2m above the bottom

    top_hs = KrakenTopHalfspace()
    z_ssp = np.array([0.0, max_depth])
    cp_ssp = np.array([1500.0, 1500.0])

    medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

    bott_hs_properties = SAND_PROPERTIES
    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(halfspace_properties=bott_hs_properties)

    att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)

    n_rcv_z = default_nb_rcv_z(max(freq), max_depth)
    field = KrakenField(
        src_depth=src_depth,
        n_rcv_z=n_rcv_z,
        rcv_z_max=max_depth,
    )

    env = KrakenEnv(
        title=title,
        env_root=env_root,
        env_filename=env_filename,
        freq=freq,
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
    )

    flp = KrakenFlp(
        env=env,
        src_depth=src_depth,
        n_rcv_z=n_rcv_z,
        rcv_z_max=max_depth,
        rcv_r_max=30,
        mode_addition="coherent",
    )

    return env, flp


def rhum_rum_isotropic_env(
    env_root,
    env_filename,
    title="Isotropic ideal environment with real Sound Speed Profile",
    src_depth=None,
    freq=[20],
):
    swir_salinity_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp\cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m_1700737971954.nc"
    swir_temp_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp\cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m_1700737952634.nc"

    # Load temperatue and salinity
    ds_sal = xr.open_dataset(swir_salinity_path)
    ds_temp = xr.open_dataset(swir_temp_path)

    ssp = uwa.soundspeed(
        ds_sal.so.isel(latitude=0, longitude=0, time=0),
        ds_temp.thetao.isel(latitude=0, longitude=0, time=0),
        ds_sal.depth,
    )
    ssp = ssp.dropna(dim="depth")
    # ssp.<
    # ssp.plot(y="depth", yincrease=False)

    max_depth = ssp.depth.max().round(2).values
    if src_depth is None:
        src_depth = max_depth - 2  # 2m above the bottom

    top_hs = KrakenTopHalfspace()
    z_ssp = ssp.depth.round(2).values
    cp_ssp = ssp.round(2).values

    medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

    bott_hs_properties = SAND_PROPERTIES
    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(halfspace_properties=bott_hs_properties)

    att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)

    n_rcv_z = default_nb_rcv_z(max(freq), max_depth)
    field = KrakenField(
        src_depth=src_depth,
        n_rcv_z=n_rcv_z,
        rcv_z_max=max_depth,
    )

    env = KrakenEnv(
        title=title,
        env_root=env_root,
        env_filename=env_filename,
        freq=freq,
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
    )

    flp = KrakenFlp(
        env=env,
        src_depth=src_depth,
        n_rcv_z=n_rcv_z,
        rcv_z_max=max_depth,
        rcv_r_max=50,
        mode_addition="coherent",
    )

    return env, flp


##########################################################################################
# Test case 1 : Shallow water environment (close from Verlinden environment)
##########################################################################################


def testcase1_common(freq, z_ssp, cp_ssp, bathy, title, testcase_name="testcase1"):
    # Create environment directory
    env_dir = os.path.join(ENV_ROOT, testcase_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)

    max_depth = max(z_ssp)  # Depth at src position (m)
    src_depth = (
        max_depth - 7
    )  # 7m above the bottom (Assume Verlinden used Hydrophone #1 presentend in fig 3.8)

    top_hs = KrakenTopHalfspace()

    medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

    # First simple model : 1 layer bottom
    bott_hs_properties = {
        "rho": 1.9 * RHO_W,
        "c_p": 1650,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
        "a_p": 0.8,  # Compression wave attenuation (dB/wavelength)
        "a_s": 2.5,  # Shear wave attenuation (dB/wavelength)
    }
    bott_hs_properties["z"] = max(z_ssp)
    bott_hs = KrakenBottomHalfspace(halfspace_properties=bott_hs_properties)

    att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)

    n_rcv_z = default_nb_rcv_z(max(freq), max_depth, n_per_l=15)
    field = KrakenField(
        src_depth=src_depth,
        n_rcv_z=n_rcv_z,
        rcv_z_max=max_depth,
    )

    env = KrakenEnv(
        title=title,
        env_root=env_dir,
        env_filename=testcase_name,
        freq=freq,
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
        kraken_bathy=bathy,
    )

    rcv_r_max = 50
    dr = 10  # 10m resolution
    n_rcv_r = rcv_r_max * 1000 / dr + 1
    flp = KrakenFlp(
        env=env,
        src_depth=src_depth,
        n_rcv_z=n_rcv_z,
        rcv_z_max=max_depth,
        rcv_r_max=rcv_r_max,
        n_rcv_r=n_rcv_r,
        mode_addition="coherent",
    )

    return env, flp


def testcase1_0(freq=[20], min_waveguide_depth=150):
    """
    Test case 1.0 : Flat and isotopric environment. 1 layer bottom and constant sound speed profile
    """
    name = "testcase1_0"
    title = "Test case 1.0: Flat and isotopric environment. 1 layer bottom and constant sound speed profile"

    max_depth = min_waveguide_depth
    z_ssp = [0, max_depth]
    cp_ssp = [1500, 1500]

    bathy = Bathymetry()  # Default bathymetry is flat
    env, flp = testcase1_common(
        freq=freq,
        z_ssp=z_ssp,
        cp_ssp=cp_ssp,
        bathy=bathy,
        title=title,
        testcase_name=name,
    )

    return env, flp


def bathy_sin_slope(
    testcase_name="testcase1",
    min_waveguide_depth=150,
    max_range=50,
    theta=94,
    range_periodicity=6,
):
    # Define bathymetry
    fr = 1 / range_periodicity
    dr = 1 / (20 * fr)
    r = np.arange(0, max_range + dr, dr)

    alpha = 50
    h = min_waveguide_depth - alpha * (
        -1
        + np.sin(
            2 * np.pi * r * np.cos(theta * np.pi / 180) / range_periodicity - np.pi / 2
        )
    )

    # Save bathymetry
    env_dir = os.path.join(ENV_ROOT, testcase_name)
    pd.DataFrame({"r": np.round(r, 3), "h": np.round(h, 3)}).to_csv(
        os.path.join(env_dir, "bathy.csv"), index=False, header=False
    )

    max_depth = min_waveguide_depth + 2 * alpha
    plt.figure(figsize=(16, 8))
    plt.plot(r, h, color="k", linewidth=2, marker="o", markersize=2)
    plt.ylim([0, max_depth])
    plt.fill_between(r, h, max_depth, color="lightgrey")
    plt.gca().invert_yaxis()
    plt.xlabel("Range (km)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Depth (m)", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.grid()
    plt.savefig(os.path.join(env_dir, "bathy.png"))


def bathy_seamount(
    testcase_name="testcase1",
    min_waveguide_depth=150,
    max_range=50,
    max_depth=250,
    seamount_width=6,
):
    # Define bathymetry
    fr = 1 / seamount_width
    dr = 1 / (20 * fr)
    r = np.arange(0, max_range + dr, dr)

    r_seamount = r.max() / 2
    r0 = r_seamount - seamount_width / 2
    r1 = r_seamount + seamount_width / 2

    h_seamount = min_waveguide_depth
    h = np.ones(r.size) * max_depth

    alpha = (h_seamount - max_depth) / (r_seamount - r0)
    upslope = alpha * (r - r0) + max_depth
    downslope = -alpha * (r - r_seamount) + h_seamount

    idx_r_before = (r0 < r) * (r <= r_seamount)
    h[idx_r_before] = upslope[idx_r_before]
    idx_r_after = (r_seamount <= r) * (r < r1)
    h[idx_r_after] = downslope[idx_r_after]

    # Save bathymetry
    env_dir = os.path.join(ENV_ROOT, testcase_name)
    pd.DataFrame({"r": np.round(r, 3), "h": np.round(h, 3)}).to_csv(
        os.path.join(env_dir, "bathy.csv"), index=False, header=False
    )

    plt.figure(figsize=(16, 8))
    plt.plot(r, h, color="k", linewidth=2, marker="o", markersize=2)
    plt.ylim([0, max_depth])
    plt.fill_between(r, h, max_depth, color="lightgrey")
    plt.gca().invert_yaxis()
    plt.xlabel("Range (km)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Depth (m)", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.grid()
    plt.savefig(os.path.join(env_dir, "bathy.png"))


def testcase1_1(freq=[20], min_waveguide_depth=100):
    """
    Test case 1.1: Isotopric environment with sinusoidal bottom. 1 layer bottom and constant sound speed profile
    """
    name = "testcase1_1"
    title = "Test case 1.1: Isotopric environment with sinusoidal bottom. 1 layer bottom and constant sound speed profile"

    if not os.path.exists(os.path.join(ENV_ROOT, name)):
        os.makedirs(os.path.join(ENV_ROOT, name))

    # Create a slope bottom
    bathy_sin_slope(
        testcase_name=name,
        min_waveguide_depth=min_waveguide_depth,
        max_range=50,
        theta=94,
        range_periodicity=6,
    )

    bathy = Bathymetry(data_file=os.path.join(ENV_ROOT, name, "bathy.csv"))

    z_ssp = [0, bathy.bathy_depth.max()]
    cp_ssp = [1500, 1500]

    env, flp = testcase1_common(
        freq=freq,
        z_ssp=z_ssp,
        cp_ssp=cp_ssp,
        bathy=bathy,
        title=title,
        testcase_name=name,
    )

    return env, flp


def testcase1_2(freq=[20], min_waveguide_depth=100):
    """
    Test case 1.2: Isotopric environment with sinusoidal bottom. 1 layer bottom and constant sound speed profile
    """
    name = "testcase1_2"
    title = "Test case 1.2: Isotopric environment with sinusoidal bottom. 1 layer bottom and constant sound speed profile"

    if not os.path.exists(os.path.join(ENV_ROOT, name)):
        os.makedirs(os.path.join(ENV_ROOT, name))

    # Create a slope bottom
    bathy_seamount(
        testcase_name=name,
        min_waveguide_depth=min_waveguide_depth,
        max_range=50,
        max_depth=250,
        seamount_width=6,
    )

    bathy = Bathymetry(data_file=os.path.join(ENV_ROOT, name, "bathy.csv"))

    z_ssp = [0, bathy.bathy_depth.max()]
    cp_ssp = [1500, 1500]

    env, flp = testcase1_common(
        freq=freq,
        z_ssp=z_ssp,
        cp_ssp=cp_ssp,
        bathy=bathy,
        title=title,
        testcase_name=name,
    )

    return env, flp


if __name__ == "__main__":
    env, flp = testcase1_2()
