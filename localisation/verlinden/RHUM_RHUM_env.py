import numpy as np
import xarray as xr
from arlpy import uwa

from cst import SAND_PROPERTIES, RHO_W
from propa.kraken_toolbox.kraken_env import (
    KrakenEnv,
    KrakenTopHalfspace,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
    KrakenFlp,
)
from propa.kraken_toolbox.utils import default_nb_rcv_z


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


def verlinden_test_case_env(
    env_root,
    env_filename,
    title="Test case environnment based on Verlinden 2017 scenario",
    src_depth=None,
    freq=[20],
):
    max_depth = 150
    src_depth = (
        max_depth - 7
    )  # 7m above the bottom (Assume Verlinden used Hydrophone #1 presentend in fig 3.8)

    top_hs = KrakenTopHalfspace()
    z_ssp = [0, max_depth]
    cp_ssp = [1500, 1500]

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

    n_rcv_z = default_nb_rcv_z(max(freq), max_depth)
    n_rcv_z = 1001
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


if __name__ == "__main__":
    rhum_rum_isotropic_env()
