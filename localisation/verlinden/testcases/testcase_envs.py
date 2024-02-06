import os

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
from localisation.verlinden.testcases.testcase_bathy import (
    bathy_sin_slope,
    bathy_seamount,
    mmdpm_profile,
)
from localisation.verlinden.verlinden_path import TC_WORKING_DIR


# def isotropic_ideal_env(
#     env_root,
#     env_filename,
#     title="Isotropic ideal environment",
#     max_depth=3000,
#     src_depth=None,
#     freq=[20],
# ):
#     if src_depth is None:
#         src_depth = max_depth - 2  # 2m above the bottom

#     top_hs = KrakenTopHalfspace()
#     z_ssp = np.array([0.0, max_depth])
#     cp_ssp = np.array([1500.0, 1500.0])

#     medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

#     bott_hs_properties = SAND_PROPERTIES
#     bott_hs_properties["z"] = z_ssp.max()
#     bott_hs = KrakenBottomHalfspace(halfspace_properties=bott_hs_properties)

#     att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)

#     n_rcv_z = default_nb_rcv_z(max(freq), max_depth)
#     field = KrakenField(
#         src_depth=src_depth,
#         n_rcv_z=n_rcv_z,
#         rcv_z_max=max_depth,
#     )

#     env = KrakenEnv(
#         title=title,
#         env_root=env_root,
#         env_filename=env_filename,
#         freq=freq,
#         kraken_top_hs=top_hs,
#         kraken_medium=medium,
#         kraken_attenuation=att,
#         kraken_bottom_hs=bott_hs,
#         kraken_field=field,
#     )

#     flp = KrakenFlp(
#         env=env,
#         src_depth=src_depth,
#         n_rcv_z=n_rcv_z,
#         rcv_z_max=max_depth,
#         rcv_r_max=30,
#         mode_addition="coherent",
#     )

#     return env, flp


# def rhum_rum_isotropic_env(
#     env_root,
#     env_filename,
#     title="Isotropic ideal environment with real Sound Speed Profile",
#     src_depth=None,
#     freq=[20],
# ):
#     swir_salinity_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp\cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m_1700737971954.nc"
#     swir_temp_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp\cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m_1700737952634.nc"

#     # Load temperatue and salinity
#     ds_sal = xr.open_dataset(swir_salinity_path)
#     ds_temp = xr.open_dataset(swir_temp_path)

#     ssp = uwa.soundspeed(
#         ds_sal.so.isel(latitude=0, longitude=0, time=0),
#         ds_temp.thetao.isel(latitude=0, longitude=0, time=0),
#         ds_sal.depth,
#     )
#     ssp = ssp.dropna(dim="depth")
#     # ssp.<
#     # ssp.plot(y="depth", yincrease=False)

#     max_depth = ssp.depth.max().round(2).values
#     if src_depth is None:
#         src_depth = max_depth - 2  # 2m above the bottom

#     top_hs = KrakenTopHalfspace()
#     z_ssp = ssp.depth.round(2).values
#     cp_ssp = ssp.round(2).values

#     medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

#     bott_hs_properties = SAND_PROPERTIES
#     bott_hs_properties["z"] = z_ssp.max()
#     bott_hs = KrakenBottomHalfspace(halfspace_properties=bott_hs_properties)

#     att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)

#     n_rcv_z = default_nb_rcv_z(max(freq), max_depth)
#     field = KrakenField(
#         src_depth=src_depth,
#         n_rcv_z=n_rcv_z,
#         rcv_z_max=max_depth,
#     )

#     env = KrakenEnv(
#         title=title,
#         env_root=env_root,
#         env_filename=env_filename,
#         freq=freq,
#         kraken_top_hs=top_hs,
#         kraken_medium=medium,
#         kraken_attenuation=att,
#         kraken_bottom_hs=bott_hs,
#         kraken_field=field,
#     )

#     flp = KrakenFlp(
#         env=env,
#         src_depth=src_depth,
#         n_rcv_z=n_rcv_z,
#         rcv_z_max=max_depth,
#         rcv_r_max=50,
#         mode_addition="coherent",
#     )

#     return env, flp


##########################################################################################
# Test case 1 : Shallow water environment (close from Verlinden environment)
##########################################################################################


def testcase1_common(freq, z_ssp, cp_ssp, bathy, title, testcase_name="testcase1"):
    # Create environment directory
    env_dir = os.path.join(TC_WORKING_DIR, testcase_name)
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

    flp_nrcv_z = 1
    flp_rcv_z_min = 5
    flp_rcv_z_max = 5

    flp = KrakenFlp(
        env=env,
        src_depth=src_depth,
        n_rcv_z=flp_nrcv_z,
        rcv_z_min=flp_rcv_z_min,
        rcv_z_max=flp_rcv_z_max,
        rcv_r_max=rcv_r_max,
        n_rcv_r=n_rcv_r,
        mode_addition="coherent",
    )

    # flp = KrakenFlp(
    #     env=env,
    #     src_depth=src_depth,
    #     n_rcv_z=n_rcv_z,
    #     rcv_z_max=max_depth,
    #     rcv_r_max=rcv_r_max,
    #     n_rcv_r=n_rcv_r,
    #     mode_addition="coherent",
    # )

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


def testcase1_1(freq=[20], min_waveguide_depth=100):
    """
    Test case 1.1: Isotopric environment with sinusoidal bottom. 1 layer bottom and constant sound speed profile
    """
    name = "testcase1_1"
    title = "Test case 1.1: Isotopric environment with sinusoidal bottom. 1 layer bottom and constant sound speed profile"

    if not os.path.exists(os.path.join(TC_WORKING_DIR, name)):
        os.makedirs(os.path.join(TC_WORKING_DIR, name))

    # Create a slope bottom
    bathy_sin_slope(
        testcase_name=name,
        min_waveguide_depth=min_waveguide_depth,
        max_range=50,
        theta=94,
        range_periodicity=6,
    )

    bathy = Bathymetry(data_file=os.path.join(TC_WORKING_DIR, name, "bathy.csv"))

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

    if not os.path.exists(os.path.join(TC_WORKING_DIR, name)):
        os.makedirs(os.path.join(TC_WORKING_DIR, name))

    # Create a slope bottom
    bathy_seamount(
        testcase_name=name,
        min_waveguide_depth=min_waveguide_depth,
        max_range=50,
        max_depth=250,
        seamount_width=6,
    )

    bathy = Bathymetry(data_file=os.path.join(TC_WORKING_DIR, name, "bathy.csv"))

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


def testcase1_3(freq=[20], min_waveguide_depth=100):
    """
    Test case 1.3: Isotopric environment with real bathy profile extracted using MMDPM app around OBS RR48. 1 layer bottom and constant sound speed profile
    """
    name = "testcase1_3"
    title = "Test case 1.3: Isotopric environment with real bathy profile extracted using MMDPM app around OBS RR48. 1 layer bottom and constant sound speed profile"

    if not os.path.exists(os.path.join(TC_WORKING_DIR, name)):
        os.makedirs(os.path.join(TC_WORKING_DIR, name))

    # Create a slope bottom
    mmdpm_profile(testcase_name=name, mmdpm_testname="PVA_RR48", azimuth=360)
    bathy = Bathymetry(data_file=os.path.join(TC_WORKING_DIR, name, "bathy.csv"))
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
    env, flp = testcase1_3()
