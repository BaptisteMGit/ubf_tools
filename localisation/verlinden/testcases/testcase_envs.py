import os
import scipy.io as sio
import matplotlib.pyplot as plt

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
    bathy_flat_seabed,
    bathy_sin_slope,
    bathy_seamount,
    mmdpm_profile,
    extract_2D_bathy_profile,
)
from localisation.verlinden.verlinden_path import TC_WORKING_DIR

##########################################################################################
# Useful constants
##########################################################################################
ENV_DESC_IMG_FOLDER = "env_desc_img"


##########################################################################################
# Utils functions
##########################################################################################
def get_img_path(testcase_name, type="medium", azimuth=None):
    root = os.path.join(TC_WORKING_DIR, testcase_name, ENV_DESC_IMG_FOLDER)
    if not os.path.exists(root):
        os.makedirs(root)

    if type == "medium":
        return os.path.join(root, "medium_properties.png")
    elif type == "bottom":
        return os.path.join(root, "bottom_properties.png")
    elif type == "bathy":
        if azimuth is not None:
            return os.path.join(root, f"bathy_az{azimuth:.2f}.png")
        else:
            return os.path.join(root, "bathy.png")


def get_bathy_path(testcase_name):
    return os.path.join(TC_WORKING_DIR, testcase_name, "bathy.csv")


def plot_env_properties(env, plot_medium, plot_bottom):
    if plot_medium:
        env.medium.plot_medium()
        plt.savefig(get_img_path(env.filename, type="medium"))
        plt.close()

    if plot_bottom:
        env.bottom_hs.plot_bottom_halfspace()
        plt.savefig(get_img_path(env.filename, type="bottom"))
        plt.close()


##########################################################################################
# Test case 1 : Isotropic environment
##########################################################################################


def testcase1_common(freq, z_ssp, cp_ssp, bathy, title, testcase_name="testcase1"):
    """
    Test case 1.x common properties.
    """

    # Create environment directory
    env_dir = os.path.join(TC_WORKING_DIR, testcase_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)

    max_depth = max(z_ssp)  # Depth at src position (m)
    src_depth = (
        max_depth - 1
    )  # Assume hydrophone is 1m above the bottom (which seems to be reasonable in the case of a hydrophone mounted on a OBS)
    # src_depth = (
    #     max_depth - 7
    # )  # 7m above the bottom (Assume Verlinden used Hydrophone #1 presentend in fig 3.8)

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

    return env, flp


def testcase1_0(testcase_varin, plot_bathy=False, plot_medium=False, plot_bottom=False):
    """
    Test case 1.0 :
        Environment: isotopric
        Bathymetry: flat bottom
        SSP: c = 1500 m/s
        Sediment: One layer bottom with constant properties
    """

    name = "testcase1_0"
    title = "Test case 1.0: Flat and isotopric environment. 1 layer bottom and constant sound speed profile"

    if "freq" not in testcase_varin:
        freq = [20]
    else:
        freq = testcase_varin["freq"]
    if "max_range_m" not in testcase_varin:
        max_range_m = 50 * 1e3
    else:
        max_range_m = testcase_varin["max_range_m"]

    waveguide_depth = 200
    cst_celerity = 1500
    z_ssp = [0, waveguide_depth]
    cp_ssp = [cst_celerity, cst_celerity]

    # Create plane bathymetry
    max_range_km = max_range_m * 1e-3
    bathy_flat_seabed(
        testcase_name=name,
        waveguide_depth=waveguide_depth,
        max_range=max_range_km,
        plot=plot_bathy,
        bathy_path=get_img_path(name, type="bathy"),
    )

    bathy = Bathymetry(get_bathy_path(testcase_name=name))
    env, flp = testcase1_common(
        freq=freq,
        z_ssp=z_ssp,
        cp_ssp=cp_ssp,
        bathy=bathy,
        title=title,
        testcase_name=name,
    )

    # Plot properties
    plot_env_properties(env, plot_medium, plot_bottom)

    return env, flp


def testcase1_1(
    testcase_varin,
    plot_bathy=False,
    plot_medium=False,
    plot_bottom=False,
):
    """
    Test case 1.1 :
        Environment: isotopric
        Bathymetry: sinusoidal bottom
        SSP: c = 1500 m/s
        Sediment: One layer bottom with constant properties
    """

    name = "testcase1_1"
    title = "Test case 1.1: Isotopric environment with sinusoidal bottom. 1 layer bottom and constant sound speed profile"

    if "freq" not in testcase_varin:
        freq = [20]
    else:
        freq = testcase_varin["freq"]
    if "max_range_m" not in testcase_varin:
        max_range_m = 50 * 1e3
    else:
        max_range_m = testcase_varin["max_range_m"]
    if "min_depth" not in testcase_varin:
        min_depth = 100
    else:
        min_depth = testcase_varin["min_depth"]

    max_range_km = max_range_m * 1e-3

    # Create a slope bottom
    bathy_sin_slope(
        testcase_name=name,
        min_depth=min_depth,
        max_range=max_range_km,
        theta=94,
        range_periodicity=6,
        plot=plot_bathy,
        bathy_path=get_img_path(name, type="bathy"),
    )

    bathy = Bathymetry(data_file=get_bathy_path(testcase_name=name))

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

    # Plot properties
    plot_env_properties(env, plot_medium, plot_bottom)

    return env, flp


def testcase1_2(testcase_varin, plot_bathy=False, plot_medium=False, plot_bottom=False):
    """
    Test case 1.2 :
        Environment: isotopric
        Bathymetry: seamount bottom
        SSP: c = 1500 m/s
        Sediment: One layer bottom with constant properties
    """

    name = "testcase1_2"
    title = "Test case 1.2: Isotopric environment with seamount bottom. 1 layer bottom and constant sound speed profile"

    if "freq" not in testcase_varin:
        freq = [20]
    else:
        freq = testcase_varin["freq"]
    if "max_range_m" not in testcase_varin:
        max_range_m = 50 * 1e3
    else:
        max_range_m = testcase_varin["max_range_m"]
    if "min_depth" not in testcase_varin:
        min_depth = 100
    else:
        min_depth = testcase_varin["min_depth"]

    max_range_km = max_range_m * 1e-3

    # Create a slope bottom
    bathy_seamount(
        testcase_name=name,
        min_depth=min_depth,
        max_range=max_range_km,
        seamount_height=100,
        seamount_width=6,
        plot=plot_bathy,
        bathy_path=get_img_path(name, type="bathy"),
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

    # Plot properties
    plot_env_properties(env, plot_medium, plot_bottom)

    return env, flp


def testcase1_3(testcase_varin, plot_bathy=False, plot_medium=False, plot_bottom=False):
    """
    Test case 1.3 :
        Environment: isotopric
        Bathymetry: real bathy profile around OBS RR48 (extracted from GEBCO grid using MMDPM app)
        SSP: c = 1500 m/s
        Sediment: One layer bottom with constant properties
    """

    name = "testcase1_3"
    title = "Test case 1.3: Isotopric environment with real bathy profile extracted using MMDPM app around OBS RR48. 1 layer bottom and constant sound speed profile"

    if "freq" not in testcase_varin:
        freq = [20]
    else:
        freq = testcase_varin["freq"]
    if "max_range_m" not in testcase_varin:
        max_range_m = 50 * 1e3
    else:
        max_range_m = testcase_varin["max_range_m"]

    max_range_km = max_range_m * 1e-3

    # Create a slope bottom
    mmdpm_profile(
        testcase_name=name,
        mmdpm_testname="PVA_RR48",
        azimuth=360,
        max_range_km=max_range_km,
        plot=plot_bathy,
        bathy_path=get_img_path(name, type="bathy"),
    )

    bathy = Bathymetry(data_file=get_bathy_path(testcase_name=name))
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

    # Plot properties
    plot_env_properties(env, plot_medium, plot_bottom)

    return env, flp


def testcase1_4(testcase_varin, plot_bathy=False, plot_medium=False, plot_bottom=False):
    """
    Test case 1.4 :
        Environment: isotopric
        Bathymetry: real bathy profile around OBS RR48 (extracted from GEBCO grid using MMDPM app)
        SSP: Realistic sound speed profile  (from Copernicus Marine Service)
        Sediment: One layer bottom with constant properties
    """
    name = "testcase1_4"
    title = "Test case 1.4: Isotopric environment with real bathy profile extracted using MMDPM app around OBS RR48. 1 layer bottom and realistic sound speed profile"

    if "freq" not in testcase_varin:
        freq = [20]
    else:
        freq = testcase_varin["freq"]
    if "max_range_m" not in testcase_varin:
        max_range_m = 50 * 1e3
    else:
        max_range_m = testcase_varin["max_range_m"]

    max_range_km = max_range_m * 1e-3

    # Load bathy profile
    mmdpm_profile(
        testcase_name=name,
        mmdpm_testname="PVA_RR48",
        azimuth=360,
        max_range_km=max_range_km,
        plot=plot_bathy,
        bathy_path=get_img_path(name, type="bathy"),
    )
    bathy = Bathymetry(data_file=get_bathy_path(testcase_name=name))

    # Load ssp mat file
    data_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp\mmdpm"
    fpath = os.path.join(data_dir, "PVA_RR48", f"mmdpm_test_PVA_RR48_ssp.mat")
    ssp_mat = sio.loadmat(fpath)
    z_ssp = ssp_mat["ssp"]["z"][0, 0].flatten()
    cp_ssp = ssp_mat["ssp"]["c"][0, 0].flatten()

    env, flp = testcase1_common(
        freq=freq,
        z_ssp=z_ssp,
        cp_ssp=cp_ssp,
        bathy=bathy,
        title=title,
        testcase_name=name,
    )

    # Plot properties
    plot_env_properties(env, plot_medium, plot_bottom)

    return env, flp


##########################################################################################
# Test case 2 : Anisotropic environment with shallow water environment
##########################################################################################


def testcase2_common(
    freq, bathy, title, testcase_name="testcase2", rcv_r_max=50, ssp_profile="constant"
):
    """
    Test case 2.x common properties.
    """
    # Create environment directory
    env_dir = os.path.join(TC_WORKING_DIR, testcase_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)

    if ssp_profile == "constant":
        z_ssp = [0, bathy.bathy_depth.max()]
        cp_ssp = [1500, 1500]
    elif ssp_profile == "real":
        # Load ssp mat file
        data_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp\mmdpm"
        fpath = os.path.join(
            data_dir,
            "shallow_water",
            f"fevrier_lon_-5.87_-2.87_lat_51.02_54.02_ssp.mat",
        )
        ssp_mat = sio.loadmat(fpath)
        z_ssp = ssp_mat["ssp"]["z"][0, 0].flatten()
        cp_ssp = ssp_mat["ssp"]["c"][0, 0].flatten()

    max_depth = max(z_ssp)  # Depth at src position (m)
    src_depth = max_depth - 1  # Assume hydrophone is 1m above the bottom

    top_hs = KrakenTopHalfspace()
    medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

    # 1 layer bottom
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

    # rcv_r_max = 50
    dr = 100  # 100m resolution
    n_rcv_r = rcv_r_max * 1000 / dr + 1

    # One receiver at 5m depth
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

    return env, flp


def testcase2_0(
    testcase_varin,
    plot_bathy=False,
    plot_medium=False,
    plot_bottom=False,
):
    """
    Test case 2.0:
        Environment: Anisotropic
        Bathymetry: flat bottom
        SSP: c = 1500 m/s
        Sediment: One layer bottom with constant properties
    """

    if "freq" not in testcase_varin:
        freq = [20]
    else:
        freq = testcase_varin["freq"]

    if "max_range_m" not in testcase_varin:
        max_range_m = 50 * 1e3
    else:
        max_range_m = testcase_varin["max_range_m"]

    waveguide_depth = 200
    max_range_km = max_range_m * 1e-3

    name = "testcase2_0"
    title = "Test case 2.0: Dummy test case equivalent to 1.0 to ensure the anisotropic process is producing results equivalent to the isotropic process."

    # Create flat bottom
    bathy_flat_seabed(
        testcase_name=name,
        waveguide_depth=waveguide_depth,
        max_range=max_range_km,
        plot=plot_bathy,
        bathy_path=get_img_path(name, type="bathy"),
    )

    bathy = Bathymetry(data_file=get_bathy_path(testcase_name=name))

    env, flp = testcase2_common(
        freq=freq,
        bathy=bathy,
        title=title,
        testcase_name=name,
        rcv_r_max=max_range_km,
        ssp_profile="constant",
    )

    # Plot properties
    plot_env_properties(env, plot_medium, plot_bottom)

    return env, flp


def testcase2_1(
    testcase_varin,
    plot_bathy=False,
    plot_medium=False,
    plot_bottom=False,
):
    """
    Test case 2.1:
        Environment: Anisotropic
        Bathymetry: sinusoidal bottom
        SSP: c = 1500 m/s
        Sediment: One layer bottom with constant properties
    """

    if "freq" not in testcase_varin:
        freq = [20]
    else:
        freq = testcase_varin["freq"]

    if "max_range_m" not in testcase_varin:
        max_range_m = 50 * 1e3
    else:
        max_range_m = testcase_varin["max_range_m"]

    if "azimuth" not in testcase_varin:
        azimuth = 0
    else:
        azimuth = testcase_varin["azimuth"]

    if "min_depth" not in testcase_varin:
        min_depth = 100
    else:
        min_depth = testcase_varin["min_depth"]

    max_range_km = max_range_m * 1e-3

    name = "testcase2_1"
    title = "Test case 2.1: Shallow water environment with sinusoidal bathy profile (same as test case 1.1). 1 layer bottom and constant sound speed profile"

    # Create a slope bottom
    bathy_sin_slope(
        testcase_name=name,
        min_depth=min_depth,
        max_range=max_range_km,
        theta=azimuth,
        range_periodicity=6,
        plot=plot_bathy,
        bathy_path=get_img_path(name, type="bathy", azimuth=azimuth),
    )

    bathy = Bathymetry(data_file=get_bathy_path(testcase_name=name))

    env, flp = testcase2_common(
        freq=freq,
        bathy=bathy,
        title=title,
        testcase_name=name,
        rcv_r_max=max_range_km,
        ssp_profile="constant",
    )

    # Plot properties
    plot_env_properties(env, plot_medium, plot_bottom)

    return env, flp


def testcase2_2(
    testcase_varin,
    plot_bathy=False,
    plot_medium=False,
    plot_bottom=False,
):
    """
    Test case 2.2:
        Environment: Anisotropic
        Bathymetry: real shallow water bathy profile (extracted from GEBCO grid using MMDPM app)
        SSP: Realistic sound speed profile  (from Copernicus Marine Service)
        Sediment: One layer bottom with constant properties
    """

    if "freq" not in testcase_varin:
        freq = [20]
    else:
        freq = testcase_varin["freq"]

    if "max_range_m" not in testcase_varin:
        max_range_m = 50 * 1e3
    else:
        max_range_m = testcase_varin["max_range_m"]

    if "azimuth" not in testcase_varin:
        azimuth = 0
    else:
        azimuth = testcase_varin["azimuth"]

    if "rcv_lon" not in testcase_varin:
        rcv_lon = -4.87
    else:
        rcv_lon = testcase_varin["rcv_lon"]

    if "rcv_lat" not in testcase_varin:
        rcv_lat = 52.22
    else:
        rcv_lat = testcase_varin["rcv_lat"]

    name = "testcase2_2"
    title = "Test case 2.2: Anisotropic environment with real bathy profile extracted using MMDPM app. 1 layer bottom and realistic sound speed profile"

    max_range_km = max_range_m * 1e-3
    range_resolution = 500  # 500m resolution
    # Load bathy bathy profile
    bathy_nc_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\bathy\shallow_water\GEBCO_2021_lon_-5.87_-2.87_lat_51.02_54.02.nc"
    extract_2D_bathy_profile(
        bathy_nc_path=bathy_nc_path,
        testcase_name=name,
        obs_lon=rcv_lon,
        obs_lat=rcv_lat,
        azimuth=azimuth,
        max_range_km=max_range_km,
        range_resolution=range_resolution,
        plot=plot_bathy,
        bathy_path=get_img_path(name, type="bathy", azimuth=azimuth),
    )
    bathy = Bathymetry(data_file=get_bathy_path(testcase_name=name))

    env, flp = testcase2_common(
        freq=freq,
        bathy=bathy,
        title=title,
        testcase_name=name,
        rcv_r_max=max_range_km,
        ssp_profile="real",
    )

    # Plot properties
    plot_env_properties(env, plot_medium, plot_bottom)

    return env, flp


##########################################################################################
# Test case 3 : Anisotropic environment with RHUM RUM environment
##########################################################################################


def testcase3_common(freq, bathy, title, testcase_name="testcase1", rcv_r_max=50):
    """
    Test case 3.x common properties.
    """
    # Create environment directory
    env_dir = os.path.join(TC_WORKING_DIR, testcase_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)

    # Load ssp mat file
    data_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp\mmdpm"
    fpath = os.path.join(data_dir, "PVA_RR48", f"mmdpm_test_PVA_RR48_ssp.mat")
    ssp_mat = sio.loadmat(fpath)
    z_ssp = ssp_mat["ssp"]["z"][0, 0].flatten()
    cp_ssp = ssp_mat["ssp"]["c"][0, 0].flatten()

    max_depth = max(z_ssp)  # Depth at src position (m)
    src_depth = max_depth - 1  # Assume hydrophone is 1m above the bottom

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

    # rcv_r_max = 50
    dr = 100  # 100m resolution
    n_rcv_r = rcv_r_max * 1000 / dr + 1

    # One receiver at 5m depth
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

    return env, flp


def testcase3_1(
    testcase_varin,
    plot_bathy=False,
    plot_medium=False,
    plot_bottom=False,
):
    """
    Test case 3.1:
        Environment: Anisotropic
        Bathymetry: real depp water bathy profile around OBS RR48 (extracted from GEBCO grid using MMDPM app)
        SSP: Realistic sound speed profile  (from Copernicus Marine Service)
        Sediment: One layer bottom with constant properties
    """

    if "freq" not in testcase_varin:
        freq = [20]
    else:
        freq = testcase_varin["freq"]

    if "max_range_m" not in testcase_varin:
        max_range_m = 50 * 1e3
    else:
        max_range_m = testcase_varin["max_range_m"]

    if "azimuth" not in testcase_varin:
        azimuth = 0
    else:
        azimuth = testcase_varin["azimuth"]

    if "rcv_lon" not in testcase_varin:
        rcv_lon = 65.94
    else:
        rcv_lon = testcase_varin["rcv_lon"]

    if "rcv_lat" not in testcase_varin:
        rcv_lat = -27.58
    else:
        rcv_lat = testcase_varin["rcv_lat"]

    name = "testcase3_1"
    title = "Test case 3.1: Anisotropic environment with real bathy profile extracted using MMDPM app around OBS RR48. 1 layer bottom and realistic sound speed profile"

    max_range_km = max_range_m * 1e-3
    range_resolution = 1000  # 1km resolution
    # Load bathy bathy profile
    bathy_nc_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\bathy\mmdpm\PVA_RR48\GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"
    extract_2D_bathy_profile(
        bathy_nc_path=bathy_nc_path,
        testcase_name=name,
        obs_lon=rcv_lon,
        obs_lat=rcv_lat,
        azimuth=azimuth,
        max_range_km=max_range_km,
        range_resolution=range_resolution,
        plot=plot_bathy,
        bathy_path=get_img_path(name, type="bathy", azimuth=azimuth),
    )
    bathy = Bathymetry(data_file=get_bathy_path(testcase_name=name))

    env, flp = testcase3_common(
        freq=freq,
        bathy=bathy,
        title=title,
        testcase_name=name,
        rcv_r_max=max_range_km,
    )

    # Plot properties
    plot_env_properties(env, plot_medium, plot_bottom)

    return env, flp


if __name__ == "__main__":

    # Test case 1.0
    tc_varin = {"freq": [20], "max_range_m": 50 * 1e3}
    env, flp = testcase1_0(
        tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True
    )
    env.write_env()
    flp.write_flp()

    # Test case 1.1
    tc_varin = {"freq": [20], "min_depth": 150}
    env, flp = testcase1_1(
        tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True
    )
    env.write_env()
    flp.write_flp()

    # Test case 1.2
    tc_varin = {"freq": [20], "min_depth": 100}
    env, flp = testcase1_2(
        tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True
    )
    env.write_env()
    flp.write_flp()

    # Test case 1.3
    tc_varin = {"freq": [20], "max_range_m": 50 * 1e3}
    env, flp = testcase1_3(
        tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True
    )
    env.write_env()
    flp.write_flp()

    # Test case 1.4
    tc_varin = {"freq": [20], "max_range_m": 50 * 1e3}
    env, flp = testcase1_4(
        tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True
    )
    env.write_env()
    flp.write_flp()

    # Test case 2.0
    tc_varin = {
        "freq": [20],
        "max_range_m": 15 * 1e3,
    }
    for az in range(0, 360, 30):
        tc_varin["azimuth"] = az
        env, flp = testcase2_0(
            tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True
        )
    env.write_env()
    flp.write_flp()

    # Test case 2.1
    tc_varin = {
        "freq": [20],
        "max_range_m": 15 * 1e3,
        "azimuth": 0,
    }
    for az in range(0, 360, 30):
        tc_varin["azimuth"] = az
        env, flp = testcase2_1(
            tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True
        )
    env.write_env()
    flp.write_flp()

    # Test case 2.2
    tc_varin = {
        "freq": [20],
        "max_range_m": 15 * 1e3,
        "azimuth": 0,
        "rcv_lon": -4.7,
        "rcv_lat": 52.6,
    }
    for az in range(0, 360, 30):
        tc_varin["azimuth"] = az
        env, flp = testcase2_2(
            tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True
        )
    env.write_env()
    flp.write_flp()

    # Test case 3.1
    tc_varin = {
        "freq": [20],
        "max_range_m": 15 * 1e3,
        "azimuth": 0,
        "rcv_lon": 65.943,
        "rcv_lat": -27.5792,
    }
    for az in range(0, 360, 30):
        tc_varin["azimuth"] = az
        env, flp = testcase3_1(
            tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True
        )
    env.write_env()
    flp.write_flp()

    # from propa.kraken_toolbox.utils import waveguide_cutoff_freq

    # tc_varin = {
    #     "freq": [20],
    #     "max_range_m": 15 * 1e3,
    #     "azimuth": 0,
    #     "rcv_lon": -4.7,
    #     "rcv_lat": 52.6,
    # }
    # for az in range(0, 360, 10):
    #     tc_varin["azimuth"] = az
    #     env, flp = testcase2_1(tc_varin, plot_bathy=True)

    #     print(f"fc = {waveguide_cutoff_freq(env.bathy.bathy_depth.min()):.2f} Hz")


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
