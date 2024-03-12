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
    elif type == "env":
        return os.path.join(root, "env_properties.png")
    elif type == "bathy":
        if azimuth is not None:
            return os.path.join(root, f"bathy_az{azimuth:.2f}.png")
        else:
            return os.path.join(root, "bathy.png")


def get_bathy_path(testcase_name):
    return os.path.join(TC_WORKING_DIR, testcase_name, "bathy.csv")


def plot_env_properties(env, plot_medium, plot_bottom, plot_env):
    if plot_medium:
        env.medium.plot_medium()
        plt.savefig(get_img_path(env.filename, type="medium"))
        plt.close()

    if plot_bottom:
        env.bottom_hs.plot_bottom_halfspace()
        plt.savefig(get_img_path(env.filename, type="bottom"))
        plt.close()

    if plot_env:
        env.plot_env()
        plt.savefig(get_img_path(env.filename, type="env"))
        plt.close()


##########################################################################################
# Main class
##########################################################################################
class TestCase:
    def __init__(
        self, name, testcase_varin={}, title="Default testcase", desc="", mode="prod"
    ):
        self.name = name
        self.testcase_varin = testcase_varin
        self.title = title
        self.desc = desc
        self.mode = mode

        ### Define variables ###
        # Varins
        self.freq = None
        self.max_range_m = None
        self.min_depth = None
        self.azimuth = None
        self.rcv_lon = None
        self.rcv_lat = None
        self.plot_medium = False
        self.plot_bottom = False
        self.plot_bathy = False
        self.plot_env = False
        # Ssp
        self.z_ssp = None
        self.cp_ssp = None
        # Kraken objects
        self.top_hs = None
        self.medium = None
        self.bott_hs = None
        self.att = None
        self.field = None
        self.bathy = None
        self.env = None
        self.flp = None
        # Environment directory
        self.env_dir = ""
        # Max depth
        self.max_depth = None
        self.src_depth = None

        # Flag for range-dependence and isotropic/anisotropic
        self.range_dependence = False
        self.isotropic = True

        # Default values
        self.default_varin = {
            "freq": [20],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
            "azimuth": 0,
            "rcv_lon": -4.87,
            "rcv_lat": 52.22,
        }

        # Set env directory
        self.set_env_dir()

        # Bottom properties (1 layer bottom)
        self.bott_hs_properties = {
            "rho": 1.9 * RHO_W * 1e-3,  # Density (g/cm^3)
            "c_p": 1650,  # P-wave celerity (m/s)
            "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
            "a_p": 0.8,  # Compression wave attenuation (dB/wavelength)
            "a_s": 2.5,  # Shear wave attenuation (dB/wavelength)
            "z": None,
        }

        self.process_testcase()

    def process_testcase(self):
        # Read variables
        self.read_varin()
        # Load testcase
        self.load_testcase()
        # Write flp and env files
        self.write_testcase_files()
        # Plot env
        self.plot_testcase_env()

    def set_env_dir(self):
        self.env_dir = os.path.join(TC_WORKING_DIR, self.name)
        if not os.path.exists(self.env_dir):
            os.makedirs(self.env_dir)

    def read_varin(self):
        for key, default_value in self.default_varin.items():
            if key in self.testcase_varin:
                setattr(self, key, self.testcase_varin[key])
            else:
                setattr(self, key, default_value)

        # In production mode no plots should be used (using plot functions creates conflics between subprocesses)
        if self.mode == "prod":
            plot_bool = False
        else:
            plot_bool = True
        for key_plot in ["bathy", "medium", "bottom", "env"]:
            setattr(self, f"plot_{key_plot}", plot_bool)

    def plot_testcase_env(self):
        plot_env_properties(
            self.env,
            plot_medium=self.plot_medium,
            plot_bottom=self.plot_bottom,
            plot_env=self.plot_env,
        )

    def load_testcase(self):
        # Load bathy
        self.write_bathy()
        self.set_bathy()
        # Load ssp
        self.load_ssp()
        # Set kraken objects
        self.set_top_hs()
        self.set_medium()
        self.set_att()
        self.set_field()
        self.set_bott_hs()
        self.set_env()
        self.set_flp()

    def load_ssp(self):
        # Default sound speed profile
        c0 = 1500
        self.z_ssp = [0, self.max_depth]
        self.cp_ssp = [c0, c0]

    def set_top_hs(self):
        self.top_hs = KrakenTopHalfspace()

    def set_bott_hs(self):
        self.bott_hs = KrakenBottomHalfspace(
            halfspace_properties=self.bott_hs_properties
        )

    def set_att(self):
        self.att = KrakenAttenuation(
            units="dB_per_wavelength", use_volume_attenuation=False
        )

    def set_field(self):
        n_rcv_z = default_nb_rcv_z(max(self.freq), self.max_depth, n_per_l=15)
        self.field = KrakenField(
            n_rcv_z=n_rcv_z,
            src_depth=self.src_depth,
            rcv_z_max=self.max_depth,
        )

    def set_medium(self):
        self.medium = KrakenMedium(
            ssp_interpolation_method="C_linear", z_ssp=self.z_ssp, c_p=self.cp_ssp
        )

    def write_bathy(self):
        # Default bathy
        bathy_flat_seabed(
            testcase_name=self.name,
            waveguide_depth=self.min_depth,
            max_range=self.max_range_m * 1e-3,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy"),
        )

    def set_bathy(self):
        self.bathy = Bathymetry(get_bathy_path(testcase_name=self.name))
        self.max_depth = self.bathy.bathy_depth.max()
        self.min_depth = self.bathy.bathy_depth.min()
        self.src_depth = self.max_depth - 1  # Assume hydrophone is 1m above the bottom
        self.bott_hs_properties["z"] = self.max_depth

    def set_env(self):
        self.env = KrakenEnv(
            title=self.title,
            env_root=self.env_dir,
            env_filename=self.name,
            freq=self.freq,
            kraken_top_hs=self.top_hs,
            kraken_medium=self.medium,
            kraken_attenuation=self.att,
            kraken_bottom_hs=self.bott_hs,
            kraken_field=self.field,
            kraken_bathy=self.bathy,
        )

    def set_flp(self):
        dr = 10  # 10m resolution
        n_rcv_r = self.max_range_m * 1000 / dr + 1

        # Source = ship radiating sound at 5m depth
        flp_nrcv_z = 1
        flp_rcv_z_min = 5
        flp_rcv_z_max = 5

        self.flp = KrakenFlp(
            env=self.env,
            src_depth=self.src_depth,
            n_rcv_z=flp_nrcv_z,
            rcv_z_min=flp_rcv_z_min,
            rcv_z_max=flp_rcv_z_max,
            rcv_r_max=self.max_range_m * 1e-3,
            n_rcv_r=n_rcv_r,
            mode_addition="coherent",
        )

    def write_testcase_files(self):
        self.env.write_env()
        self.flp.write_flp()


##########################################################################################
# Test case 1 : Isotropic environment
##########################################################################################
class TestCase1(TestCase):
    def __init__(
        self,
        name,
        testcase_varin={},
        title="Test case 1: Isotropic environment",
        desc="",
        mode="prod",
    ):
        super().__init__(name, testcase_varin, title, desc, mode)
        self.isotropic = True


class TestCase1_0(TestCase1):
    def __init__(self, testcase_varin={}, mode="prod"):
        name = "testcase1_0"
        title = "Test case 1.0: Flat and isotopric environment. 1 layer bottom and constant sound speed profile"
        desc = "Environment: isotopric, Bathymetry: flat bottom, SSP: c = 1500 m/s, Sediment: One layer bottom with constant properties"
        super().__init__(
            name, testcase_varin=testcase_varin, title=title, desc=desc, mode=mode
        )

        # Update default values with values testcase specific values
        self.default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
        }
        # Flat bottom
        self.range_dependence = False

        # Process all info
        self.process_testcase()


class TestCase1_1(TestCase1):
    def __init__(self, testcase_varin={}, mode="prod"):
        name = "testcase1_1"
        title = "Test case 1.1: Isotopric environment with sinusoidal bottom. 1 layer bottom and constant sound speed profile"
        desc = "Environment: isotopric, Bathymetry: sinusoidal bottom, SSP: c = 1500 m/s, Sediment: One layer bottom with constant properties"

        super().__init__(
            name, testcase_varin=testcase_varin, title=title, desc=desc, mode=mode
        )

        # Update default values with values testcase specific values
        self.default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
        }
        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process_testcase()

    def write_bathy(self):
        # Create a slope bottom
        bathy_sin_slope(
            testcase_name=self.name,
            min_depth=self.min_depth,
            max_range=self.max_range_m * 1e-3,
            theta=94,
            range_periodicity=6,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy"),
        )


class TestCase1_2(TestCase1):
    def __init__(self, testcase_varin={}, mode="prod"):
        name = "testcase1_2"
        title = "Test case 1.2: Isotopric environment with seamount bottom. 1 layer bottom and constant sound speed profile"
        desc = "Environment: isotopric, Bathymetry: seamount bottom, SSP: c = 1500 m/s, Sediment: One layer bottom with constant properties"

        super().__init__(
            name, testcase_varin=testcase_varin, title=title, desc=desc, mode=mode
        )

        # Update default values with values testcase specific values
        self.default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
        }
        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process_testcase()

    def write_bathy(self):
        # Create a seamount bathy profile
        bathy_seamount(
            testcase_name=self.name,
            min_depth=self.min_depth,
            max_range=self.max_range_m * 1e-3,
            seamount_height=100,
            seamount_width=6,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy"),
        )


class TestCase1_3(TestCase1):
    def __init__(self, testcase_varin={}, mode="prod"):
        name = "testcase1_3"
        title = "Test case 1.3: Isotopric environment with real bathy profile extracted using MMDPM app around OBS RR48. 1 layer bottom and constant sound speed profile"
        desc = "Environment: isotopric, Bathymetry: real bathy profile around OBS RR48 (extracted from GEBCO grid using MMDPM app), SSP: c = 1500 m/s, Sediment: One layer bottom with constant properties"

        super().__init__(
            name, testcase_varin=testcase_varin, title=title, desc=desc, mode=mode
        )

        # Update default values with values testcase specific values
        self.default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
        }
        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process_testcase()

    def write_bathy(self):
        # Load real profile around OBS RR48
        mmdpm_profile(
            testcase_name=self.name,
            mmdpm_testname="PVA_RR48",
            azimuth=360,
            max_range_km=self.max_range_m * 1e-3,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy"),
        )


class TestCase1_4(TestCase1):
    def __init__(self, testcase_varin={}, mode="prod"):
        name = "testcase1_4"
        title = "Test case 1.4: Isotopric environment with real bathy profile extracted using MMDPM app around OBS RR48. 1 layer bottom and realistic sound speed profile"
        desc = "Environment: isotopric, Bathymetry: real bathy profile around OBS RR48 (extracted from GEBCO grid using MMDPM app), SSP: Realistic sound speed profile  (from Copernicus Marine Service), Sediment: One layer bottom with constant properties"

        super().__init__(
            name, testcase_varin=testcase_varin, title=title, desc=desc, mode=mode
        )

        # Update default values with values testcase specific values
        self.default_varin = {
            "freq": [20],
            "max_range_m": 50 * 1e3,
        }
        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process_testcase()

    def write_bathy(self):
        # Load bathy profile
        mmdpm_profile(
            testcase_name=self.name,
            mmdpm_testname="PVA_RR48",
            azimuth=360,
            max_range_km=self.max_range_m * 1e-3,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy"),
        )

    def load_ssp(self):
        # Load ssp mat file
        data_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp\mmdpm"
        fpath = os.path.join(data_dir, "PVA_RR48", f"mmdpm_test_PVA_RR48_ssp.mat")
        ssp_mat = sio.loadmat(fpath)
        self.z_ssp = ssp_mat["ssp"]["z"][0, 0].flatten()
        self.cp_ssp = ssp_mat["ssp"]["c"][0, 0].flatten()


##########################################################################################
# Test case 2 : Anisotropic and shallow water environment
##########################################################################################


class TestCase2(TestCase):
    def __init__(
        self,
        name,
        testcase_varin={},
        title="Test case 1: Isotropic environment",
        desc="",
        mode="prod",
    ):
        super().__init__(name, testcase_varin, title, desc, mode)
        self.isotropic = False


##########################################################################################
# Test case 3 : Anisotropic and deep whater environment (RHUM RUM)
##########################################################################################


class TestCase3(TestCase):
    def __init__(
        self,
        name,
        testcase_varin={},
        title="Test case 3: Real environment (real bathy and ssp)",
    ):
        super().__init__(name, testcase_varin, title)
        self.range_dependence = True


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
        "rho": 1.9 * RHO_W / 1000,
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
    plot_env=False,
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
    plot_env_properties(env, plot_medium, plot_bottom, plot_env)

    return env, flp


def testcase2_1(
    testcase_varin,
    plot_bathy=False,
    plot_medium=False,
    plot_bottom=False,
    plot_env=False,
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
    plot_env_properties(env, plot_medium, plot_bottom, plot_env)

    return env, flp


def testcase2_2(
    testcase_varin,
    plot_bathy=False,
    plot_medium=False,
    plot_bottom=False,
    plot_env=False,
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
    plot_env_properties(env, plot_medium, plot_bottom, plot_env)

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
        "rho": 1.9 * RHO_W / 1000,
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
    plot_env=False,
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
    plot_env_properties(env, plot_medium, plot_bottom, plot_env)

    return env, flp


if __name__ == "__main__":

    # Test class
    # tc1_0 = TestCase1_0(mode="show")
    # tc1_1 = TestCase1_1(mode="show")
    # tc1_2 = TestCase1_2(mode="show")
    # tc1_3 = TestCase1_3(mode="show")
    tc1_ = TestCase1_4(mode="show")

    print()

    # # Test case 1.0
    # tc_varin = {"freq": [20], "max_range_m": 50 * 1e3}
    # env, flp = testcase1_0(
    #     tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True, plot_env=True
    # )
    # env.write_env()
    # flp.write_flp()

    # # Test case 1.1
    # tc_varin = {"freq": [20], "min_depth": 150}
    # env, flp = testcase1_1(
    #     tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True, plot_env=True
    # )
    # env.write_env()
    # flp.write_flp()

    # # Test case 1.2
    # tc_varin = {"freq": [20], "min_depth": 100}
    # env, flp = testcase1_2(
    #     tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True, plot_env=True
    # )
    # env.write_env()
    # flp.write_flp()

    # # Test case 1.3
    # tc_varin = {"freq": [20], "max_range_m": 50 * 1e3}
    # env, flp = testcase1_3(
    #     tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True, plot_env=True
    # )
    # env.write_env()
    # flp.write_flp()

    # # Test case 1.4
    # tc_varin = {"freq": [20], "max_range_m": 50 * 1e3}
    # env, flp = testcase1_4(
    #     tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True, plot_env=True
    # )
    # env.write_env()
    # flp.write_flp()

    # # Test case 2.0
    # tc_varin = {
    #     "freq": [20],
    #     "max_range_m": 15 * 1e3,
    # }
    # for az in range(0, 360, 30):
    #     tc_varin["azimuth"] = az
    #     env, flp = testcase2_0(
    #         tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True, plot_env=True
    #     )
    # env.write_env()
    # flp.write_flp()

    # # Test case 2.1
    # tc_varin = {
    #     "freq": [20],
    #     "max_range_m": 15 * 1e3,
    #     "azimuth": 0,
    # }
    # for az in range(0, 360, 30):
    #     tc_varin["azimuth"] = az
    #     env, flp = testcase2_1(
    #         tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True, plot_env=True
    #     )
    # env.write_env()
    # flp.write_flp()

    # # Test case 2.2
    # tc_varin = {
    #     "freq": [20],
    #     "max_range_m": 15 * 1e3,
    #     "azimuth": 0,
    #     "rcv_lon": -4.7,
    #     "rcv_lat": 52.6,
    # }
    # for az in range(0, 360, 30):
    #     tc_varin["azimuth"] = az
    #     env, flp = testcase2_2(
    #         tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True, plot_env=True
    #     )
    # env.write_env()
    # flp.write_flp()

    # # Test case 3.1
    # tc_varin = {
    #     "freq": [20],
    #     "max_range_m": 15 * 1e3,
    #     "azimuth": 0,
    #     "rcv_lon": 65.943,
    #     "rcv_lat": -27.5792,
    # }
    # for az in range(0, 360, 30):
    #     tc_varin["azimuth"] = az
    #     env, flp = testcase3_1(
    #         tc_varin, plot_bathy=True, plot_medium=True, plot_bottom=True, plot_env=True
    #     )
    # env.write_env()
    # flp.write_flp()
