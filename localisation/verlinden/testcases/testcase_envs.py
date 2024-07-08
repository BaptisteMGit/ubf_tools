import os
import socket
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from cst import SAND_PROPERTIES, RHO_W
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
from propa.kraken_toolbox.run_kraken import get_subprocess_working_dir

from localisation.verlinden.testcases.testcase_bathy import (
    bathy_flat_seabed,
    bathy_sin_slope,
    bathy_seamount,
    mmdpm_profile,
    extract_2D_bathy_profile,
)
from localisation.verlinden.misc.params import (
    TC_WORKING_DIR,
    PROJECT_ROOT,
    BATHY_FILENAME,
)

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
        self.name = f"{name}_{socket.gethostname()}"
        # self.name = name

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
        self.dr_flp = None
        self.dr_bathy = None
        self.nb_modes = None
        self.plot_medium = False
        self.plot_bottom = False
        self.plot_bathy = False
        self.plot_env = False
        # Src
        self.z_src = None
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

        # Flp config
        self.flp_n_rcv_r = None
        self.flp_n_rcv_z = None
        self.flp_rcv_z_min = None
        self.flp_rcv_z_max = None
        self.mode_theory = None
        self.mode_addition = None

        # Environment directory
        self.env_dir = ""
        # Max depth
        self.max_depth = None
        self.src_depth = None

        # Flag for range-dependence and isotropic/anisotropic
        self.range_dependence = False
        self.isotropic = True
        self.run_parallel = True
        self.called_by_subprocess = False

        # Default values
        self.default_varin = {
            "freq": [20],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
            "azimuth": 0,
            "rcv_lon": -4.87,
            "rcv_lat": 52.22,
            "dr_flp": 50,
            "dr_bathy": 500,
            "nb_modes": 100,
            "called_by_subprocess": False,
            "mode_theory": "adiabatic",
            "mode_addition": "coherent",
            "z_src": 5,
            "src_depth": None,
            "flp_n_rcv_z": None,
            "flp_rcv_z_min": None,
            "flp_rcv_z_max": None,
        }

        # Set env directory
        self.set_env_dir()

        # Bottom properties (1 layer bottom)
        self.bott_hs_properties = {
            "rho": 1.9 * RHO_W * 1e-3,  # Density (g/cm^3)
            "c_p": 1650,  # P-wave celerity (m/s)
            "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
            "a_p": 0.8,  # Compression wave attenuation (dB/wavelength)
            "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
            "z": None,
        }

        # self.process()

    def process(self):
        # Read variables
        self.read_varin()
        # Load testcase
        self.load()
        # Write flp and env files
        self.write_kraken_files()
        # Plot env
        self.plot_testcase_env()

    def update(self, varin):
        self.testcase_varin.update(varin)
        self.process()

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

        if self.isotropic and not self.range_dependence:
            self.run_parallel = True
        else:
            self.run_parallel = False

    def plot_testcase_env(self):
        plot_env_properties(
            self.env,
            plot_medium=self.plot_medium,
            plot_bottom=self.plot_bottom,
            plot_env=self.plot_env,
        )

    def load(self):
        # Load bathy
        self.write_bathy()
        self.set_bathy()
        # Load ssp
        self.load_ssp()
        # Set kraken objects
        self.set_top_hs()
        self.set_medium()
        self.set_att()
        self.set_bott_hs()
        self.set_field()
        self.set_env()
        # Write env to set dependent to true
        self.env.write_env()
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
        self.bott_hs.derive_sedim_layer_max_depth(z_max=self.max_depth)

    def set_att(self):
        self.att = KrakenAttenuation(
            units="dB_per_wavelength", use_volume_attenuation=False
        )

    def set_field(self):
        z_max = np.ceil(self.bott_hs.sedim_layer_max_depth + 5)
        n_rcv_z = default_nb_rcv_z(max(self.freq), z_max, n_per_l=12)

        c_offset = 200
        c_low = min(
            np.min(self.cp_ssp), np.min(self.bott_hs_properties["c_p"])
        )  # Minimum p-wave speed in the problem as recommanded by Kraken manual to exclude interfacial waves
        c_low = np.max(c_low - c_offset, 0)
        c_high = (
            np.max(self.bott_hs_properties["c_p"]) + c_offset
        )  # Maximum p-wave speed in the bottom to limit the number of modes computed

        self.field = KrakenField(
            n_rcv_z=n_rcv_z,
            src_depth=self.src_depth,
            rcv_z_max=z_max,
            phase_speed_limits=[c_low, c_high],
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
            called_by_subprocess=self.called_by_subprocess,
        )

    def get_bathy_path(self):
        if self.called_by_subprocess:
            bathy_dir = get_subprocess_working_dir(
                env_root=os.path.join(TC_WORKING_DIR, self.name), worker_pid=os.getpid()
            )
        else:
            bathy_dir = os.path.join(TC_WORKING_DIR, self.name)

        return os.path.join(bathy_dir, "bathy.csv")

    def set_bathy(self):
        self.bathy = Bathymetry(self.get_bathy_path())
        self.max_depth = self.bathy.bathy_depth.max()
        self.min_depth = self.bathy.bathy_depth.min()
        if self.src_depth is None:
            self.src_depth = (
                self.max_depth - 1
            )  # Assume hydrophone is 1m above the bottom
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
        self.flp_n_rcv_r = self.max_range_m / self.dr_flp + 1

        # # Source = ship radiating sound at 5m depth
        if self.flp_n_rcv_z is None:
            self.flp_n_rcv_z = 1
        if self.flp_rcv_z_min is None:
            self.flp_rcv_z_min = 5
        if self.flp_rcv_z_max is None:
            self.flp_rcv_z_max = 5

        # self.flp_n_rcv_z = default_nb_rcv_z(
        #     max(self.freq), self.bott_hs.sedim_layer_max_depth, n_per_l=12
        # )
        # self.flp_rcv_z_min = 0
        # self.flp_rcv_z_max = self.bott_hs.sedim_layer_max_depth

        self.flp = KrakenFlp(
            env=self.env,
            src_depth=self.src_depth,
            n_rcv_z=self.flp_n_rcv_z,
            rcv_z_min=self.flp_rcv_z_min,
            rcv_z_max=self.flp_rcv_z_max,
            rcv_r_max=self.max_range_m * 1e-3,
            n_rcv_r=self.flp_n_rcv_r,
            nb_modes=self.nb_modes,
            mode_addition=self.mode_addition,
            mode_theory=self.mode_theory,
        )

    def write_kraken_files(self):
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
        tc_default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
            "dr_flp": 50,
            "nb_modes": 100,
        }
        for key, default_value in tc_default_varin.items():
            self.default_varin[key] = default_value

        # Flat bottom
        self.range_dependence = False

        # Process all info
        self.process()


class TestCase1_1(TestCase1):
    def __init__(self, testcase_varin={}, mode="prod"):
        name = "testcase1_1"
        title = "Test case 1.1: Isotopric environment with sinusoidal bottom. 1 layer bottom and constant sound speed profile"
        desc = "Environment: isotopric, Bathymetry: sinusoidal bottom, SSP: c = 1500 m/s, Sediment: One layer bottom with constant properties"

        super().__init__(
            name, testcase_varin=testcase_varin, title=title, desc=desc, mode=mode
        )

        # Update default values with values testcase specific values
        tc_default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
            "dr_flp": 50,
            "dr_bathy": 500,
            "nb_modes": 100,
        }
        for key, default_value in tc_default_varin.items():
            self.default_varin[key] = default_value

        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process()

    def write_bathy(self):
        # Create a slope bottom
        bathy_sin_slope(
            testcase_name=self.name,
            min_depth=self.min_depth,
            max_range=self.max_range_m * 1e-3,
            theta=94,
            dr=self.dr_bathy,
            range_periodicity=6,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy"),
            called_by_subprocess=self.called_by_subprocess,
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
        tc_default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
            "dr_flp": 50,
            "dr_bathy": 500,
            "nb_modes": 100,
        }
        for key, default_value in tc_default_varin.items():
            self.default_varin[key] = default_value

        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process()

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
            called_by_subprocess=self.called_by_subprocess,
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
        tc_default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
            "dr_flp": 50,
            "dr_bathy": 500,
            "nb_modes": 100,
        }
        for key, default_value in tc_default_varin.items():
            self.default_varin[key] = default_value
        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process()

    def write_bathy(self):
        # Load real profile around OBS RR48
        mmdpm_profile(
            testcase_name=self.name,
            mmdpm_testname="PVA_RR48",
            azimuth=360,
            max_range_km=self.max_range_m * 1e-3,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy"),
            called_by_subprocess=self.called_by_subprocess,
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
        tc_default_varin = {
            "freq": [20],
            "max_range_m": 50 * 1e3,
            "dr_flp": 50,
            "dr_bathy": 500,
            "nb_modes": 100,
        }
        for key, default_value in tc_default_varin.items():
            self.default_varin[key] = default_value
        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process()

    def write_bathy(self):
        # Load bathy profile
        mmdpm_profile(
            testcase_name=self.name,
            mmdpm_testname="PVA_RR48",
            azimuth=360,
            max_range_km=self.max_range_m * 1e-3,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy"),
            called_by_subprocess=self.called_by_subprocess,
        )

    def load_ssp(self):
        # Load ssp mat file
        data_dir = os.path.join(PROJECT_ROOT, "data", "ssp", "mmdpm")
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


class TestCase2_0(TestCase2):
    def __init__(self, testcase_varin={}, mode="prod"):
        name = "testcase2_0"
        title = "Test case 2.0: Anisotropic environment with flat bottom. 1 layer bottom and constant sound speed profile"
        desc = "Environment: Anisotropic, Bathymetry: flat bottom, SSP: c = 1500 m/s, Sediment: One layer bottom with constant properties"
        super().__init__(
            name, testcase_varin=testcase_varin, title=title, desc=desc, mode=mode
        )

        # Update default values with values testcase specific values
        tc_default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
            "dr_flp": 50,
            "dr_bathy": 500,
            "nb_modes": 100,
        }
        for key, default_value in tc_default_varin.items():
            self.default_varin[key] = default_value
        # Flat bottom
        self.range_dependence = False

        # Process all info
        self.process()


class TestCase2_1(TestCase2):
    def __init__(self, testcase_varin={}, mode="prod"):
        name = "testcase2_1"
        title = "Test case 2.1:  Shallow water environment with sinusoidal bathy profile (same as test case 1.1). 1 layer bottom and constant sound speed profile"
        desc = "Environment: Anisotropic, Bathymetry: sinusoidal bottom, SSP: c = 1500 m/s, Sediment: One layer bottom with constant properties"

        super().__init__(
            name, testcase_varin=testcase_varin, title=title, desc=desc, mode=mode
        )

        # Update default values with values testcase specific values
        tc_default_varin = {
            "freq": [25],
            "max_range_m": 50 * 1e3,
            "min_depth": 100,
            "azimuth": 0,
            "dr_flp": 50,
            "dr_bathy": 500,
            "nb_modes": 100,
        }
        for key, default_value in tc_default_varin.items():
            self.default_varin[key] = default_value
        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process()

    def write_bathy(self):
        # Create a slope bottom
        bathy_sin_slope(
            testcase_name=self.name,
            min_depth=self.min_depth,
            max_range=self.max_range_m * 1e-3,
            theta=self.azimuth,
            range_periodicity=6,
            dr=self.dr_bathy,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy", azimuth=self.azimuth),
            called_by_subprocess=self.called_by_subprocess,
        )


class TestCase2_2(TestCase2):
    def __init__(self, testcase_varin={}, mode="prod"):
        name = "testcase2_2"
        title = "Test case 2.2: Anisotropic shallow water environment with real bathy profile extracted from GEBCO 2021 grid. 1 layer bottom and realistic sound speed profile"
        desc = "Environment: Anisotropic, Bathymetry: real bathy profile extractedfrom GEBCO 2021 grid, SSP: Realistic sound speed profile  (from Copernicus Marine Service), Sediment: One layer bottom with constant properties"

        super().__init__(
            name, testcase_varin=testcase_varin, title=title, desc=desc, mode=mode
        )

        # Update default values with values testcase specific values
        tc_default_varin = {
            "freq": [20],
            "max_range_m": 50 * 1e3,
            "azimuth": 0,
            "rcv_lon": -4.87,
            "rcv_lat": 52.22,
            "dr_flp": 50,
            "dr_bathy": 500,
            "nb_modes": 100,
        }
        for key, default_value in tc_default_varin.items():
            self.default_varin[key] = default_value
        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process()

    def write_bathy(self):
        bathy_nc_path = os.path.join(
            PROJECT_ROOT, "data", "bathy", "mmdpm", "PVA_RR48", BATHY_FILENAME
        )

        # Load real profile around OBS RR48
        extract_2D_bathy_profile(
            bathy_nc_path=bathy_nc_path,
            testcase_name=self.name,
            obs_lon=self.rcv_lon,
            obs_lat=self.rcv_lat,
            azimuth=self.azimuth,
            max_range_km=self.max_range_m * 1e-3,
            range_resolution=self.dr_bathy,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy", azimuth=self.azimuth),
            called_by_subprocess=self.called_by_subprocess,
        )

    def load_ssp(self):
        # Load ssp mat file
        data_dir = os.path.join(PROJECT_ROOT, "data", "ssp", "mmdpm")
        fpath = os.path.join(
            data_dir,
            "shallow_water",
            f"july_lon_-5.87_-2.87_lat_50.72_53.72_ssp.mat",
        )
        ssp_mat = sio.loadmat(fpath)
        self.z_ssp = ssp_mat["ssp"]["z"][0, 0].flatten()
        self.cp_ssp = ssp_mat["ssp"]["c"][0, 0].flatten()


##########################################################################################
# Test case 3 : Anisotropic and deep whater environment (RHUM RUM)
##########################################################################################


class TestCase3(TestCase):

    def __init__(
        self,
        name,
        testcase_varin={},
        title="Test case 3: Real environment (real bathy and ssp)",
        desc="",
        mode="prod",
    ):
        super().__init__(name, testcase_varin, title, desc, mode)
        self.range_dependence = True
        self.isotropic = False


class TestCase3_1(TestCase3):
    def __init__(self, testcase_varin={}, mode="prod"):
        name = "testcase3_1"
        title = "Test case 3.1: Anisotropic environment with real bathy profile extracted from GEBCO 2021 global grid around OBS RR48. 1 layer bottom and realistic sound speed profile"
        desc = "Environment: Anisotropic, Bathymetry: real bathy profile around OBS RR48 (extracted from GEBCO 2021 grid), SSP: Realistic sound speed profile  (from Copernicus Marine Service), Sediment: One layer bottom with constant properties"
        super().__init__(
            name, testcase_varin=testcase_varin, title=title, desc=desc, mode=mode
        )

        # Update default values with values testcase specific values
        tc_default_varin = {
            "freq": [20],
            "max_range_m": 50 * 1e3,
            "azimuth": 0,
            "rcv_lon": 65.94,
            "rcv_lat": -27.58,
            "dr_flp": 50,
            "dr_bathy": 1000,
            "nb_modes": 100,
            "called_by_subprocess": False,
            "mode_theory": "coupled",
            # "mode_theory": "adiabatic",
        }
        for key, default_value in tc_default_varin.items():
            self.default_varin[key] = default_value

        # Flat bottom
        self.range_dependence = True

        # Process all info
        self.process()

    def write_bathy(self):
        # bathy_nc_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\bathy\mmdpm\PVA_RR48\GEBCO_2021_lon_64.44_67.44_lat_-29.08_-26.08.nc"
        bathy_nc_path = os.path.join(
            PROJECT_ROOT, "data", "bathy", "mmdpm", "PVA_RR48", BATHY_FILENAME
        )

        # Load real profile around OBS RR48
        extract_2D_bathy_profile(
            bathy_nc_path=bathy_nc_path,
            testcase_name=self.name,
            obs_lon=self.rcv_lon,
            obs_lat=self.rcv_lat,
            azimuth=self.azimuth,
            max_range_km=self.max_range_m * 1e-3,
            range_resolution=self.dr_bathy,
            plot=self.plot_bathy,
            bathy_path=get_img_path(self.name, type="bathy", azimuth=self.azimuth),
            called_by_subprocess=self.called_by_subprocess,
        )

    def load_ssp(self):
        # Load ssp mat file
        data_dir = os.path.join(PROJECT_ROOT, "data", "ssp", "mmdpm")
        fpath = os.path.join(data_dir, "PVA_RR48", f"mmdpm_test_PVA_RR48_ssp.mat")
        ssp_mat = sio.loadmat(fpath)
        self.z_ssp = ssp_mat["ssp"]["z"][0, 0].flatten()
        self.cp_ssp = ssp_mat["ssp"]["c"][0, 0].flatten()


if __name__ == "__main__":

    # Test class
    # tc1_0 = TestCase1_0(mode="show")
    # tc1_1 = TestCase1_1(mode="show")
    # tc1_2 = TestCase1_2(mode="show")
    # tc1_3 = TestCase1_3(mode="show")
    # tc1_4 = TestCase1_4(mode="show")
    # tc2_0 = TestCase2_0(mode="show")

    tc_varin = {
        "freq": [20],
        "max_range_m": 15 * 1e3,
        "azimuth": 0,
        "rcv_lon": 65.943,
        "rcv_lat": -27.5792,
        "mode_theory": "coupled",
    }
    tc2_1 = TestCase2_1(mode="show", testcase_varin=tc_varin)
    tc2_2 = TestCase2_2(mode="show", testcase_varin=tc_varin)
    tc3_1 = TestCase3_1(mode="show", testcase_varin=tc_varin)

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

    # Test case 3.1
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
