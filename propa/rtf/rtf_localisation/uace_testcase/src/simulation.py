#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   simulation.py
@Time    :   2025/05/06 21:51:22
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to handle the simulation properties
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import shutil

# import inspect

import numpy as np
import xarray as xr

from propa.rtf.rtf_localisation.uace_testcase.src.antenna import Antenna, SparseAntenna
from propa.rtf.rtf_localisation.uace_testcase.src.acoustic_source import (
    AcousticSource,
    Ship,
)

from propa.kraken_toolbox.src.kraken_env import KrakenEnv, KrakenFlp

# import source.global_constants as g
# from source.signal_generator import SignalGenerator
import propa.rtf.rtf_localisation.uace_testcase.src.params as p

# from propa.kraken_toolbox.src.kraken_manager import KrakenManager
# from misc import cast_matrix_to_target_shape, mult_along_axis


class Simulation:
    """
    Class to handle the simulation properties
    """

    def __init__(
        self,
        name: str = p.name,
        root_img: str = p.root_img,
        root_tmp: str = p.root_tmp,
        root_data: str = p.root_data,
        fmin: float = p.fmin,
        fmax: float = p.fmax,
        fs: float = p.fs,
        signal_duration: float = p.duration,
        antenna: Antenna = p.antenna,
        library_ship: np.ndarray[Ship] = p.library_ship,
        event_ship: Ship = p.event_ship,
        event_ship_x: float = p.event_ship_x,
        event_ship_y: float = p.event_ship_y,
        event_ship_z: float = p.event_ship_z,
        interferer: AcousticSource = None,
        sir: float = None,
        dx: float = p.dx,
        dy: float = p.dy,
        search_area_length: float = p.search_area_length,
        cmin: float = p.cmin,
        monte_carlo_iterations: int = p.monte_carlo_iterations,
        frequency_drawing_method: str = p.frequency_drawing_method,
        number_of_drawn_frequencies: int = p.number_of_drawn_frequencies,
        use_weighted_rtf: bool = p.use_weighted_rtf,
        kraken_env: KrakenEnv = None,
        kraken_flp: KrakenFlp = None,
        feature_nperseg: int = p.nperseg,
        feature_overlap_ratio: float = p.alpha_overlap,
        check_features: bool = False,
        plot_library_ship_distribution: bool = True,
        debug: bool = False,
        verbose: bool = False,
    ):
        """
        Constructor
        """
        self.name = name
        self.root_img = root_img
        self.root_tmp = root_tmp
        self.root_data = root_data
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.signal_duration = signal_duration

        # Window length used to derive features
        self.feature_nperseg = feature_nperseg
        self.feature_overlap_ratio = feature_overlap_ratio
        self.feature_noverlap = int(feature_nperseg * feature_overlap_ratio)

        self.antenna = antenna
        self.library_ship = np.atleast_1d(library_ship)
        self.event_ship = event_ship
        self.library_stype = "ship"
        self.event_stype = "ship"

        # Usefull flags
        self.use_weighted_rtf = use_weighted_rtf
        self.debug = debug
        self.verbose = verbose
        self.check_features = check_features
        self.plot_library_ship_distribution = plot_library_ship_distribution

        # Position of the event ship to localize
        self.event_ship_x = event_ship_x
        self.event_ship_y = event_ship_y
        self.event_ship_z = event_ship_z

        # Interferer
        self.interferer = interferer
        # Signal to interference ratio
        self.sir = sir

        # Grid properties
        self.dx = dx
        self.dy = dy
        self.grid_res_label = f"dx{dx}m_dy{dy}m"
        self.search_area_length = search_area_length

        # Init grid properties
        self.grid_x = None
        self.grid_y = None
        self.grid_rmax = None
        self.grid_rmin = None
        self.grid_ranges_from_rcv = None

        # Environment properties
        self.cmin = cmin

        # Localization params
        self.monte_carlo_iterations = monte_carlo_iterations
        self.frequency_drawing_method = frequency_drawing_method
        self.number_of_drawn_frequencies = number_of_drawn_frequencies

        # Kraken env and flp (only need to be provided for rd simulations)
        self.kraken_env = kraken_env
        self.kraken_flp = kraken_flp
        self._is_range_dependent = None

        # Paths
        self._env_file = None
        self._data_folder = None
        self._img_folder = None
        self._tmp_folder = None

        self.init()

    # =======================================================================================================================
    # Initialize data builder
    # =======================================================================================================================
    def init(self):
        """Initialize data builder"""
        # Set debug config
        if self.debug:
            self.set_debug_config()

        # Set file paths
        self.init_file_paths()

        # Set grid
        self.init_grid()

        # Write logs
        self.write_logs()

    @property
    def env_file(self):
        self._env_file = os.path.join(self.tmp_folder, f"{self.name}.env")
        return self._env_file

    @env_file.setter
    def env_file(self, value):
        """
        Set path to kraken env file
        :param value: path
        """
        self._env_file = value

    @property
    def data_folder(self):
        self._data_folder = os.path.join(self.root_data, self.name)
        return self._data_folder

    @data_folder.setter
    def data_folder(self, value):
        """
        Path to data folder
        :param value: path
        """
        self._data_folder = value

    @property
    def img_folder(self):
        self._img_folder = os.path.join(self.root_img, self.name)
        return self._img_folder

    @img_folder.setter
    def img_folder(self, value):
        """
        Path to the img folder
        :param value: path
        """
        self._img_folder = value

    @property
    def tmp_folder(self):
        self._tmp_folder = os.path.join(self.root_tmp, self.name, "io_files")
        return self._tmp_folder

    @tmp_folder.setter
    def tmp_folder(self, value):
        """
        Path to the tmp folder
        :param value: path
        """
        self._tmp_folder = value

    @property
    def is_range_dependent(self):
        """
        Return True if the environment is range dependent
        :return: bool
        """
        if self.kraken_env is None:
            return False
        else:
            return self.kraken_env.range_dependent_env

    @is_range_dependent.setter
    def is_range_dependent(self, value):
        """
        Set the range dependent flag
        :param value: bool
        """
        self._is_range_dependent = value

    def init_file_paths(self):

        # Create folders if they do not exist
        for folder in [self.tmp_folder, self.data_folder, self.img_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # tf dataset
        self.tf_dataset_fpath = os.path.join(self.data_folder, f"tf.nc")

        # Gridded tf dataset
        if self.debug:
            tf_grid_dataset_fname = f"tf_grid_{self.grid_res_label}_debug"
        else:
            tf_grid_dataset_fname = f"tf_grid_{self.grid_res_label}"
        self.tf_grid_dataset_fpath = os.path.join(
            self.data_folder, tf_grid_dataset_fname + ".nc"  #
        )

        # Library dataset
        if self.debug:
            library_dataset_fname = f"library_{self.grid_res_label}_debug"
        else:
            library_dataset_fname = f"library_{self.grid_res_label}"

        self.library_dataset_fpath = os.path.join(
            self.data_folder, library_dataset_fname + ".nc"
        )

        # Feature dataset
        if self.debug:
            feature_dataset_fname = f"features_{self.grid_res_label}_debug"
        else:
            feature_dataset_fname = f"features_{self.grid_res_label}"

        self.feature_dataset_fpath = os.path.join(
            self.data_folder, feature_dataset_fname + ".nc"
        )

        # Rtf weights dataset
        if self.debug:
            rtf_weights_dataset_fname = f"rtf_weights_{self.grid_res_label}_debug"
        else:
            rtf_weights_dataset_fname = f"rtf_weights_{self.grid_res_label}"
        self.rtf_weights_dataset_fpath = os.path.join(
            self.data_folder, rtf_weights_dataset_fname + ".nc"
        )

        # Feature kraken dataset
        if self.debug:
            kraken_feature_dataset_fname = (
                f"kraken_features_{self.grid_res_label}_debug"
            )
        else:
            kraken_feature_dataset_fname = f"kraken_features_{self.grid_res_label}"

        self.kraken_feature_dataset_fpath = os.path.join(
            self.data_folder, kraken_feature_dataset_fname + ".nc"
        )

        # Localization dataset
        if self.debug:
            localization_dataset_fname = f"localization_{self.grid_res_label}_debug"
        else:
            localization_dataset_fname = f"localization_{self.grid_res_label}"
        self.localization_dataset_fname = localization_dataset_fname
        self.localization_dataset_fpath = os.path.join(
            self.data_folder, localization_dataset_fname + ".nc"
        )

        self.from_sig_foldername = f"from_signal_{self.grid_res_label}"
        # Folder to store img derived from signal
        self.root_img_from_sig = os.path.join(self.img_folder, self.from_sig_foldername)

    def init_grid(self):
        # Derive range from each receiver
        r_src_rcv = np.sqrt(
            (self.event_ship_x - self.antenna.x) ** 2
            + (self.event_ship_y - self.antenna.y) ** 2
        )
        self.event_ship_ranges_from_rcv = r_src_rcv  # (nrcv,)

        # Set coordinates of the left bottom corner of the grid
        x_bott_left_corner = self.event_ship_x - self.search_area_length / 2
        y_bott_left_corner = self.event_ship_y - self.search_area_length / 2
        # Round to the nearest grid point
        x_bott_left_corner = np.round(x_bott_left_corner / self.dx) * self.dx
        y_bott_left_corner = np.round(y_bott_left_corner / self.dy) * self.dy

        # Define grid
        x_search_area = np.arange(
            x_bott_left_corner,
            x_bott_left_corner + self.search_area_length + self.dx,
            self.dx,
        )
        y_search_area = np.arange(
            y_bott_left_corner,
            y_bott_left_corner + self.search_area_length + self.dy,
            self.dy,
        )
        x_grid, y_grid = np.meshgrid(x_search_area, y_search_area)  # x_grid = (ny, nx)
        r_grid = np.zeros((self.antenna.n_elements,) + x_grid.shape)  # (nr, ny, nx)
        for i in range(self.antenna.n_elements):
            r_grid[i] = np.sqrt(
                (x_grid - self.antenna.x[i]) ** 2 + (y_grid - self.antenna.y[i]) ** 2
            )

        # Set grid parameters
        self.grid_rmin = np.floor(np.min(r_grid) * 1e-3) * 1e3
        self.grid_rmax = np.ceil(np.max(r_grid) * 1e-3) * 1e3
        self.grid_x = x_search_area
        self.grid_y = y_search_area
        self.grid_ranges_from_rcv = r_grid

    def set_debug_config(self):
        """Set debug config for the test case"""

        # Set debug parameters
        self.search_area_length = 200
        self.monte_carlo_iterations = 2

    def write_logs(self, filename: str = "simulation_config.txt"):
        """
        Logs structured simulation properties to a formatted text file.

        Parameters
        ----------
        filename : str
            Name of the output text file (placed in tmp_folder).
        """

        log_path_data = os.path.join(self.data_folder, filename)

        def safe_serialize(value):
            if isinstance(value, (int, float, str, bool, type(None))):
                return value
            elif isinstance(value, np.ndarray):
                return f"ndarray, shape={value.shape}, dtype={value.dtype}"
            elif hasattr(value, "__class__"):
                return f"{value.__class__.__name__}"
            else:
                return str(value)

        def format_array(arr):
            if arr is None:
                return "None"
            if isinstance(arr, np.ndarray):
                return np.array2string(arr, separator=", ", threshold=5)
            return str(arr)

        sections = {
            "General Settings": [
                "name",
                "root_img",
                "root_tmp",
                "root_data",
                "fs",
                "fmin",
                "fmax",
                "signal_duration",
            ],
            "Feature Extraction": [
                "feature_nperseg",
                "feature_overlap_ratio",
                "feature_noverlap",
                "check_features",
            ],
            "Environment & Grid": [
                "dx",
                "dy",
                "search_area_length",
                "cmin",
                "grid_rmax",
            ],
            "Simulation Parameters": [
                "monte_carlo_iterations",
                "frequency_drawing_method",
                "number_of_drawn_frequencies",
                "use_weighted_rtf",
                "plot_library_ship_distribution",
                "debug",
                "verbose",
            ],
            "Event Ship Configuration": [
                "event_ship_x",
                "event_ship_y",
                "event_ship_z",
                "event_stype",
                "library_stype",
            ],
            "Environment Modules": ["kraken_env", "kraken_flp", "is_range_dependent"],
            "Paths": [
                "data_folder",
                "img_folder",
                "tmp_folder",
                "tf_dataset_fpath",
                "tf_grid_dataset_fpath",
                "library_dataset_fpath",
                "feature_dataset_fpath",
                "rtf_weights_dataset_fpath",
                "kraken_feature_dataset_fpath",
                "localization_dataset_fpath",
                "root_img_from_sig",
            ],
        }

        with open(log_path_data, "w") as f:
            f.write("SIMULATION CONFIGURATION LOG\n")
            f.write("=" * 50 + "\n\n")

            for section_title, keys in sections.items():
                f.write(f"[{section_title}]\n")
                f.write("-" * (len(section_title) + 2) + "\n")
                for key in keys:
                    val = getattr(self, key, "N/A")
                    f.write(f"{key}: {safe_serialize(val)}\n")
                f.write("\n")

            # Dedicated Antenna Section
            f.write("[Antenna Configuration]\n")
            f.write("------------------------\n")
            if self.antenna is not None:
                f.write(f"name: {self.antenna.name}\n")
                f.write(f"n_elements: {self.antenna.n_elements}\n")
                f.write(
                    f"antenna_radius: {safe_serialize(self.antenna.antenna_radius)}\n"
                )
                f.write(f"x: {format_array(self.antenna.x)}\n")
                f.write(f"y: {format_array(self.antenna.y)}\n")
                f.write(f"rcv_idx: {format_array(self.antenna.rcv_idx)}\n")
            else:
                f.write("No antenna configuration provided.\n")
            f.write("\n")

            # Event Ship Signal Section
            f.write("[Event Ship Signal]\n")
            f.write("-------------------\n")
            if self.event_ship is not None:
                s = self.event_ship
                f.write(f"name: {s.name}\n")
                f.write(f"f0 (Hz): {s.f0}\n")
                f.write(f"fs (Hz): {s.fs}\n")
                f.write(f"duration (s): {s.duration}\n")
                f.write(f"std_fi: {s.std_fi}\n")
                f.write(f"tau_corr_fi: {s.tau_corr_fi}\n")
                f.write(f"n_samples: {s.n_samples}\n")
                f.write(
                    f"signal: {'available' if s.signal is not None else 'not generated'}\n"
                )
                f.write(
                    f"spectrum: {'available' if s.spectrum is not None else 'not generated'}\n"
                )
            else:
                f.write("No event ship signal provided.\n")
            f.write("\n")

            # Ship Library Section
            f.write("[Ship Signal Library]\n")
            f.write("---------------------\n")
            if self.library_ship.size > 0:
                for i, s in enumerate(self.library_ship):
                    f.write(f"--- Ship #{i + 1} ---\n")
                    f.write(f"name: {s.name}\n")
                    f.write(f"f0 (Hz): {s.f0}\n")
                    f.write(f"fs (Hz): {s.fs}\n")
                    f.write(f"duration (s): {s.duration}\n")
                    f.write(f"std_fi: {s.std_fi}\n")
                    f.write(f"tau_corr_fi: {s.tau_corr_fi}\n")
                    f.write(f"n_samples: {s.n_samples}\n")
                    f.write(
                        f"signal: {'available' if s.signal is not None else 'not generated'}\n"
                    )
                    f.write(
                        f"spectrum: {'available' if s.spectrum is not None else 'not generated'}\n"
                    )
                    f.write("\n")
            else:
                f.write("No ship library signals provided.\n\n")

            # Optional log footer
            f.write("=" * 50 + "\n")
            f.write("End of simulation log\n")

        # Copy the log file to the img folder
        log_path_img = os.path.join(self.img_folder, filename)
        shutil.copy(log_path_data, log_path_img)

        if self.verbose:
            print(f"Structured simulation properties logged to {log_path_data}")


if __name__ == "__main__":
    name = "test_simulation_class"
    simu = Simulation(name=name)
