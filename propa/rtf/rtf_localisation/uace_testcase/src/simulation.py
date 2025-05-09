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
import numpy as np
import xarray as xr

from propa.rtf.rtf_localisation.uace_testcase.src.antenna import Antenna, SparseAntenna

import source.global_constants as g
from source.signal_generator import SignalGenerator
import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from misc import cast_matrix_to_target_shape, mult_along_axis


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
        library_ship: str = p.library_ship,
        event_ship: str = p.event_ship,
        event_ship_x: float = p.event_ship_x,
        event_ship_y: float = p.event_ship_y,
        event_ship_z: float = p.event_ship_z,
        dx: float = p.dx,
        dy: float = p.dy,
        search_area_length: float = p.search_area_length,
        cmin: float = p.cmin,
        monte_carlo_iterations: int = p.monte_carlo_iterations,
        frequency_drawing_method: str = p.frequency_drawing_method,
        number_of_drawn_frequencies: int = p.number_of_drawn_frequencies,
        check_features: bool = False,
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

        self.antenna = antenna
        self.library_ship = library_ship
        self.event_ship = event_ship
        self.library_stype = "ship"
        self.event_stype = "ship"

        # Usefull flags
        self.debug = debug
        self.verbose = verbose
        self.check_features = check_features

        # Position of the event ship to localize
        self.event_ship_x = event_ship_x
        self.event_ship_y = event_ship_y
        self.event_ship_z = event_ship_z

        # Grid properties
        self.dx = dx
        self.dy = dy
        self.grid_res_label = f"dx{dx}m_dy{dy}m"
        self.search_area_length = search_area_length

        # Init grid properties
        self.grid_x = None
        self.grid_y = None
        self.grid_rmax = None
        self.grid_ranges_from_rcv = None

        # Environment properties
        self.cmin = cmin

        # Localization params
        self.monte_carlo_iterations = monte_carlo_iterations
        self.frequency_drawing_method = frequency_drawing_method
        self.number_of_drawn_frequencies = number_of_drawn_frequencies

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

        # Kraken env file
        self.env_file = os.path.join(self.root_tmp, f"{self.name}.env")

    def init_file_paths(self):
        # tf dataset
        self.tf_dataset_fpath = os.path.join(self.root_data, f"{self.name}_tf.nc")

        # Gridded tf dataset
        if self.debug:
            tf_grid_dataset_fname = f"{self.name}_tf_grid_{self.grid_res_label}_debug"
        else:
            tf_grid_dataset_fname = f"{self.name}_tf_grid_{self.grid_res_label}"
        self.tf_grid_dataset_fpath = os.path.join(
            self.root_data, tf_grid_dataset_fname + ".nc"  #
        )

        # Library dataset
        if self.debug:
            library_dataset_fname = f"{self.name}_library_{self.grid_res_label}_debug"
        else:
            library_dataset_fname = f"{self.name}_library_{self.grid_res_label}"

        self.library_dataset_fpath = os.path.join(
            self.root_data, library_dataset_fname + ".nc"
        )

        # Feature dataset
        if self.debug:
            feature_dataset_fname = f"{self.name}_features_{self.grid_res_label}_debug"
        else:
            feature_dataset_fname = f"{self.name}_features_{self.grid_res_label}"

        self.feature_dataset_fpath = os.path.join(
            self.root_data, feature_dataset_fname + ".nc"
        )

        # Feature kraken dataset
        if self.debug:
            kraken_feature_dataset_fname = (
                f"{self.name}_kraken_features_{self.grid_res_label}_debug"
            )
        else:
            kraken_feature_dataset_fname = (
                f"{self.name}_kraken_features_{self.grid_res_label}"
            )

        self.kraken_feature_dataset_fpath = os.path.join(
            self.root_data, kraken_feature_dataset_fname + ".nc"
        )

        # Localization dataset
        if self.debug:
            localization_dataset_fname = (
                f"{self.name}_localization_{self.grid_res_label}_debug"
            )
        else:
            localization_dataset_fname = (
                f"{self.name}_localization_{self.grid_res_label}"
            )
        self.localization_dataset_fname = localization_dataset_fname
        self.localization_dataset_fpath = os.path.join(
            self.root_data, localization_dataset_fname + ".nc"
        )

        self.from_sig_foldername = f"from_signal_{self.grid_res_label}"
        # Folder to store img derived from signal
        self.root_img_from_sig = os.path.join(self.root_img, self.from_sig_foldername)

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
        self.grid_rmax = np.ceil(np.max(r_grid) * 1e-3) * 1e3
        self.grid_x = x_search_area
        self.grid_y = y_search_area
        self.grid_ranges_from_rcv = r_grid

    def set_debug_config(self):
        """Set debug config for the test case"""

        # Set debug parameters
        self.search_area_length = 200
        self.monte_carlo_iterations = 2


if __name__ == "__main__":
    simu = Simulation()
