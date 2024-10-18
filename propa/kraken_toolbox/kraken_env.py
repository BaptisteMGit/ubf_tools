#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   kraken_env.py
@Time    :   2024/07/08 09:06:58
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Kraken environment class
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import copy
import warnings
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cst import SAND_PROPERTIES, C0
from publication.PublicationFigure import PubFigure
from propa.kraken_toolbox.utils import align_var_description
from propa.kraken_toolbox.plot_utils import plot_ssp, plot_attenuation, plot_density


class KrakenMedium:
    def __init__(
        self,
        ssp_interpolation_method="C_linear",
        z_ssp=[0.0, 100.0],
        c_p=[1500.0, 1500.0],
        c_s=0.0,
        rho=1.0,
        a_p=0.0,
        a_s=0.0,
        nmesh=0,
        sigma=0.0,
    ):
        self.interpolation_method = ssp_interpolation_method

        self.z_ssp = np.array(z_ssp)  # Depth (m)
        self.cp_ssp = np.array(c_p)  # Compression waves celerity (m/s)
        self.cs_ssp = np.array(c_s)  # Shear waves celerity (m/s)
        self.rho = np.array(rho)  # Density (g/cm3)
        self.ap = np.array(
            a_p
        )  # Compressional wave attenuation (defined in KrakenAttenuation)
        self.ash = np.array(
            a_s
        )  # Shear wave attenuation (defined in KrakenAttenuation)

        self.nmesh = nmesh  # Number of mesh points to use initially, should be about 10 per vertical wavelenght (0 let KRAKEN decide)
        self.sigma = sigma  # RMS roughness at the surface

        self.interp_code = None
        self.available_interpolation_methods = [
            "C_linear",
            "N2_linear",
            "cubic_spline",
            "analytic",
        ]

        self.set_interp_code()

    def set_interp_code(self):
        if self.interpolation_method == "C_linear":
            self.interp_code = "C"
        elif self.interpolation_method == "N2_linear":
            self.interp_code = "N"
        elif self.interpolation_method == "cubic_spline":
            self.interp_code = "S"
        elif (
            self.interpolation_method == "analytic"
        ):  # Not recommended -> needs to "modify the analytic formulas in PROFIL.FOR in recompile and link"
            self.interp_code = "A"
            warnings.warn(
                "'analytic' interpolation method is not recommended, you need to modify the analytic formulas in PROFIL.FOR in recompile and link (see KRAKEN doc)"
            )

        else:
            raise ValueError(
                f"Unknown interpolation method '{self.interpolation_method}'. Please pick one of the following: {self.available_interpolation_methods}"
            )

    def write_lines(self, bottom_hs=None):
        # Medim info
        medium_info = align_var_description(
            f"{self.nmesh} {self.sigma} {self.z_ssp.max():.2f}",
            "Number of mesh points, RMS surface roughness, Max depth (units: m)",
        )

        # SSP bloc
        # Check variables size consistency
        cp_check = (self.z_ssp.size == self.cp_ssp.size) or (self.cp_ssp.size == 1)
        cs_check = (self.z_ssp.size == self.cs_ssp.size) or (self.cs_ssp.size == 1)
        rhocheck = (self.z_ssp.size == self.rho.size) or (self.rho.size == 1)
        apcheck = (self.z_ssp.size == self.ap.size) or (self.ap.size == 1)
        ashcheck = (self.z_ssp.size == self.ash.size) or (self.ash.size == 1)

        if not cp_check:
            raise ValueError(
                "Inconsistent SSP data: 'z_ssp', 'c_p' must have the same size"
            )
        if not cs_check:
            raise ValueError(
                "Inconsistent SSP data: 'z_ssp', 'c_s' must have the same size"
            )

        if not rhocheck:
            raise ValueError(
                "Inconsistent SSP data: 'z_ssp' and 'rho' must have the same size or 'rho' must be a scalar"
            )
        if not apcheck:
            raise ValueError(
                "Inconsistent SSP data: 'z_ssp' and 'a_p' must have the same size or 'a_p' must be a scalar"
            )
        if not ashcheck:
            raise ValueError(
                "Inconsistent SSP data: 'z_ssp' and 'a_s' must have the same size or 'a_s' must be a scalar"
            )

        if self.rho.size == 1 and (self.ash.size != 1 or self.ap.size != 1):
            self.rho = np.ones(self.z_ssp.size) * self.rho
        if self.ap.size == 1 and (self.rho.size != 1 or self.ash.size != 1):
            self.ap = np.ones(self.z_ssp.size) * self.ap
        if self.ash.size == 1 and (self.rho.size != 1 or self.ap.size != 1):
            self.ash = np.ones(self.z_ssp.size) * self.ash
        if self.rho.size == 1 and self.ap.size == 1 and self.ash.size == 1:
            scalar_flag = True

        # Write water column SSP bloc
        ssp_desc = "Depth (m), C-wave celerity (m/s), S-wave celerity (m/s), Density (g/cm3), C-wave attenuation , S- wave attenuation"
        if not scalar_flag:
            ssp_bloc = [
                align_var_description(
                    f"{self.z_ssp[0]:.2f} {self.cp_ssp[0]:.2f} {self.cs_ssp[0]:.2f} {self.rho[0]:.2f} {self.ap[0]:.2f} {self.ash[0]:.2f}",
                    ssp_desc,
                )
            ]
        else:
            ssp_bloc = [
                align_var_description(
                    f"{self.z_ssp[0]:.2f} {self.cp_ssp[0]:.2f} {self.cs_ssp:.2f} {self.rho:.2f} {self.ap:.2f} {self.ash:.2f}",
                    ssp_desc,
                )
            ]

        for i in range(1, self.z_ssp.size):
            if not scalar_flag:
                ssp_bloc.append(
                    f"{self.z_ssp[i]:.3} {self.cp_ssp[i]:.2f} {self.cs_ssp[i]:.2f} {self.ap[i]:.2f} {self.ash[i]:.2f}\n"
                )
            else:
                ssp_bloc.append(f"{self.z_ssp[i]:.2f} {self.cp_ssp[i]:.2f} / \n")

        # Write ssp in sediment media layer bloc
        if bottom_hs.write_sedim_layer_bloc:
            sedim_medium_info = align_var_description(
                f"{self.nmesh} {self.sigma} {bottom_hs.sedim_layer_max_depth:.2f}",
                "Number of mesh points in sediment layer, RMS surface roughness, Max depth (units: m)",
            )
            sedim_layer_prop_1 = align_var_description(
                f"{self.z_ssp.max():.2f} {bottom_hs.cp_bot_halfspace:.2f} {bottom_hs.cs_bot_halfspace:.2f} {bottom_hs.rhobot_halfspace:.2f} {bottom_hs.apbot_halfspace:.2f} {bottom_hs.ashbot_halfspace:.2f}",
                ssp_desc,
            )
            sedim_layer_prop_2 = align_var_description(
                f"{bottom_hs.sedim_layer_max_depth:.2f} {bottom_hs.cp_bot_halfspace:.2f} {bottom_hs.cs_bot_halfspace:.2f} {bottom_hs.rhobot_halfspace:.2f} {bottom_hs.apbot_halfspace:.2f} {bottom_hs.ashbot_halfspace:.2f}",
                ssp_desc,
            )
            ssp_sedim_bloc = [sedim_medium_info, sedim_layer_prop_1, sedim_layer_prop_2]
        else:
            ssp_sedim_bloc = []

        self.lines = [medium_info] + ssp_bloc + ssp_sedim_bloc

    def set_default(self):
        self.interpolation_method = "C_linear"
        self.set_interp_code()
        self.z_ssp = np.array([0.0, 100.0])
        self.cp_ssp = np.array([1500.0, 1500.0])
        self.cs_ssp = np.array([0.0, 0.0])
        self.rho = np.array([1.0, 1.0])
        self.ap = np.array([0.0, 0.0])
        self.ash = np.array([0.0, 0.0])
        self.nmesh = 0
        self.sigma = 0.0

    def plot_medium(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
        axs[0].set_ylabel("Depth [m]")
        plot_ssp(cp_ssp=self.cp_ssp, cs_ssp=self.cs_ssp, z=self.z_ssp, ax=axs[0])
        plot_attenuation(ap=self.ap, ash=self.ash, z=self.z_ssp, ax=axs[1])
        plot_density(rho=self.rho, z=self.z_ssp, ax=axs[2])
        plt.suptitle("Medium properties")
        plt.tight_layout()


class KrakenTopHalfspace:
    def __init__(
        self,
        boundary_condition="vacuum",
        halfspace_properties=None,
        twersky_scatter_properties=None,
    ):
        self.boundary_condition = boundary_condition
        self.halfspace_properties = halfspace_properties
        self.twersky_scatter_properties = twersky_scatter_properties

        self.boundary_code = None
        self.available_boundary_conditions = [
            "vacuum",
            "acousto_elastic",
            "perfectly_rigid",
            "reflection_coefficient",
            "soft_boss_Twersky_scatter",
            "hard_boss_Twersky_scatter",
            "soft_boss_Twersky_scatter_amplitude_only",
            "hard_boss_Twersky_scatter_amplitude_only",
        ]

        self.set_boundary_code()

    def set_boundary_code(self):
        if self.boundary_condition == "vacuum":
            self.boundary_code = "V"

        elif self.boundary_condition == "acousto_elastic":
            self.boundary_code = "A"
            self.set_halfspace_properties()

        elif self.boundary_condition == "perfectly_rigid":
            self.boundary_code = "R"
        elif self.boundary_condition == "reflection_coefficient":
            self.boundary_code = "F"
            warnings.warn(
                "reflection_coefficient' boundary condition requires top reflection coefficient to be provided in a separeted .'TRC' file"
            )
        elif self.boundary_condition == "soft_boss_Twersky_scatter":
            self.boundary_code = "S"
            self.set_twersky_scatter()
        elif self.boundary_condition == "hard_boss_Twersky_scatter":
            self.boundary_code = "H"
            self.set_twersky_scatter()
        elif self.boundary_condition == "soft_boss_Twersky_scatter_amplitude_only":
            self.boundary_code = "T"
            self.set_twersky_scatter()
        elif self.boundary_condition == "hard_boss_Twersky_scatter_amplitude_only":
            self.boundary_code = "I"
            self.set_twersky_scatter()
        else:
            raise ValueError(
                f"Unknown interpolation method '{self.boundary_condition}'. Please pick one of the following: {self.available_boundary_conditions}"
            )

    def set_halfspace_properties(self):
        if self.halfspace_properties is None:
            raise ValueError(
                "You need to provide top halfspace properties when using 'acousto_elastic' boundary condition"
            )
        else:
            self.z_top_halfspace = self.halfspace_properties["z"]  # Depth (units: m)
            self.cp_top_halfspace = self.halfspace_properties[
                "c_p"
            ]  # Compression waves celerity (units: m/s)
            self.cs_top_halfspace = self.halfspace_properties[
                "c_s"
            ]  # Shear waves celerity (units: m/s)
            self.rhotop_halfspace = self.halfspace_properties[
                "rho"
            ]  # Density (units: g/cm3)
            self.aptop_halfspace = self.halfspace_properties[
                "a_p"
            ]  # Top compressional wave attenuation (units: self.units)
            self.ashtop_halfspace = self.halfspace_properties[
                "a_s"
            ]  # Top shear wave attenuation (units: self.units)

    def set_twersky_scatter(self):
        if self.twersky_scatter_properties is None:
            raise ValueError(
                "You need to provide Twersky scatter properties when using 'soft_boss_Twersky_scatter', 'hard_boss_Twersky_scatter', 'soft_boss_Twersky_scatter_amplitude_only', 'hard_boss_Twersky_scatter_amplitude_only' boundary condition"
            )
        else:
            self.bumden = self.twersky_scatter_properties[
                "bumden"
            ]  # Bump density in ridges/km
            self.eta = self.twersky_scatter_properties[
                "eta"
            ]  # Principal radius 1 of bump
            self.xi = self.twersky_scatter_properties[
                "xi"
            ]  # Principal radius 2 of bump

    def write_lines(
        self,
        kraken_medium,
        kraken_attenuation,
        slow_rootfinder=False,
        broadband_run=False,
    ):
        desc = "SSP interpolation, Top boundary condition, Attenuation units, Volume attenuation"
        if slow_rootfinder:
            slow_rootfinder_code = "."
            desc += ", Slow rootfinder"
        else:
            slow_rootfinder_code = " "

        if broadband_run:
            broadband_code = "B"
            desc += ", Broadband run"
        else:
            broadband_code = ""

        # Top halfspace info
        top_halfspace_info = align_var_description(
            f"'{kraken_medium.interp_code}{self.boundary_code}{kraken_attenuation.unitscode}{kraken_attenuation.thorp_code}{slow_rootfinder_code}{broadband_code}'",
            desc,
        )

        self.lines = [top_halfspace_info]

    def set_default(self):
        self.boundary_condition = "vacuum"
        self.set_boundary_code()


class KrakenBottomHalfspace:
    def __init__(
        self,
        boundary_condition="acousto_elastic",
        sigma=0.0,
        halfspace_properties=SAND_PROPERTIES,
        fmin=10,
        alpha_wavelength=10,
    ):
        self.sigma = sigma
        self.boundary_condition = boundary_condition
        self.halfspace_properties = halfspace_properties

        # Sedim layer depth
        self.sedim_layer_depth = alpha_wavelength * C0 / fmin
        self.z_in_bottom = np.array(
            [0, self.sedim_layer_depth]
        )  # Depth from bottom water/sediment interface (m)
        self.sedim_layer_max_z = 10000  # Maximum depth of the sediment layer
        self.sedim_layer_max_depth = None

        # Halfspace properties
        self.write_sedim_layer_bloc = False
        self.use_halfspace_properties = False

        # Boundary code
        self.boundary_code = None
        self.available_boundary_conditions = [
            "vacuum",
            "acousto_elastic",
            "perfectly_rigid",
            "reflection_coefficient",
            "precalculated_reflection_coefficient",
        ]
        self.set_boundary_code()

        # Bathymetry code
        self.bathymetry_code = ""

    def set_boundary_code(self):
        if self.boundary_condition == "vacuum":
            self.boundary_code = "V"
            self.sedim_layer_depth = 0
            self.z_in_bottom = np.array(
                [0, 0]
            )  # Depth from bottom water/sediment interface (m)

        elif self.boundary_condition == "acousto_elastic":
            self.boundary_code = "A"
            self.set_halfspace_properties()
            self.write_sedim_layer_bloc = True

        elif self.boundary_condition == "perfectly_rigid":
            self.boundary_code = "R"
        elif self.boundary_condition == "reflection_coefficient":
            self.boundary_code = "F"
            warnings.warn(
                "reflection_coefficient' boundary condition requires bottom reflection coefficient to be provided in a separeted .'TRC' file"
            )
        elif self.boundary_condition == "precalculated_reflection_coefficient":
            self.boundary_code = "P"
            warnings.warn(
                "precalculated_reflection_coefficient' boundary condition requires bottom reflection coefficient to precalculated by BOUNCE"
            )
        else:
            raise ValueError(
                f"Unknown boundary condition '{self.boundary_condition}'. Please pick one of the following: {self.available_boundary_conditions}"
            )

    def set_bathymetry_code(self, use_bathymetry):
        if not use_bathymetry:
            self.bathymetry_code = ""  # Don't use bathymetry
        else:
            self.bathymetry_code = "~"  # Use bathymetry

    def set_halfspace_properties(self):
        if self.halfspace_properties is None:
            raise ValueError(
                "You need to provide bottom halfspace properties when using 'acousto_elastic' boundary condition"
            )
        else:
            self.cp_bot_halfspace = self.halfspace_properties[
                "c_p"
            ]  # Compression waves celerity (units: m/s)
            self.cs_bot_halfspace = self.halfspace_properties[
                "c_s"
            ]  # Shear waves celerity (units: m/s)
            self.rhobot_halfspace = self.halfspace_properties[
                "rho"
            ]  # Density (units: g/cm3)
            self.apbot_halfspace = self.halfspace_properties[
                "a_p"
            ]  # Bottom compressional wave attenuation (units: self.units)
            self.ashbot_halfspace = self.halfspace_properties[
                "a_s"
            ]  # Bottom shear wave attenuation (units: self.units)

            self.use_halfspace_properties = True

    def derive_sedim_layer_max_depth(self, z_max):
        sedim_layer_z = z_max + self.sedim_layer_depth
        self.sedim_layer_max_depth = np.ceil(min(sedim_layer_z, self.sedim_layer_max_z))

    def write_lines(self, use_bathymetry=False):
        # Get bathymetry code
        self.set_bathymetry_code(use_bathymetry)

        # Bottom halfspace info
        bottom_halfspace_info = align_var_description(
            f"'{self.boundary_code+self.bathymetry_code}' {self.sigma}",
            "Type of bottom boundary condition, Interfacial roughness",
        )
        self.lines = [bottom_halfspace_info]

        if self.use_halfspace_properties:
            ssp_desc = "Depth (m), C-wave celerity (m/s), S-wave celerity (m/s), Density (g/cm3), C-wave attenuation , S-wave attenuation"
            half_space_prop = align_var_description(
                f"{self.sedim_layer_max_depth:.2f} {self.cp_bot_halfspace:.2f} {self.cs_bot_halfspace:.2f} {self.rhobot_halfspace:.2f} {self.apbot_halfspace:.2f} {self.ashbot_halfspace:.2f}",
                ssp_desc,
            )
            self.lines.append(half_space_prop)

    def set_default(self):
        self.halfspace_properties = SAND_PROPERTIES
        self.boundary_condition = "acousto_elastic"
        self.set_boundary_code()
        self.sigma = 0.0
        self.bathymetry_code = ""

    """ Associated plotting tools to represent bottom properties """

    def plot_bottom_halfspace(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
        axs[0].set_ylabel("Depth (from water/sediment interface) [m]")
        plot_ssp(
            cp_ssp=self.cp_bot_halfspace,
            cs_ssp=self.cs_bot_halfspace,
            z=self.z_in_bottom,
            ax=axs[0],
        )
        plot_attenuation(
            ap=self.apbot_halfspace,
            ash=self.ashbot_halfspace,
            z=self.z_in_bottom,
            ax=axs[1],
        )
        plot_density(rho=self.rhobot_halfspace, z=self.z_in_bottom, ax=axs[2])
        plt.suptitle("Bottom properties")
        # plt.tight_layout()


class KrakenAttenuation:
    def __init__(self, units="dB_per_wavelength", use_volume_attenuation=False):
        self.units = units
        self.volume_attenuation = use_volume_attenuation
        self.unitscode = None

        self.available_units = [
            "neper_per_m",
            "dB_per_kmhz",
            "dB_per_m",
            "dB_per_wavelength",
            "quality_factor",
            "thorp",
        ]
        self.set_unitscode()
        self.set_thorp_code()

    def set_unitscode(self):
        if self.units == "nepers_per_m":
            self.unitscode = "N"
        elif self.units == "dB_per_kmhz":
            self.unitscode = "F"
        elif self.units == "dB_per_m":
            self.unitscode = "M"
        elif self.units == "dB_per_wavelength":
            self.unitscode = "W"
        elif self.units == "quality_factor":
            self.unitscode = "Q"
        elif self.units == "thorp":
            self.unitscode = "T"
        else:
            raise ValueError(
                f"Unknown interpolation method '{self.units}'. Please pick one of the following: {self.available_units}"
            )

    def set_thorp_code(self):
        if self.volume_attenuation:
            self.thorp_code = "T"
        else:
            self.thorp_code = " "

    def set_default(self):
        self.units = "dB_per_wavelength"
        self.set_unitscode()
        self.volume_attenuation = False
        self.set_thorp_code()


class KrakenField:
    def __init__(
        self,
        phase_speed_limits=None,
        src_depth=[5],
        n_rcv_z=1000,
        rcv_z_min=0.0,
        rcv_z_max=1000.0,
        rcv_r_max=0.0,
    ):
        if phase_speed_limits is None:
            self.phase_speed_limits = [0.0, 2000.0]

        self.phase_speed_limits = np.array(phase_speed_limits)

        self.src_depth = np.array(src_depth)
        if self.src_depth.size == 1:
            self.src_depth = np.array([src_depth])

        self.n_rcv_z = n_rcv_z
        self.rcv_depth_min = rcv_z_min
        self.rcv_depth_max = rcv_z_max
        self.rcv_range_max = rcv_r_max

    def write_lines(self):
        self.lines = []
        self.lines.append(
            align_var_description(
                f"{self.phase_speed_limits[0]} {self.phase_speed_limits[1]}",
                "Phase speed limits (min, max) (m/s)",
            )
        )
        self.lines.append(
            align_var_description(f"{self.rcv_range_max}", "Maximum range (km)")
        )
        self.lines.append(
            align_var_description(
                f"{self.src_depth.size}", "Number of source depth (m)"
            )
        )
        self.lines.append(
            align_var_description(
                "".join([str(src_d) + " " for src_d in self.src_depth]),
                "Source depths (m)",
            )
        )
        self.lines.append(
            align_var_description(f"{self.n_rcv_z}", "Number of receiver depths (m)")
        )
        self.lines.append(
            align_var_description(
                f"{self.rcv_depth_min} {self.rcv_depth_max} /",
                "Minimum and maximum receiver depths (m)",
            )
        )


class Bathymetry:
    def __init__(self, data_file=None, units="km", interpolation_method="linear"):
        self.data_file = data_file
        self.units = units
        self.interpolation_method = interpolation_method

        if self.data_file is None:
            self.use_bathy = False
        else:
            self.load_data()

    def load_data(self):
        if os.path.exists(self.data_file):
            data = pd.read_csv(self.data_file, sep=",", header=None)

            # Ensure range are in km and depth in m
            if self.units == "km":
                self.bathy_range = data[0].values
            elif self.units == "m":
                self.bathy_range = data[0].values / 1000
            else:
                raise ValueError(
                    f"Unknown units '{self.units}'. Please pick one of the following: 'km', 'm'"
                )

            self.bathy_depth = data[1].values
            # Check if bathy is constant (i.e. no depth variation)
            if np.all(self.bathy_depth == self.bathy_depth[0]):
                self.use_bathy = False
            else:
                self.interpolator = sp.interpolate.interp1d(
                    self.bathy_range,
                    self.bathy_depth,
                    kind=self.interpolation_method,
                    fill_value=(self.bathy_depth[0], self.bathy_depth[-1]),
                    bounds_error=False,
                )
                self.use_bathy = True
        else:
            raise ValueError(f"Data file '{self.data_file}' does not exist")


class KrakenEnv:
    def __init__(
        self,
        title="",
        env_root="",
        env_filename="",
        freq=50.0,
        kraken_top_hs=KrakenTopHalfspace(),
        kraken_medium=KrakenMedium(),
        kraken_attenuation=KrakenAttenuation(),
        kraken_bottom_hs=KrakenBottomHalfspace(),
        kraken_field=KrakenField(),
        kraken_bathy=Bathymetry(),
        rModes=None,
        rModes_units="km",
        nmedia=1,
    ):
        self.simulation_title = title

        self.root_ = env_root
        self.filename = env_filename
        # .env file path
        self.env_fpath = os.path.join(self.root_, self.filename + ".env")
        # .flp file path
        self.flp_fpath = os.path.join(self.root_, self.filename + ".flp")
        # .shd file path
        self.shd_fpath = os.path.join(self.root_, self.filename + ".shd")

        # List of ordered frequencies
        self.freq = np.array(freq)
        self.freq = np.unique(self.freq)
        self.freq.sort()

        if self.freq.size > 1:
            self.broadband_run = True
            self.nominal_frequency = float(self.freq[0])
        else:
            self.broadband_run = False
            self.nominal_frequency = float(self.freq)

        self.top_hs = kraken_top_hs
        self.medium = kraken_medium
        self.att = kraken_attenuation
        self.bottom_hs = kraken_bottom_hs
        self.field = kraken_field
        self.bathy = kraken_bathy
        self.nmedia = nmedia

        if rModes is not None and self.bathy.use_bathy:
            self.modes_range = rModes
            if rModes_units == "m":
                self.modes_range = rModes / 1000  # Convert to km
            # Sort by ascending ranges
            self.modes_range.sort()

        elif rModes is None and self.bathy.use_bathy:
            self.modes_range = self.bathy.bathy_range  # Already in km
            # Sort by ascending ranges
            self.modes_range.sort()

        # Ensure modes_range contains 0
        if self.bathy.use_bathy and self.modes_range[0] != 0:
            self.modes_range = np.append(0, self.modes_range)

        # if self.bathy.use_bathy:
        # Defined max depth of the sediment layer
        if self.bottom_hs.sedim_layer_max_depth is None:
            self.bottom_hs.derive_sedim_layer_max_depth(
                z_max=self.bathy.bathy_depth.max()
            )

        self.range_dependent_env = False

    def write_range_independent_lines(self):
        # Init lines list
        self.env_lines = []

        # Write top halfspace lines
        self.top_hs.write_lines(
            kraken_medium=self.medium,
            kraken_attenuation=self.att,
            broadband_run=self.broadband_run,
            slow_rootfinder=False,
        )
        # Write medium lines
        self.medium.write_lines(bottom_hs=self.bottom_hs)
        # Write bottom halfspace lines
        self.bottom_hs.write_lines(use_bathymetry=self.bathy.use_bathy)

        # self.bottom_hs.write_lines(
        #     kraken_medium=self.medium, use_bathymetry=self.bathy.use_bathy
        # )
        # Write field lines
        self.field.write_lines()
        self.write_lines(title=self.simulation_title, medium=self.medium)

        self.range_dependent_env = False

    def write_range_dependent_lines(self):
        # Init lines list
        self.env_lines = []

        # Write top halfspace lines
        self.top_hs.write_lines(
            kraken_medium=self.medium,
            kraken_attenuation=self.att,
            broadband_run=self.broadband_run,
            slow_rootfinder=False,
        )

        # Write field lines
        self.field.write_lines()

        for i in range(self.modes_range.size):
            depth = self.bathy.interpolator(self.modes_range[i])

            medium_copy = copy.deepcopy(self.medium)

            # Remove depths that exceed the bathymetry
            idx = medium_copy.z_ssp <= depth

            medium_copy.z_ssp = medium_copy.z_ssp[idx]
            medium_copy.cp_ssp = medium_copy.cp_ssp[idx]

            # Add a new SSP point interpolated to the bathymetry
            if (
                depth > medium_copy.z_ssp[-1]
            ):  # make sure added point is greater in depth
                medium_copy.cp_ssp = np.append(
                    medium_copy.cp_ssp,
                    np.interp(depth, medium_copy.z_ssp, medium_copy.cp_ssp),
                )

                if medium_copy.cs_ssp.size == self.medium.z_ssp.size:
                    medium_copy.cs_ssp = medium_copy.cs_ssp[idx]
                    medium_copy.cs_ssp = np.append(
                        depth, medium_copy.z_ssp, medium_copy.cs_ssp
                    )

                if medium_copy.rho.size == self.medium.z_ssp.size:
                    medium_copy.rho = medium_copy.rho[idx]
                    medium_copy.rho = np.append(
                        depth, medium_copy.z_ssp, medium_copy.rho
                    )

                if medium_copy.ap.size == self.medium.z_ssp.size:
                    medium_copy.ap = medium_copy.ap[idx]
                    medium_copy.ap = np.append(depth, medium_copy.z_ssp, medium_copy.ap)

                if medium_copy.ash.size == self.medium.z_ssp.size:
                    medium_copy.ash = medium_copy.ash[idx]
                    medium_copy.ash = np.append(
                        depth, medium_copy.z_ssp, medium_copy.ash
                    )

                medium_copy.z_ssp = np.append(medium_copy.z_ssp, depth)

            # Write medium lines
            medium_copy.write_lines(bottom_hs=self.bottom_hs)

            # Write bottom halfspace lines
            self.bottom_hs.write_lines(use_bathymetry=self.bathy.use_bathy)

            # Change title to include range
            title = self.simulation_title + f" - r = {self.modes_range[i]:.2f} km"
            self.write_lines(title=title, medium=medium_copy)

        self.range_dependent_env = True

    def write_lines(self, title, medium):
        # Write env lines
        # Bloc 1
        self.env_lines.append(f"'{title}'\n")
        self.env_lines.append(
            align_var_description(f"{self.nominal_frequency}", "Nominal frequency (Hz)")
        )
        self.env_lines.append(
            align_var_description(f"{self.nmedia}", "Number of media")
        )
        # Bloc 2
        self.env_lines += self.top_hs.lines
        # Bloc 3
        self.env_lines += medium.lines
        # Bloc 4
        self.env_lines += self.bottom_hs.lines
        # Bloc 5
        self.env_lines += self.field.lines
        # Bloc 6
        if self.broadband_run:
            self.env_lines.append(
                align_var_description(f"{self.freq.size}", "Number of frequencies")
            )
            self.env_lines.append(
                align_var_description(
                    " ".join([str(f) for f in self.freq]), "Frequencies (Hz)"
                )
            )

    def write_env(self):
        # TODO: handle range dependent ssp
        if self.bathy.use_bathy:
            self.write_range_dependent_lines()
        else:
            self.write_range_independent_lines()

        # Write lines to .env file
        with open(self.env_fpath, "w") as f_out:
            f_out.writelines(self.env_lines)

    @property
    def root(self):
        return self.root_

    @root.setter
    def root(self, root):
        self.root_ = root
        self.env_fpath = os.path.join(self.root_, self.filename + ".env")
        self.flp_fpath = os.path.join(self.root_, self.filename + ".flp")
        self.shd_fpath = os.path.join(self.root_, self.filename + ".shd")

    # Plotting tools
    def plot_env(self, plot_src=False, src_depth=None):

        pfig = PubFigure(titlepad=50, labelpad=25)
        fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
        axs[0].set_ylabel("Depth [m]")
        # Plot ssp
        if np.array(self.medium.cp_ssp).size == 1:
            cp_med = np.ones(self.medium.z_ssp.size) * self.medium.cp_ssp
        else:
            cp_med = self.medium.cp_ssp

        if np.array(self.medium.cs_ssp).size == 1:
            cs_med = np.ones(self.medium.z_ssp.size) * self.medium.cs_ssp
        else:
            cs_med = self.medium.cs_ssp

        if np.array(self.bottom_hs.cp_bot_halfspace).size == 1:
            cp_bot = (
                np.ones(self.bottom_hs.z_in_bottom.size)
                * self.bottom_hs.cp_bot_halfspace
            )
        else:
            cp_bot = self.bottom_hs.cp_bot_halfspace

        if np.array(self.bottom_hs.cs_bot_halfspace).size == 1:
            cs_bot = (
                np.ones(self.bottom_hs.z_in_bottom.size)
                * self.bottom_hs.cs_bot_halfspace
            )
        else:
            cs_bot = self.bottom_hs.cs_bot_halfspace

        cp_env = np.append(cp_med, cp_bot)
        cs_env = np.append(cs_med, cs_bot)
        z_bottom = self.medium.z_ssp[-1]
        z_env = np.append(self.medium.z_ssp, self.bottom_hs.z_in_bottom + z_bottom)
        plot_ssp(
            cp_ssp=cp_env,
            cs_ssp=cs_env,
            z=z_env,
            z_bottom=z_bottom,
            ax=axs[0],
        )

        # Plot attenuation
        if np.array(self.medium.ap).size == 1:
            apmed = np.ones(self.medium.z_ssp.size) * self.medium.ap
        else:
            apmed = self.medium.ap

        if np.array(self.medium.ash).size == 1:
            ashmed = np.ones(self.medium.z_ssp.size) * self.medium.ash
        else:
            ashmed = self.medium.ash

        if np.array(self.bottom_hs.apbot_halfspace).size == 1:
            apbot = (
                np.ones(self.bottom_hs.z_in_bottom.size)
                * self.bottom_hs.apbot_halfspace
            )
        else:
            apbot = self.bottom_hs.apbot_halfspace

        if np.array(self.bottom_hs.ashbot_halfspace).size == 1:
            ashbot = (
                np.ones(self.bottom_hs.z_in_bottom.size)
                * self.bottom_hs.ashbot_halfspace
            )
        else:
            ashbot = self.bottom_hs.ashbot_halfspace

        apenv = np.append(apmed, apbot)
        ashenv = np.append(ashmed, ashbot)
        plot_attenuation(
            ap=apenv,
            ash=ashenv,
            z=z_env,
            z_bottom=z_bottom,
            ax=axs[1],
        )

        # Plot density
        if np.array(self.medium.rho).size == 1:
            rhomed = np.ones(self.medium.z_ssp.size) * self.medium.rho
        else:
            rhomed = self.medium.rho

        if np.array(self.bottom_hs.rhobot_halfspace).size == 1:
            rhobot = (
                np.ones(self.bottom_hs.z_in_bottom.size)
                * self.bottom_hs.rhobot_halfspace
            )
        else:
            rhobot = self.bottom_hs.rhobot_halfspace

        rhoenv = np.append(rhomed, rhobot)
        plot_density(rho=rhoenv, z=z_env, z_bottom=z_bottom, ax=axs[2])

        if plot_src:
            for i in range(3):
                xmin = axs[i].get_xlim()[0]
                axs[i].scatter(xmin, src_depth, s=30, color="k")
                # Circles
                for s in [200, 500]:
                    axs[i].scatter(
                        xmin,
                        src_depth,
                        s=s,
                        facecolors="None",
                        edgecolors="k",
                        linewidths=0.5,
                    )

        # if plot_rcv:
        #     for i in range(3):
        #         xmax = axs[i].get_xlim()[1]
        #         axs[i].scatter(xmax, rcv_depth, s=50, color="k", marker=">")

        plt.suptitle("Waveguide properties")
        # plt.tight_layout(w_pad=3)


class KrakenFlp:
    def __init__(
        self,
        env,
        src_type="point_source",
        mode_theory="adiabatic",
        mode_addition="coherent",
        nb_modes=9999,
        src_depth=[5],
        n_rcv_z=1000,
        rcv_z_min=0.0,
        rcv_z_max=1000.0,
        n_rcv_r=1001,
        rcv_r_min=0.0,
        rcv_r_max=50.0,
        rcv_dist_offset=0.0,
    ):
        self.env = env
        self.flp_fpath = self.env.flp_fpath
        self.title = self.env.simulation_title
        self.src_type = src_type
        self.mode_theory = mode_theory
        self.mode_addition = mode_addition
        self.nb_modes = nb_modes

        # Profile info ( for range dependent env)
        if self.env.range_dependent_env:
            self.n_profiles = self.env.modes_range.size
            self.profiles_ranges = self.env.modes_range
            # rcv_z_max = self.env.bottom_hs.sedim_layer_max_depth
        else:
            self.n_profiles = 1
            self.profiles_ranges = np.array([0.0])

        self.src_z = np.array(src_depth)
        if self.src_z.size == 1:
            self.src_z = np.array([src_depth])

        # Receiver depth info
        self.n_rcv_z = int(n_rcv_z)
        self.rcv_z_min = int(np.floor(rcv_z_min))
        self.rcv_z_max = int(np.ceil(rcv_z_max))
        # Receiver range info
        self.n_rcv_r = int(n_rcv_r)
        self.rcv_r_min = int(np.floor(rcv_r_min))
        self.rcv_r_max = int(np.ceil(rcv_r_max))
        self.rcv_dist_offset = int(rcv_dist_offset)

        self.set_codes()

    # TODO : add decorator to uptdate attributes on change ?

    def set_codes(self):
        # source type
        if self.src_type == "point_source":
            self.src_code = "R"
        elif self.src_type == "line_source":
            self.src_code = "X"
        else:
            raise ValueError(
                f"Unknown mode theory method '{self.src_type}'. Please pick one of the following: 'point_source', 'line_source'"
            )

        # mode theory
        if self.mode_theory == "coupled":
            self.th_code = "C"
        elif self.mode_theory == "adiabatic":
            self.th_code = "A"
        else:
            raise ValueError(
                f"Unknown mode theory method '{self.mode_theory}'. Please pick one of the following: 'coupled', 'adiabatic'"
            )

        # addition mode
        if self.mode_addition == "coherent":
            self.add_code = "C"
        elif self.mode_addition == "incoherent":
            self.add_code = "I"
        else:
            raise ValueError(
                f"Unknown addition mode '{self.mode_addition}'. Please pick one of the following: 'coherent', 'incoherent'"
            )

    def write_lines(self):
        self.lines = []
        self.lines.append(f"'{self.title}'\n")
        self.lines.append(
            align_var_description(
                f"'{self.src_code}{self.th_code} {self.add_code}'",
                "Source type, Mode theory, Mode addition",
            )
        )
        self.lines.append(align_var_description(f"{self.nb_modes}", "Number of modes"))
        self.lines.append(
            align_var_description(f"{self.n_profiles}", "Number of profiles")
        )
        self.lines.append(
            align_var_description(
                " ".join([f"{r:.4f}" for r in self.profiles_ranges]) + " /",
                "Profile ranges (km)",
            )
        )
        self.lines.append(
            align_var_description(f"{self.n_rcv_r}", "Number of receiver ranges")
        )
        self.lines.append(
            align_var_description(
                f"{self.rcv_r_min} {self.rcv_r_max} /", "Receiver ranges (km)"
            )
        )

        self.lines.append(
            align_var_description(f"{self.src_z.size}", "Number of source depth (m)")
        )
        self.lines.append(
            align_var_description(
                "".join([str(src_d) + " " for src_d in self.src_z]) + " /",
                "Source depths (m)",
            )
        )
        self.lines.append(
            align_var_description(f"{self.n_rcv_z}", "Number of receiver depths (m)")
        )
        self.lines.append(
            align_var_description(
                f"{self.rcv_z_min} {self.rcv_z_max} /",
                "Receiver depths (m)",
            )
        )
        self.lines.append(
            align_var_description(
                f"{self.n_rcv_z}", "Number of receiver range-displacements"
            )
        )
        self.lines.append(
            align_var_description(
                f"{self.rcv_dist_offset} /", "Receiver displacements (m)"
            )
        )

    def write_flp(self):
        self.write_lines()

        with open(self.flp_fpath, "w") as f_out:
            f_out.writelines(self.lines)


if __name__ == "__main__":
    project_root = os.getcwd()
    test_root = os.path.join(project_root, r"propa\kraken_toolbox\tests\kraken_env")

    top_hs = KrakenTopHalfspace()

    z_ssp = np.array([0.0, 100.0, 500, 600, 700, 1000.0])
    cp_ssp = np.array([1500.0, 1550.0, 1540.0, 1532.0, 1522.0, 1512.0])
    medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

    # medium.plot_medium()
    # plt.show()

    bathy_fpath = os.path.join(test_root, "bathy_data.csv")
    bathy = Bathymetry(
        data_file=bathy_fpath,
        interpolation_method="linear",
        units="m",
    )

    bott_hs_properties = SAND_PROPERTIES
    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(
        halfspace_properties=bott_hs_properties,
    )

    att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)
    field = KrakenField(src_depth=50)

    env = KrakenEnv(
        title="Test de la classe KrakenEnv",
        env_root=test_root,
        env_filename="test_kraken_rd",
        freq=[10, 50, 16, 25, 20, 21, 62, 85, 93, 714, 16, 25, 20, 21, 62],
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
        kraken_bathy=bathy,
    )

    env.write_env()

    flp = KrakenFlp(env=env, src_depth=50)
    flp.write_flp()
