import os
import copy
import warnings
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cst import SAND_PROPERTIES, C0
from propa.kraken_toolbox.utils import align_var_description


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
        self.interpolation_method_ = ssp_interpolation_method

        self.z_ssp_ = np.array(z_ssp)  # Depth (m)
        self.cp_ssp_ = np.array(c_p)  # Compression waves celerity (m/s)
        self.cs_ssp_ = np.array(c_s)  # Shear waves celerity (m/s)
        self.rho_ = np.array(rho)  # Density (g/cm3)
        self.ap_ = np.array(
            a_p
        )  # Compressional wave attenuation (defined in KrakenAttenuation)
        self.as_ = np.array(
            a_s
        )  # Shear wave attenuation (defined in KrakenAttenuation)

        self.nmesh_ = nmesh  # Number of mesh points to use initially, should be about 10 per vertical wavelenght (0 let KRAKEN decide)
        self.sigma_ = sigma  # RMS roughness at the surface

        self.interp_code = None
        self.available_interpolation_methods = [
            "C_linear",
            "N2_linear",
            "cubic_spline",
            "analytic",
        ]

        self.set_interp_code()
        # self.write_lines()

    def set_interp_code(self):
        if self.interpolation_method_ == "C_linear":
            self.interp_code = "C"
        elif self.interpolation_method_ == "N2_linear":
            self.interp_code = "N"
        elif self.interpolation_method_ == "cubic_spline":
            self.interp_code = "S"
        elif (
            self.interpolation_method_ == "analytic"
        ):  # Not recommended -> needs to "modify the analytic formulas in PROFIL.FOR in recompile and link"
            self.interp_code = "A"
            warnings.warn(
                "'analytic' interpolation method is not recommended, you need to modify the analytic formulas in PROFIL.FOR in recompile and link (see KRAKEN doc)"
            )

        else:
            raise ValueError(
                f"Unknown interpolation method '{self.interpolation_method_}'. Please pick one of the following: {self.available_interpolation_methods}"
            )

    def write_lines(self, bottom_hs=None):
        # Medim info
        medium_info = align_var_description(
            f"{self.nmesh_} {self.sigma_} {self.z_ssp_.max():.2f}",
            "Number of mesh points, RMS surface roughness, Max depth (units: m)",
        )

        # SSP bloc
        # Check variables size consistency
        cp_check = (self.z_ssp_.size == self.cp_ssp_.size) or (self.cp_ssp_.size == 1)
        cs_check = (self.z_ssp_.size == self.cs_ssp_.size) or (self.cs_ssp_.size == 1)
        rho_check = (self.z_ssp_.size == self.rho_.size) or (self.rho_.size == 1)
        ap_check = (self.z_ssp_.size == self.ap_.size) or (self.ap_.size == 1)
        as_check = (self.z_ssp_.size == self.as_.size) or (self.as_.size == 1)

        if not cp_check:
            raise ValueError(
                "Inconsistent SSP data: 'z_ssp', 'c_p' must have the same size"
            )
        if not cs_check:
            raise ValueError(
                "Inconsistent SSP data: 'z_ssp', 'c_s' must have the same size"
            )

        if not rho_check:
            raise ValueError(
                "Inconsistent SSP data: 'z_ssp' and 'rho' must have the same size or 'rho' must be a scalar"
            )
        if not ap_check:
            raise ValueError(
                "Inconsistent SSP data: 'z_ssp' and 'a_p' must have the same size or 'a_p' must be a scalar"
            )
        if not as_check:
            raise ValueError(
                "Inconsistent SSP data: 'z_ssp' and 'a_s' must have the same size or 'a_s' must be a scalar"
            )

        if self.rho_.size == 1 and (self.as_.size != 1 or self.ap_.size != 1):
            self.rho_ = np.ones(self.z_ssp_.size) * self.rho_
        if self.ap_.size == 1 and (self.rho_.size != 1 or self.as_.size != 1):
            self.ap_ = np.ones(self.z_ssp_.size) * self.ap_
        if self.as_.size == 1 and (self.rho_.size != 1 or self.ap_.size != 1):
            self.as_ = np.ones(self.z_ssp_.size) * self.as_
        if self.rho_.size == 1 and self.ap_.size == 1 and self.as_.size == 1:
            scalar_flag = True

        # Write water column SSP bloc
        ssp_desc = "Depth (m), C-wave celerity (m/s), S-wave celerity (m/s), Density (g/cm3), C-wave attenuation , S- wave attenuation"
        if not scalar_flag:
            ssp_bloc = [
                align_var_description(
                    f"{self.z_ssp_[0]:.2f} {self.cp_ssp_[0]:.2f} {self.cs_ssp_[0]:.2f} {self.rho_[0]:.2f} {self.ap_[0]:.2f} {self.as_[0]:.2f}",
                    ssp_desc,
                )
            ]
        else:
            ssp_bloc = [
                align_var_description(
                    f"{self.z_ssp_[0]:.2f} {self.cp_ssp_[0]:.2f} {self.cs_ssp_:.2f} {self.rho_:.2f} {self.ap_:.2f} {self.as_:.2f}",
                    ssp_desc,
                )
            ]

        for i in range(1, self.z_ssp_.size):
            if not scalar_flag:
                ssp_bloc.append(
                    f"{self.z_ssp_[i]:.3} {self.cp_ssp_[i]:.2f} {self.cs_ssp_[i]:.2f} {self.ap_[i]:.2f} {self.as_[i]:.2f}\n"
                )
            else:
                ssp_bloc.append(f"{self.z_ssp_[i]:.2f} {self.cp_ssp_[i]:.2f} / \n")

        # Write ssp in sediment media layer bloc
        sedim_medium_info = align_var_description(
            f"{self.nmesh_} {self.sigma_} {bottom_hs.sedim_layer_max_depth:.2f}",
            "Number of mesh points in sediment layer, RMS surface roughness, Max depth (units: m)",
        )
        sedim_layer_prop_1 = align_var_description(
            f"{self.z_ssp_.max():.2f} {bottom_hs.cp_bot_halfspace:.2f} {bottom_hs.cs_bot_halfspace:.2f} {bottom_hs.rho_bot_halfspace:.2f} {bottom_hs.ap_bot_halfspace:.2f} {bottom_hs.as_bot_halfspace:.2f}",
            ssp_desc,
        )
        sedim_layer_prop_2 = align_var_description(
            f"{bottom_hs.sedim_layer_max_depth:.2f} {bottom_hs.cp_bot_halfspace:.2f} {bottom_hs.cs_bot_halfspace:.2f} {bottom_hs.rho_bot_halfspace:.2f} {bottom_hs.ap_bot_halfspace:.2f} {bottom_hs.as_bot_halfspace:.2f}",
            ssp_desc,
        )
        ssp_sedim_bloc = [sedim_medium_info, sedim_layer_prop_1, sedim_layer_prop_2]

        self.lines = [medium_info] + ssp_bloc + ssp_sedim_bloc

    def set_default(self):
        self.interpolation_method_ = "C_linear"
        self.set_interp_code()
        self.z_ssp_ = np.array([0.0, 100.0])
        self.cp_ssp_ = np.array([1500.0, 1500.0])
        self.cs_ssp_ = np.array([0.0, 0.0])
        self.rho_ = np.array([1.0, 1.0])
        self.ap_ = np.array([0.0, 0.0])
        self.as_ = np.array([0.0, 0.0])
        self.nmesh_ = 0
        self.sigma_ = 0.0

    def plot_medium(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
        axs[0].set_ylabel("Depth [m]")
        self.plot_spp(ax=axs[0])
        self.plot_attenuation(ax=axs[1])
        self.plot_density(ax=axs[2])
        plt.suptitle("Medium properties")
        plt.tight_layout()
        # plt.show()

    def plot_spp(self, ax=None):
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.set_ylabel("Depth [m]")
        ax.invert_yaxis()
        col1 = "red"
        ax.plot(self.cp_ssp_, self.z_ssp_, color=col1)
        ax.tick_params(axis="x", labelcolor=col1)
        ax.set_xlabel("C-wave celerity " + r"[$m.s^{-1}$]", color=col1)
        # ax.set_title("Sound speed profile")

        if self.cs_ssp_.size == 1:
            cs = np.ones(self.z_ssp_.size) * self.cs_ssp_
        else:
            cs = self.cs_ssp_
        ax_bis = ax.twiny()  # instantiate a second axes that shares the same y-axis
        col2 = "blue"
        ax_bis.plot(cs, self.z_ssp_, color=col2)
        ax_bis.tick_params(axis="x", labelcolor=col2)
        ax_bis.set_xlabel("S-wave celerity " + r"[$m.s^{-1}$]", color=col2)

    def plot_attenuation(self, ax=None):
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.set_ylabel("Depth [m]")

        if self.ap_.size == 1:
            ap = np.ones(self.z_ssp_.size) * self.ap_
        else:
            ap = self.ap_

        ax.invert_yaxis()
        col1 = "red"
        ax.plot(ap, self.z_ssp_, color=col1)
        ax.tick_params(axis="x", labelcolor=col1)
        ax.set_xlabel("C-wave attenuation " + r"[$dB.\lambda^{-1}$]", color=col1)

        if self.as_.size == 1:
            as_ = np.ones(self.z_ssp_.size) * self.as_
        else:
            as_ = self.as_

        ax_bis = ax.twiny()  # instantiate a second axes that shares the same y-axis
        col2 = "blue"
        ax_bis.plot(as_, self.z_ssp_, color=col2)
        ax_bis.tick_params(axis="x", labelcolor=col2)
        ax_bis.set_xlabel("S-wave attenuation " + r"[$dB.\lambda^{-1}$]", color=col2)

    def plot_density(self, ax=None):
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.set_ylabel("Depth [m]")

        if self.rho_.size == 1:
            rho = np.ones(self.z_ssp_.size) * self.rho_
        else:
            rho = self.rho_
        ax.plot(rho, self.z_ssp_, label="Density", color="blue")
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelcolor="blue")
        ax.set_xlabel("Density " + r"[$g.cm^{-3}$]", color="blue")


class KrakenTopHalfspace:
    def __init__(
        self,
        boundary_condition="vacuum",
        halfspace_properties=None,
        twersky_scatter_properties=None,
    ):
        self.boundary_condition_ = boundary_condition
        self.halfspace_properties_ = halfspace_properties
        self.twersky_scatter_properties_ = twersky_scatter_properties

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
        if self.boundary_condition_ == "vacuum":
            self.boundary_code = "V"
        elif self.boundary_condition_ == "acousto_elastic":
            self.boundary_code = "A"
            self.set_halfspace_properties()
        elif self.boundary_condition_ == "perfectly_rigid":
            self.boundary_code = "R"
        elif self.boundary_condition_ == "reflection_coefficient":
            self.boundary_code = "F"
            warnings.warn(
                "reflection_coefficient' boundary condition requires top reflection coefficient to be provided in a separeted .'TRC' file"
            )
        elif self.boundary_condition_ == "soft_boss_Twersky_scatter":
            self.boundary_code = "S"
            self.set_twersky_scatter()
        elif self.boundary_condition_ == "hard_boss_Twersky_scatter":
            self.boundary_code = "H"
            self.set_twersky_scatter()
        elif self.boundary_condition_ == "soft_boss_Twersky_scatter_amplitude_only":
            self.boundary_code = "T"
            self.set_twersky_scatter()
        elif self.boundary_condition_ == "hard_boss_Twersky_scatter_amplitude_only":
            self.boundary_code = "I"
            self.set_twersky_scatter()
        else:
            raise ValueError(
                f"Unknown interpolation method '{self.boundary_condition_}'. Please pick one of the following: {self.available_boundary_conditions}"
            )

    def set_halfspace_properties(self):
        if self.halfspace_properties_ is None:
            raise ValueError(
                "You need to provide top halfspace properties when using 'acousto_elastic' boundary condition"
            )
        else:
            self.z_top_halfspace = self.halfspace_properties_["z"]  # Depth (units: m)
            self.cp_top_halfspace = self.halfspace_properties_[
                "c_p"
            ]  # Compression waves celerity (units: m/s)
            self.cs_top_halfspace = self.halfspace_properties_[
                "c_s"
            ]  # Shear waves celerity (units: m/s)
            self.rho_top_halfspace = self.halfspace_properties_[
                "rho"
            ]  # Density (units: g/cm3)
            self.ap_top_halfspace = self.halfspace_properties_[
                "a_p"
            ]  # Top compressional wave attenuation (units: self.units)
            self.as_top_halfspace = self.halfspace_properties_[
                "a_s"
            ]  # Top shear wave attenuation (units: self.units)

    def set_twersky_scatter(self):
        if self.twersky_scatter_properties_ is None:
            raise ValueError(
                "You need to provide Twersky scatter properties when using 'soft_boss_Twersky_scatter', 'hard_boss_Twersky_scatter', 'soft_boss_Twersky_scatter_amplitude_only', 'hard_boss_Twersky_scatter_amplitude_only' boundary condition"
            )
        else:
            self.bumden = self.twersky_scatter_properties_[
                "bumden"
            ]  # Bump density in ridges/km
            self.eta = self.twersky_scatter_properties_[
                "eta"
            ]  # Principal radius 1 of bump
            self.xi = self.twersky_scatter_properties_[
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
            f"'{kraken_medium.interp_code}{self.boundary_code}{kraken_attenuation.units_code}{kraken_attenuation.thorp_code}{slow_rootfinder_code}{broadband_code}'",
            desc,
        )

        self.lines = [top_halfspace_info]

    def set_default(self):
        self.boundary_condition_ = "vacuum"
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
        self.boundary_condition_ = boundary_condition
        self.sigma_ = sigma
        self.halfspace_properties_ = halfspace_properties

        # Sedim layer depth
        self.sedim_layer_depth = alpha_wavelength * C0 / fmin
        self.z_in_bottom = np.array(
            [0, self.sedim_layer_depth]
        )  # Depth from bottom water/sediment interface (m)
        self.sedim_layer_max_z = 10000  # Maximum depth of the sediment layer
        self.sedim_layer_max_depth = None

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
        if self.boundary_condition_ == "vacuum":
            self.boundary_code = "V"
        elif self.boundary_condition_ == "acousto_elastic":
            self.boundary_code = "A"
            self.set_halfspace_properties()
        elif self.boundary_condition_ == "perfectly_rigid":
            self.boundary_code = "R"
        elif self.boundary_condition_ == "reflection_coefficient":
            self.boundary_code = "F"
            warnings.warn(
                "reflection_coefficient' boundary condition requires bottom reflection coefficient to be provided in a separeted .'TRC' file"
            )
        elif self.boundary_condition_ == "precalculated_reflection_coefficient":
            self.boundary_code = "P"
            warnings.warn(
                "precalculated_reflection_coefficient' boundary condition requires bottom reflection coefficient to precalculated by BOUNCE"
            )
        else:
            raise ValueError(
                f"Unknown boundary condition '{self.boundary_condition_}'. Please pick one of the following: {self.available_boundary_conditions}"
            )

    def set_bathymetry_code(self, use_bathymetry):
        if not use_bathymetry:
            self.bathymetry_code = ""  # Don't use bathymetry
        else:
            self.bathymetry_code = "~"  # Use bathymetry

    def set_halfspace_properties(self):
        if self.halfspace_properties_ is None:
            raise ValueError(
                "You need to provide top halfspace properties when using 'acousto_elastic' boundary condition"
            )
        else:
            self.cp_bot_halfspace = self.halfspace_properties_[
                "c_p"
            ]  # Compression waves celerity (units: m/s)
            self.cs_bot_halfspace = self.halfspace_properties_[
                "c_s"
            ]  # Shear waves celerity (units: m/s)
            self.rho_bot_halfspace = self.halfspace_properties_[
                "rho"
            ]  # Density (units: g/cm3)
            self.ap_bot_halfspace = self.halfspace_properties_[
                "a_p"
            ]  # Top compressional wave attenuation (units: self.units)
            self.as_bot_halfspace = self.halfspace_properties_[
                "a_s"
            ]  # Top shear wave attenuation (units: self.units)

            self.use_halfspace_properties = True

    def derive_sedim_layer_max_depth(self, z_max):
        sedim_layer_z = z_max + self.sedim_layer_depth
        self.sedim_layer_max_depth = min(sedim_layer_z, self.sedim_layer_max_z)

    def write_lines(self, use_bathymetry=False):
        # Get bathymetry code
        self.set_bathymetry_code(use_bathymetry)

        # Bottom halfspace info
        bottom_halfspace_info = align_var_description(
            f"'{self.boundary_code+self.bathymetry_code}' {self.sigma_}",
            "Type of bottom boundary condition, Interfacial roughness",
        )
        self.lines = [bottom_halfspace_info]

        if self.use_halfspace_properties:
            ssp_desc = "Depth (m), C-wave celerity (m/s), S-wave celerity (m/s), Density (g/cm3), C-wave attenuation , S-wave attenuation"
            half_space_prop = align_var_description(
                f"{self.sedim_layer_max_depth:.2f} {self.cp_bot_halfspace:.2f} {self.cs_bot_halfspace:.2f} {self.rho_bot_halfspace:.2f} {self.ap_bot_halfspace:.2f} {self.as_bot_halfspace:.2f}",
                ssp_desc,
            )
            self.lines.append(half_space_prop)

    def set_default(self):
        self.halfspace_properties_ = SAND_PROPERTIES
        self.boundary_condition_ = "acousto_elastic"
        self.set_boundary_code()
        self.sigma_ = 0.0
        self.bathymetry_code = ""

    """ Associated plotting tools to represent bottom properties """

    def plot_bottom_halfspace(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
        axs[0].set_ylabel("Depth (from water/sediment interface) [m]")
        self.plot_ssp(ax=axs[0])
        self.plot_attenuation(ax=axs[1])
        self.plot_density(ax=axs[2])
        plt.suptitle("Bottom properties")
        plt.tight_layout()

    def plot_ssp(self, ax=None):
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.set_ylabel("Depth (from water/sediment interface) [m]")
        ax.invert_yaxis()
        col1 = "red"

        # C-wave speed
        # TODO: change if several layer in bottom
        # if self.cp_bot_halfspace.size == 1:
        #     cp = np.ones(self.z_in_bottom.size) * self.cp_bot_halfspace
        # else:
        #     cp = self.cp_bot_halfspace
        cp = np.ones(self.z_in_bottom.size) * self.cp_bot_halfspace

        ax.plot(cp, self.z_in_bottom, color=col1)
        ax.tick_params(axis="x", labelcolor=col1)
        ax.set_xlabel("C-wave celerity " + r"[$m.s^{-1}$]", color=col1)

        # S-wave speed
        # if self.cs_bot_halfspace.size == 1:
        #     cs = np.ones(self.z_in_bottom.size) * self.cs_bot_halfspace
        # else:
        #     cs = self.cs_bot_halfspace
        cs = np.ones(self.z_in_bottom.size) * self.cs_bot_halfspace
        ax_bis = ax.twiny()  # instantiate a second axes that shares the same y-axis
        col2 = "blue"
        ax_bis.plot(cs, self.z_in_bottom, color=col2)
        ax_bis.tick_params(axis="x", labelcolor=col2)
        ax_bis.set_xlabel("S-wave celerity " + r"[$m.s^{-1}$]", color=col2)

    def plot_attenuation(self, ax=None):
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.set_ylabel("Depth (from water/sediment interface) [m]")

        # if self.ap_bot_halfspace.size == 1:
        #     ap = np.ones(self.z_in_bottom.size) * self.ap_bot_halfspace
        # else:
        #     ap = self.ap_bot_halfspace
        xlim_offset = 0.5 * abs(self.as_bot_halfspace - self.ap_bot_halfspace)
        min_att = min(self.as_bot_halfspace, self.ap_bot_halfspace)
        max_att = max(self.as_bot_halfspace, self.ap_bot_halfspace)

        ap = np.ones(self.z_in_bottom.size) * self.ap_bot_halfspace
        ax.invert_yaxis()
        col1 = "red"
        ax.plot(ap, self.z_in_bottom, color=col1)
        ax.tick_params(axis="x", labelcolor=col1)
        ax.set_xlabel("C-wave attenuation " + r"[$dB.\lambda^{-1}$]", color=col1)
        ax.set_xlim([min_att - xlim_offset, max_att + xlim_offset])

        # ax.set_title("Attenuation profile")

        # if self.as_bot_halfspace.size == 1:
        #     as_ = np.ones(self.z_in_bottom.size) * self.as_bot_halfspace
        # else:
        #     as_ = self.as_bot_halfspace
        as_ = np.ones(self.z_in_bottom.size) * self.as_bot_halfspace

        ax_bis = ax.twiny()  # instantiate a second axes that shares the same y-axis
        col2 = "blue"
        ax_bis.plot(as_, self.z_in_bottom, color=col2)
        ax_bis.tick_params(axis="x", labelcolor=col2)
        ax_bis.set_xlabel("S-wave attenuation " + r"[$dB.\lambda^{-1}$]", color=col2)
        ax_bis.set_xlim([min_att - xlim_offset, max_att + xlim_offset])

    def plot_density(self, ax=None):
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.set_ylabel("Depth [m]")

        # if self.rho_bot_halfspace.size == 1:
        #     rho = np.ones(self.z_in_bottom.size) * self.rho_bot_halfspace
        # else:
        #     rho = self.rho_bot_halfspace
        rho = np.ones(self.z_in_bottom.size) * self.rho_bot_halfspace
        ax.plot(rho, self.z_in_bottom, label="Density", color="blue")
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelcolor="blue")
        ax.set_xlabel("Density " + r"[$g.cm^{-3}$]", color="blue")


class KrakenAttenuation:
    def __init__(self, units="dB_per_wavelength", use_volume_attenuation=False):
        self.units_ = units
        self.volume_attenuation_ = use_volume_attenuation
        self.units_code = None

        self.available_units = [
            "neper_per_m",
            "dB_per_kmhz",
            "dB_per_m",
            "dB_per_wavelength",
            "quality_factor",
            "thorp",
        ]
        self.set_units_code()
        self.set_thorp_code()

    def set_units_code(self):
        if self.units_ == "nepers_per_m":
            self.units_code = "N"
        elif self.units_ == "dB_per_kmhz":
            self.units_code = "F"
        elif self.units_ == "dB_per_m":
            self.units_code = "M"
        elif self.units_ == "dB_per_wavelength":
            self.units_code = "W"
        elif self.units_ == "quality_factor":
            self.units_code = "Q"
        elif self.units_ == "thorp":
            self.units_code = "T"
        else:
            raise ValueError(
                f"Unknown interpolation method '{self.units_}'. Please pick one of the following: {self.available_units}"
            )

    def set_thorp_code(self):
        if self.volume_attenuation_:
            self.thorp_code = "T"
        else:
            self.thorp_code = " "

    def set_default(self):
        self.units_ = "dB_per_wavelength"
        self.set_units_code()
        self.volume_attenuation_ = False
        self.set_thorp_code()


class KrakenField:
    def __init__(
        self,
        phase_speed_limits=[0.0, 20000],
        src_depth=[5],
        n_rcv_z=1000,
        rcv_z_min=0.0,
        rcv_z_max=1000.0,
        rcv_r_max=0.0,
    ):
        self.phase_speed_limits_ = np.array(phase_speed_limits)

        self.src_depth_ = np.array(src_depth)
        if self.src_depth_.size == 1:
            self.src_depth_ = np.array([src_depth])

        self.n_rcv_z_ = n_rcv_z
        self.rcv_depth_min_ = rcv_z_min
        self.rcv_depth_max_ = rcv_z_max
        self.rcv_range_max_ = rcv_r_max

    def write_lines(self):
        self.lines = []
        self.lines.append(
            align_var_description(
                f"{self.phase_speed_limits_[0]} {self.phase_speed_limits_[1]}",
                "Phase speed limits (min, max) (m/s)",
            )
        )
        self.lines.append(
            align_var_description(f"{self.rcv_range_max_}", "Maximum range (km)")
        )
        self.lines.append(
            align_var_description(
                f"{self.src_depth_.size}", "Number of source depth (m)"
            )
        )
        self.lines.append(
            align_var_description(
                "".join([str(src_d) + " " for src_d in self.src_depth_]),
                "Source depths (m)",
            )
        )
        self.lines.append(
            align_var_description(f"{self.n_rcv_z_}", "Number of receiver depths (m)")
        )
        self.lines.append(
            align_var_description(
                f"{self.rcv_depth_min_} {self.rcv_depth_max_} /",
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
        nmedia=2,
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

        if self.bathy.use_bathy:
            # Defined max depth of the sediment layer
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
        self.bottom_hs.write_lines(
            kraken_medium=self.medium, use_bathymetry=self.bathy.use_bathy
        )
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
            idx = medium_copy.z_ssp_ <= depth

            medium_copy.z_ssp_ = medium_copy.z_ssp_[idx]
            medium_copy.cp_ssp_ = medium_copy.cp_ssp_[idx]

            # Add a new SSP point interpolated to the bathymetry
            if (
                depth > medium_copy.z_ssp_[-1]
            ):  # make sure added point is greater in depth
                medium_copy.cp_ssp_ = np.append(
                    medium_copy.cp_ssp_,
                    np.interp(depth, medium_copy.z_ssp_, medium_copy.cp_ssp_),
                )

                if medium_copy.cs_ssp_.size == self.medium.z_ssp_.size:
                    medium_copy.cs_ssp_ = medium_copy.cs_ssp_[idx]
                    medium_copy.cs_ssp_ = np.append(
                        depth, medium_copy.z_ssp_, medium_copy.cs_ssp_
                    )

                if medium_copy.rho_.size == self.medium.z_ssp_.size:
                    medium_copy.rho_ = medium_copy.rho_[idx]
                    medium_copy.rho_ = np.append(
                        depth, medium_copy.z_ssp_, medium_copy.rho_
                    )

                if medium_copy.ap_.size == self.medium.z_ssp_.size:
                    medium_copy.ap_ = medium_copy.ap_[idx]
                    medium_copy.ap_ = np.append(
                        depth, medium_copy.z_ssp_, medium_copy.ap_
                    )

                if medium_copy.as_.size == self.medium.z_ssp_.size:
                    medium_copy.as_ = medium_copy.as_[idx]
                    medium_copy.as_ = np.append(
                        depth, medium_copy.z_ssp_, medium_copy.as_
                    )

                medium_copy.z_ssp_ = np.append(medium_copy.z_ssp_, depth)

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

    def plot_env(self):
        self.medium.plot_medium()

    @property
    def root(self):
        return self.root_

    @root.setter
    def root(self, root):
        self.root_ = root
        self.env_fpath = os.path.join(self.root_, self.filename + ".env")
        self.flp_fpath = os.path.join(self.root_, self.filename + ".flp")
        self.shd_fpath = os.path.join(self.root_, self.filename + ".shd")


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
        self.title_ = self.env.simulation_title
        self.src_type_ = src_type
        self.mode_theory_ = mode_theory
        self.mode_addition_ = mode_addition
        self.nb_modes_ = nb_modes

        # Profile info ( for range dependent env)
        if self.env.range_dependent_env:
            self.n_profiles_ = self.env.modes_range.size
            self.profiles_ranges_ = self.env.modes_range
            rcv_z_max = self.env.bottom_hs.sedim_layer_max_depth
        else:
            self.n_profiles_ = 1
            self.profiles_ranges_ = np.array([0.0])

        self.src_z_ = np.array(src_depth)
        if self.src_z_.size == 1:
            self.src_z_ = np.array([src_depth])

        # Receiver depth info
        self.n_rcv_z_ = int(n_rcv_z)
        self.rcv_z_min_ = int(np.floor(rcv_z_min))
        self.rcv_z_max_ = int(np.ceil(rcv_z_max))
        # Receiver range info
        self.n_rcv_r_ = int(n_rcv_r)
        self.rcv_r_min_ = int(np.floor(rcv_r_min))
        self.rcv_r_max_ = int(np.ceil(rcv_r_max))
        self.rcv_dist_offset_ = int(rcv_dist_offset)

        self.set_codes()
        # self.write_lines()

    # TODO : add decorator to uptdate attributes on change

    def set_codes(self):
        # source type
        if self.src_type_ == "point_source":
            self.src_code = "R"
        elif self.src_type_ == "line_source":
            self.src_code = "X"
        else:
            raise ValueError(
                f"Unknown mode theory method '{self.src_type_}'. Please pick one of the following: 'point_source', 'line_source'"
            )

        # mode theory
        if self.mode_theory_ == "coupled":
            self.th_code = "C"
        elif self.mode_theory_ == "adiabatic":
            self.th_code = "A"
        else:
            raise ValueError(
                f"Unknown mode theory method '{self.mode_theory_}'. Please pick one of the following: 'coupled', 'adiabatic'"
            )

        # addition mode
        if self.mode_addition_ == "coherent":
            self.add_code = "C"
        elif self.mode_addition_ == "incoherent":
            self.add_code = "I"
        else:
            raise ValueError(
                f"Unknown addition mode '{self.mode_addition_}'. Please pick one of the following: 'coherent', 'incoherent'"
            )

    def write_lines(self):
        self.lines = []
        self.lines.append(f"'{self.title_}'\n")
        self.lines.append(
            align_var_description(
                f"'{self.src_code}{self.th_code} {self.add_code}'",
                "Source type, Mode theory, Mode addition",
            )
        )
        self.lines.append(align_var_description(f"{self.nb_modes_}", "Number of modes"))
        self.lines.append(
            align_var_description(f"{self.n_profiles_}", "Number of profiles")
        )
        self.lines.append(
            align_var_description(
                " ".join([f"{r:.4f}" for r in self.profiles_ranges_]) + " /",
                "Profile ranges (km)",
            )
        )
        self.lines.append(
            align_var_description(f"{self.n_rcv_r_}", "Number of receiver ranges")
        )
        self.lines.append(
            align_var_description(
                f"{self.rcv_r_min_} {self.rcv_r_max_} /", "Receiver ranges (km)"
            )
        )

        self.lines.append(
            align_var_description(f"{self.src_z_.size}", "Number of source depth (m)")
        )
        self.lines.append(
            align_var_description(
                "".join([str(src_d) + " " for src_d in self.src_z_]) + " /",
                "Source depths (m)",
            )
        )
        self.lines.append(
            align_var_description(f"{self.n_rcv_z_}", "Number of receiver depths (m)")
        )
        self.lines.append(
            align_var_description(
                f"{self.rcv_z_min_} {self.rcv_z_max_} /",
                "Receiver depths (m)",
            )
        )
        self.lines.append(
            align_var_description(
                f"{self.n_rcv_z_}", "Number of receiver range-displacements"
            )
        )
        self.lines.append(
            align_var_description(
                f"{self.rcv_dist_offset_} /", "Receiver displacements (m)"
            )
        )

    def write_flp(self):
        self.write_lines()

        with open(self.flp_fpath, "w") as f_out:
            f_out.writelines(self.lines)


if __name__ == "__main__":
    top_hs = KrakenTopHalfspace()

    z_ssp = np.array([0.0, 100.0, 500, 600, 700, 1000.0])
    cp_ssp = np.array([1500.0, 1550.0, 1540.0, 1532.0, 1522.0, 1512.0])
    medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

    # medium.plot_medium()
    # plt.show()

    bathy = Bathymetry(
        data_file=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd\bathy_data.csv",
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
        env_root=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd",
        env_filename="test_kraken_rd",
        freq=[10, 50, 16, 25, 20, 21, 62, 85, 93, 714, 16, 25, 20, 21, 62],
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
    )

    env.write_env()

    flp = KrakenFlp(env=env, src_depth=50)
    flp.write_flp()
