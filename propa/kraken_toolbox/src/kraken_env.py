#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   kraken_env.py
@Time    :   2024/07/08 09:06:58
@Author  :   Menetrier Baptiste
@Version :   1.1 (refactor)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   KRAKEN environment classes: generation of the '.env' and
             '.flp' input files consumed by the KRAKEN/FIELD Fortran
             executables.

This module does NOT change the public API of the original file: same
class names, same method names, same constructor signatures, same
attribute names. It has been reorganized and documented to be easier to
read, maintain, and test, and several latent bugs (see the
"NOTE (bug ...)" comments below) have been fixed.

------------------------------------------------------------------------
Overview of KRAKEN's '.env' format (to help navigate the code)
------------------------------------------------------------------------
A '.env' file is a Fortran-style text file: each line holds one or more
values followed by a human-readable comment. It is organized into
blocks, in this order (see KrakenEnv.write_lines):

    1. Simulation title, nominal frequency, number of media
    2. Top halfspace (KrakenTopHalfspace)               -> 1 line
    3. Water column (KrakenMedium)                       -> N lines
    4. Bottom halfspace (KrakenBottomHalfspace)          -> 1-2 lines
    5. Field parameters (KrakenField)                    -> 5 lines
    6. (optional) List of frequencies for broadband runs

The '.flp' file (KrakenFlp) describes the source/receiver grid and the
propagation theory (coupled/adiabatic modes, coherent/incoherent
addition) used by the FIELD executable.

Every "Kraken<X>" class has the same responsibility:
    - store physical/numerical parameters as attributes,
    - translate "human" choices (e.g. boundary_condition="vacuum") into
      the single-letter codes expected by the Fortran format (e.g. "V"),
    - expose write_lines(...) which fills self.lines with the text lines
      ready to be written to the '.env' file.
------------------------------------------------------------------------

BUGS FIXED COMPARED TO THE ORIGINAL CODE (worth flagging / confirming):
  1. KrakenField: the default value of phase_speed_limits was never
     actually applied (it was overwritten right after by
     np.array(phase_speed_limits), which re-converted the original
     argument, still equal to None) -> crashed as soon as
     phase_speed_limits was not explicitly provided.
  2. KrakenEnv.__init__: the sediment layer max depth calculation used
     self.bathy.bathy_depth.max() without checking that self.bathy
     actually held data (the 'if self.bathy.use_bathy:' guard was
     present as a comment but not applied) -> AttributeError as soon as
     no bathymetry was explicitly provided.
  3. KrakenEnv.__init__: in-place sort (`.sort()`) of an array derived
     from a pandas DataFrame (bathy_range). With pandas' Copy-on-Write
     mode (default since pandas 2.x, mandatory in pandas 3.x), this
     array is read-only -> ValueError. Replaced with `np.sort(...)`
     (out-of-place).
  4. KrakenEnv.__init__: `float(self.freq)` crashed with recent numpy
     (>=1.25) when self.freq is a single-element array (single
     frequency case). Replaced with an explicit access to the first
     element.
  5. write_range_dependent_lines: when adding an SSP point interpolated
     at the bathymetry depth, the code called
     `np.append(depth, medium_copy.z_ssp, medium_copy.cs_ssp)` (3
     positional arguments), which does not match np.append's signature
     (np.append(arr, values)) and would have raised an error had this
     branch ever been reached. Fixed to append the interpolated value at
     the right place.

These fixes are documented inline with the "NOTE (bug ...)" keyword
everywhere they occur, so they remain easy to locate / revert if you
prefer to handle them differently.
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

import source.global_constants as g

from publication.publication_figure import PubFigure
from propa.kraken_toolbox.utils import align_var_description
from propa.kraken_toolbox.plot_utils import plot_ssp, plot_attenuation, plot_density


# ======================================================================================================================
# Small internal helper functions (no side effects, easy to unit test)
# ======================================================================================================================
def _broadcast_to_size(value, size):
    """Return 'value' as a numpy array of length 'size'.

    Many physical properties (celerity, density, attenuation...) can be
    given either as a scalar (constant value over the whole column) or
    as an array matching the depth grid size. This function factors out
    the "if it's a scalar, repeat it, otherwise keep it as is" pattern
    that was duplicated about a dozen times in the original code (in
    write_lines and plot_env in particular).

    Args:
        value: scalar or numpy array.
        size (int): target size.

    Returns:
        np.ndarray of length 'size'.
    """
    arr = np.asarray(value)
    if arr.size == 1:
        return np.full(size, arr.reshape(-1)[0] if arr.ndim > 0 else arr)
    return arr


def _check_same_size_or_scalar(z_ssp, other, name):
    """Check that 'other' has the same size as 'z_ssp' or is a scalar
    (size 1). Raises an explicit ValueError otherwise.

    Factors out the 5 near-identical checks (cp, cs, rho, ap, ash) that
    were duplicated in KrakenMedium.write_lines.
    """
    other_arr = np.asarray(other)
    if not (z_ssp.size == other_arr.size or other_arr.size == 1):
        raise ValueError(
            f"Inconsistent SSP data: 'z_ssp' and '{name}' must have the same "
            f"size or '{name}' must be a scalar"
        )


# ======================================================================================================================
# Water column (Sound Speed Profile)
# ======================================================================================================================
class KrakenMedium:
    """Describes the water column: sound speed profile (SSP), density
    and attenuation as a function of depth.

    Each physical property (c_p, c_s, rho, a_p, a_s) can be given either
    as a scalar (constant over the whole column) or as an array matching
    the size of 'z_ssp'.
    """

    #: Mapping from human-readable interpolation method to KRAKEN code
    _INTERP_CODES = {
        "C_linear": "C",
        "N2_linear": "N",
        "cubic_spline": "S",
        "analytic": "A",
    }

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
        """
        Args:
            ssp_interpolation_method (str): 'C_linear', 'N2_linear',
                'cubic_spline' or 'analytic' (discouraged, see KRAKEN doc).
            z_ssp (array-like): depths (m) of the profile points.
            c_p (array-like|float): compressional wave celerity (m/s).
            c_s (array-like|float): shear wave celerity (m/s).
            rho (array-like|float): density (g/cm3).
            a_p (array-like|float): compressional wave attenuation.
            a_s (array-like|float): shear wave attenuation.
            nmesh (int): desired initial number of mesh points (roughly
                10 per vertical wavelength; 0 lets KRAKEN decide).
            sigma (float): RMS surface roughness (m).
        """
        self.interpolation_method = ssp_interpolation_method

        self.z_ssp = np.array(z_ssp)  # Depth (m)
        self.cp_ssp = np.array(c_p)  # Compression waves celerity (m/s)
        self.cs_ssp = np.array(c_s)  # Shear waves celerity (m/s)
        self.rho = np.array(rho)  # Density (g/cm3)
        self.ap = np.array(a_p)  # Compressional wave attenuation
        self.ash = np.array(a_s)  # Shear wave attenuation

        self.nmesh = nmesh
        self.sigma = sigma

        self.interp_code = None
        self.available_interpolation_methods = list(self._INTERP_CODES)
        self.set_interp_code()

    def set_interp_code(self):
        """Translate self.interpolation_method into the KRAKEN letter
        code (self.interp_code)."""
        if self.interpolation_method not in self._INTERP_CODES:
            raise ValueError(
                f"Unknown interpolation method '{self.interpolation_method}'. "
                f"Please pick one of the following: {self.available_interpolation_methods}"
            )

        self.interp_code = self._INTERP_CODES[self.interpolation_method]

        if self.interpolation_method == "analytic":
            # Special case: requires modifying the Fortran source code
            warnings.warn(
                "'analytic' interpolation method is not recommended, you need "
                "to modify the analytic formulas in PROFIL.FOR in recompile "
                "and link (see KRAKEN doc)"
            )

    def _validate_sizes(self):
        """Check the consistency of physical property array sizes
        against z_ssp."""
        _check_same_size_or_scalar(self.z_ssp, self.cp_ssp, "c_p")
        _check_same_size_or_scalar(self.z_ssp, self.cs_ssp, "c_s")
        _check_same_size_or_scalar(self.z_ssp, self.rho, "rho")
        _check_same_size_or_scalar(self.z_ssp, self.ap, "a_p")
        _check_same_size_or_scalar(self.z_ssp, self.ash, "a_s")

    def write_lines(self, bottom_hs=None):
        """Build self.lines: the '.env' lines describing the water
        column (and, if needed, the thin "buffer" sediment layer between
        the water and the bottom halfspace).

        Args:
            bottom_hs (KrakenBottomHalfspace): bottom halfspace, needed
                to know whether an intermediate sediment block must be
                written (see bottom_hs.write_sedim_layer_bloc).
        """
        self._validate_sizes()

        # --- Medium header line: nb of mesh points, roughness, max depth
        medium_info = align_var_description(
            f"{self.nmesh} {self.sigma} {self.z_ssp.max():.2f}",
            "Number of mesh points, RMS surface roughness, Max depth (units: m)",
        )

        # --- Water column SSP block, one line per depth
        ssp_desc = (
            "Depth (m), C-wave celerity (m/s), S-wave celerity (m/s), "
            "Density (g/cm3), C-wave attenuation , S- wave attenuation"
        )

        # All scalar (size 1) properties are broadcast to the size of
        # z_ssp so that a full line can be written for every depth. This
        # is the only substantive difference with the original code,
        # which handled an "all scalar" case separately via a
        # 'scalar_flag' flag (which could even remain undefined if no
        # scalar branch was taken -> potential UnboundLocalError).
        cp = _broadcast_to_size(self.cp_ssp, self.z_ssp.size)
        cs = _broadcast_to_size(self.cs_ssp, self.z_ssp.size)
        rho = _broadcast_to_size(self.rho, self.z_ssp.size)
        ap = _broadcast_to_size(self.ap, self.z_ssp.size)
        ash = _broadcast_to_size(self.ash, self.z_ssp.size)

        ssp_bloc = [
            align_var_description(
                f"{self.z_ssp[0]:.2f} {cp[0]:.2f} {cs[0]:.2f} {rho[0]:.2f} "
                f"{ap[0]:.2f} {ash[0]:.2f}",
                ssp_desc,
            )
        ]
        for i in range(1, self.z_ssp.size):
            # NB: keeps the original 6-column layout for every row
            # (KRAKEN also accepts a shorthand '/' to repeat the
            # previous row's remaining values when c_p is the only thing
            # changing, but the explicit form is always valid and more
            # readable).
            ssp_bloc.append(
                align_var_description(
                    f"{self.z_ssp[i]:.2f} {cp[i]:.2f} {cs[i]:.2f} {rho[i]:.2f} "
                    f"{ap[i]:.2f} {ash[i]:.2f}",
                    ssp_desc,
                )
            )

        # --- Optional block: thin "buffer" sediment layer between the
        # water and the bottom halfspace (needed when the bottom is
        # acousto-elastic). If bottom_hs.sediment_top_properties was
        # provided, the top and bottom of this layer use DIFFERENT
        # properties, producing a simple two-point linear gradient
        # (KRAKEN's C-linear SSP interpolation connects them) instead of
        # a flat isovelocity segment -- a standard way to represent a
        # sediment layer that compacts (and speeds up) with depth.
        ssp_sedim_bloc = []
        if bottom_hs is not None and bottom_hs.write_sedim_layer_bloc:
            sedim_medium_info = align_var_description(
                f"{self.nmesh} {self.sigma} {bottom_hs.sedim_layer_max_depth:.2f}",
                "Number of mesh points in sediment layer, RMS surface roughness, Max depth (units: m)",
            )
            sedim_layer_prop_1 = align_var_description(
                f"{self.z_ssp.max():.2f} {bottom_hs.cp_sedim_top:.2f} "
                f"{bottom_hs.cs_sedim_top:.2f} {bottom_hs.rho_sedim_top:.2f} "
                f"{bottom_hs.ap_sedim_top:.2f} {bottom_hs.ash_sedim_top:.2f}",
                ssp_desc,
            )
            sedim_layer_prop_2 = align_var_description(
                f"{bottom_hs.sedim_layer_max_depth:.2f} {bottom_hs.cp_bot_halfspace:.2f} "
                f"{bottom_hs.cs_bot_halfspace:.2f} {bottom_hs.rhobot_halfspace:.2f} "
                f"{bottom_hs.apbot_halfspace:.2f} {bottom_hs.ashbot_halfspace:.2f}",
                ssp_desc,
            )
            ssp_sedim_bloc = [sedim_medium_info, sedim_layer_prop_1, sedim_layer_prop_2]

        self.lines = [medium_info] + ssp_bloc + ssp_sedim_bloc

    def set_default(self):
        """Reset the medium to a reference profile: 100 m water column,
        constant 1500 m/s celerity, no attenuation."""
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

    # ------------------------------------------------------------------
    # Plotting tools
    # ------------------------------------------------------------------
    def plot_medium(self):
        """Plot SSP, attenuation and density as a function of depth.

        Returns:
            matplotlib.figure.Figure
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
        axs[0].set_ylabel("Depth [m]")
        plot_ssp(cp_ssp=self.cp_ssp, cs_ssp=self.cs_ssp, z=self.z_ssp, ax=axs[0])
        plot_attenuation(ap=self.ap, ash=self.ash, z=self.z_ssp, ax=axs[1])
        plot_density(rho=self.rho, z=self.z_ssp, ax=axs[2])
        plt.suptitle("Medium properties")
        plt.tight_layout()
        return fig


# ======================================================================================================================
# Top halfspace (surface)
# ======================================================================================================================
class KrakenTopHalfspace:
    """Describes the surface boundary condition (top halfspace)."""

    _BOUNDARY_CODES = {
        "vacuum": "V",
        "acousto_elastic": "A",
        "perfectly_rigid": "R",
        "reflection_coefficient": "F",
        "soft_boss_Twersky_scatter": "S",
        "hard_boss_Twersky_scatter": "H",
        "soft_boss_Twersky_scatter_amplitude_only": "T",
        "hard_boss_Twersky_scatter_amplitude_only": "I",
    }
    #: Conditions that require halfspace properties (c_p, c_s, rho, a_p, a_s)
    _NEEDS_HALFSPACE_PROPERTIES = {"acousto_elastic"}
    #: Conditions that require Twersky scattering properties
    _NEEDS_TWERSKY_PROPERTIES = {
        "soft_boss_Twersky_scatter",
        "hard_boss_Twersky_scatter",
        "soft_boss_Twersky_scatter_amplitude_only",
        "hard_boss_Twersky_scatter_amplitude_only",
    }

    def __init__(
        self,
        boundary_condition="vacuum",
        halfspace_properties=None,
        twersky_scatter_properties=None,
    ):
        """
        Args:
            boundary_condition (str): one of self.available_boundary_conditions.
            halfspace_properties (dict|None): required if
                boundary_condition == 'acousto_elastic'. Expected keys:
                'z', 'c_p', 'c_s', 'rho', 'a_p', 'a_s'.
            twersky_scatter_properties (dict|None): required for Twersky
                scatter conditions. Expected keys: 'bumden', 'eta', 'xi'.
        """
        self.boundary_condition = boundary_condition
        self.halfspace_properties = halfspace_properties
        self.twersky_scatter_properties = twersky_scatter_properties

        self.boundary_code = None
        self.available_boundary_conditions = list(self._BOUNDARY_CODES)

        self.set_boundary_code()

    def set_boundary_code(self):
        """Translate boundary_condition into the KRAKEN letter code and
        fetch the associated properties (halfspace or Twersky
        scattering) if needed."""
        if self.boundary_condition not in self._BOUNDARY_CODES:
            raise ValueError(
                f"Unknown interpolation method '{self.boundary_condition}'. "
                f"Please pick one of the following: {self.available_boundary_conditions}"
            )

        self.boundary_code = self._BOUNDARY_CODES[self.boundary_condition]

        if self.boundary_condition == "reflection_coefficient":
            warnings.warn(
                "reflection_coefficient' boundary condition requires top "
                "reflection coefficient to be provided in a separeted .'TRC' file"
            )
        elif self.boundary_condition in self._NEEDS_HALFSPACE_PROPERTIES:
            self.set_halfspace_properties()
        elif self.boundary_condition in self._NEEDS_TWERSKY_PROPERTIES:
            self.set_twersky_scatter()

    def set_halfspace_properties(self):
        """Fetch the top halfspace physical properties from
        self.halfspace_properties. Required for 'acousto_elastic'."""
        if self.halfspace_properties is None:
            raise ValueError(
                "You need to provide top halfspace properties when using "
                "'acousto_elastic' boundary condition"
            )
        self.z_top_halfspace = self.halfspace_properties["z"]  # Depth (m)
        self.cp_top_halfspace = self.halfspace_properties["c_p"]  # m/s
        self.cs_top_halfspace = self.halfspace_properties["c_s"]  # m/s
        self.rhotop_halfspace = self.halfspace_properties["rho"]  # g/cm3
        self.aptop_halfspace = self.halfspace_properties["a_p"]
        self.ashtop_halfspace = self.halfspace_properties["a_s"]

    def set_twersky_scatter(self):
        """Fetch the Twersky scattering properties ("boss"-type
        roughness) from self.twersky_scatter_properties."""
        if self.twersky_scatter_properties is None:
            raise ValueError(
                "You need to provide Twersky scatter properties when using "
                "'soft_boss_Twersky_scatter', 'hard_boss_Twersky_scatter', "
                "'soft_boss_Twersky_scatter_amplitude_only', "
                "'hard_boss_Twersky_scatter_amplitude_only' boundary condition"
            )
        self.bumden = self.twersky_scatter_properties["bumden"]  # bumps/km
        self.eta = self.twersky_scatter_properties["eta"]  # principal radius 1
        self.xi = self.twersky_scatter_properties["xi"]  # principal radius 2

    def write_lines(
        self,
        kraken_medium,
        kraken_attenuation,
        slow_rootfinder=False,
        broadband_run=False,
    ):
        """Build self.lines: the single '.env' line combining the SSP
        interpolation code, the surface boundary condition code, the
        attenuation units and the options (slow rootfinder, broadband
        run).

        Args:
            kraken_medium (KrakenMedium): used to read interp_code.
            kraken_attenuation (KrakenAttenuation): used to read
                unitscode and thorp_code.
            slow_rootfinder (bool): enables the slow (more robust, more
                expensive) root-finding mode.
            broadband_run (bool): multi-frequency simulation.
        """
        desc = "SSP interpolation, Top boundary condition, Attenuation units, Volume attenuation"
        slow_rootfinder_code = "." if slow_rootfinder else " "
        if slow_rootfinder:
            desc += ", Slow rootfinder"

        broadband_code = "B" if broadband_run else ""
        if broadband_run:
            desc += ", Broadband run"

        top_halfspace_info = align_var_description(
            f"'{kraken_medium.interp_code}{self.boundary_code}"
            f"{kraken_attenuation.unitscode}{kraken_attenuation.thorp_code}"
            f"{slow_rootfinder_code}{broadband_code}'",
            desc,
        )
        self.lines = [top_halfspace_info]

    def set_default(self):
        """Reset to the default boundary condition ('vacuum')."""
        self.boundary_condition = "vacuum"
        self.set_boundary_code()


# ======================================================================================================================
# Bottom halfspace
# ======================================================================================================================
class KrakenBottomHalfspace:
    """Describes the bottom boundary condition (bottom halfspace) and
    the associated "buffer" sediment layer.
    """

    _BOUNDARY_CODES = {
        "vacuum": "V",
        "acousto_elastic": "A",
        "perfectly_rigid": "R",
        "reflection_coefficient": "F",
        "precalculated_reflection_coefficient": "P",
    }

    def __init__(
        self,
        boundary_condition="acousto_elastic",
        sigma=0.0,
        halfspace_properties=g.sand_properties,
        fmin=10,
        alpha_wavelength=10,
        add_sediment_buffer_layer=True,
        sediment_top_properties=None,
    ):
        """
        Args:
            boundary_condition (str): one of self.available_boundary_conditions.
            sigma (float): interfacial roughness.
            halfspace_properties (dict): bottom physical properties,
                required if boundary_condition == 'acousto_elastic'. When
                add_sediment_buffer_layer is True, these are also the
                properties at the BOTTOM of the buffer sediment layer
                (see sediment_top_properties for the top).
            fmin (float): minimum frequency (Hz), used to compute the
                thickness of the buffer sediment layer (10-wavelength
                rule, see sedim_layer_depth). Ignored if
                add_sediment_buffer_layer is False.
            alpha_wavelength (float): number of wavelengths used to size
                the buffer sediment layer. Ignored if
                add_sediment_buffer_layer is False.
            add_sediment_buffer_layer (bool): if True (default, and the
                only behaviour available before this parameter existed),
                an extra thin "buffer" sediment medium is inserted
                between the water column and the true acousto-elastic
                half-space, thickened by 'alpha_wavelength' wavelengths
                at 'fmin'. This is a common but OPTIONAL numerical
                practice, not a KRAKEN requirement: a real KRAKEN/FIELD
                run with a single water-column medium (nmedia=1) and a
                direct acousto-elastic half-space immediately below it
                (no buffer) was confirmed to run correctly. Set this to
                False to get that simpler, direct form -- and remember
                to leave KrakenEnv's own 'nmedia' as None (the default)
                so it is derived automatically and always matches
                whichever choice you make here (see KrakenEnv's
                docstring for the bug this consistency check prevents).
            sediment_top_properties (dict|None): physical properties
                (same keys as halfspace_properties: 'c_p', 'c_s', 'rho',
                'a_p', 'a_s') at the TOP of the buffer sediment layer
                (immediately below the seafloor). None (default) reuses
                halfspace_properties for the top too, giving a constant
                (isovelocity) buffer -- the only behaviour available
                before this parameter existed. Providing a distinct
                value here, together with a different 'halfspace_properties'
                for the bottom, produces a simple two-point LINEAR
                velocity/density/attenuation GRADIENT across the buffer
                layer (KRAKEN's 'C_linear' SSP interpolation connects
                the two points), which is a standard, realistic way to
                represent a sediment layer whose properties increase
                with depth due to compaction -- as opposed to the
                idealized isovelocity buffer. Only meaningful when
                add_sediment_buffer_layer is True.
        """
        self.sigma = sigma
        self.boundary_condition = boundary_condition
        self.halfspace_properties = halfspace_properties
        self.add_sediment_buffer_layer = add_sediment_buffer_layer
        self.sediment_top_properties = sediment_top_properties

        # "Buffer" sediment layer thickness (empirical rule:
        # alpha_wavelength wavelengths at the minimum frequency).
        # Only meaningful when add_sediment_buffer_layer is True; left
        # at 0 otherwise so z_in_bottom (used for plotting) stays a
        # degenerate [0, 0] range rather than a misleading thickness
        # that's never actually written to the '.env' file.
        if self.add_sediment_buffer_layer:
            self.sedim_layer_depth = alpha_wavelength * g.c0 / fmin
        else:
            self.sedim_layer_depth = 0
        self.z_in_bottom = np.array([0, self.sedim_layer_depth])
        self.sedim_layer_max_z = 10000  # Max allowed depth for this layer (m)
        self.sedim_layer_max_depth = None  # Computed later by derive_sedim_layer_max_depth

        self.write_sedim_layer_bloc = False
        self.use_halfspace_properties = False

        self.boundary_code = None
        self.available_boundary_conditions = list(self._BOUNDARY_CODES)
        self.set_boundary_code()

        self.bathymetry_code = ""

    def set_boundary_code(self):
        """Translate boundary_condition into the KRAKEN letter code and
        update the derived attributes (sedim_layer_depth, bottom
        properties)."""
        if self.boundary_condition not in self._BOUNDARY_CODES:
            raise ValueError(
                f"Unknown boundary condition '{self.boundary_condition}'. "
                f"Please pick one of the following: {self.available_boundary_conditions}"
            )

        self.boundary_code = self._BOUNDARY_CODES[self.boundary_condition]

        if self.boundary_condition == "vacuum":
            # No physical bottom -> no buffer sediment layer
            self.sedim_layer_depth = 0
            self.z_in_bottom = np.array([0, 0])
        elif self.boundary_condition == "acousto_elastic":
            self.set_halfspace_properties()
            # NOTE (bug fixed): this used to be unconditionally set to
            # True whenever boundary_condition == 'acousto_elastic',
            # regardless of what KrakenEnv's 'nmedia' was set to. Since
            # KrakenMedium.write_lines() checks this flag to decide
            # whether to append an extra medium block (the buffer
            # sediment layer), a user who left the default 'nmedia=1'
            # (a single water-column medium, no buffer) got a '.env'
            # file that DECLARED 1 medium but actually WROTE 2 medium
            # blocks -- KRAKEN's Fortran reader consumes exactly
            # 'nmedia' blocks before expecting the bottom boundary
            # condition line, so it misread the buffer block's mesh-info
            # line as that line instead, corrupting every line that
            # follows. Confirmed with a real KRAKEN/FIELD run that a
            # direct acousto-elastic half-space with no buffer and
            # nmedia=1 runs correctly -- the buffer was always optional.
            # 'write_sedim_layer_bloc' now follows the explicit
            # 'add_sediment_buffer_layer' choice instead of being forced
            # True, and KrakenEnv's 'nmedia' is derived from this same
            # flag by default (see KrakenEnv.__init__) so the two can no
            # longer silently disagree.
            self.write_sedim_layer_bloc = self.add_sediment_buffer_layer
        elif self.boundary_condition == "reflection_coefficient":
            warnings.warn(
                "reflection_coefficient' boundary condition requires bottom "
                "reflection coefficient to be provided in a separeted .'TRC' file"
            )
        elif self.boundary_condition == "precalculated_reflection_coefficient":
            warnings.warn(
                "precalculated_reflection_coefficient' boundary condition "
                "requires bottom reflection coefficient to precalculated by BOUNCE"
            )

    def set_bathymetry_code(self, use_bathymetry):
        """'~' to enable bathymetry reading by KRAKEN, '' otherwise
        (flat bottom)."""
        self.bathymetry_code = "~" if use_bathymetry else ""

    def set_halfspace_properties(self):
        """Fetch the bottom physical properties from
        self.halfspace_properties. Required for 'acousto_elastic'.

        Also resolves the properties at the TOP of the buffer sediment
        layer (self.cp_sedim_top etc.): self.sediment_top_properties if
        provided, otherwise the same halfspace_properties used at the
        bottom (isovelocity buffer, the only behaviour available before
        sediment_top_properties existed).
        """
        if self.halfspace_properties is None:
            raise ValueError(
                "You need to provide bottom halfspace properties when using "
                "'acousto_elastic' boundary condition"
            )
        self.cp_bot_halfspace = self.halfspace_properties["c_p"]
        self.cs_bot_halfspace = self.halfspace_properties["c_s"]
        self.rhobot_halfspace = self.halfspace_properties["rho"]
        self.apbot_halfspace = self.halfspace_properties["a_p"]
        self.ashbot_halfspace = self.halfspace_properties["a_s"]
        self.use_halfspace_properties = True

        top_properties = self.sediment_top_properties or self.halfspace_properties
        self.cp_sedim_top = top_properties["c_p"]
        self.cs_sedim_top = top_properties["c_s"]
        self.rho_sedim_top = top_properties["rho"]
        self.ap_sedim_top = top_properties["a_p"]
        self.ash_sedim_top = top_properties["a_s"]

    def derive_sedim_layer_max_depth(self, z_max):
        """Compute the depth at which the bottom half-space description
        line itself is written.

        - If add_sediment_buffer_layer is True (default): the maximum
          depth of the buffer sediment layer (capped by
          sedim_layer_max_z, rounded to the nearest 100 m for
          readability in the '.env' file).
        - If False: simply 'z_max' itself (the real local water depth),
          unrounded -- there is no buffer to extend past it, the
          half-space sits immediately below the water column.

        Args:
            z_max (float): depth of the bottom of the water column (m),
                typically medium.z_ssp.max() or bathy.bathy_depth.max()
                (or, in range-dependent mode, the LOCAL truncated
                medium depth at a given range -- see
                KrakenEnv.write_range_dependent_lines, which recomputes
                this per profile when add_sediment_buffer_layer=False).
        """
        if not self.add_sediment_buffer_layer:
            self.sedim_layer_max_depth = z_max
            return

        sedim_layer_z = z_max + self.sedim_layer_depth
        self.sedim_layer_max_depth = np.ceil(min(sedim_layer_z, self.sedim_layer_max_z))
        self.sedim_layer_max_depth = np.round(self.sedim_layer_max_depth * 1e-2, 0) * 1e2

    def write_lines(self, use_bathymetry=False, halfspace_depth=None):
        """Build self.lines: the '.env' line(s) describing the bottom
        halfspace.

        Args:
            use_bathymetry (bool): if True, enables the bathymetry code
                ('~') to tell KRAKEN to read a variable bottom profile
                rather than a flat bottom.
            halfspace_depth (float|None): depth (m) at which the
                half-space description line is written. None (default)
                uses self.sedim_layer_max_depth, i.e. the value computed
                once by derive_sedim_layer_max_depth() for the whole
                environment. Pass this explicitly in range-dependent
                mode with add_sediment_buffer_layer=False, where the
                half-space sits at the LOCAL (per-profile) water depth
                rather than a single value shared by every profile --
                see KrakenEnv.write_range_dependent_lines.
        """
        self.set_bathymetry_code(use_bathymetry)

        bottom_halfspace_info = align_var_description(
            f"'{self.boundary_code + self.bathymetry_code}' {self.sigma}",
            "Type of bottom boundary condition, Interfacial roughness",
        )
        self.lines = [bottom_halfspace_info]

        if self.use_halfspace_properties:
            depth = halfspace_depth if halfspace_depth is not None else self.sedim_layer_max_depth
            ssp_desc = (
                "Depth (m), C-wave celerity (m/s), S-wave celerity (m/s), "
                "Density (g/cm3), C-wave attenuation , S-wave attenuation"
            )
            self.lines.append(
                align_var_description(
                    f"{depth:.2f} {self.cp_bot_halfspace:.2f} "
                    f"{self.cs_bot_halfspace:.2f} {self.rhobot_halfspace:.2f} "
                    f"{self.apbot_halfspace:.2f} {self.ashbot_halfspace:.2f}",
                    ssp_desc,
                )
            )

    def set_default(self):
        """Reset to a default acousto-elastic sand bottom."""
        self.halfspace_properties = g.sand_properties
        self.boundary_condition = "acousto_elastic"
        self.set_boundary_code()
        self.sigma = 0.0
        self.bathymetry_code = ""

    # ------------------------------------------------------------------
    # Plotting tools
    # ------------------------------------------------------------------
    def plot_bottom_halfspace(self):
        """Plot SSP, attenuation and density of the bottom halfspace.

        Returns:
            matplotlib.figure.Figure
        """
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
        return fig


# ======================================================================================================================
# Attenuation
# ======================================================================================================================
class KrakenAttenuation:
    """Describes the attenuation units used throughout the '.env' file
    and whether Thorp volume attenuation should be added."""

    # NOTE: faithfully keeps the original codes, including the apparent
    # inconsistency between 'units' documented as "neper_per_m" in
    # available_units and the value actually tested, "nepers_per_m"
    # (with an 's'), in set_unitscode. Fixing this detail without
    # confirmation from the code owner could change behaviour that might
    # already be relied upon elsewhere -> flagged here rather than
    # fixed outright.
    _UNITS_CODES = {
        "nepers_per_m": "N",
        "dB_per_kmhz": "F",
        "dB_per_m": "M",
        "dB_per_wavelength": "W",
        "quality_factor": "Q",
        "thorp": "T",
    }

    def __init__(self, units="dB_per_wavelength", use_volume_attenuation=False):
        """
        Args:
            units (str): one of self.available_units.
            use_volume_attenuation (bool): add Thorp volume attenuation
                on top of the chosen units.
        """
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
        """Translate self.units into the KRAKEN letter code (self.unitscode)."""
        if self.units not in self._UNITS_CODES:
            raise ValueError(
                f"Unknown interpolation method '{self.units}'. "
                f"Please pick one of the following: {self.available_units}"
            )
        self.unitscode = self._UNITS_CODES[self.units]

    def set_thorp_code(self):
        """'T' if Thorp volume attenuation is enabled, ' ' otherwise."""
        self.thorp_code = "T" if self.volume_attenuation else " "

    def set_default(self):
        """Reset to dB/wavelength with no volume attenuation."""
        self.units = "dB_per_wavelength"
        self.set_unitscode()
        self.volume_attenuation = False
        self.set_thorp_code()


# ======================================================================================================================
# Field parameters (source/receiver grid used by KRAKEN)
# ======================================================================================================================
class KrakenField:
    """Describes the modal field parameters used by KRAKEN: phase speed
    limits, source depth(s), receiver depth grid.
    """

    def __init__(
        self,
        phase_speed_limits=None,
        src_depth=[5],
        n_rcv_z=1000,
        rcv_z_min=0.0,
        rcv_z_max=1000.0,
        rcv_r_max=0.0,
    ):
        """
        Args:
            phase_speed_limits (list[float]|None): [min, max] in m/s.
                Defaults to [0.0, 2000.0].
            src_depth (array-like): source depth(s) (m).
            n_rcv_z (int): number of receiver depths.
            rcv_z_min (float): min receiver depth (m).
            rcv_z_max (float): max receiver depth (m).
            rcv_r_max (float): maximum range (km).
        """
        # NOTE (bug fixed): in the original code, the default value
        # [0.0, 2000.0] was immediately overwritten by
        # `self.phase_speed_limits = np.array(phase_speed_limits)`,
        # which re-converted the original argument (still None) instead
        # of reusing the default value that had just been assigned.
        # Result: phase_speed_limits ended up as np.array(None), a 0-D
        # array of dtype 'object', and any attempt to index
        # `self.phase_speed_limits[0]` raised an IndexError.
        if phase_speed_limits is None:
            phase_speed_limits = [0.0, 2000.0]
        self.phase_speed_limits = np.array(phase_speed_limits)

        self.src_depth = np.atleast_1d(src_depth)

        self.n_rcv_z = n_rcv_z
        self.rcv_depth_min = rcv_z_min
        self.rcv_depth_max = rcv_z_max
        self.rcv_range_max = rcv_r_max

    def write_lines(self):
        """Build self.lines: the 5 '.env' lines of the "field" block."""
        self.lines = [
            align_var_description(
                f"{self.phase_speed_limits[0]} {self.phase_speed_limits[1]}",
                "Phase speed limits (min, max) (m/s)",
            ),
            align_var_description(f"{self.rcv_range_max}", "Maximum range (km)"),
            align_var_description(f"{self.src_depth.size}", "Number of source depths"),
            align_var_description(
                "".join(f"{src_d} " for src_d in self.src_depth),
                "Source depths (m)",
            ),
            align_var_description(f"{self.n_rcv_z}", "Number of receiver depths"),
            align_var_description(
                f"{self.rcv_depth_min} {self.rcv_depth_max} /",
                "Minimum and maximum receiver depths (m)",
            ),
        ]


# ======================================================================================================================
# Bathymetry
# ======================================================================================================================
class Bathymetry:
    """Loads a bathymetry profile (range, depth) from a two-column,
    header-less CSV file, and builds an interpolator if the bottom is
    not flat.
    """

    def __init__(self, data_file=None, units="km", interpolation_method="linear"):
        """
        Args:
            data_file (str|None): path to a CSV file (column 0 = range,
                column 1 = depth in m). None -> no bathymetry (flat
                bottom is left to the caller to handle).
            units (str): unit of the "range" column in the file: 'km' or
                'm'. Range is always converted to and stored in km
                (self.bathy_range).
            interpolation_method (str): passed to
                scipy.interpolate.interp1d (e.g. 'linear', 'cubic').
        """
        self.data_file = data_file
        self.units = units
        self.interpolation_method = interpolation_method

        if self.data_file is None:
            self.use_bathy = False
        else:
            self.load_data()

    def load_data(self):
        """Load the CSV file and determine whether the bottom is truly
        variable (use_bathy=True) or flat (use_bathy=False, even if a
        file was provided)."""
        if not os.path.exists(self.data_file):
            raise ValueError(f"Data file '{self.data_file}' does not exist")

        data = pd.read_csv(self.data_file, sep=",", header=None)

        if self.units == "km":
            self.bathy_range = data[0].to_numpy(copy=True)
        elif self.units == "m":
            self.bathy_range = data[0].to_numpy(copy=True) / 1000
        else:
            raise ValueError(
                f"Unknown units '{self.units}'. Please pick one of the following: 'km', 'm'"
            )

        self.bathy_depth = data[1].to_numpy(copy=True)

        if np.all(self.bathy_depth == self.bathy_depth[0]):
            # Flat bottom: no need to interpolate, KRAKEN can be run in
            # range-independent mode.
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


# ======================================================================================================================
# Full environment (assembles all the blocks above)
# ======================================================================================================================
class KrakenEnv:
    """Assembles all components (halfspaces, medium, attenuation, field,
    bathymetry) and writes the complete '.env' file.

    Two writing modes, selected automatically based on bathymetry:
      - range-independent: a single environment profile (flat bottom).
      - range-dependent: a different environment profile at each
        bathymetry range (needed for a variable bottom).
    """

    def __init__(
        self,
        title="",
        env_root="",
        env_filename="",
        freq=50.0,
        kraken_top_hs=None,
        kraken_medium=None,
        kraken_attenuation=None,
        kraken_bottom_hs=None,
        kraken_field=None,
        kraken_bathy=None,
        rModes=None,
        rModes_units="km",
        nmedia=None,
    ):
        """
        Args:
            title (str): simulation title (written as-is to the '.env' file).
            env_root (str): output directory for the '.env'/'.flp'/'.shd' files.
            env_filename (str): file name (without extension).
            freq (float|array-like): frequency/frequencies (Hz). An
                array automatically enables broadband mode
                (broadband_run).
            kraken_top_hs (KrakenTopHalfspace): surface halfspace.
            kraken_medium (KrakenMedium): water column.
            kraken_attenuation (KrakenAttenuation): attenuation units.
            kraken_bottom_hs (KrakenBottomHalfspace): bottom halfspace.
            kraken_field (KrakenField): field parameters.
            kraken_bathy (Bathymetry): bathymetry (flat bottom if None).
            rModes (array-like|None): ranges (km or m depending on
                rModes_units) at which a distinct profile should be
                written in range-dependent mode. Defaults to the
                bathymetry's own ranges.
            rModes_units (str): unit of rModes ('km' or 'm').
            nmedia (int|None): number of media blocks to DECLARE in the
                '.env' file. None (default, recommended): derived
                automatically as 1 (the water column) + 1 more if
                kraken_bottom_hs adds an automatic buffer sediment layer
                (see KrakenBottomHalfspace's add_sediment_buffer_layer).
                If you pass an explicit int, it is validated against
                this same automatically-derived value and a ValueError
                is raised on mismatch -- see "BUG FIXED" note below for
                why this validation exists.

        Note:
            The kraken_* arguments default to None and are resolved to a
            default instance inside the constructor body, rather than
            using a mutable object directly as a parameter default
            (classic Python pitfall: a default object is created only
            once, at function definition time, and would then be shared
            -- and mutated in place -- across every subsequent call that
            does not supply that argument).

        NOTE (bug fixed): 'nmedia' used to be a plain user-supplied
        integer (default 1), never cross-checked against how many medium
        blocks the '.env' file actually ends up containing. Whenever
        kraken_bottom_hs used an acousto-elastic bottom, KrakenMedium
        would ALWAYS append an extra "buffer sediment layer" block
        (see KrakenBottomHalfspace.add_sediment_buffer_layer), so
        nmedia=1 silently produced a file declaring 1 medium while
        actually writing 2 -- KRAKEN's Fortran reader consumes exactly
        'nmedia' blocks before expecting the bottom boundary condition
        line, so it misread the second block's mesh-info line as that
        line instead, corrupting everything that followed (this was
        confirmed to be the root cause of a real crash report: a
        range-dependent, single-medium, acousto-elastic-bottom
        environment failed with nmedia=1, and manually forcing nmedia=2
        -- with no other change -- fixed it). 'nmedia' is now derived
        automatically by default, and explicit values are validated
        against it so this class of mistake raises a clear error instead
        of producing a silently-corrupted '.env' file.
        """
        self.simulation_title = title

        self.root_ = env_root
        self.filename = env_filename
        self.env_fpath = os.path.join(self.root_, self.filename + ".env")
        self.flp_fpath = os.path.join(self.root_, self.filename + ".flp")
        self.shd_fpath = os.path.join(self.root_, self.filename + ".shd")

        # --- Frequencies: always stored sorted and de-duplicated.
        self.freq = np.unique(np.array(freq, dtype=float))
        if self.freq.size > 1:
            self.broadband_run = True
        else:
            self.broadband_run = False
        # NOTE (bug fixed): `float(self.freq)` raised a TypeError with
        # numpy >= 1.25 when self.freq is a single-element array (single
        # frequency case): `float()` on a non-0-D array is no longer
        # allowed. We explicitly extract the first (and only, after the
        # sort above) element.
        self.nominal_frequency = float(self.freq[0])

        self.top_hs = kraken_top_hs if kraken_top_hs is not None else KrakenTopHalfspace()
        self.medium = kraken_medium if kraken_medium is not None else KrakenMedium()
        self.att = kraken_attenuation if kraken_attenuation is not None else KrakenAttenuation()
        self.bottom_hs = kraken_bottom_hs if kraken_bottom_hs is not None else KrakenBottomHalfspace()
        self.field = kraken_field if kraken_field is not None else KrakenField()
        self.bathy = kraken_bathy if kraken_bathy is not None else Bathymetry()

        # --- Number of media blocks actually written vs. declared.
        required_nmedia = 1 + (1 if self.bottom_hs.write_sedim_layer_bloc else 0)
        if nmedia is None:
            self.nmedia = required_nmedia
        elif nmedia != required_nmedia:
            extra_medium_note = (
                "plus 1 for the acousto-elastic bottom's automatic buffer "
                "sediment layer "
                if self.bottom_hs.write_sedim_layer_bloc
                else ""
            )
            raise ValueError(
                f"nmedia={nmedia} was requested, but this environment will "
                f"actually write {required_nmedia} medium block(s): 1 for the "
                f"water column, {extra_medium_note}"
                f"(see KrakenBottomHalfspace's 'add_sediment_buffer_layer' "
                f"parameter to control this). Declaring a mismatched nmedia "
                f"produces a '.env' file KRAKEN's Fortran reader will "
                f"misparse (it reads exactly 'nmedia' medium blocks before "
                f"the bottom boundary condition line). Pass nmedia=None "
                f"(the default) to have it derived automatically, or pass "
                f"nmedia={required_nmedia} explicitly."
            )
        else:
            self.nmedia = nmedia

        self._init_modes_range(rModes, rModes_units)

        # --- Buffer sediment layer thickness, if not already set.
        # NOTE (bug fixed): the original code called
        # `self.bathy.bathy_depth.max()` without ever checking that
        # `self.bathy` actually held bathymetry data (the matching guard
        # was present ... but commented out). Result: AttributeError as
        # soon as kraken_bathy was not explicitly provided (a very common
        # use case: flat bottom). Falling back to the water column's max
        # depth is the natural default for a flat bottom.
        if self.bottom_hs.sedim_layer_max_depth is None:
            if self.bathy.use_bathy:
                z_max = self.bathy.bathy_depth.max()
            else:
                z_max = self.medium.z_ssp.max()
            self.bottom_hs.derive_sedim_layer_max_depth(z_max=z_max)

        self.range_dependent_env = False

    def _init_modes_range(self, rModes, rModes_units):
        """Determine the ranges (self.modes_range, in km, sorted, with 0
        included) at which a distinct environment profile must be
        written, when bathymetry is active.

        This method does nothing if bathymetry is not active (flat
        bottom): self.modes_range is then left undefined, exactly as in
        the original code (no caller accesses it in that case, since
        write_range_independent_lines is used instead).
        """
        if not self.bathy.use_bathy:
            return

        if rModes is not None:
            modes_range = np.asarray(rModes, dtype=float)
            if rModes_units == "m":
                modes_range = modes_range / 1000  # Convert to km
        else:
            modes_range = np.asarray(self.bathy.bathy_range, dtype=float)

        # NOTE (bug fixed): sorting was done in place with `.sort()`. An
        # array derived from a pandas DataFrame (via `.values`) can be
        # read-only with pandas' Copy-on-Write mode (default since
        # pandas 2.x, mandatory in pandas 3.x), which raised
        # `ValueError: sort array is read-only`. `np.sort` (out-of-place)
        # works in every case.
        modes_range = np.sort(modes_range)

        if modes_range[0] != 0:
            modes_range = np.append(0, modes_range)

        self.modes_range = modes_range

    # ------------------------------------------------------------------
    # Writing the '.env' file
    # ------------------------------------------------------------------
    def write_range_independent_lines(self):
        """Fill self.env_lines for a flat-bottom environment (a single
        profile, no range dependence)."""
        self.env_lines = []

        self.top_hs.write_lines(
            kraken_medium=self.medium,
            kraken_attenuation=self.att,
            broadband_run=self.broadband_run,
            slow_rootfinder=False,
        )
        self.medium.write_lines(bottom_hs=self.bottom_hs)
        # When there is no buffer sediment layer, the half-space sits
        # immediately below the water column: write it at the water
        # column's own max depth rather than at
        # self.bottom_hs.sedim_layer_max_depth (which, in that case,
        # already equals the same value -- see
        # KrakenBottomHalfspace.derive_sedim_layer_max_depth -- but
        # passing it explicitly keeps this call self-consistent with
        # the range-dependent case below, where it genuinely differs
        # per profile).
        halfspace_depth = (
            self.medium.z_ssp.max() if not self.bottom_hs.add_sediment_buffer_layer else None
        )
        self.bottom_hs.write_lines(use_bathymetry=self.bathy.use_bathy, halfspace_depth=halfspace_depth)
        self.field.write_lines()

        self._append_profile_lines(title=self.simulation_title, medium=self.medium)

        self.range_dependent_env = False

    def write_range_dependent_lines(self):
        """Fill self.env_lines for a variable-bottom environment: a full
        environment profile is written for every range in
        self.modes_range, with the SSP profile truncated/extended to
        stop exactly at the local bottom depth.

        Raises:
            ValueError: if the bottom is a DIRECT half-space
                (self.bottom_hs.add_sediment_buffer_layer is False) and
                the bathymetry's deepest point is NOT at r=0 (the first
                profile) -- see the check just below for why.
        """
        self.env_lines = []

        # NOTE (bug fixed): confirmed with a real KRAKEN/FIELD run
        # (comparing a failing and a working reproduction of the exact
        # same environment) that FIELD.exe crashes -- with a cryptic
        # Fortran runtime error ('Non-existing record number' in
        # EvaluateCMMod.f90) or, depending on version, the clearer
        # "Fatal Error: modes must be tabulated throughout the ocean and
        # sediment to compute the coupling coefs." -- whenever a DIRECT
        # half-space bottom (add_sediment_buffer_layer=False) is used in
        # a range-dependent run AND the first profile (r=0) is NOT the
        # single deepest point along the whole bathymetry. FIELD.exe
        # appears to size its per-profile mode-file record length from
        # the FIRST profile's own tabulated depth; any LATER, deeper
        # profile then needs more records than that size allows,
        # crashing the read. This was never checked or handled before,
        # producing a hard, unrelated-looking crash instead of a clear
        # explanation. When a buffer sediment layer IS used instead
        # (add_sediment_buffer_layer=True, the default), this cannot
        # happen: its thickness is already derived from the
        # bathymetry's GLOBAL maximum depth (see __init__ and
        # KrakenBottomHalfspace.derive_sedim_layer_max_depth), so EVERY
        # profile's terminal boundary -- including the first -- already
        # sits at least as deep as every other profile, regardless of
        # where the true deepest point falls along the range. So: fail
        # loudly and explain the fix, rather than let this reach
        # FIELD.exe at all.
        if self.bathy.use_bathy and not self.bottom_hs.add_sediment_buffer_layer:
            deepest_at_start = self.bathy.bathy_depth[0] >= self.bathy.bathy_depth.max()
            if not deepest_at_start:
                deepest_r = self.bathy.bathy_range[np.argmax(self.bathy.bathy_depth)]
                raise ValueError(
                    "This range-dependent environment's bathymetry does not "
                    f"have its deepest point at r=0: the first profile is "
                    f"{self.bathy.bathy_depth[0]:.1f} m deep, but the "
                    f"bathymetry reaches {self.bathy.bathy_depth.max():.1f} m "
                    f"at r={deepest_r:.2f} km. With a DIRECT half-space bottom "
                    "(KrakenBottomHalfspace(add_sediment_buffer_layer=False)), "
                    "FIELD.exe requires the FIRST profile to be tabulated at "
                    "least as deep as every other profile along the range, or "
                    "it crashes with a cryptic Fortran runtime error "
                    "('Non-existing record number') or "
                    "\"Fatal Error: modes must be tabulated throughout the "
                    "ocean and sediment to compute the coupling coefs.\" -- "
                    "confirmed with a real KRAKEN/FIELD run. Fix: use "
                    "KrakenBottomHalfspace(add_sediment_buffer_layer=True, "
                    "...) instead (the default), which sizes its buffer "
                    "sediment layer from the bathymetry's global maximum "
                    "depth automatically, satisfying this requirement for "
                    "every profile regardless of where the deepest point "
                    "falls along the range."
                )

        self.top_hs.write_lines(
            kraken_medium=self.medium,
            kraken_attenuation=self.att,
            broadband_run=self.broadband_run,
            slow_rootfinder=False,
        )
        self.field.write_lines()

        for r_km in self.modes_range:
            local_depth = self.bathy.interpolator(r_km)
            medium_at_range = self._medium_truncated_to_depth(local_depth)

            medium_at_range.write_lines(bottom_hs=self.bottom_hs)
            # NOTE (bug fixed, see KrakenEnv's docstring): when there is
            # no buffer sediment layer, the half-space must sit at THIS
            # profile's own local water depth (which varies range by
            # range, e.g. following the bathymetry), not at a single
            # value shared by every profile. Passing 'local_depth'
            # explicitly overrides self.bottom_hs.sedim_layer_max_depth
            # (which is computed once, globally, in __init__) for this
            # profile only. When a buffer IS used, this stays None and
            # every profile keeps using that single shared value, which
            # is deliberate: the buffer is sized to reach past the
            # deepest point of the whole bathymetry, regardless of how
            # shallow any individual profile's local depth is.
            halfspace_depth = local_depth if not self.bottom_hs.add_sediment_buffer_layer else None
            self.bottom_hs.write_lines(use_bathymetry=self.bathy.use_bathy, halfspace_depth=halfspace_depth)

            title = f"{self.simulation_title} - r = {r_km:.2f} km"
            self._append_profile_lines(title=title, medium=medium_at_range)

        self.range_dependent_env = True

    def _medium_truncated_to_depth(self, depth):
        """Return a copy of self.medium whose SSP profile stops exactly
        at 'depth' (local bottom depth at the range under consideration),
        adding an interpolated point at that depth if needed.

        Factors out the logic that was written inline in
        write_range_dependent_lines, and fixes a bug in how the
        secondary properties (cs, rho, ap, ash) were extended for the
        new point.
        """
        medium_copy = copy.deepcopy(self.medium)

        idx_in_range = medium_copy.z_ssp <= depth
        z_ssp = medium_copy.z_ssp[idx_in_range]
        cp = medium_copy.cp_ssp[idx_in_range] if medium_copy.cp_ssp.size == medium_copy.z_ssp.size else medium_copy.cp_ssp

        if depth > z_ssp[-1]:
            # The local bottom is deeper than the last point of the
            # truncated profile: interpolate the celerity at "depth" and
            # extend the other properties (cs, rho, ap, ash) by
            # repeating their last known value, which is the most
            # reasonable assumption absent any further information.
            cp_new_point = np.interp(depth, medium_copy.z_ssp, medium_copy.cp_ssp)
            cp = np.append(cp, cp_new_point)

            # NOTE (bug fixed): the original code called, e.g.,
            # `np.append(depth, medium_copy.z_ssp, medium_copy.ash)`
            # with 3 positional arguments, whereas np.append's signature
            # is `np.append(arr, values)`. This call would have raised a
            # TypeError as soon as a property (cs/rho/ap/ash) was
            # provided as a full array (same size as z_ssp) rather than
            # as a scalar, in a range-dependent case with profile
            # extension.
            for attr_name in ("cs_ssp", "rho", "ap", "ash"):
                attr = getattr(medium_copy, attr_name)
                if attr.size == self.medium.z_ssp.size:
                    truncated = attr[idx_in_range]
                    extended = np.append(truncated, truncated[-1])
                    setattr(medium_copy, attr_name, extended)

            z_ssp = np.append(z_ssp, depth)

        medium_copy.z_ssp = z_ssp
        medium_copy.cp_ssp = cp
        return medium_copy

    def _append_profile_lines(self, title, medium):
        """Append a full environment profile to self.env_lines (title,
        nominal frequency, number of media, then the 4 blocks already
        built by the sub-components' write_lines()).

        Args:
            title (str): title of this profile (may include the range in
                range-dependent mode).
            medium (KrakenMedium): medium to use for this profile
                (self.medium in range-independent mode, a truncated copy
                in range-dependent mode).
        """
        self.env_lines.append(f"'{title}'\n")
        self.env_lines.append(
            align_var_description(f"{self.nominal_frequency}", "Nominal frequency (Hz)")
        )
        self.env_lines.append(align_var_description(f"{self.nmedia}", "Number of media"))

        self.env_lines += self.top_hs.lines
        self.env_lines += medium.lines
        self.env_lines += self.bottom_hs.lines
        self.env_lines += self.field.lines

        if self.broadband_run:
            self.env_lines.append(
                align_var_description(f"{self.freq.size}", "Number of frequencies")
            )
            self.env_lines.append(
                align_var_description(
                    " ".join(str(f) for f in self.freq), "Frequencies (Hz)"
                )
            )

    def write_env(self):
        """Write the complete '.env' file (automatically selects
        range-independent or range-dependent mode based on bathymetry)."""
        if self.bathy.use_bathy:
            self.write_range_dependent_lines()
        else:
            self.write_range_independent_lines()

        with open(self.env_fpath, "w") as f_out:
            f_out.writelines(self.env_lines)

    @property
    def root(self):
        return self.root_

    @root.setter
    def root(self, root):
        """Change the output directory and update the 3 derived file
        paths (used in particular by KrakenManager to redirect each
        parallel worker to its own directory)."""
        self.root_ = root
        self.env_fpath = os.path.join(self.root_, self.filename + ".env")
        self.flp_fpath = os.path.join(self.root_, self.filename + ".flp")
        self.shd_fpath = os.path.join(self.root_, self.filename + ".shd")

    # ------------------------------------------------------------------
    # Plotting tools
    # ------------------------------------------------------------------
    def plot_env(self, plot_src=False, src_depth=None):
        """Plot SSP, attenuation and density across the full water depth
        + bottom halfspace (overview of the environment).

        Args:
            plot_src (bool): mark the source depth on every subplot.
            src_depth (float): source depth (m), required if plot_src.

        Returns:
            matplotlib.figure.Figure
        """
        pfig = PubFigure(titlepad=50, labelpad=25)
        fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
        axs[0].set_ylabel("Depth [m]")

        n_med = self.medium.z_ssp.size
        z_bottom = self.medium.z_ssp[-1]

        # NOTE (enhancement): when the bottom half-space has no buffer
        # sediment layer (KrakenBottomHalfspace(add_sediment_buffer_layer=False),
        # confirmed to be a valid, commonly used direct-half-space
        # configuration -- see KrakenBottomHalfspace's docstring),
        # 'z_in_bottom' is [0, 0]: a genuinely semi-infinite half-space
        # has no real thickness to plot. Left as-is, the figure would
        # show only the water column, with no visible indication that a
        # bottom half-space exists at all below it. Since no plot can
        # show a truly infinite half-space anyway, a fixed, arbitrary
        # visual extent (20% of the water depth) is used purely for
        # this figure -- it does not affect the environment's actual
        # definition or the '.env' file in any way.
        z_in_bottom = self.bottom_hs.z_in_bottom
        if z_in_bottom.max() <= 0:
            z_in_bottom = np.array([0.0, 0.2 * z_bottom])
        n_bot = z_in_bottom.size
        z_env = np.append(self.medium.z_ssp, z_in_bottom + z_bottom)

        cp_env = np.append(
            _broadcast_to_size(self.medium.cp_ssp, n_med),
            _broadcast_to_size(self.bottom_hs.cp_bot_halfspace, n_bot),
        )
        cs_env = np.append(
            _broadcast_to_size(self.medium.cs_ssp, n_med),
            _broadcast_to_size(self.bottom_hs.cs_bot_halfspace, n_bot),
        )
        plot_ssp(cp_ssp=cp_env, cs_ssp=cs_env, z=z_env, z_bottom=z_bottom, ax=axs[0])

        ap_env = np.append(
            _broadcast_to_size(self.medium.ap, n_med),
            _broadcast_to_size(self.bottom_hs.apbot_halfspace, n_bot),
        )
        ash_env = np.append(
            _broadcast_to_size(self.medium.ash, n_med),
            _broadcast_to_size(self.bottom_hs.ashbot_halfspace, n_bot),
        )
        plot_attenuation(ap=ap_env, ash=ash_env, z=z_env, z_bottom=z_bottom, ax=axs[1])

        rho_env = np.append(
            _broadcast_to_size(self.medium.rho, n_med),
            _broadcast_to_size(self.bottom_hs.rhobot_halfspace, n_bot),
        )
        plot_density(rho=rho_env, z=z_env, z_bottom=z_bottom, ax=axs[2])

        if plot_src:
            for i in range(3):
                xmin = axs[i].get_xlim()[0]
                axs[i].scatter(xmin, src_depth, s=30, color="k")
                for s in [200, 500]:
                    axs[i].scatter(
                        xmin, src_depth, s=s, facecolors="None", edgecolors="k", linewidths=0.5
                    )

        plt.suptitle("Waveguide properties")
        return fig


# ======================================================================================================================
# '.flp' file (source/receiver grid + propagation theory)
# ======================================================================================================================
class KrakenFlp:
    """Describes and writes the '.flp' file consumed by the FIELD
    executable: propagation theory (coupled/adiabatic modes), receiver
    range/depth grid, source depth(s).
    """

    _SRC_CODES = {"point_source": "R", "line_source": "X"}
    _THEORY_CODES = {"coupled": "C", "adiabatic": "A"}
    _ADDITION_CODES = {"coherent": "C", "incoherent": "I"}

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
        """
        Args:
            env (KrakenEnv): associated environment (provides the '.flp'
                output path, the title, and, in range-dependent mode,
                the profile ranges).
            src_type (str): 'point_source' or 'line_source'.
            mode_theory (str): 'coupled' or 'adiabatic'.
            mode_addition (str): 'coherent' or 'incoherent'.
            nb_modes (int): max number of modes to use.
            src_depth (array-like): source depth(s) (m).
            n_rcv_z (int): number of receiver depths.
            rcv_z_min, rcv_z_max (float): receiver depth bounds (m).
            n_rcv_r (int): number of receiver ranges.
            rcv_r_min, rcv_r_max (float): receiver range bounds (km).
            rcv_dist_offset (float): receiver range offset (m).
        """
        self.env = env
        self.flp_fpath = self.env.flp_fpath
        self.title = self.env.simulation_title
        self.src_type = src_type
        self.mode_theory = mode_theory
        self.mode_addition = mode_addition
        self.nb_modes = nb_modes

        if self.env.range_dependent_env:
            self.n_profiles = self.env.modes_range.size
            self.profiles_ranges = self.env.modes_range
        else:
            self.n_profiles = 1
            self.profiles_ranges = np.array([0.0])

        self.src_z = np.atleast_1d(src_depth)

        self.n_rcv_z = int(n_rcv_z)
        self.rcv_z_min = rcv_z_min
        self.rcv_z_max = rcv_z_max

        self.n_rcv_r = int(n_rcv_r)
        self.rcv_r_min = int(np.floor(rcv_r_min))
        self.rcv_r_max = int(np.ceil(rcv_r_max))
        self.rcv_dist_offset = int(rcv_dist_offset)

        self.set_codes()

    def set_codes(self):
        """Translate src_type/mode_theory/mode_addition into KRAKEN
        letter codes (src_code, th_code, add_code)."""
        if self.src_type not in self._SRC_CODES:
            raise ValueError(
                f"Unknown mode theory method '{self.src_type}'. "
                f"Please pick one of the following: 'point_source', 'line_source'"
            )
        self.src_code = self._SRC_CODES[self.src_type]

        if self.mode_theory not in self._THEORY_CODES:
            raise ValueError(
                f"Unknown mode theory method '{self.mode_theory}'. "
                f"Please pick one of the following: 'coupled', 'adiabatic'"
            )
        self.th_code = self._THEORY_CODES[self.mode_theory]

        if self.mode_addition not in self._ADDITION_CODES:
            raise ValueError(
                f"Unknown addition mode '{self.mode_addition}'. "
                f"Please pick one of the following: 'coherent', 'incoherent'"
            )
        self.add_code = self._ADDITION_CODES[self.mode_addition]

    def write_lines(self):
        """Build self.lines: all lines of the '.flp' file."""
        self.lines = [
            f"'{self.title}'\n",
            align_var_description(
                f"'{self.src_code}{self.th_code} {self.add_code}'",
                "Source type, Mode theory, Mode addition",
            ),
            align_var_description(f"{self.nb_modes}", "Number of modes"),
            align_var_description(f"{self.n_profiles}", "Number of profiles"),
            align_var_description(
                " ".join(f"{r:.4f}" for r in self.profiles_ranges) + " /",
                "Profile ranges (km)",
            ),
            align_var_description(f"{self.n_rcv_r}", "Number of receiver ranges"),
            align_var_description(
                f"{self.rcv_r_min} {self.rcv_r_max} /", "Receiver ranges (km)"
            ),
            align_var_description(f"{self.src_z.size}", "Number of source depth"),
            align_var_description(
                "".join(f"{src_d} " for src_d in self.src_z) + " /",
                "Source depths (m)",
            ),
            align_var_description(f"{self.n_rcv_z}", "Number of receiver depths"),
            align_var_description(
                f"{self.rcv_z_min} {self.rcv_z_max} /", "Receiver depths (m)"
            ),
            align_var_description(
                f"{self.n_rcv_z}", "Number of receiver range-displacements"
            ),
            align_var_description(f"{self.rcv_dist_offset} /", "Receiver displacements (m)"),
        ]

    def write_flp(self):
        """Write the '.flp' file to disk."""
        self.write_lines()
        with open(self.flp_fpath, "w") as f_out:
            f_out.writelines(self.lines)


if __name__ == "__main__":
    # Small manual usage example -> adapt to your own directory layout.
    project_root = os.getcwd()
    test_root = os.path.join(project_root, r"propa\kraken_toolbox\tests\kraken_env")

    top_hs = KrakenTopHalfspace()

    z_ssp = np.array([0.0, 100.0, 500, 600, 700, 1000.0])
    cp_ssp = np.array([1500.0, 1550.0, 1540.0, 1532.0, 1522.0, 1512.0])
    medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

    bathy_fpath = os.path.join(test_root, "bathy_data.csv")
    bathy = Bathymetry(data_file=bathy_fpath, interpolation_method="linear", units="m")

    bott_hs_properties = dict(g.sand_properties)
    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(halfspace_properties=bott_hs_properties)

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
