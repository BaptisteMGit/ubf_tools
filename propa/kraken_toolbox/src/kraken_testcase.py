#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   kraken_testcase.py
@Time    :   2025/04/24 13:36:53
@Author  :   Menetrier Baptiste
@Version :   1.1 (refactor)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   High-level class assembling a complete KRAKEN test case
             (domain, source, receivers, environment) and managing the
             writing of input files and associated diagnostic plots.

This module does NOT change the public API of the original file (same
class/method/parameter names).

------------------------------------------------------------------------
IMPORTANT BUG FIXED: mutable default arguments
------------------------------------------------------------------------
The original file defined default values such as:

    class KrakenProperties:
        def __init__(self, ..., field=KrakenField(...), ...):
            ...

    class KrakenTestCase:
        def __init__(self, ..., kraken_properties=KrakenProperties(), ...):
            ...

In Python, a default value is evaluated ONCE, when the function is
defined (i.e. at module load time) -- not on every call. Every call to
`KrakenTestCase(...)` that does not explicitly supply
`kraken_properties=` (or `domain_properties=`, `src_properties=`,
`rcv_properties=`) then reuses THE EXACT SAME OBJECT.

Now, `KrakenTestCase.set_bathy()` mutates this object in place
(`self.domain.zmax_m = ...`, `self.kraken.field.n_rcv_z = ...`,
`self.kraken.field.rcv_depth_max = ...`). Concrete consequence: creating
TWO `KrakenTestCase` instances in a row without an explicit argument
means the second one silently overwrites the domain/field properties of
the first one (they share the same Python object). This is the kind of
bug that stays invisible on a single, isolated test case (hence its
presence in the original code, never triggered by the
`if __name__ == "__main__"` block, which only creates one), but that can
silently corrupt results as soon as several test cases are created
within the same session (sensitivity loop, simulation batch...).

Fixed here by using `None` as the default value and creating the object
inside the function body -- the only reliable way to get a "fresh"
object on every call (a classic pitfall documented in the Python
FAQ/Zen: "default parameter values are evaluated once").

Other clean-ups:
  - removed unused imports ('socket', 'get_subprocess_working_dir',
    neither of which was ever used in this file);
  - explicit validation of the 'unit' parameter in DomainProperties /
    ReceiverProperties (the original code raised a hard-to-understand
    UnboundLocalError if 'unit' was neither 'm' nor 'km');
  - factored out the logic shared by DomainProperties / ReceiverProperties
    (in the original code, these are two strictly identical classes --
    both names are kept to avoid breaking calling code that would
    distinguish the two by type).
------------------------------------------------------------------------
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from propa.kraken_toolbox.src.kraken_env import (
    KrakenEnv,
    KrakenTopHalfspace,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
    KrakenFlp,
    Bathymetry,
)

from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from propa.kraken_toolbox.utils import default_nb_rcv_z
from propa.kraken_toolbox.plot_utils import plotshd, plotmode

from publication.publication_figure import PubFigure


# ======================================================================================================================
# Unit conversion shared by DomainProperties / ReceiverProperties
# ======================================================================================================================
def _resolve_range_depth_units(zmin, zmax, rmin, rmax, unit):
    """Convert depth/range bounds to the internal units expected by
    KRAKEN: depths in meters, ranges in kilometers.

    Args:
        zmin, zmax: depth bounds, in unit 'unit'.
        rmin, rmax: range bounds, in unit 'unit'.
        unit (str): 'm' (depths are already in m, ranges must be
            converted from m to km) or 'km' (ranges are already in km,
            depths must be converted from km to m).

    Returns:
        tuple(zmin_m, zmax_m, rmin_km, rmax_km)
    """
    if unit == "m":
        alpha_z, alpha_r = 1, 1e-3
    elif unit == "km":
        alpha_z, alpha_r = 1e3, 1
    else:
        # NOTE (bug fixed): the original code only defined
        # alpha_z/alpha_r in the 'm'/'km' branches; any other value of
        # 'unit' triggered an UnboundLocalError with no obvious link to
        # the real cause (an invalid 'unit' parameter).
        raise ValueError(f"Unknown unit '{unit}'. Please pick one of: 'm', 'km'")

    return zmin * alpha_z, zmax * alpha_z, rmin * alpha_r, rmax * alpha_r


class DomainProperties:
    """Depth/range bounds of the simulation domain."""

    def __init__(self, zmin=0, zmax=1000, rmin=0, rmax=10 * 1e3, unit="m"):
        """
        Args:
            zmin, zmax: domain depth bounds, in unit 'unit'.
            rmin, rmax: domain range bounds, in unit 'unit'.
            unit (str): 'm' or 'km', unit of zmin/zmax/rmin/rmax on input.
        """
        self.zmin_m, self.zmax_m, self.rmin_km, self.rmax_km = _resolve_range_depth_units(
            zmin, zmax, rmin, rmax, unit
        )


class ReceiverProperties:
    """Depth/range bounds of the receiver grid.

    NOTE: strictly identical to DomainProperties in the original code
    (same parameters, same logic); kept as a separate class so as not to
    change the public API, but relies on the same unit-conversion
    function.
    """

    def __init__(self, zmin=0, zmax=1000, rmin=0, rmax=10 * 1e3, unit="m"):
        self.zmin_m, self.zmax_m, self.rmin_km, self.rmax_km = _resolve_range_depth_units(
            zmin, zmax, rmin, rmax, unit
        )


class SourceProperties:
    """Source type, depth, and frequency/frequencies."""

    def __init__(self, src_type="point_source", src_depth=100, freq=50):
        """
        Args:
            src_type (str): 'point_source' or 'line_source'.
            src_depth (float): source depth (m).
            freq (float|array-like): frequency/frequencies (Hz).
        """
        self.type = src_type
        self.depth = src_depth
        self.freq = np.atleast_1d(freq)


class KrakenProperties:
    """Groups all the KRAKEN parameters needed to build a
    KrakenEnv/KrakenFlp: propagation theory, grid resolution,
    halfspaces, medium, and field.
    """

    def __init__(
        self,
        mode_coupling="adiabatic",
        mode_addition="coherent",
        n_mode=100,
        nr=1000,
        nz=1000,
        nmedia=None,
        top_hs=None,
        bott_hs=None,
        att=None,
        medium=None,
        field=None,
    ):
        """
        Args:
            mode_coupling (str): 'adiabatic' or 'coupled'.
            mode_addition (str): 'coherent' or 'incoherent'.
            n_mode (int): max number of modes.
            nr (int): number of receiver ranges (FIELD grid).
            nz (int): number of receiver depths (FIELD grid).
            nmedia (int|None): number of media blocks to declare in the
                '.env' file, passed straight through to KrakenEnv. None
                (default, recommended) lets KrakenEnv derive it
                automatically from bott_hs (1, or 2 if bott_hs adds an
                automatic buffer sediment layer -- see
                KrakenBottomHalfspace's add_sediment_buffer_layer and
                KrakenEnv's docstring for the bug this prevents). Only
                pass an explicit value if you have a specific reason to
                override the derived one; it will be validated and
                raise a clear error on mismatch.
            top_hs (KrakenTopHalfspace|None): defaults to a new
                KrakenTopHalfspace() (vacuum).
            bott_hs (KrakenBottomHalfspace|None): defaults to a new
                KrakenBottomHalfspace() (sand, acousto-elastic, with the
                automatic buffer sediment layer enabled -- hence why the
                default 'nmedia' derivation above resolves to 2 unless
                you pass a different bott_hs or nmedia yourself).
            att (KrakenAttenuation|None): defaults to a new
                KrakenAttenuation() (dB/wavelength).
            medium (KrakenMedium|None): defaults to a 1000 m water
                column at a constant 1500 m/s.
            field (KrakenField|None): defaults to a KrakenField
                calibrated for a 1000 m domain and a 100 m source depth.

        Note:
            As with KrakenTestCase (see module docstring), default
            values are resolved HERE (at call time) rather than in the
            function signature, to avoid every KrakenProperties()
            created without explicit arguments sharing (and mutating in
            place) the same KrakenField / KrakenBottomHalfspace / etc.
            objects.
        """
        self.mode_coupling = mode_coupling
        self.mode_addition = mode_addition
        self.n_mode = n_mode
        self.nr = nr
        self.nz = nz
        self.nmedia = nmedia

        self.top_hs = top_hs if top_hs is not None else KrakenTopHalfspace()
        self.bott_hs = bott_hs if bott_hs is not None else KrakenBottomHalfspace()
        self.att = att if att is not None else KrakenAttenuation()

        if medium is not None:
            self.medium = medium
        else:
            default_domain = DomainProperties()
            self.medium = KrakenMedium(z_ssp=[0, default_domain.zmax_m])

        if field is not None:
            self.field = field
        else:
            default_domain = DomainProperties()
            default_src = SourceProperties()
            self.field = KrakenField(
                n_rcv_z=1000,
                src_depth=default_src.depth,
                rcv_z_max=default_domain.zmax_m + KrakenBottomHalfspace().sedim_layer_depth,
                phase_speed_limits=[1000, 20000],
            )


class KrakenTestCase:
    """Assembles a complete KRAKEN test case: creates the output
    directory tree, the bathymetry, the environment ('.env') and the
    field parameters ('.flp'), and provides utilities to run the
    simulation and plot diagnostic figures.
    """

    def __init__(
        self,
        name: str,
        root_dir: str,
        domain_properties: DomainProperties = None,
        src_properties: SourceProperties = None,
        rcv_properties: ReceiverProperties = None,
        kraken_properties: KrakenProperties = None,
        bathy: Bathymetry = None,
        title: str = "Default testcase",
        mode: str = "run",
    ):
        """
        Args:
            name (str): test case name (used as the '.env'/'.flp' file
                name and as the output subdirectory name).
            root_dir (str): root directory in which the test case
                subdirectory is created.
            domain_properties (DomainProperties|None): defaults to a new
                DomainProperties().
            src_properties (SourceProperties|None): defaults to a new
                SourceProperties().
            rcv_properties (ReceiverProperties|None): defaults to a new
                ReceiverProperties().
            kraken_properties (KrakenProperties|None): defaults to a new
                KrakenProperties().
            bathy (Bathymetry|None): bathymetry to use. None -> flat
                bottom, at the domain's max depth.
            title (str): simulation title (written to the '.env' file).
            mode (str): 'run' (default, just writes the input files) or
                'demo' (also writes the environment diagnostic plots).
                NB: regardless of 'mode', self.run() must be called
                explicitly to actually launch KRAKEN/FIELD -- the name
                "run" does not trigger it automatically (same behaviour
                as the original code, just documented here).
        """
        self.name = name
        self.root_dir = root_dir
        self.title = title
        self.mode = mode

        self.init_testcase_dirs()

        # NOTE (bug fixed): see module docstring -- these objects are
        # created HERE (a fresh object per test case) rather than being a
        # shared default value in the signature.
        self.src = src_properties if src_properties is not None else SourceProperties()
        self.rcv = rcv_properties if rcv_properties is not None else ReceiverProperties()
        self.domain = domain_properties if domain_properties is not None else DomainProperties()
        self.kraken = kraken_properties if kraken_properties is not None else KrakenProperties()

        self.bathy = bathy

        # Flags used by plot_testcase_env() in 'demo' mode
        self.plot_medium = True
        self.plot_bottom = True
        self.plot_bathy = True
        self.plot_env = True

        self.pre_process_testcase()

    # ------------------------------------------------------------------
    # Building the bathymetry / environment / field
    # ------------------------------------------------------------------
    def set_bathy(self):
        """Determine the bathymetry to use (flat by default, or
        user-supplied and truncated to the range domain), save it to
        io_files_dir/bathy.csv (for traceability: the file actually used
        is always visible alongside the '.env'/'.flp' files), then
        reload a clean Bathymetry object from that file.

        If a variable bathymetry is supplied, also updates:
          - self.domain.zmax_m (actual max depth reached);
          - self.kraken.field.n_rcv_z / rcv_depth_max (receiver grid
            recomputed to cover the new depth domain).
        """
        if self.bathy is None:
            # Flat bottom over the whole range domain
            r_km = np.array([self.domain.rmin_km, self.domain.rmax_km])
            h_m = np.array([self.domain.zmax_m, self.domain.zmax_m])
        else:
            r_km, h_m = self._bathy_truncated_to_domain()

        bathy_path = os.path.join(self.io_files_dir, "bathy.csv")
        pd.DataFrame({"r": np.round(r_km, 3), "h": np.round(h_m, 3)}).to_csv(
            bathy_path, index=False, header=False
        )
        # Reload from the written file: guarantees that self.bathy
        # exactly reflects what was written to disk (and therefore what
        # KRAKEN will actually use), while also benefiting from the
        # "flat bottom" detection performed by Bathymetry.load_data().
        self.bathy = Bathymetry(bathy_path)

    def _bathy_truncated_to_domain(self):
        """Truncate the user-supplied bathymetry to the range domain
        (self.domain.rmax_km), and update self.domain / self.kraken.field
        accordingly.

        Returns:
            tuple(r_km, h_m): truncated ranges (km) and depths (m).
        """
        r_km = np.asarray(self.bathy.bathy_range, dtype=float)
        h_m = np.asarray(self.bathy.bathy_depth, dtype=float)

        idx_in_range_domain = r_km <= self.domain.rmax_km
        r_km = r_km[idx_in_range_domain]
        h_m = h_m[idx_in_range_domain]

        self.domain.zmax_m = np.max(h_m)

        # New receiver grid max depth: domain depth + buffer sediment
        # layer, rounded to the nearest 100 m for readability.
        max_domain_depth = np.round(
            (self.domain.zmax_m + self.kraken.bott_hs.sedim_layer_depth) * 1e-2, 0
        ) * 1e2
        n_rcv_z = default_nb_rcv_z(fmax=np.max(self.src.freq), max_depth=max_domain_depth)

        self.kraken.field.n_rcv_z = n_rcv_z
        self.kraken.field.rcv_depth_max = max_domain_depth

        return r_km, h_m

    def set_env(self):
        """Build self.env (KrakenEnv) from the test case properties, and
        write it once (needed so that env.range_dependent_env is
        correctly set before building the '.flp', which depends on it --
        see KrakenFlp.__init__)."""
        self.env = KrakenEnv(
            title=self.title,
            env_root=self.io_files_dir,
            env_filename=self.name,
            freq=self.src.freq,
            kraken_top_hs=self.kraken.top_hs,
            kraken_medium=self.kraken.medium,
            kraken_attenuation=self.kraken.att,
            kraken_bottom_hs=self.kraken.bott_hs,
            kraken_field=self.kraken.field,
            kraken_bathy=self.bathy,
            nmedia=self.kraken.nmedia,
        )
        self.env.write_env()

    def set_flp(self):
        """Build self.flp (KrakenFlp) from self.env (already written
        once by set_env) and the test case's source/receiver
        properties."""
        self.flp = KrakenFlp(
            env=self.env,
            src_type=self.src.type,
            src_depth=self.src.depth,
            mode_theory=self.kraken.mode_coupling,
            mode_addition=self.kraken.mode_addition,
            nb_modes=self.kraken.n_mode,
            rcv_r_min=self.rcv.rmin_km,
            rcv_r_max=self.rcv.rmax_km,
            rcv_z_min=self.rcv.zmin_m,
            rcv_z_max=self.rcv.zmax_m,
            n_rcv_r=self.kraken.nr,
            n_rcv_z=self.kraken.nz,
        )

    def load(self):
        """Successively build bathymetry, environment, and field."""
        self.set_bathy()
        self.set_env()
        self.set_flp()

    def write_kraken_files(self):
        """(Re-)write the '.env' and '.flp' files to disk."""
        self.env.write_env()
        self.flp.write_flp()

    def pre_process_testcase(self):
        """Pipeline automatically run by __init__: builds and writes the
        input files, then ('demo' mode only) plots the environment
        diagnostic figures."""
        self.load()
        self.write_kraken_files()
        if self.mode == "demo":
            self.plot_testcase_env()

    # ------------------------------------------------------------------
    # Output directory tree
    # ------------------------------------------------------------------
    def init_testcase_dirs(self):
        """Create the test case output directory tree:

            <root_dir>/<name>/
                io_files/          '.env', '.flp', '.shd' files, bathy.csv
                imgs/env/          environment diagnostic figures
                imgs/outputs/      result figures (modes, TL...)
        """
        self.testcase_directory = os.path.join(self.root_dir, self.name)
        self.io_files_dir = os.path.join(self.testcase_directory, "io_files")
        self.imgs_dir = os.path.join(self.testcase_directory, "imgs")
        self.imgs_env_dir = os.path.join(self.imgs_dir, "env")
        self.imgs_outputs_dir = os.path.join(self.imgs_dir, "outputs")

        for d in (
            self.testcase_directory,
            self.io_files_dir,
            self.imgs_dir,
            self.imgs_env_dir,
            self.imgs_outputs_dir,
        ):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self):
        """Run KRAKEN/FIELD on the test case's environment/field (see
        KrakenManager.runkraken). Must be called explicitly: neither
        __init__ nor mode='run' trigger it automatically."""
        manager = KrakenManager()
        manager.runkraken(env=self.env, flp=self.flp, frequencies=self.src.freq)

    # ------------------------------------------------------------------
    # Diagnostic figures
    # ------------------------------------------------------------------
    def plot_testcase_env(self):
        """Plot (based on the self.plot_* flags) the environment
        diagnostic figures: SSP/attenuation/density of the medium, of
        the bottom, of the full environment, and the bathymetry."""
        if self.plot_medium:
            self.env.medium.plot_medium()
            plt.savefig(os.path.join(self.imgs_env_dir, "medium_properties.png"))
            plt.close()

        if self.plot_bottom:
            self.env.bottom_hs.plot_bottom_halfspace()
            plt.savefig(os.path.join(self.imgs_env_dir, "bottom_properties.png"))
            plt.close()

        if self.plot_env:
            self.env.plot_env()
            plt.savefig(os.path.join(self.imgs_env_dir, "env_properties.png"))
            plt.close()

        if self.plot_bathy:
            self._plot_bathy_profile()

    def _plot_bathy_profile(self):
        """Plot the bathymetry profile (range vs depth), shaded seabed,
        inverted depth axis (0 at the top)."""
        plt.figure(figsize=(16, 8))
        plt.plot(
            self.bathy.bathy_range,
            self.bathy.bathy_depth,
            color="k",
            linewidth=2,
            marker="o",
            markersize=2,
        )
        plt.ylim([0, self.domain.zmax_m + 10])
        plt.fill_between(
            self.bathy.bathy_range,
            self.bathy.bathy_depth,
            self.domain.zmax_m + 10,
            color="lightgrey",
        )
        plt.gca().invert_yaxis()
        plt.xlabel("Range (km)")
        plt.ylabel("Depth (m)")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.imgs_env_dir, "bathy.png"))
        plt.close()

    def plot_diags(self, tl_min=None, tl_max=None, modes=[1, 2, 3, 4]):
        """Plot the eigenmodes and the transmission loss (TL) obtained
        after a KRAKEN/FIELD run (self.run())."""
        fpath = os.path.join(self.io_files_dir, self.env.filename)

        # NOTE (bug fixed): the bathymetry line used to be added
        # manually, via plt.gca(), AFTER plotmode() returned -- meaning
        # it only ever landed on the LAST subplot plotmode() had
        # created, not on every mode's panel. plotmode() now draws this
        # line (and a single legend entry for it, instead of a
        # duplicated per-call plt.legend()) on every panel itself.
        fig_modes = plotmode(
            fpath, freq=self.src.freq, modes=modes, bathy_depth=self.bathy.bathy_depth[0]
        )
        fig_modes.savefig(os.path.join(self.imgs_outputs_dir, "modes.png"))

        fig_tl = plotshd(
            fpath + ".shd",
            title=f"{self.name} - f={self.src.freq}Hz",
            bathy=self.bathy,
            tl_min=tl_min,
            tl_max=tl_max,
        )
        fig_tl.savefig(os.path.join(self.imgs_outputs_dir, "tl.png"))

    def plot_ssp_tl(self, tl_min=None, tl_max=None, publi=False):
        """Plot a publication-ready figure combining the SSP profile and
        the transmission loss (TL) map, side by side."""
        fpath = os.path.join(self.io_files_dir, self.env.filename)
        PubFigure(label_fontsize=32, ticks_fontsize=30, title_fontsize=32)

        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            sharey=True,
            gridspec_kw={"width_ratios": [2, 7]},
            figsize=(18, 6),
        )

        plotshd(
            fpath + ".shd",
            title="",
            bathy=self.bathy,
            tl_min=tl_min,
            tl_max=tl_max,
            axis=axs[1],
            units="km",
        )
        axs[1].set_ylabel("")
        axs[1].set_title("(b)")

        axs[0].plot(
            self.kraken.medium.cp_ssp, self.kraken.medium.z_ssp, color="k", linewidth=2
        )
        axs[0].set_xlabel(r"Celerity [$\textrm{m~s}^{-1}$]")
        axs[0].set_xlim(
            [self.kraken.medium.cp_ssp.min() - 5, self.kraken.medium.cp_ssp.max() + 5]
        )
        plt.ylim([self.bathy.bathy_depth.max() * 1.15, 0])
        axs[0].set_ylabel("Depth [m]")
        axs[0].set_title("(a)")

        plt.savefig(os.path.join(self.imgs_outputs_dir, "ssp_tl.png"), dpi=300)
        if publi:
            plt.savefig(os.path.join(self.imgs_outputs_dir, "ssp_tl.pdf"), dpi=300)


if __name__ == "__main__":
    root_dir = (
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\testcases"
    )

    k_tc = KrakenTestCase(name="default", root_dir=root_dir)

    k_tc.run()
    k_tc.plot_diags()
