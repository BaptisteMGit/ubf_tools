#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   kraken_testcase.py
@Time    :   2025/04/24 13:36:53
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to handle kraken simulation testcases
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import source.global_constants as g

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
from propa.kraken_toolbox.run_kraken import get_subprocess_working_dir
from propa.kraken_toolbox.plot_utils import plotshd, plotmode

from publication.publication_figure import PubFigure


class DomainProperties:
    """
    Class to handle kraken simulation domain properties
    """

    def __init__(self, zmin=0, zmax=1000, rmin=0, rmax=10 * 1e3, unit="m"):
        """
        Initialize the DomainProperties object.


        """
        if unit == "m":
            alpha_z = 1
            alpha_r = 1e-3
        elif unit == "km":
            alpha_z = 1e3
            alpha_r = 1

        self.zmin_m = zmin * alpha_z
        self.zmax_m = zmax * alpha_z
        self.rmin_km = rmin * alpha_r
        self.rmax_km = rmax * alpha_r


class SourceProperties:
    """
    Class to handle kraken simulation source properties
    """

    def __init__(self, src_type="point_source", src_depth=100, freq=50):
        """
        Initialize the SourceProperties object.

        Args:
            src_type (str): Type of the source.
            src_depth (float): Depth of the source.
        """
        self.type = src_type
        self.depth = src_depth
        self.freq = np.atleast_1d(freq)


class ReceiverProperties:
    """
    Class to handle kraken simulation receiver properties
    """

    def __init__(self, zmin=0, zmax=1000, rmin=0, rmax=10 * 1e3, unit="m"):
        """
        Initialize the ReceiverProperties object.
        """
        if unit == "m":
            alpha_z = 1
            alpha_r = 1e-3
        elif unit == "km":
            alpha_z = 1e3
            alpha_r = 1

        self.zmin_m = zmin * alpha_z
        self.zmax_m = zmax * alpha_z
        self.rmin_km = rmin * alpha_r
        self.rmax_km = rmax * alpha_r


class KrakenProperties:
    """
    Class to handle kraken simulation properties
    """

    def __init__(
        self,
        mode_coupling="adiabatic",
        mode_addition="coherent",
        n_mode=100,
        nr=1000,
        nz=1000,
        nmedia=2,
        top_hs=KrakenTopHalfspace(),
        bott_hs=KrakenBottomHalfspace(),
        att=KrakenAttenuation(),
        medium=KrakenMedium(z_ssp=[0, DomainProperties().zmax_m]),
        field=KrakenField(
            n_rcv_z=1000,
            src_depth=SourceProperties().depth,
            rcv_z_max=DomainProperties().zmax_m
            + KrakenBottomHalfspace().sedim_layer_depth,
            phase_speed_limits=[1000, 20000],
        ),
    ):
        """
        Initialize the KrakenProperties object.

        """

        self.mode_coupling = mode_coupling
        self.mode_addition = mode_addition
        self.n_mode = n_mode
        self.nr = nr
        self.nz = nz
        self.nmedia = nmedia

        self.top_hs = top_hs
        self.bott_hs = bott_hs
        self.att = att
        self.medium = medium
        self.field = field


class KrakenTestCase:
    """
    Class to handle kraken simulation testcases
    """

    def __init__(
        self,
        name: str,
        root_dir: str,
        domain_properties: DomainProperties = DomainProperties(),
        src_properties: SourceProperties = SourceProperties(),
        rcv_properties: ReceiverProperties = ReceiverProperties(),
        kraken_properties: KrakenProperties = KrakenProperties(),
        bathy: Bathymetry = None,
        title: str = "Default testcase",
        mode: str = "run",
    ):
        """
        Initialize the TestCase object.

        Args:
            name (str): Name of the test case.
            root_dir (str): Directory where the test case files will be stored.
        """
        self.name = name
        self.root_dir = root_dir
        self.title = title
        self.mode = mode

        # Initialize directories
        self.init_testcase_dirs()

        # Source properties
        self.src = src_properties

        # Receiver properties
        self.rcv = rcv_properties

        # Domain properties
        self.domain = domain_properties

        # Kraken properties
        self.kraken = kraken_properties

        self.bathy = bathy

        # Plotting flags
        self.plot_medium = True
        self.plot_bottom = True
        self.plot_bathy = True
        self.plot_env = True

        self.pre_process_testcase()

    def set_bathy(self):
        if self.bathy is None:
            # Define flat bathymetry
            r_km = [self.domain.rmin_km, self.domain.rmax_km]
            h_m = [self.domain.zmax_m, self.domain.zmax_m]
        else:
            r_km = self.bathy.bathy_range
            h_m = self.bathy.bathy_depth

            # Limit to range domain
            idx_in_range_domain = r_km <= self.domain.rmax_km
            r_km = r_km[idx_in_range_domain]
            h_m = h_m[idx_in_range_domain]

            # Update domain depth limit
            self.domain.zmax_m = np.max(h_m)

            # Update field depth limit and resolution
            max_domain_depth = (
                np.round(
                    (self.domain.zmax_m + self.kraken.bott_hs.sedim_layer_depth) * 1e-2,
                    0,
                )
                * 1e2
            )  # Round to closet 100m
            n_rcv_z = default_nb_rcv_z(
                fmax=np.max(self.src.freq), max_depth=max_domain_depth
            )
            self.kraken.field.n_rcv_z = n_rcv_z
            self.kraken.field.rcv_depth_max = max_domain_depth

        # Save or copy bathymetry to io_files directory for clarity
        bathy_path = os.path.join(self.io_files_dir, "bathy.csv")
        pd.DataFrame({"r": np.round(r_km, 3), "h": np.round(h_m, 3)}).to_csv(
            bathy_path, index=False, header=False
        )
        self.bathy = Bathymetry(bathy_path)

    def set_env(self):
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
        # Dummy write env to set range_dependent_env flag
        self.env.write_env()

    def set_flp(self):
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

    def pre_process_testcase(self):
        # Load testcase
        self.load()
        # Write flp and env files
        self.write_kraken_files()
        if self.mode == "demo":
            # Plot env
            self.plot_testcase_env()

    def plot_testcase_env(self):
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

    def load(self):
        self.set_bathy()
        self.set_env()
        self.set_flp()

    def write_kraken_files(self):
        self.env.write_env()
        self.flp.write_flp()

    def init_testcase_dirs(self):
        self.testcase_directory = os.path.join(self.root_dir, self.name)
        self.io_files_dir = os.path.join(self.testcase_directory, "io_files")
        self.imgs_dir = os.path.join(self.testcase_directory, "imgs")
        self.imgs_env_dir = os.path.join(self.imgs_dir, "env")
        self.imgs_outputs_dir = os.path.join(self.imgs_dir, "outputs")

        # Create directories if they do not exist
        dirs = [
            self.testcase_directory,
            self.io_files_dir,
            self.imgs_dir,
            self.imgs_env_dir,
            self.imgs_outputs_dir,
        ]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)

    def run(self):
        manager = KrakenManager()
        manager.runkraken(env=self.env, flp=self.flp, frequencies=self.src.freq)

    def plot_diags(self, tl_min=None, tl_max=None, modes=[1, 2, 3, 4]):
        # Plotting diagnostics
        fpath = os.path.join(self.io_files_dir, self.env.filename)
        plotmode(fpath, freq=self.src.freq, modes=modes)
        plt.savefig(os.path.join(self.imgs_outputs_dir, "modes.png"))

        plotshd(
            fpath + ".shd",
            title=f"{self.name} - f={self.src.freq}Hz",
            bathy=self.bathy,
            tl_min=tl_min,
            tl_max=tl_max,
        )
        plt.savefig(os.path.join(self.imgs_outputs_dir, "tl.png"))

    def plot_ssp_tl(self, tl_min=None, tl_max=None, publi=False):
        fpath = os.path.join(self.io_files_dir, self.env.filename)
        pfig = PubFigure(label_fontsize=30)
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            sharey=True,
            gridspec_kw={"width_ratios": [1, 6]},
            figsize=(16, 8),
            # wspace=0.05,
        )
        # Plot transmission loss map
        plotshd(
            fpath + ".shd",
            title="",
            bathy=self.bathy,
            tl_min=tl_min,
            tl_max=tl_max,
            axis=axs[1],
            units="km",
            rasterized=False,
        )
        axs[1].set_ylabel("")
        axs[1].set_title("(b)")

        # Add ssp beside transmission loss
        axs[0].plot(
            self.kraken.medium.cp_ssp, self.kraken.medium.z_ssp, color="k", linewidth=1
        )
        axs[0].set_xlabel(r"Celerity [$ms^{-1}$]")
        axs[0].set_xlim(
            [self.kraken.medium.cp_ssp.min() - 5, self.kraken.medium.cp_ssp.max() + 5]
        )
        # Rotate x ticks for better readability
        axs[0].tick_params(axis="x", rotation=30)
        axs[1].tick_params(axis="x", rotation=30)
        plt.ylim([self.bathy.bathy_depth.max() * 1.15, 0])

        axs[0].set_ylabel("Depth [m]")
        axs[0].set_title("(a)")

        # # Adjust spacing
        # plt.subplots_adjust(
        #     wspace=0.05,
        # )

        plt.savefig(os.path.join(self.imgs_outputs_dir, "ssp_tl.png"))
        if publi:
            plt.savefig(os.path.join(self.imgs_outputs_dir, "ssp_tl.pdf"))


if __name__ == "__main__":
    root_dir = (
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\testcases"
    )

    k_tc = KrakenTestCase(name="default", root_dir=root_dir)

    k_tc.run()
    k_tc.plot_diags()
