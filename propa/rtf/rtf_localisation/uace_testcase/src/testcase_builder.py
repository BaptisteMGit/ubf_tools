#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   testcase_builder.py
@Time    :   2025/05/06 09:21:47
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to build the test case env files
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import time
import numpy as np
import matplotlib.pyplot as plt


import source.global_constants as g
import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from source.ssp_profiles import SSPProfile
from propa.kraken_toolbox.src.kraken_testcase import (
    KrakenTestCase,
    DomainProperties,
    SourceProperties,
    ReceiverProperties,
    KrakenProperties,
)
from propa.kraken_toolbox.src.kraken_env import (
    KrakenTopHalfspace,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
    Bathymetry,
)
from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.kraken_toolbox.utils import default_nb_rcv_z


class DeepWaterPekerisMunk(KrakenTestCase):

    def __init__(
        self,
        simulation: Simulation = Simulation(),
        mode="run",
        name: str = "dw_pekeris_munk",
    ):
        """
        Constructor
        """

        # Initialize the simulation object
        # name = "dw_pekeris_munk"

        # Common properties
        zmin = 0
        zmax = 2000
        z_channel = 500.0
        # zmax = 5000
        # z_channel = 1000.0
        rmax = 50 * 1e3

        min_phase_speed = 1000
        max_phase_speed = 20000

        bott_props = g.sand_properties
        bott_props["a_s"] = 0.0  # No shear wave
        zs = 5

        # TODO vérifier les paramètres dans Jensen
        # # Properties from https://oalib-acoustics.org/website_resources/AcousticsToolbox/manual/node8.html
        # bott_props = {
        #     "rho": g.rho_w * 1e-3,  # Density (g/cm^3)
        #     "c_p": 1600,  # P-wave celerity (m/s)
        #     "c_s": 0.0,  # S-wave celerity (m/s)
        #     "a_p": 0.0,  # Compression wave attenuation (dB/wavelength)
        #     "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
        # }
        # zs = 100  # Source depth

        # Set domain properties
        domain_properties = DomainProperties(
            zmin=zmin, zmax=zmax, rmin=0, rmax=rmax, unit="m"
        )

        # Set source properties
        if mode == "run":
            freq = [20, 50]
        else:
            freq = 20
        src_properties = SourceProperties(
            src_type="point_source", src_depth=zs, freq=freq
        )

        # Set kraken properties
        nmedia = 2
        top_hs = KrakenTopHalfspace(
            boundary_condition="vacuum",
            halfspace_properties=None,
            twersky_scatter_properties=None,
        )

        bott_hs = KrakenBottomHalfspace(
            boundary_condition="acousto_elastic",
            sigma=0.0,
            halfspace_properties=bott_props,
            fmin=simulation.fmin,
            alpha_wavelength=10,
        )

        # Set receiver properties
        if mode == "run":
            # In run mode we only need to derive the transfert functions at a single receiver depth
            z_rcv = zmax - 1
            rcv_z_min = z_rcv
            rcv_z_max = z_rcv
            # Number of receiver depths / ranges (flp file)
            dr = 5
            nr_flp = int(rmax / dr) + 1
            nz_flp = 1

        else:
            # In demo mode we need to derive the transfert functions at all receiver depths to plot tl profile
            rcv_z_min = zmin
            rcv_z_max = zmax + bott_hs.sedim_layer_depth / 2
            # Number of receiver depths / ranges (flp file)
            dr = 50
            dz = 5
            nr_flp = int(rmax / dr) + 1
            nz_flp = int(rcv_z_max / dz) + 1

        rcv_properties = ReceiverProperties(
            zmin=rcv_z_min, zmax=rcv_z_max, rmin=0, rmax=rmax, unit="m"
        )

        att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)
        ssp = SSPProfile()
        ssp.set_munk_profile(
            zmin=zmin,
            zmax=zmax,
            z_channel=z_channel,
            nz=200,
        )
        medium = KrakenMedium(
            ssp_interpolation_method="C_linear",
            z_ssp=ssp.z,
            c_p=ssp.c,
            c_s=0.0,
            rho=1.0,
            a_p=0.0,
            a_s=0.0,
            nmesh=0,
            sigma=0.0,
        )

        max_domain_depth = domain_properties.zmax_m + bott_hs.sedim_layer_depth
        n_rcv_z = default_nb_rcv_z(
            fmax=simulation.fmax, max_depth=max_domain_depth, n_per_l=10
        )
        field = KrakenField(
            phase_speed_limits=[min_phase_speed, max_phase_speed],
            src_depth=[src_properties.depth],
            n_rcv_z=n_rcv_z,
            rcv_z_min=0,
            rcv_z_max=max_domain_depth,
            rcv_r_max=0.0,
        )

        kraken_properties = KrakenProperties(
            mode_coupling="adiabatic",
            mode_addition="coherent",
            n_mode=100,
            nr=nr_flp,
            nz=nz_flp,
            nmedia=nmedia,
            top_hs=top_hs,
            bott_hs=bott_hs,
            att=att,
            medium=medium,
            field=field,
        )

        title = "Deep Water Pekeris waveguide with Munk celerity profile"

        super().__init__(
            name=name,
            root_dir=p.root_tmp,
            domain_properties=domain_properties,
            src_properties=src_properties,
            rcv_properties=rcv_properties,
            kraken_properties=kraken_properties,
            title=title,
        )

        # Update simulation
        simulation.name = name
        simulation.init()
        simulation.kraken_env = self.env
        simulation.kraken_flp = self.flp
        self.simulation = simulation

        # Run and plot diags if "demo" mode is selected
        if mode == "demo":
            self.run()
            # self.plot_diags(tl_min=60, tl_max=110, modes=[1, 30, 90])
            self.plot_diags(modes=[1, 30, 90])


class DeepWaterPekerisRhumrumSSP(KrakenTestCase):

    def __init__(
        self,
        simulation: Simulation = Simulation(),
        mode="run",
        name: str = "dw_pekeris_rhumrum_ssp",
    ):
        """
        Constructor
        """

        # Initialize the simulation object
        # name = "dw_pekeris_rhumrum_ssp"

        # Common properties
        zmin = 0
        zmax = 2000
        rmax = 50 * 1e3

        min_phase_speed = 1000
        max_phase_speed = 20000

        bott_props = g.sand_properties
        bott_props["a_s"] = 0.0  # No shear wave
        zs = 5

        # Set domain properties
        domain_properties = DomainProperties(
            zmin=zmin, zmax=zmax, rmin=0, rmax=rmax, unit="m"
        )

        # Set source properties
        if mode == "run":
            freq = [20, 50]
        else:
            freq = 20
        src_properties = SourceProperties(
            src_type="point_source", src_depth=zs, freq=freq
        )

        # Set kraken properties
        nmedia = 2
        top_hs = KrakenTopHalfspace(
            boundary_condition="vacuum",
            halfspace_properties=None,
            twersky_scatter_properties=None,
        )

        bott_hs = KrakenBottomHalfspace(
            boundary_condition="acousto_elastic",
            sigma=0.0,
            halfspace_properties=bott_props,
            fmin=simulation.fmin,
            alpha_wavelength=10,
        )

        # Set receiver properties
        if mode == "run":
            # In run mode we only need to derive the transfert functions at a single receiver depth
            z_rcv = zmax - 1
            rcv_z_min = z_rcv
            rcv_z_max = z_rcv
            # Number of receiver depths / ranges (flp file)
            dr = 5
            nr_flp = int(rmax / dr) + 1
            nz_flp = 1

        else:
            # In demo mode we need to derive the transfert functions at all receiver depths to plot tl profile
            rcv_z_min = zmin
            rcv_z_max = zmax + bott_hs.sedim_layer_depth / 2
            # Number of receiver depths / ranges (flp file)
            dr = 50
            dz = 5
            nr_flp = int(rmax / dr) + 1
            nz_flp = int(rcv_z_max / dz) + 1

        rcv_properties = ReceiverProperties(
            zmin=rcv_z_min, zmax=rcv_z_max, rmin=0, rmax=rmax, unit="m"
        )

        att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)
        ssp = SSPProfile()
        ssp.set_rhumrum_ssp(
            zmin=zmin,
            zmax=zmax,
            nz=None,
        )
        medium = KrakenMedium(
            ssp_interpolation_method="C_linear",
            z_ssp=ssp.z,
            c_p=ssp.c,
            c_s=0.0,
            rho=1.0,
            a_p=0.0,
            a_s=0.0,
            nmesh=0,
            sigma=0.0,
        )

        max_domain_depth = domain_properties.zmax_m + bott_hs.sedim_layer_depth
        n_rcv_z = default_nb_rcv_z(
            fmax=simulation.fmax, max_depth=max_domain_depth, n_per_l=10
        )
        field = KrakenField(
            phase_speed_limits=[min_phase_speed, max_phase_speed],
            src_depth=[src_properties.depth],
            n_rcv_z=n_rcv_z,
            rcv_z_min=0,
            rcv_z_max=max_domain_depth,
            rcv_r_max=0.0,
        )

        kraken_properties = KrakenProperties(
            mode_coupling="adiabatic",
            mode_addition="coherent",
            n_mode=100,
            nr=nr_flp,
            nz=nz_flp,
            nmedia=nmedia,
            top_hs=top_hs,
            bott_hs=bott_hs,
            att=att,
            medium=medium,
            field=field,
        )

        title = "Deep Water Pekeris waveguide with real ssp (RR48)"

        super().__init__(
            name=name,
            root_dir=p.root_tmp,
            domain_properties=domain_properties,
            src_properties=src_properties,
            rcv_properties=rcv_properties,
            kraken_properties=kraken_properties,
            title=title,
        )

        # Update simulation
        simulation.name = name
        simulation.init()
        simulation.kraken_env = self.env
        simulation.kraken_flp = self.flp
        self.simulation = simulation

        # Run and plot diags if "demo" mode is selected
        if mode == "demo":
            self.run()
            # self.plot_diags(tl_min=60, tl_max=110, modes=[1, 30, 90])
            self.plot_diags(modes=[1, 30, 90])


class DeepWaterRealEnv(KrakenTestCase):

    def __init__(
        self,
        simulation: Simulation = Simulation(),
        mode="run",
        name: str = "dw_real_env",
        depth_offset: float = 0,
        n_bathy_subsample: int = 1,
    ):
        """
        Constructor
        """

        # Set name and title
        title = "Deep Water waveguide with real bathy profile and real ssp"

        if mode == "demo":
            rmin = 0
            rmax = 50 * 1e3
            name = "dw_real_env_demo"
        else:
            rmax = simulation.grid_rmax
            rmin = simulation.grid_rmin

        # Common properties
        zmin = 0
        min_phase_speed = 1000
        max_phase_speed = 20000
        # bott_props = g.sand_properties
        # bott_props = g.boulders_bedrock_propertiesc
        bott_props = g.coarse_sediment_properties
        bott_props["a_s"] = 0.0  # No shear wave
        zs = 5

        # Set bathy
        fpath_parts = os.path.normpath(p.bathy_fpath).split("\\")
        bathy = Bathymetry(
            data_file=os.path.join(p.project_root, *fpath_parts),
            units="km",  # Units of the bathy file
        )

        # Limit bathy to range domain
        idx_in_range_domain = bathy.bathy_range <= rmax * 1e-3
        bathy.bathy_range = bathy.bathy_range[idx_in_range_domain]
        bathy.bathy_depth = bathy.bathy_depth[idx_in_range_domain]

        # Subsample bathy profile
        # nsubsample = 5
        bathy.bathy_range = bathy.bathy_range[::n_bathy_subsample]
        bathy.bathy_depth = bathy.bathy_depth[::n_bathy_subsample]

        # Add off set to bathy depth to reach approximate zmax
        # z_offset = bathy.bathy_depth.max() - zmax
        bathy.bathy_depth = bathy.bathy_depth - depth_offset
        zmax = bathy.bathy_depth.max()

        # Set domain properties
        domain_properties = DomainProperties(
            zmin=zmin, zmax=zmax, rmin=0, rmax=rmax, unit="m"
        )

        # Set source properties
        if mode == "run":
            freq = [20, 50]
        else:
            freq = 50
        src_properties = SourceProperties(
            src_type="point_source", src_depth=zs, freq=freq
        )

        # Set kraken properties
        nmedia = 2
        top_hs = KrakenTopHalfspace(
            boundary_condition="vacuum",
            halfspace_properties=None,
            twersky_scatter_properties=None,
        )

        bott_hs = KrakenBottomHalfspace(
            boundary_condition="acousto_elastic",
            sigma=0.0,
            halfspace_properties=bott_props,
            fmin=simulation.fmin,
            alpha_wavelength=10,
        )

        # Set receiver properties
        if mode == "run":
            # In run mode we only need to derive the transfert functions at a single receiver depth
            z_rcv = (
                bathy.bathy_depth.min() - 1
            )  # Ensure rcv is not lying inside sediment
            rcv_z_min = z_rcv
            rcv_z_max = z_rcv
            # Number of receiver depths / ranges (flp file)
            dr = 5
            nr_flp = int(rmax / dr) + 1
            nz_flp = 1

        else:
            # In demo mode we need to derive the transfert functions at all receiver depths to plot tl profile
            rcv_z_min = zmin
            rcv_z_max = zmax + bott_hs.sedim_layer_depth / 2
            rcv_z_max = np.round(rcv_z_max * 1e-2, 0) * 1e2
            # Number of receiver depths / ranges (flp file)
            dr = 50
            dz = 5
            nr_flp = int(rmax / dr) + 1
            nz_flp = int(rcv_z_max / dz) + 1

        rcv_properties = ReceiverProperties(
            zmin=rcv_z_min, zmax=rcv_z_max, rmin=rmin, rmax=rmax, unit="m"
        )

        att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)
        ssp = SSPProfile()
        ssp.set_rhumrum_ssp(
            zmin=zmin,
            zmax=zmax,
            nz=None,
        )
        medium = KrakenMedium(
            ssp_interpolation_method="C_linear",
            z_ssp=ssp.z,
            c_p=ssp.c,
            c_s=0.0,
            rho=1.0,
            a_p=0.0,
            a_s=0.0,
            nmesh=0,
            sigma=0.0,
        )

        # Round the max depth to the nearest 100 m
        max_domain_depth = (
            np.round((domain_properties.zmax_m + bott_hs.sedim_layer_depth) * 1e-2, 0)
            * 1e2
        )
        n_rcv_z = default_nb_rcv_z(
            fmax=simulation.fmax, max_depth=max_domain_depth, n_per_l=7
        )
        field = KrakenField(
            phase_speed_limits=[min_phase_speed, max_phase_speed],
            src_depth=[src_properties.depth],
            n_rcv_z=n_rcv_z,
            rcv_z_min=0,
            rcv_z_max=max_domain_depth,
            rcv_r_max=0.0,
        )

        kraken_properties = KrakenProperties(
            mode_coupling="coupled",
            mode_addition="coherent",
            n_mode=100,
            nr=nr_flp,
            nz=nz_flp,
            nmedia=nmedia,
            top_hs=top_hs,
            bott_hs=bott_hs,
            att=att,
            medium=medium,
            field=field,
        )

        super().__init__(
            name=name,
            root_dir=p.root_tmp,
            domain_properties=domain_properties,
            src_properties=src_properties,
            rcv_properties=rcv_properties,
            kraken_properties=kraken_properties,
            bathy=bathy,
            title=title,
            mode=mode,
        )

        # Update simulation
        simulation.name = name
        simulation.init()
        simulation.kraken_env = self.env
        simulation.kraken_flp = self.flp
        self.simulation = simulation

        self.simulation.write_logs()

        # Run and plot diags if "demo" mode is selected
        if mode == "demo":
            t0 = time.time()
            self.run()
            print(
                "Ellapsed time to derive single frequency tl profile : {:.2f} s".format(
                    time.time() - t0
                )
            )
            self.plot_diags(tl_min=60, tl_max=120, modes=[1, 30, 90])
            # self.plot_diags(modes=[1, 30, 90])


if __name__ == "__main__":
    # Example usage

    # test_case = DeepWaterPekerisMunk(mode="demo")
    # test_case = DeepWaterPekerisRhumrumSSP(mode="demo")
    # test_case = DeepWaterRealEnv(mode="demo", depth_offset=0)
    test_case = DeepWaterRealEnv(mode="demo", depth_offset=0)
    test_case.plot_ssp_tl(publi=True)
    # test_case.run()


# test_case = DeepWaterPekerisMunk(mode="run")

# print(test_case.name)
# print(test_case.simulation.tmp_folder)
# print(test_case.io_files_dir)
# print(test_case.root_dir)
# print(test_case.domain_properties.zmin)
