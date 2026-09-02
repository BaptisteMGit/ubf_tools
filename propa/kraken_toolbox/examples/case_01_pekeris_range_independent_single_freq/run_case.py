#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Case 1 -- Pekeris waveguide, range-independent, single frequency.

The Pekeris waveguide is the simplest canonical underwater acoustics
problem: a homogeneous (isovelocity) water layer of constant depth,
lying directly over a homogeneous (isovelocity) fluid half-space
(the sediment, extending to infinite depth). It has a well-known
analytic normal-mode solution, which makes it the standard first test
for any normal-mode code.

This example also showcases the simplest possible bottom description
supported by the API: a DIRECT acousto-elastic half-space right below
the water column, with NO artificial "buffer" sediment layer
(add_sediment_buffer_layer=False) and a single medium (nmedia=1) --
confirmed against a real KRAKEN/FIELD run to be a valid, working
configuration (see the project's bug-fix notes for the nmedia /
sediment-buffer issue this avoids).

Environment:
    - Water column: 0-100 m, constant c = 1500 m/s, rho = 1.0 g/cm3.
    - Bottom: semi-infinite fluid half-space, c = 1700 m/s,
      rho = 1.5 g/cm3, attenuation 0.5 dB/wavelength.
    - Single frequency: 100 Hz.
    - Range-independent (flat bottom).

Run this script directly to write the '.env'/'.flp' files and the
environment overview figure. Set the KRAKEN_EXAMPLES_RUN_KRAKEN=1
environment variable (and make sure kraken.exe/field.exe are reachable)
to also run the simulation and produce the mode-shape/TL figures --
or run every case at once with examples/run_all_cases.py.
"""

import os

import matplotlib.pyplot as plt

from propa.kraken_toolbox.src.kraken_env import (
    KrakenEnv,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenField,
    KrakenFlp,
)
from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from propa.kraken_toolbox import plot_utils as pu

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILENAME = "case_01_pekeris_ri_single_freq"
RUN_KRAKEN = os.environ.get("KRAKEN_EXAMPLES_RUN_KRAKEN", "0") == "1"
RUN_KRAKEN = True
# ----------------------------------------------------------------------
# 1. Environment: Pekeris waveguide
# ----------------------------------------------------------------------
WATER_DEPTH = 100.0  # m
FREQ = 100.0  # Hz
SRC_DEPTH = 25.0  # m

medium = KrakenMedium(
    ssp_interpolation_method="C_linear",
    z_ssp=[0.0, WATER_DEPTH],
    c_p=[1500.0, 1500.0],  # isovelocity water column
    rho=1.0,
)

bottom_hs = KrakenBottomHalfspace(
    halfspace_properties={
        "z": WATER_DEPTH,
        "c_p": 1700.0,
        "c_s": 0.0,  # fluid sediment: no shear waves
        "rho": 1.5,
        "a_p": 0.5,  # dB/wavelength
        "a_s": 0.0,
    },
    add_sediment_buffer_layer=False,  # direct half-space -> classic Pekeris model
)

field = KrakenField(
    phase_speed_limits=[0, 2000],
    src_depth=SRC_DEPTH,
    n_rcv_z=201,
    rcv_z_min=0.0,
    rcv_z_max=WATER_DEPTH,
    rcv_r_max=5.0,
)

env = KrakenEnv(
    title="Case 1 - Pekeris waveguide (range-independent, single frequency)",
    env_root=HERE,
    env_filename=ENV_FILENAME,
    freq=FREQ,
    kraken_medium=medium,
    kraken_bottom_hs=bottom_hs,
    kraken_field=field,
    nmedia=None,  # derived automatically -> 1 (no buffer layer)
)
assert env.nmedia == 1
env.write_env()
print(f"Wrote {env.env_fpath} (nmedia={env.nmedia})")

flp = KrakenFlp(
    env=env,
    src_type="point_source",
    mode_theory="adiabatic",  # irrelevant for a range-independent run, kept simple
    mode_addition="coherent",
    nb_modes=9999,
    src_depth=SRC_DEPTH,
    n_rcv_z=201,
    rcv_z_min=0.0,
    rcv_z_max=WATER_DEPTH,
    n_rcv_r=501,
    rcv_r_min=0.0,
    rcv_r_max=5.0,
)
flp.write_flp()
print(f"Wrote {flp.flp_fpath}")

# ----------------------------------------------------------------------
# 2. Run KRAKEN + FIELD (requires real binaries -- see KrakenManager /
#    propa.kraken_toolbox.params.KRAKEN_BIN_DIRECTORY).
# ----------------------------------------------------------------------
if RUN_KRAKEN:
    manager = KrakenManager(verbose=True)
    manager.runkraken(env=env, flp=flp, frequencies=env.freq)
    print("KRAKEN/FIELD run completed.")


if __name__ == "__main__":
    # Standardized environment overview (SSP/attenuation/density across
    # water + bottom half-space) -- available immediately, no KRAKEN
    # run needed. Same method (KrakenEnv.plot_env) used in every case
    # of this gallery.
    fig_env = env.plot_env(plot_src=True, src_depth=SRC_DEPTH)
    fig_env.savefig(os.path.join(HERE, "environment.png"))
    plt.close(fig_env)

    if RUN_KRAKEN:
        mod_fpath = env.env_fpath.replace(".env", ".mod")
        shd_fpath = env.shd_fpath

        fig1 = pu.plotmode(mod_fpath, freq=FREQ)
        fig1.savefig(os.path.join(HERE, "mode_shapes.png"))
        plt.close(fig1)

        fig2 = pu.plotshd(shd_fpath, freq=FREQ, units="km")
        fig2.savefig(os.path.join(HERE, "transmission_loss.png"))
        plt.close(fig2)

        fig3 = pu.plot_tl_profile(
            shd_fpath,
            freq=FREQ,
            rcv_depth=SRC_DEPTH,
            units="km",
            spherical_loss=True,
            cylindrical_loss=True,
        )
        fig3.savefig(os.path.join(HERE, "tl_profile_at_source_depth.png"))
        plt.close(fig3)

        print(
            "Figures saved: environment.png, mode_shapes.png, "
            "transmission_loss.png, tl_profile_at_source_depth.png"
        )
    else:
        print(
            "RUN_KRAKEN is False: wrote '.env'/'.flp' and environment.png "
            "only. Set KRAKEN_EXAMPLES_RUN_KRAKEN=1 (with kraken.exe/"
            "field.exe reachable) to also run the simulation and produce "
            "the mode-shape/TL figures."
        )
