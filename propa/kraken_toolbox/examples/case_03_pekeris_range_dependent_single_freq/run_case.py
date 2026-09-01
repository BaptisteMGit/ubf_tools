#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Case 3 -- Pekeris-type waveguide, range-dependent (sloping bottom),
single frequency.

A classic sloping-bottom ("wedge") geometry, inspired by the standard
ASA wedge benchmark used throughout the ocean acoustics literature
(Jensen, Kuperman, Porter & Schmidt, "Computational Ocean Acoustics"):
an isovelocity water column shoaling from 200 m to 50 m over 10 km,
above a homogeneous fluid sediment half-space. This geometry couples
energy between modes as the water shallows, which a range-independent
model cannot capture -- a good first range-dependent test after the
flat-bottom Pekeris case.

NOTE: the numeric values below (depths, distances, sediment properties)
are a representative, simplified version of this classic benchmark
geometry, not a certified reproduction of a specific published
reference case -- adjust them if you need to match a particular
published result exactly.

Environment:
    - Water column: isovelocity c = 1500 m/s, depth shoaling from
      200 m (r=0) to 50 m (r=10 km) -- see bathy.csv.
    - Bottom: semi-infinite fluid half-space, c = 1700 m/s,
      rho = 1.5 g/cm3, attenuation 0.5 dB/wavelength (no buffer layer,
      single medium per profile -- see Case 1's docstring).
    - Single frequency: 25 Hz (low enough to keep the number of modes,
      and hence the mode-coupling effect, easy to visualize).
    - Range-dependent (coupled-mode theory in the '.flp').

Run this script directly to write the '.env'/'.flp' files and the
environment overview figures. Set the KRAKEN_EXAMPLES_RUN_KRAKEN=1
environment variable (and make sure kraken.exe/field.exe are reachable)
to also run the simulation and produce the mode-shape/TL figures -- or
run every case at once with examples/run_all_cases.py.
"""
import os

import matplotlib.pyplot as plt

from propa.kraken_toolbox.src.kraken_env import (
    KrakenEnv,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenField,
    KrakenFlp,
    Bathymetry,
)
from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from propa.kraken_toolbox import plot_utils as pu

HERE = os.path.dirname(os.path.abspath(__file__))
BATHY_CSV = os.path.join(HERE, "bathy.csv")  # range_km, depth_m : 0,200 / 5,200 / 10,50
ENV_FILENAME = "case_03_pekeris_rd_single_freq"
RUN_KRAKEN = os.environ.get("KRAKEN_EXAMPLES_RUN_KRAKEN", "0") == "1"

# ----------------------------------------------------------------------
# 1. Environment: sloping-bottom Pekeris-type wedge
# ----------------------------------------------------------------------
MAX_WATER_DEPTH = 200.0  # m, at r=0 (see bathy.csv)
FREQ = 25.0  # Hz
SRC_DEPTH = 50.0  # m
MAX_RANGE_KM = 10.0

medium = KrakenMedium(
    ssp_interpolation_method="C_linear",
    z_ssp=[0.0, MAX_WATER_DEPTH],  # truncated automatically at each range's local depth
    c_p=[1500.0, 1500.0],
    rho=1.0,
)

bottom_hs = KrakenBottomHalfspace(
    halfspace_properties={
        "z": 0, "c_p": 1700.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.5, "a_s": 0.0,
    },
    add_sediment_buffer_layer=False,
)

bathy = Bathymetry(data_file=BATHY_CSV, units="km")

field = KrakenField(
    phase_speed_limits=[0, 2000],
    src_depth=SRC_DEPTH,
    n_rcv_z=201,
    rcv_z_min=0.0,
    rcv_z_max=MAX_WATER_DEPTH,
    rcv_r_max=0.0,
)

env = KrakenEnv(
    title="Case 3 - Sloping-bottom wedge (range-dependent, single frequency)",
    env_root=HERE,
    env_filename=ENV_FILENAME,
    freq=FREQ,
    kraken_medium=medium,
    kraken_bottom_hs=bottom_hs,
    kraken_field=field,
    kraken_bathy=bathy,
    nmedia=None,
)
assert env.nmedia == 1
assert env.bathy.use_bathy
env.write_env()
assert env.range_dependent_env
print(f"Wrote {env.env_fpath} (nmedia={env.nmedia}, {env.modes_range.size} profiles)")

flp = KrakenFlp(
    env=env,
    src_type="point_source",
    mode_theory="coupled",  # required to capture mode coupling on the slope
    mode_addition="coherent",
    nb_modes=9999,
    src_depth=SRC_DEPTH,
    n_rcv_z=201,
    rcv_z_min=0.0,
    rcv_z_max=MAX_WATER_DEPTH,
    n_rcv_r=1001,
    rcv_r_min=0.0,
    rcv_r_max=MAX_RANGE_KM,
)
flp.write_flp()
print(f"Wrote {flp.flp_fpath}")

# ----------------------------------------------------------------------
# 2. Run KRAKEN + FIELD
# ----------------------------------------------------------------------
if RUN_KRAKEN:
    manager = KrakenManager(verbose=True)
    manager.runkraken(env=env, flp=flp, frequencies=env.freq)
    print("KRAKEN/FIELD run completed.")


if __name__ == "__main__":
    # Standardized environment overview: SSP/attenuation/density (for
    # the r=0 profile, the deepest point) via KrakenEnv.plot_env(), plus
    # the bathymetry profile via plot_utils.plot_bathymetry() -- the one
    # part of a range-dependent environment plot_env() does not itself
    # show.
    fig_env = env.plot_env(plot_src=True, src_depth=SRC_DEPTH)
    fig_env.savefig(os.path.join(HERE, "environment.png"))
    plt.close(fig_env)

    fig_bathy = pu.plot_bathymetry(bathy)
    fig_bathy.savefig(os.path.join(HERE, "bathymetry.png"))
    plt.close(fig_bathy)

    if RUN_KRAKEN:
        mod_fpath = env.env_fpath.replace(".env", ".mod")
        shd_fpath = env.shd_fpath

        fig1 = pu.plotmode(mod_fpath, freq=FREQ, bathy_depth=bathy.bathy_depth[0])
        fig1.savefig(os.path.join(HERE, "mode_shapes.png"))
        plt.close(fig1)

        fig2 = pu.plotshd(shd_fpath, freq=FREQ, units="km", bathy=bathy)
        fig2.savefig(os.path.join(HERE, "transmission_loss.png"))
        plt.close(fig2)

        fig3 = pu.plot_tl_profile(shd_fpath, freq=FREQ, rcv_depth=SRC_DEPTH, units="km")
        fig3.savefig(os.path.join(HERE, "tl_profile_at_source_depth.png"))
        plt.close(fig3)

        print("Figures saved.")
    else:
        print(
            "RUN_KRAKEN is False: wrote '.env'/'.flp', environment.png and "
            "bathymetry.png only. Set KRAKEN_EXAMPLES_RUN_KRAKEN=1 (with "
            "kraken.exe/field.exe reachable) to also run the simulation "
            "and produce the mode-shape/TL figures."
        )
