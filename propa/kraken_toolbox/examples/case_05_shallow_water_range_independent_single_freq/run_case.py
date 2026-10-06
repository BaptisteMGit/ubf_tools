#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Case 5 -- Shallow-water waveguide, range-independent, single frequency,
realistic sound-speed profile, semi-infinite isovelocity sediment.

Unlike the idealized isovelocity Pekeris cases (1-4), this uses a
depth-varying sound-speed profile (SSP) representative of a summer
shallow-water environment: a warm, faster surface layer, a thermocline
where sound speed drops sharply, and a nearly isovelocity layer near
the bottom.

NOTE: the SSP values below are a representative, illustrative synthetic
profile (typical shape of a summer shallow-water thermocline), NOT
measured data from a specific site -- replace them with your own
measured/modeled profile for real studies.

Environment:
    - Water column: 0-100 m, depth-varying SSP (see SSP_Z/SSP_CP below).
    - Bottom: semi-infinite fluid half-space (sand-like), c = 1650 m/s,
      rho = 1.8 g/cm3, attenuation 0.8 dB/wavelength.
    - Single frequency: 300 Hz (shallow-water studies typically use
      higher frequencies than deep-water ones).
    - Range-independent (flat bottom).

Run this script directly to write the '.env'/'.flp' files and the
environment overview figure. Set the KRAKEN_EXAMPLES_RUN_KRAKEN=1
environment variable (and make sure kraken.exe/field.exe are reachable)
to also run the simulation and produce the mode-shape/TL figures -- or
run every case at once with examples/run_all_cases.py.
"""
import os

import numpy as np
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
ENV_FILENAME = "case_05_shallow_ri_single_freq"
RUN_KRAKEN = os.environ.get("KRAKEN_EXAMPLES_RUN_KRAKEN", "0") == "1"

# ----------------------------------------------------------------------
# 1. Environment: shallow water, realistic (illustrative) summer SSP
# ----------------------------------------------------------------------
WATER_DEPTH = 100.0  # m
FREQ = 300.0  # Hz
SRC_DEPTH = 20.0  # m

# Representative summer shallow-water profile: warm surface layer,
# thermocline down to ~40 m, near-isovelocity below.
SSP_Z = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0])
SSP_CP = np.array([1520.0, 1518.0, 1512.0, 1498.0, 1490.0, 1487.0, 1486.0, 1485.0])

medium = KrakenMedium(
    ssp_interpolation_method="C_linear",
    z_ssp=SSP_Z,
    c_p=SSP_CP,
    rho=1.0,
)

bottom_hs = KrakenBottomHalfspace(
    halfspace_properties={
        "z": WATER_DEPTH, "c_p": 1650.0, "c_s": 0.0, "rho": 1.8, "a_p": 0.8, "a_s": 0.0,
    },
    add_sediment_buffer_layer=False,
)

field = KrakenField(
    phase_speed_limits=[0, 2000],
    src_depth=SRC_DEPTH,
    n_rcv_z=201,
    rcv_z_min=0.0,
    rcv_z_max=WATER_DEPTH,
    rcv_r_max=10.0,
)

env = KrakenEnv(
    title="Case 5 - Shallow water, realistic SSP (range-independent, single frequency)",
    env_root=HERE,
    env_filename=ENV_FILENAME,
    freq=FREQ,
    kraken_medium=medium,
    kraken_bottom_hs=bottom_hs,
    kraken_field=field,
    nmedia=None,
)
assert env.nmedia == 1
env.write_env()
print(f"Wrote {env.env_fpath} (nmedia={env.nmedia})")

flp = KrakenFlp(
    env=env,
    src_type="point_source",
    mode_theory="adiabatic",
    mode_addition="coherent",
    nb_modes=9999,
    src_depth=SRC_DEPTH,
    n_rcv_z=201,
    rcv_z_min=0.0,
    rcv_z_max=WATER_DEPTH,
    n_rcv_r=1001,
    rcv_r_min=0.0,
    rcv_r_max=10.0,
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

        fig3 = pu.plot_tl_profile(shd_fpath, freq=FREQ, rcv_depth=SRC_DEPTH, units="km")
        fig3.savefig(os.path.join(HERE, "tl_profile_at_source_depth.png"))
        plt.close(fig3)

        print("Figures saved.")
    else:
        print(
            "RUN_KRAKEN is False: wrote '.env'/'.flp' and environment.png "
            "only. Set KRAKEN_EXAMPLES_RUN_KRAKEN=1 (with kraken.exe/"
            "field.exe reachable) to also run the simulation and produce "
            "the mode-shape/TL figures."
        )
