#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Case 2 -- Pekeris waveguide, range-independent, broadband (multi-frequency).

Same canonical Pekeris waveguide as Case 1 (homogeneous water layer over
a homogeneous fluid half-space), but run over several frequencies at
once. KRAKEN natively supports broadband runs for a range-independent
environment (a single '.env' profile, several frequencies listed at the
end) -- no special handling needed compared to Case 1.

Environment:
    - Water column: 0-100 m, constant c = 1500 m/s, rho = 1.0 g/cm3.
    - Bottom: semi-infinite fluid half-space, c = 1700 m/s,
      rho = 1.5 g/cm3, attenuation 0.5 dB/wavelength.
    - Frequencies: 25, 50, 100, 200 Hz.
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
ENV_FILENAME = "case_02_pekeris_ri_broadband"
RUN_KRAKEN = os.environ.get("KRAKEN_EXAMPLES_RUN_KRAKEN", "0") == "1"

# ----------------------------------------------------------------------
# 1. Environment: same Pekeris waveguide as Case 1, several frequencies
# ----------------------------------------------------------------------
WATER_DEPTH = 100.0  # m
FREQS = np.array([25.0, 50.0, 100.0, 200.0])  # Hz
SRC_DEPTH = 25.0  # m

medium = KrakenMedium(
    ssp_interpolation_method="C_linear",
    z_ssp=[0.0, WATER_DEPTH],
    c_p=[1500.0, 1500.0],
    rho=1.0,
)

bottom_hs = KrakenBottomHalfspace(
    halfspace_properties={
        "z": WATER_DEPTH, "c_p": 1700.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.5, "a_s": 0.0,
    },
    add_sediment_buffer_layer=False,
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
    title="Case 2 - Pekeris waveguide (range-independent, broadband)",
    env_root=HERE,
    env_filename=ENV_FILENAME,
    freq=FREQS,  # an array -> broadband_run is enabled automatically
    kraken_medium=medium,
    kraken_bottom_hs=bottom_hs,
    kraken_field=field,
    nmedia=None,
)
assert env.nmedia == 1
assert env.broadband_run
env.write_env()
print(f"Wrote {env.env_fpath} (nmedia={env.nmedia}, {env.freq.size} frequencies)")

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
    n_rcv_r=501,
    rcv_r_min=0.0,
    rcv_r_max=5.0,
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
        ref_freq = float(FREQS[0])

        fig1 = pu.plotmode_several_freqs(mod_fpath, freq=FREQS)
        fig1.savefig(os.path.join(HERE, "mode_shapes_all_frequencies.png"))
        plt.close(fig1)

        fig2 = pu.plotshd(shd_fpath, freq=ref_freq, units="km")
        fig2.savefig(os.path.join(HERE, f"transmission_loss_{ref_freq:.0f}Hz.png"))
        plt.close(fig2)

        fig3 = pu.plot_tl_profile_multi_freq(shd_fpath, freqs=FREQS, rcv_depth=SRC_DEPTH, units="km")
        fig3.savefig(os.path.join(HERE, "tl_profiles_all_frequencies.png"))
        plt.close(fig3)

        print("Figures saved.")
    else:
        print(
            "RUN_KRAKEN is False: wrote '.env'/'.flp' and environment.png "
            "only. Set KRAKEN_EXAMPLES_RUN_KRAKEN=1 (with kraken.exe/"
            "field.exe reachable) to also run the simulation and produce "
            "the mode-shape/TL figures."
        )
