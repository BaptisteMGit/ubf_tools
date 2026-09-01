#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Case 8 -- Shallow-water waveguide, range-dependent, broadband, realistic
bathymetry, realistic sound-speed profile, sediment layer WITH a
realistic sound-speed profile (gradient).

Identical environment to Case 7, except the sediment is no longer a
flat, isovelocity half-space: it now has its own velocity/density/
attenuation GRADIENT, softer and more attenuative near the seafloor
(freshly deposited mud/sand) and harder, faster, less attenuative at
depth (compacted sediment, approaching basement). This is done with
KrakenBottomHalfspace's 'sediment_top_properties' parameter, which adds
a two-point linear gradient across the (still automatically-sized)
buffer sediment layer, terminated by a true half-space using the
'bottom' properties -- see KrakenBottomHalfspace's docstring.

Because a buffer layer is used here (add_sediment_buffer_layer=True,
the default), nmedia is derived as 2 (water + sediment) instead of 1,
automatically -- no manual bookkeeping required.

NOTE: the SSP, bathymetry and sediment gradient values below are
representative, illustrative synthetic data, NOT measurements from a
specific site -- replace them with your own measured/modeled data for
real studies.

Environment:
    - Water column: same as Case 7 (illustrative summer SSP, 0-120 m).
    - Bathymetry: same as Case 7 (100 -> 80 -> 110 -> 120 m over 15 km).
    - Sediment: gradient layer from c=1600 m/s, rho=1.5 g/cm3 (top,
      just below the seafloor) to c=1800 m/s, rho=1.9 g/cm3 (bottom,
      basement), attenuation decreasing from 0.8 to 0.3 dB/wavelength.
    - Frequencies: 100, 200, 300 Hz.
    - Range-dependent, coupled-mode theory.

Run this script directly to write the '.env'/'.flp' files and the
environment overview figures. Set the KRAKEN_EXAMPLES_RUN_KRAKEN=1
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
    Bathymetry,
)
from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from propa.kraken_toolbox import plot_utils as pu

HERE = os.path.dirname(os.path.abspath(__file__))
BATHY_CSV = os.path.join(HERE, "bathy.csv")
ENV_FILENAME = "case_08_shallow_rd_broadband_gradient_sediment"
RUN_KRAKEN = os.environ.get("KRAKEN_EXAMPLES_RUN_KRAKEN", "0") == "1"
PARALLEL = os.environ.get("KRAKEN_EXAMPLES_PARALLEL", "0") == "1"

# ----------------------------------------------------------------------
# 1. Environment: same as Case 7, but with a sediment gradient
# ----------------------------------------------------------------------
MAX_WATER_DEPTH = 120.0  # m
FREQS = np.array([100.0, 200.0, 300.0])  # Hz
SRC_DEPTH = 20.0  # m
MAX_RANGE_KM = 15.0

SSP_Z = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 120.0])
SSP_CP = np.array([1520.0, 1518.0, 1512.0, 1498.0, 1490.0, 1487.0, 1486.0, 1485.0, 1484.0])

medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=SSP_Z, c_p=SSP_CP, rho=1.0)

# Sediment gradient: soft/slow/attenuative at the top, hard/fast/less
# attenuative near the basement -- a standard, simplified geoacoustic
# compaction model.
bottom_hs = KrakenBottomHalfspace(
    halfspace_properties={  # bottom of the sediment layer / basement
        "z": 0, "c_p": 1800.0, "c_s": 0.0, "rho": 1.9, "a_p": 0.3, "a_s": 0.0,
    },
    sediment_top_properties={  # top of the sediment layer, just below the seafloor
        "c_p": 1600.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.8, "a_s": 0.0,
    },
    add_sediment_buffer_layer=True,  # needed to have a sediment layer at all
    fmin=100.0,  # sizes the sediment layer thickness (10 wavelengths at 100 Hz)
    alpha_wavelength=10,
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
    title="Case 8 - Shallow water, gradient sediment (range-dependent, broadband)",
    env_root=HERE,
    env_filename=ENV_FILENAME,
    freq=FREQS,
    kraken_medium=medium,
    kraken_bottom_hs=bottom_hs,
    kraken_field=field,
    kraken_bathy=bathy,
    nmedia=None,  # derived automatically -> 2 (water + sediment gradient layer)
)
assert env.nmedia == 2
assert env.bathy.use_bathy
assert env.broadband_run
env.write_env()
assert env.range_dependent_env
print(f"Wrote {env.env_fpath} (nmedia={env.nmedia}, {env.modes_range.size} profiles, "
      f"{env.freq.size} frequencies, sediment thickness = "
      f"{bottom_hs.sedim_layer_depth:.1f} m)")

flp = KrakenFlp(
    env=env,
    src_type="point_source",
    mode_theory="coupled",
    mode_addition="coherent",
    nb_modes=9999,
    src_depth=SRC_DEPTH,
    n_rcv_z=201,
    rcv_z_min=0.0,
    rcv_z_max=MAX_WATER_DEPTH,
    n_rcv_r=1501,
    rcv_r_min=0.0,
    rcv_r_max=MAX_RANGE_KM,
)
flp.write_flp()
print(f"Wrote {flp.flp_fpath}")

# ----------------------------------------------------------------------
# 2. Run KRAKEN + FIELD
# ----------------------------------------------------------------------
if RUN_KRAKEN:
    manager = KrakenManager(verbose=True, parallel=PARALLEL)
    manager.runkraken(env=env, flp=flp, frequencies=env.freq)
    print("KRAKEN/FIELD run completed.")


if __name__ == "__main__":
    # Standardized environment overview: since this case uses a real
    # (gradient) sediment MEDIUM rather than a bare half-space,
    # env.plot_env() now shows that gradient too (bottom_hs's
    # cp_bot_halfspace/etc are the deep-basement end of the gradient;
    # the shallow, top-of-sediment end comes from
    # sediment_top_properties -- both are drawn automatically).
    fig_env = env.plot_env(plot_src=True, src_depth=SRC_DEPTH)
    fig_env.savefig(os.path.join(HERE, "environment.png"))
    plt.close(fig_env)

    fig_bathy = pu.plot_bathymetry(bathy)
    fig_bathy.savefig(os.path.join(HERE, "bathymetry.png"))
    plt.close(fig_bathy)

    if RUN_KRAKEN:
        mod_fpath = env.env_fpath.replace(".env", ".mod")
        shd_fpath = env.shd_fpath
        ref_freq = float(FREQS[1])

        fig1 = pu.plotmode_several_freqs(mod_fpath, freq=FREQS)
        fig1.savefig(os.path.join(HERE, "mode_shapes_all_frequencies.png"))
        plt.close(fig1)

        fig2 = pu.plotshd(shd_fpath, freq=ref_freq, units="km", bathy=bathy)
        fig2.savefig(os.path.join(HERE, f"transmission_loss_{ref_freq:.0f}Hz.png"))
        plt.close(fig2)

        fig3 = pu.plot_tl_profile_multi_freq(shd_fpath, freqs=FREQS, rcv_depth=SRC_DEPTH, units="km")
        fig3.savefig(os.path.join(HERE, "tl_profiles_all_frequencies.png"))
        plt.close(fig3)

        print("Figures saved.")
    else:
        print(
            "RUN_KRAKEN is False: wrote '.env'/'.flp', environment.png and "
            "bathymetry.png only. Set KRAKEN_EXAMPLES_RUN_KRAKEN=1 (with "
            "kraken.exe/field.exe reachable) to also run the simulation "
            "and produce the mode-shape/TL figures."
        )
