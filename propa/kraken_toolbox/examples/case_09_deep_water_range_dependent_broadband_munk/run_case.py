#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Case 9 -- Deep-water waveguide, range-dependent, broadband, realistic
bathymetry, canonical Munk sound-speed profile, semi-infinite
isovelocity sediment.

The Munk profile is THE canonical analytic deep-water sound-speed
profile used throughout ocean acoustics (Munk, 1974), combining a
near-surface layer, a sound channel axis (the local sound-speed
minimum, here at 1300 m), and an increasing sound speed both above
(warm surface layer) and below (increasing pressure) that axis. It
produces the classic deep-water "convergence zone" propagation pattern.
Here it is combined with a simple, realistic-looking continental-slope
bathymetry (constant abyssal depth, then shoaling towards a shelf).

The Munk formula used below (c1, eps, zaxis, B parameters) is the
standard canonical form found in the ocean acoustics literature (e.g.
Jensen, Kuperman, Porter & Schmidt, "Computational Ocean Acoustics").
The bathymetry values are representative/illustrative, not measured
data from a specific site.

Environment:
    - Water column: Munk profile, sound channel axis at 1300 m,
      computed over the full 0-5000 m depth range (truncated
      automatically to each profile's local depth).
    - Bathymetry: 5000 m (r=0-20 km, abyssal plain) shoaling to 2000 m
      at r=50 km (continental slope) -- see bathy.csv.
    - Bottom: semi-infinite fluid half-space (abyssal sediment),
      c = 1600 m/s, rho = 1.8 g/cm3, attenuation 0.2 dB/wavelength.
    - Frequencies: 50, 100, 200 Hz (typical of long-range deep-water
      propagation studies).
    - Range-dependent, coupled-mode theory.

Run this script directly to write the '.env'/'.flp' files and the
environment overview figures. Set the KRAKEN_EXAMPLES_RUN_KRAKEN=1
environment variable (and make sure kraken.exe/field.exe are reachable)
to also run the simulation and produce the mode-shape/TL figures -- or
run every case at once with examples/run_all_cases.py. This is the
largest case in this set (5000 m water column, 50 km range, 3
frequencies): a KRAKEN/FIELD run will take noticeably longer than the
shallow-water cases; set 'parallel=True' in KrakenManager(...) below to
speed it up.
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
ENV_FILENAME = "case_09_deepwater_rd_broadband_munk"
RUN_KRAKEN = os.environ.get("KRAKEN_EXAMPLES_RUN_KRAKEN", "0") == "1"
PARALLEL = os.environ.get("KRAKEN_EXAMPLES_PARALLEL", "0") == "1"


def munk_profile(z, c1=1500.0, eps=0.00737, zaxis=1300.0, B=1300.0):
    """Canonical Munk (1974) deep-water sound-speed profile.

    Args:
        z (array-like): depths (m).
        c1 (float): sound speed at the channel axis (m/s).
        eps (float): perturbation coefficient (dimensionless).
        zaxis (float): depth of the sound channel axis (m).
        B (float): depth scale of the profile (m).

    Returns:
        np.ndarray: sound speed (m/s) at each depth in 'z'.
    """
    eta = 2 * (np.asarray(z, dtype=float) - zaxis) / B
    return c1 * (1 + eps * (eta - 1 + np.exp(-eta)))


# ----------------------------------------------------------------------
# 1. Environment: deep water, Munk profile, continental-slope bathymetry
# ----------------------------------------------------------------------
MAX_WATER_DEPTH = 5000.0  # m, deepest point of the bathymetry
FREQS = np.array([50.0, 100.0, 200.0])  # Hz
SRC_DEPTH = 1000.0  # m, in the SOFAR
MAX_RANGE_KM = 100.0

SSP_Z = np.linspace(0.0, MAX_WATER_DEPTH, 51)
SSP_CP = munk_profile(SSP_Z)

medium = KrakenMedium(
    ssp_interpolation_method="C_linear", z_ssp=SSP_Z, c_p=SSP_CP, rho=1.0
)

bottom_hs = KrakenBottomHalfspace(
    halfspace_properties={
        "z": 0,
        "c_p": 1600.0,
        "c_s": 0.0,
        "rho": 1.8,
        "a_p": 0.2,
        "a_s": 0.0,
    },
    add_sediment_buffer_layer=False,
)

bathy = Bathymetry(data_file=BATHY_CSV, units="km")

field = KrakenField(
    phase_speed_limits=[1400, 20000],
    src_depth=SRC_DEPTH,
    n_rcv_z=501,
    rcv_z_min=0.0,
    rcv_z_max=MAX_WATER_DEPTH,
    rcv_r_max=0.0,
)

env = KrakenEnv(
    title="Case 9 - Deep water, Munk profile (range-dependent, broadband)",
    env_root=HERE,
    env_filename=ENV_FILENAME,
    freq=FREQS,
    kraken_medium=medium,
    kraken_bottom_hs=bottom_hs,
    kraken_field=field,
    kraken_bathy=bathy,
    nmedia=None,
)
assert env.nmedia == 1
assert env.bathy.use_bathy
assert env.broadband_run
env.write_env()
assert env.range_dependent_env
print(
    f"Wrote {env.env_fpath} (nmedia={env.nmedia}, {env.modes_range.size} profiles, "
    f"{env.freq.size} frequencies)"
)

flp = KrakenFlp(
    env=env,
    src_type="point_source",
    mode_theory="coupled",
    mode_addition="coherent",
    nb_modes=9999,
    src_depth=SRC_DEPTH,
    n_rcv_z=501,
    rcv_z_min=0.0,
    rcv_z_max=MAX_WATER_DEPTH,
    n_rcv_r=2001,
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
    pressure_field, field_pos = manager.runkraken(
        env=env, flp=flp, frequencies=env.freq
    )
    print("KRAKEN/FIELD run completed.")


if __name__ == "__main__":
    fig_env = env.plot_env(plot_src=True, src_depth=SRC_DEPTH)
    fig_env.savefig(os.path.join(HERE, "environment.png"))
    plt.close(fig_env)

    fig_bathy = pu.plot_bathymetry(bathy)
    fig_bathy.savefig(os.path.join(HERE, "bathymetry.png"))
    plt.close(fig_bathy)

    if RUN_KRAKEN:
        ref_freq = float(FREQS[1])  # 100 Hz

        # NOTE: this is a broadband + range-dependent run, so KRAKEN was
        # actually re-run once per frequency (see
        # KrakenManager.runkraken_broadband_range_dependent's module
        # docstring), each time overwriting the SAME '.mod'/'.shd'
        # files -- there is no single on-disk file left containing
        # every frequency's data to read back here. Use the in-memory
        # results collected during that per-frequency loop instead:
        # manager.last_modes (mode shapes) and pressure_field/field_pos
        # (already returned by runkraken() above, aggregated across
        # every frequency).
        fig1 = pu.plotmode_from_data(
            manager.last_modes, freq=FREQS, bathy_depth=bathy.bathy_depth[0]
        )
        fig1.savefig(os.path.join(HERE, "mode_shapes_all_frequencies.png"))
        plt.close(fig1)

        ref_freq_idx = int(np.argmin(np.abs(FREQS - ref_freq)))
        fig2 = pu.plotshd_from_pressure_field(
            None,
            pressure_field=pressure_field[ref_freq_idx],
            freq=ref_freq,
            pos=field_pos,
            base_title=env.simulation_title,
            units="km",
            bathy=bathy,
        )
        fig2.savefig(os.path.join(HERE, f"transmission_loss_{ref_freq:.0f}Hz.png"))
        plt.close(fig2)

        fig3 = pu.plot_tl_profile_multi_freq_from_data(
            pressure_field, FREQS, field_pos, rcv_depth=SRC_DEPTH, units="km"
        )
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
