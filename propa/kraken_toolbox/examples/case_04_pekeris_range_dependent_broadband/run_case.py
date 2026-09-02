#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Case 4 -- Pekeris-type waveguide, range-dependent (sloping bottom),
broadband (multi-frequency).

Same sloping-bottom "wedge" geometry as Case 3, run over several
frequencies. This is the combination KRAKEN does NOT support natively
in a single pass (range-dependent + broadband) -- see
KrakenManager.runkraken / run_kraken.runkraken's module docstring: the
library works around it by re-running KRAKEN/FIELD once per frequency
(same range-dependent environment, one frequency at a time) and
stacking the resulting pressure fields. From the outside, calling
KrakenManager.runkraken() with a broadband range-dependent env is
exactly the same call as any other case; the workaround is transparent.

NOTE: as in Case 3, the numeric values are a representative, simplified
version of the classic ASA-wedge-style benchmark geometry, not a
certified reproduction of a specific published reference case.

Environment:
    - Water column: isovelocity c = 1500 m/s, depth shoaling from
      200 m (r=0) to 50 m (r=10 km) -- see bathy.csv.
    - Bottom: semi-infinite fluid half-space, c = 1700 m/s,
      rho = 1.5 g/cm3, attenuation 0.5 dB/wavelength.
    - Frequencies: 15, 25, 50 Hz.
    - Range-dependent, coupled-mode theory.

Run this script directly to write the '.env'/'.flp' files and the
environment overview figures. Set the KRAKEN_EXAMPLES_RUN_KRAKEN=1
environment variable (and make sure kraken.exe/field.exe are reachable)
to also run the simulation and produce the mode-shape/TL figures --
or run every case at once with examples/run_all_cases.py. Running the
broadband + range-dependent workaround can take noticeably longer than
the other cases (one full KRAKEN/FIELD run per frequency); set
'parallel=True' in KrakenManager(...) below to speed it up on a
multi-core machine.
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
ENV_FILENAME = "case_04_pekeris_rd_broadband"
RUN_KRAKEN = os.environ.get("KRAKEN_EXAMPLES_RUN_KRAKEN", "0") == "1"
PARALLEL = os.environ.get("KRAKEN_EXAMPLES_PARALLEL", "0") == "1"

# ----------------------------------------------------------------------
# 1. Environment: same sloping-bottom wedge as Case 3, several frequencies
# ----------------------------------------------------------------------
MAX_WATER_DEPTH = 200.0  # m
FREQS = np.array([15.0, 25.0, 50.0, 100.0])  # Hz
SRC_DEPTH = 50.0  # m
MAX_RANGE_KM = 10.0

medium = KrakenMedium(
    ssp_interpolation_method="C_linear",
    z_ssp=[0.0, MAX_WATER_DEPTH],
    c_p=[1500.0, 1500.0],
    rho=1.0,
)

bottom_hs = KrakenBottomHalfspace(
    halfspace_properties={
        "z": 0,
        "c_p": 1700.0,
        "c_s": 0.0,
        "rho": 1.5,
        "a_p": 0.5,
        "a_s": 0.0,
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
    title="Case 4 - Sloping-bottom wedge (range-dependent, broadband)",
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
# 2. Run KRAKEN + FIELD (broadband + range-dependent workaround, see
#    module docstring above)
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
        ref_freq = float(FREQS[-1])

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
