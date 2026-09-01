#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Full reproduction of the user-provided reference case
(test_kraken_rd.env / .flp / .prt / field.prt): a range-dependent
environment with a SINGLE water-column medium (nmedia=1) and a direct
acousto-elastic bottom half-space (no buffer sediment layer), confirmed
by a real KRAKEN/FIELD run to work correctly.

This script:
  1. Builds the environment and field parameters with the API (using
     the 'add_sediment_buffer_layer=False' fix -- see kraken_env.py).
  2. Writes the '.env'/'.flp' files.
  3. Runs KRAKEN + FIELD (via KrakenManager) -- REQUIRES kraken.exe /
     field.exe to be available on your machine and importable project
     modules (source.global_constants, cst, propa.kraken_toolbox.params
     with KRAKEN_BIN_DIRECTORY pointing at your real binaries).
  4. Reads the resulting '.mod' (mode shapes) and '.shd' (pressure
     field / TL) and plots:
       - mode shapes (real part of phi) for the first few modes, at the
         source range profile;
       - a transmission loss (TL) map (range vs. depth).

Adjust ROOT_DIR / KRAKEN_BIN_DIRECTORY to your own setup before running.
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
from propa.kraken_toolbox.read_modes import readmodes
from propa.kraken_toolbox.read_shd import readshd

# ----------------------------------------------------------------------
# 0. Paths -- adjust to your own project layout.
# ----------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BATHY_CSV = os.path.join(
    ROOT_DIR, "bathy_data.csv"
)  # 'range_km,depth_m' rows: 0,3000 / 4,3000 / 10,2500
ENV_FILENAME = "test_kraken_rd_repro"

# ----------------------------------------------------------------------
# 1. Water column SSP (identical to the reference test_kraken_rd.env)
# ----------------------------------------------------------------------
Z_SSP = np.array(
    [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        38,
        50,
        70,
        100,
        140,
        160,
        170,
        200,
        215,
        250,
        300,
        370,
        450,
        500,
        700,
        900,
        1000,
        1250,
        1500,
        2000,
        2500,
        3000,
    ],
    dtype=float,
)
CP_SSP = np.array(
    [
        1476.70,
        1476.70,
        1476.70,
        1476.70,
        1476.70,
        1476.70,
        1476.70,
        1476.70,
        1476.70,
        1472.60,
        1468.80,
        1467.20,
        1471.60,
        1473.60,
        1473.60,
        1472.70,
        1472.20,
        1471.60,
        1471.60,
        1472.00,
        1472.70,
        1473.10,
        1474.90,
        1477.00,
        1478.10,
        1480.70,
        1483.80,
        1490.50,
        1498.30,
        1506.50,
    ]
)

medium = KrakenMedium(
    ssp_interpolation_method="C_linear", z_ssp=Z_SSP, c_p=CP_SSP, nmesh=5000
)

# ----------------------------------------------------------------------
# 2. Bottom half-space: acousto-elastic, DIRECT (no buffer layer).
#    This is the fix: add_sediment_buffer_layer=False reproduces the
#    single-medium (nmedia=1) configuration confirmed to work.
# ----------------------------------------------------------------------
bottom_hs = KrakenBottomHalfspace(
    halfspace_properties={
        "z": Z_SSP.max(),
        "c_p": 1600.0,
        "c_s": 0.0,
        "rho": 1.5,
        "a_p": 0.5,
        "a_s": 0.0,
    },
    add_sediment_buffer_layer=False,
)

# ----------------------------------------------------------------------
# 3. Bathymetry (range-dependent) and field parameters.
# ----------------------------------------------------------------------
bathy = Bathymetry(data_file=BATHY_CSV, units="km")

field = KrakenField(
    phase_speed_limits=[0, 20000],
    src_depth=18,
    n_rcv_z=5001,
    rcv_z_min=0.0,
    rcv_z_max=3000.0,
    rcv_r_max=0.0,
)

# ----------------------------------------------------------------------
# 4. Assemble the environment. nmedia=None -> derived automatically
#    (must be 1, matching the reference file).
# ----------------------------------------------------------------------
freq = 200
env = KrakenEnv(
    title="Test de la classe KrakenEnv",
    env_root=ROOT_DIR,
    env_filename=ENV_FILENAME,
    freq=freq,
    kraken_medium=medium,
    kraken_bottom_hs=bottom_hs,
    kraken_field=field,
    kraken_bathy=bathy,
    nmedia=None,
)
assert env.nmedia == 1, f"expected nmedia=1, got {env.nmedia}"
env.write_env()
print(f"Wrote {env.env_fpath} (nmedia={env.nmedia})")

# ----------------------------------------------------------------------
# 5. Field parameters ('.flp'): coupled modes, coherent addition,
#    matching test_kraken_rd.flp ('RC C', 1001 receiver ranges over
#    0-10 km, 1000 receiver depths over 0-4500 m).
# ----------------------------------------------------------------------
flp = KrakenFlp(
    env=env,
    src_type="point_source",
    mode_theory="coupled",
    mode_addition="coherent",
    nb_modes=9999,
    src_depth=18,
    n_rcv_z=1000,
    rcv_z_min=0.0,
    rcv_z_max=4500.0,
    n_rcv_r=1001,
    rcv_r_min=0.0,
    rcv_r_max=10.0,
    rcv_dist_offset=0.0,
)
flp.write_flp()
print(f"Wrote {flp.flp_fpath}")

# ----------------------------------------------------------------------
# 6. Run KRAKEN + FIELD.
#    Requires kraken.exe / field.exe to be reachable (see
#    propa.kraken_toolbox.params.KRAKEN_BIN_DIRECTORY and your PATH).
# ----------------------------------------------------------------------
RUN_KRAKEN = True  # set to True once you have kraken.exe/field.exe available
if RUN_KRAKEN:
    manager = KrakenManager(verbose=True)
    pressure_field, field_pos = manager.runkraken(
        env=env, flp=flp, frequencies=env.freq
    )
    print("KRAKEN/FIELD run completed.")


# ----------------------------------------------------------------------
# 7. Read and plot mode shapes (from the '.mod' file, first profile).
# ----------------------------------------------------------------------
def plot_mode_shapes(mod_fpath, freq, n_modes=4):
    """Plot the real part of the first 'n_modes' mode shapes (pressure
    field eigenfunctions) as a function of depth."""
    Modes = readmodes(mod_fpath, freq=freq)
    n_modes = min(n_modes, Modes["M"])

    fig, ax = plt.subplots(figsize=(6, 8))
    for i in range(n_modes):
        ax.plot(Modes["phi"][:, i].real, Modes["z"], label=f"Mode {i + 1}")
    ax.invert_yaxis()
    ax.set_xlabel("Mode amplitude (real part)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"KRAKEN mode shapes at {freq} Hz\n(first profile, r = 0 km)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 8. Read and plot transmission loss (from the '.shd' file).
# ----------------------------------------------------------------------
def plot_transmission_loss(shd_fpath, freq):
    """Plot a range-depth transmission loss (TL) map:
    TL = -20*log10(|p|), clipped to a sensible dynamic range."""
    _, _, _, _, _, _, Pos, pressure = readshd(shd_fpath, freq=freq)

    # For a single frequency, readshd already drops the frequency axis,
    # but the theta and source-depth axes remain even when they only
    # have one value (single omnidirectional source, single source
    # depth here) -- squeeze them out to get a plain (depth, range) 2D
    # array for pcolormesh.
    pressure_2d = np.squeeze(pressure)

    with np.errstate(divide="ignore"):
        TL = -20 * np.log10(np.abs(pressure_2d) + 1e-30)

    r_km = Pos["r"]["r"]
    z_m = Pos["r"]["z"]

    fig, ax = plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(r_km, z_m, TL, shading="auto", cmap="jet_r", vmin=40, vmax=100)
    ax.invert_yaxis()
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Transmission Loss (dB) at {freq} Hz")
    fig.colorbar(pcm, ax=ax, label="TL (dB)")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    if RUN_KRAKEN:
        mod_fig = plot_mode_shapes(env.env_fpath.replace(".env", ".mod"), freq=freq)
        mod_fig.savefig(os.path.join(ROOT_DIR, "mode_shapes.png"))

        tl_fig = plot_transmission_loss(env.shd_fpath, freq=freq)
        tl_fig.savefig(os.path.join(ROOT_DIR, "transmission_loss.png"))

        print("Figures saved: mode_shapes.png, transmission_loss.png")
    else:
        print(
            "RUN_KRAKEN is False: only the '.env'/'.flp' files were "
            "generated. Set RUN_KRAKEN=True (and make sure kraken.exe/"
            "field.exe are reachable) to also run the simulation and "
            "plot mode shapes + TL, or point plot_mode_shapes()/"
            "plot_transmission_loss() at your own already-computed "
            "'.mod'/'.shd' files."
        )
