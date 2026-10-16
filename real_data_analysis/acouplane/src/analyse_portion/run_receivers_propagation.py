#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
One broadband, range-dependent, coupled-mode KRAKEN/FIELD simulation per
receiver (model: example case 4 of kraken-api).

Each receiver gets its own sub-folder of ROOT_SIMULATION, containing
bathy.csv, the .env/.flp files, the KRAKEN/FIELD outputs and all the
diagnostic figures.

Usage (on Windows, keep the `if __name__ == "__main__":` guard in the calling
script: the parallel mode uses multiprocessing):

    from run_receivers_propagation import run_all_receivers

    if __name__ == "__main__":
        # quick check with 3 test frequencies on one receiver
        res, failed = run_all_receivers(bathy_profiles, z_ssp, cp_ssp,
                                        test_mode=True, only=["obs1"])
        # full run (all the frequencies of the impulse-response vector)
        res, failed = run_all_receivers(bathy_profiles, z_ssp, cp_ssp)
        pressure_field, field_pos = res["obs1"]      # (n_freq, n_z, n_r), grid positions
"""

import pickle
import traceback
from pathlib import Path

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

ROOT_SIMULATION = Path(
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis"
    r"\acouplane\src\analyse_portion\portion1\propagation"
)

# ----------------------------------------------------------------------
# PARAMETERS TO ADJUST
# ----------------------------------------------------------------------
# Frequency vector used to build the impulse response (rfft grid)
FMAX = 800.0  # Hz
T_SIGNAL = 1.0  # s
FS = 2 * FMAX  # Hz (Nyquist)
NT = int(T_SIGNAL * FS)  # number of samples
# Set a lower bound (Hz) to skip the useless very-low frequencies below the waveguide
# cutoff (a few Hz in ~100 m of water); None = keep the whole vector. f = 0 is always removed.
FMIN_COMPUTE = 20.0  # Hz, used to skip the very-low frequencies (None = keep all)
TEST_FREQS = np.array([100.0, 400.0, 800.0])  # Hz, used when test_mode=True
N_DIAG_FREQS = 4  # number of frequencies shown in the diagnostic figures
# fmin (Hz) used to size the buffer sediment layer (10 wavelengths at this frequency).
# Do NOT set it to 0.2 Hz: the buffer would be tens of km thick.
BUFFER_FMIN = 100.0
SRC_DEPTH = 5.0  # m, must be < shallowest depth of every profile (~97 m)
N_RCV_Z = 201  # number of depths of the output grid
N_RCV_R = 501  # number of ranges of the output grid
# BOTTOM = {
#     "c_p": 1650.0,
#     "c_s": 0.0,
#     "rho": 1.8,
#     "a_p": 0.8,
#     "a_s": 0.0,
# }  # sandy sediment
from source.global_constants import sables_TG

BOTTOM = {
    "c_p": sables_TG["c_p"],
    "c_s": 0.0,
    "rho": sables_TG["rho"],
    "a_p": sables_TG["a_p"],
    "a_s": 0.0,
}  # sandy sediment
PARALLEL = True  # frequencies distributed over N_CORES processes (global_constants.py)
STORE_DTYPE = (
    np.complex64
)  # dtype of the returned/saved pressure field (complex128 = 2x memory)
N_WORKERS = 10


def get_frequencies(test_mode=False, fmin_compute=FMIN_COMPUTE):
    """Frequency vector of the simulation (3 test frequencies if test_mode)."""
    if test_mode:
        return TEST_FREQS.copy()
    freq = np.fft.rfftfreq(n=NT, d=1.0 / FS)
    freq = freq[freq > 0.0]  # KRAKEN cannot run at f = 0
    if fmin_compute is not None:
        freq = freq[freq >= fmin_compute]
    return freq


def diag_indices(n_freq, n_diag=N_DIAG_FREQS):
    """Indices of n_diag equally spaced frequencies between the first and last one."""
    return np.unique(
        np.round(np.linspace(0, n_freq - 1, min(n_diag, n_freq))).astype(int)
    )


def _crop_ssp(z_ssp, cp_ssp, z_max):
    """Make the SSP start at 0 and end exactly at z_max (crop or extend)."""
    z = np.asarray(z_ssp, dtype=float)
    cp = np.asarray(cp_ssp, dtype=float)
    keep = z < z_max
    z, cp = z[keep], cp[keep]
    c_end = np.interp(
        z_max, np.asarray(z_ssp, float), cp_ssp
    )  # constant beyond the last point
    if z[0] > 0.0:  # extend up to the surface
        z, cp = np.r_[0.0, z], np.r_[cp[0], cp]
    return np.r_[z, z_max], np.r_[cp, c_end]


def run_receiver(
    name,
    profile,
    z_ssp,
    cp_ssp,
    run_kraken=True,
    root=ROOT_SIMULATION,
    freqs=None,
    test_mode=False,
    src_depth=SRC_DEPTH,
    parallel=PARALLEL,
    save_results=True,
    keep_in_memory=True,
):
    name = str(name)
    freqs = (
        get_frequencies(test_mode) if freqs is None else np.asarray(freqs, dtype=float)
    )
    sim_dir = Path(root) / name
    sim_dir.mkdir(parents=True, exist_ok=True)

    # --- bathymetry: bathy.csv (range [m], depth [m]) in the receiver folder
    r_m = np.asarray(profile["range_m"], dtype=float)
    depth = np.asarray(profile["depth_m"], dtype=float)
    bathy_csv = sim_dir / "bathy.csv"
    np.savetxt(bathy_csv, np.c_[r_m, depth], delimiter=",", fmt="%.4f")
    bathy = Bathymetry(data_file=str(bathy_csv), units="m")

    max_depth = float(np.ceil(depth.max()))
    max_range_km = float(r_m.max()) / 1000.0
    if src_depth >= depth.min():
        raise ValueError(
            f"{name}: src_depth={src_depth} m >= shallowest depth {depth.min():.1f} m"
        )

    # --- environment
    z, cp = _crop_ssp(z_ssp, cp_ssp, max_depth)
    medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z, c_p=cp, rho=1.0)
    # Buffer sediment layer (as in case 7): the deepest point of these profiles is
    # not at r=0, a direct half-space would make FIELD crash.
    bottom_hs = KrakenBottomHalfspace(
        halfspace_properties={"z": 0, **BOTTOM},
        add_sediment_buffer_layer=True,
        fmin=BUFFER_FMIN,
    )
    field = KrakenField(
        phase_speed_limits=[0, 2000],
        src_depth=src_depth,
        n_rcv_z=N_RCV_Z,
        rcv_z_min=0.0,
        rcv_z_max=max_depth,
        rcv_r_max=0.0,
    )
    env = KrakenEnv(
        title=f"Receiver {name} - range-dependent, broadband",
        env_root=str(sim_dir),
        env_filename=f"{name}_rd_broadband",
        freq=freqs,
        kraken_medium=medium,
        kraken_bottom_hs=bottom_hs,
        kraken_field=field,
        kraken_bathy=bathy,
        nmedia=None,
    )
    env.write_env()

    flp = KrakenFlp(
        env=env,
        src_type="point_source",
        mode_theory="coupled",
        mode_addition="coherent",
        nb_modes=9999,
        src_depth=src_depth,
        n_rcv_z=N_RCV_Z,
        rcv_z_min=0.0,
        rcv_z_max=max_depth,
        n_rcv_r=N_RCV_R,
        rcv_r_min=0.0,
        rcv_r_max=max_range_km,
    )
    flp.write_flp()

    # --- diagnostic figures that do not need KRAKEN
    fig = env.plot_env(plot_src=True, src_depth=src_depth)
    fig.savefig(sim_dir / "environment.png")
    plt.close(fig)
    fig = pu.plot_bathymetry(bathy)
    fig.savefig(sim_dir / "bathymetry.png")
    plt.close(fig)

    if not run_kraken:
        return

    # --- KRAKEN + FIELD
    manager = KrakenManager(verbose=True, parallel=parallel, n_workers=N_WORKERS)
    pressure_field, field_pos = manager.runkraken(
        env=env, flp=flp, frequencies=env.freq
    )
    all_modes = manager.last_modes

    # Frequencies for which KRAKEN/FIELD failed are left at 0 in the pressure field
    empty = [i for i in range(len(freqs)) if not np.any(pressure_field[i])]
    if empty:
        print(
            f"WARNING {name}: {len(empty)} frequencies without result "
            f"(first ones: {freqs[empty[:5]]} Hz)"
        )

    # --- diagnostic figures on N_DIAG_FREQS equally spaced frequencies
    idx = diag_indices(len(freqs))
    f_diag = freqs[idx]
    print(f"{name}: diagnostic figures at {np.round(f_diag, 1)} Hz")

    if all_modes is not None and len(all_modes) == len(freqs):
        fig = pu.plotmode_from_data(
            [all_modes[i] for i in idx], freq=f_diag, bathy_depth=bathy.bathy_depth[0]
        )
        fig.savefig(sim_dir / "mode_shapes_diag_frequencies.png")
        plt.close(fig)
    else:
        print(
            f"WARNING {name}: modes list misaligned with frequencies (failed "
            f"frequencies): mode-shape figure skipped."
        )

    for i, f in zip(idx, f_diag):
        fig = pu.plotshd_from_pressure_field(
            None,
            pressure_field=pressure_field[i],
            freq=float(f),
            pos=field_pos,
            base_title=env.simulation_title,
            units="km",
            bathy=bathy,
        )
        fig.savefig(sim_dir / f"transmission_loss_{f:.1f}Hz.png")
        plt.close(fig)

    fig = pu.plot_tl_profile_multi_freq_from_data(
        pressure_field[idx], f_diag, field_pos, rcv_depth=src_depth, units="km"
    )
    fig.savefig(sim_dir / "tl_profiles_diag_frequencies.png")
    plt.close(fig)

    # --- storage
    if STORE_DTYPE is not None and pressure_field.dtype != STORE_DTYPE:
        pressure_field = pressure_field.astype(STORE_DTYPE)
    if save_results:
        np.save(sim_dir / "pressure_field.npy", pressure_field)
        np.save(sim_dir / "freq.npy", freqs)
        with open(sim_dir / "field_pos.pkl", "wb") as fh:
            pickle.dump(field_pos, fh)
    return (pressure_field, field_pos) if keep_in_memory else None


def run_all_receivers(
    bathy_profiles, z_ssp, cp_ssp, run_kraken=True, only=None, test_mode=False, **kwargs
):
    """Run every receiver (or those listed in `only`).

    test_mode=True: only the 3 frequencies of TEST_FREQS (to check the chain quickly).
    Extra keyword arguments go to run_receiver (root, freqs, src_depth, parallel,
    save_results, keep_in_memory).

    Returns (results, failed):
        results: {receiver_name: (pressure_field, field_pos)} (empty if run_kraken=False
                 or keep_in_memory=False; pressure_field shape = (n_freq, n_z, n_r))
        failed:  {receiver_name: traceback} -- a failure does not stop the other receivers.
    """
    results, failed = {}, {}
    for name, profile in bathy_profiles.items():
        if only is not None and str(name) not in only:
            continue
        print(f"\n===== Receiver {name} =====")
        try:
            out = run_receiver(
                name,
                profile,
                z_ssp,
                cp_ssp,
                run_kraken=run_kraken,
                test_mode=test_mode,
                **kwargs,
            )
            if out is not None:
                results[str(name)] = out
        except Exception:
            failed[str(name)] = traceback.format_exc()
            print(failed[str(name)])
    print("\nDone." if not failed else f"\nFailed receivers: {list(failed)}")
    return results, failed


if __name__ == "__main__":
    pass
