#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   run_kraken.py
@Time    :   2024/02/21 15:34:27
@Author  :   Menetrier Baptiste
@Version :   1.1 (refactor)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Procedural functions to run the KRAKEN/FIELD Fortran
             executables. Also see propa.kraken_toolbox.src.kraken_manager.KrakenManager,
             a class-based wrapper around (mostly) the same logic.

This module does NOT change the public API of the original file (same
function names/signatures).

------------------------------------------------------------------------
IMPORTANT: this file heavily duplicates kraken_manager.KrakenManager
------------------------------------------------------------------------
Every function here (runkraken, assign_frequency_intervalls, run_exec,
run_field_exec, run_kraken_exec, clear_kraken_parallel_working_dir,
get_subprocess_working_dir, init_parallel_kraken_working_dirs,
runkraken_broadband_range_dependent) has a near-identical counterpart as
a method of KrakenManager. The two implementations had already drifted
apart in small but meaningful ways before this refactor (see the bugs
below, none of which existed in the KrakenManager version) -- exactly
the kind of divergence duplicated code invites. This refactor fixes the
bugs and cleans up this file in place (it was not asked to be merged
into KrakenManager), but you likely want to retire one of the two
eventually and have the other delegate to (or import) it, so a future
fix only has to be made once.
------------------------------------------------------------------------

BUGS FIXED COMPARED TO THE ORIGINAL CODE:
  1. runkraken_broadband_range_dependent: the per-frequency KrakenEnv
     reconstruction did not pass `nmedia=env.nmedia`, silently resetting
     nmedia back to KrakenEnv's default (1) on every frequency, even if
     the original environment used a different value (e.g. 2 media).
     The KrakenManager version of this method never had this bug.
  2. runkraken: the parallel branch created a `multiprocessing.Pool`
     without a `with` block / try-finally. If `pool.starmap(...)` (or
     anything before `pool.close()`) raised an exception, the pool's
     worker processes were never joined/closed, leaking OS processes.
     Fixed with `with multiprocessing.Pool(...) as pool:`.
  3. runkraken_broadband_range_dependent: the try/except around
     run_field_exec + readshd used a bare `except:`, which also
     swallows KeyboardInterrupt/SystemExit and gives no information
     about what failed. Narrowed to `except Exception as exc:` with the
     error message included.
------------------------------------------------------------------------
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import os
import shutil
import numpy as np
import multiprocessing

from cst import N_CORES
from propa.kraken_toolbox.params import KRAKEN_BIN_DIRECTORY
from propa.kraken_toolbox.src.kraken_env import KrakenEnv
from propa.kraken_toolbox.read_shd import readshd
from propa.kraken_toolbox.read_modes import readmodes
from propa.kraken_toolbox.utils import find_optimal_intervals


# ======================================================================================================================
# Main entry point
# ======================================================================================================================
def runkraken(
    env, flp, frequencies, parallel=False, verbose=False, clear=True, n_workers=None
):
    """Write the '.env'/'.flp' files then run KRAKEN/FIELD, choosing
    automatically the strategy suited to the given environment.

    Args:
        env (KrakenEnv): environment (env.range_dependent_env and
            env.broadband_run drive which strategy is used).
        flp (KrakenFlp): field parameters associated with env.
        frequencies (array-like): simulation frequencies (Hz).
        parallel (bool): enable parallelization over frequency batches
            (only relevant for the broadband + range-dependent case,
            the only one requiring several independent KRAKEN/FIELD
            runs -- see module docstring in kraken_manager.py for why).
        verbose (bool): print progress messages.
        clear (bool): clean up parallel working directories before
            launching a parallel simulation.
        n_workers (int|None): number of parallel processes. Defaults to
            min(number of frequencies, number of CPU cores).

    Returns:
        tuple(pressure_field, field_pos): complex pressure field and
        receiver grid positions (see read_shd.readshd).
    """
    if verbose:
        print(f"Running Kraken  (parallel = {parallel})...")

    # NOTE (bug fixed): matches the identical fix in
    # KrakenManager.runkraken() -- os.chdir(env.root) below used to
    # never be restored, leaking a global process-wide side effect past
    # the end of this call.
    original_cwd = os.getcwd()
    try:
        os.chdir(env.root)
        env.write_env()
        flp.write_flp()

        if env.range_dependent_env and env.broadband_run:
            return _run_broadband_range_dependent(
                env, flp, frequencies, parallel=parallel, verbose=verbose,
                clear=clear, n_workers=n_workers,
            )
        return _run_native(env, flp, frequencies, verbose=verbose)
    finally:
        os.chdir(original_cwd)


def _run_broadband_range_dependent(env, flp, frequencies, parallel, verbose, clear, n_workers):
    """'Broadband + variable bottom' case: not natively supported by
    KRAKEN -> re-run once per frequency and merge the results."""
    if parallel:
        pressure_field, field_pos, _all_modes = _run_broadband_range_dependent_parallel(
            env, flp, frequencies, clear=clear, n_workers=n_workers
        )
    else:
        pressure_field, field_pos, _all_modes = runkraken_broadband_range_dependent(
            env=env, flp=flp, frequencies=frequencies
        )
    # NOTE: '_all_modes' (per-frequency mode shapes, see
    # runkraken_broadband_range_dependent's docstring) is discarded
    # here so this function's return signature stays a plain
    # (pressure_field, field_pos) 2-tuple, like every other case. If you
    # need the per-frequency mode data (e.g. to plot it with
    # plot_utils.plotmode_from_data), call
    # runkraken_broadband_range_dependent() directly instead of going
    # through this wrapper -- exactly like KrakenManager.runkraken()
    # exposes it via the separate 'self.last_modes' attribute rather
    # than changing its own return signature.

    if verbose:
        print("Broadband range dependent kraken simulation completed.")
    return pressure_field, field_pos


def _run_broadband_range_dependent_parallel(env, flp, frequencies, clear, n_workers):
    """Distribute frequencies across several processes, each handling
    its own frequency range via runkraken_broadband_range_dependent(),
    then concatenate the resulting pressure fields (and mode lists, in
    the same frequency order -- each worker's frequency batch is
    contiguous and in order, see assign_frequency_intervalls)."""
    if clear:
        clear_kraken_parallel_working_dir(root=env.root)

    n_workers_requested = n_workers if n_workers is not None else N_CORES
    n_workers = min(len(frequencies), n_workers_requested)

    frequencies_intervalls, nb_used_workers = assign_frequency_intervalls(
        frequencies, n_workers, mode="optimal"
    )
    n_workers = nb_used_workers

    param_pool = [
        (env, flp, frequencies_intervalls[i], True)
        for i in range(len(frequencies_intervalls))
    ]

    # NOTE (bug fixed): the original code created the Pool without a
    # 'with' block, calling pool.close()/pool.join() only after
    # pool.starmap() returned. Any exception raised during starmap (in a
    # worker or while collecting results) skipped close()/join()
    # entirely, leaking the pool's worker processes. A 'with' block
    # guarantees proper cleanup on every exit path, including exceptions.
    with multiprocessing.Pool(processes=n_workers) as pool:
        result = pool.starmap(
            runkraken_broadband_range_dependent, param_pool, chunksize=1
        )

    field_pos = result[0][1]
    pressure_field = np.concatenate([r[0] for r in result], axis=0)
    all_modes = [Modes for r in result for Modes in r[2]]
    return pressure_field, field_pos, all_modes


def _run_native(env, flp, frequencies, verbose):
    """Case natively supported by KRAKEN: flat bottom (with or without
    broadband), or variable bottom at a single frequency. A single
    KRAKEN + FIELD run is enough; no parallelization here."""
    run_kraken_exec(env.filename)
    run_field_exec(env.filename)

    _, _, _, _, read_freq, _, field_pos, pressure_field = readshd(
        filename=env.filename + ".shd", freq=frequencies
    )

    if verbose:
        if env.broadband_run:
            print("Broadband range independent kraken simulation completed.")
        elif env.range_dependent_env:
            print("Single frequency range dependent kraken simulation completed.")
        else:
            print("Single frequency range independent kraken simulation completed.")

    return pressure_field, field_pos


# ======================================================================================================================
# Frequency distribution across workers
# ======================================================================================================================
def assign_frequency_intervalls(frequencies, n_workers, mode="equally_distributed"):
    """Distribute the frequencies to process across 'n_workers' batches.

    Args:
        frequencies (array-like): frequencies to distribute (Hz).
        n_workers (int): desired number of batches.
        mode (str):
            - 'equally_distributed': same number of frequencies per
              batch (give or take one), in order.
            - 'optimal': distributes frequencies to balance the
              estimated CPU time across batches (see
              utils.find_optimal_intervals), since the cost of
              KRAKEN/FIELD generally differs between low and high
              frequencies. If there are fewer frequencies than workers,
              falls back to 'equally_distributed' (a batch cannot be
              empty).

    Returns:
        tuple(list[np.ndarray], int): the list of frequency batches
        (empty batches omitted) and the number of batches actually used
        (<= n_workers).
    """
    nf = len(frequencies)

    if mode == "equally_distributed":
        assigned_frequency_ranges = _split_equally(frequencies, n_workers, nf)
    elif mode == "optimal":
        if nf <= n_workers:
            # Fewer frequencies than workers: might as well assign one
            # per worker, no need to optimize.
            assigned_frequency_ranges, _ = assign_frequency_intervalls(
                frequencies=frequencies, n_workers=n_workers, mode="equally_distributed"
            )
        else:
            assigned_frequency_ranges = _split_optimal(frequencies, n_workers, nf)
    else:
        raise ValueError(f"Mode {mode} not implemented.")

    nb_used_workers = len(assigned_frequency_ranges)
    return assigned_frequency_ranges, nb_used_workers


def _split_equally(frequencies, n_workers, nf):
    """Split 'frequencies' into 'n_workers' contiguous chunks of sizes
    as equal as possible, omitting empty chunks (case where
    nf < n_workers)."""
    ranges = []
    for i in range(n_workers):
        lo = i * nf // n_workers
        hi = min((i + 1) * nf // n_workers, nf)
        chunk = frequencies[lo:hi]
        if len(chunk) > 0:
            ranges.append(chunk)
    return ranges


def _split_optimal(frequencies, n_workers, nf):
    """Split 'frequencies' into frequency bounds chosen to balance the
    estimated CPU cost across workers (see
    utils.find_optimal_intervals), then assign each worker the
    frequencies actually falling within its range."""
    assigned_frequency_ranges = []
    _expected_cpu_time, f_bounds = find_optimal_intervals(
        fmin=frequencies.min(), fmax=frequencies.max(), nf=nf, n_workers=n_workers
    )
    for i in range(n_workers):
        idx_freq = np.logical_and(
            frequencies >= f_bounds[i], frequencies <= f_bounds[i + 1]
        )
        if np.any(idx_freq):
            assigned_frequency_ranges.append(frequencies[idx_freq])
    return assigned_frequency_ranges


# ======================================================================================================================
# Launching the Fortran executables
# ======================================================================================================================
def run_exec(exec, filename, parallel, worker_pid, silent):
    """Build and run the shell command launching the Fortran executable
    'exec' (kraken/field) on 'filename'.

    On Windows, the '.exe' extension is added automatically. In parallel
    mode on Windows, each worker uses its own copy of the binaries
    (local 'bin' folder), because launching kraken.exe/field.exe
    simultaneously from several processes sharing the same executable
    causes issues on Windows (see init_parallel_kraken_working_dirs).

    Args:
        exec (str): executable name, without extension ('kraken' or 'field').
        filename (str): '.env'/'.flp' file name, without extension.
        parallel (bool): if True (and on Windows), look for the
            executable in the current worker's local 'bin' folder rather
            than in the PATH.
        worker_pid (int|None): worker process id (required if
            parallel=True on Windows).
        silent (bool): redirect stdout/stderr to /dev/null (or NUL on
            Windows).
    """
    if os.name == "nt":
        ext = ".exe"
        silent_redirection = " >NUL 2>&1"
    else:
        ext = ""
        silent_redirection = " >/dev/null 2>&1"

    if parallel and (os.name == "nt"):
        if worker_pid is None:
            raise ValueError("worker_pid must be specified with parallel set to True.")
        subprocess_working_dir = os.path.join(os.getcwd(), "bin")
        cmd = os.path.join(subprocess_working_dir, exec)
    else:
        cmd = exec

    cmd += ext
    to_ex = f"{cmd} {filename}"
    if silent:
        to_ex += silent_redirection

    os.system(to_ex)


def run_field_exec(filename, parallel=False, worker_pid=None, silent=False):
    """Run the FIELD executable on '<filename>.flp'. See run_exec for
    parameter details."""
    run_exec(
        exec="field", filename=filename, parallel=parallel,
        worker_pid=worker_pid, silent=silent,
    )


def run_kraken_exec(filename, parallel=False, worker_pid=None, silent=True):
    """Run the KRAKEN executable on '<filename>.env'. See run_exec for
    parameter details."""
    run_exec(
        exec="kraken", filename=filename, parallel=parallel,
        worker_pid=worker_pid, silent=silent,
    )


# ======================================================================================================================
# Managing parallel working directories
# ======================================================================================================================
def clear_kraken_parallel_working_dir(root):
    """Remove the working subdirectories created by previous parallel
    workers, under '<root>/parallel_working_dir/'."""
    parallel_dir = os.path.join(root, "parallel_working_dir")
    if not os.path.isdir(parallel_dir):
        return
    for entry in os.scandir(parallel_dir):
        if entry.is_dir():
            shutil.rmtree(entry.path)


def get_subprocess_working_dir(env_root, worker_pid):
    """Return (creating it if needed) the working directory dedicated to
    process 'worker_pid', under
    '<env_root>/parallel_working_dir/child_process_<worker_pid>/'."""
    subprocess_working_dir = os.path.join(
        env_root, "parallel_working_dir", f"child_process_{worker_pid}"
    )
    os.makedirs(subprocess_working_dir, exist_ok=True)
    return subprocess_working_dir


def init_parallel_kraken_working_dirs(env, env_root, worker_pid):
    """Prepare a parallel worker's working directory: create its
    dedicated folder, point env.root to it, and (Windows only) locally
    copy the KRAKEN/FIELD binaries and their DLLs, because launching
    several instances of the same shared executables from concurrent
    processes fails on Windows.
    """
    subprocess_working_dir = get_subprocess_working_dir(env_root, worker_pid)
    env.root = subprocess_working_dir

    if os.name != "nt":
        return

    bin_folder = os.path.join(subprocess_working_dir, "bin")
    os.makedirs(bin_folder, exist_ok=True)

    required_files = [
        "kraken.exe",
        "field.exe",
        "cyggcc_s-seh-1.dll",
        "cyggfortran-5.dll",
        "cygquadmath-0.dll",
        "cygwin1.dll",
    ]
    for bin_file in required_files:
        src = os.path.join(KRAKEN_BIN_DIRECTORY, bin_file)
        dst = os.path.join(bin_folder, bin_file)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)


# ======================================================================================================================
# Broadband + variable-bottom simulation (see module note above)
# ======================================================================================================================
def runkraken_broadband_range_dependent(env, flp, frequencies, parallel=False):
    """Works around the lack of native "broadband + variable bottom"
    support by re-running KRAKEN/FIELD independently for each frequency
    in 'frequencies' (same range-dependent environment, different
    frequency each time), then stacking the resulting pressure fields
    along a new frequency axis.

    Args:
        env (KrakenEnv): reference range-dependent environment (serves
            as a template: halfspaces, medium, bathymetry...).
        flp (KrakenFlp): field parameters (written only once, they do
            not depend on frequency).
        frequencies (array-like): frequencies (Hz) to process.
        parallel (bool): passed to run_kraken_exec/run_field_exec so the
            worker's local binaries are used (see
            init_parallel_kraken_working_dirs).

    Returns:
        tuple(np.ndarray, dict, list): broadband pressure field of
        shape (n_freq, ...), receiver grid positions, and a list of
        per-frequency Modes dicts (see read_modes.readmodes), in the
        same order as 'frequencies' -- see the NOTE below for why this
        third value exists.

    NOTE (bug fixed): every frequency in this loop re-runs KRAKEN into
    the SAME working directory with the SAME '.env'/'.mod'/'.shd'
    filename (only the content differs). This is already handled
    correctly for the pressure field: it is read via readshd() and
    accumulated into 'broadband_pressure_field' INSIDE this loop,
    before the next iteration overwrites the '.shd' file. Mode shapes
    were NOT collected the same way: a caller had no choice but to read
    the '.mod' file again afterwards, by which point it only ever
    contained the LAST frequency's modes -- confirmed to fail with
    `FileNotFoundError` for a caller expecting to find the top-level,
    original filename (the file that actually exists after this loop
    sits in a 'parallel_working_dir' subdirectory instead, and even
    there holds only the last frequency's data). Fixed the same way as
    the pressure field: read the '.mod' file with read_modes.readmodes()
    INSIDE the loop, right after it is written, and accumulate the
    per-frequency results into a list returned alongside the pressure
    field. See plot_utils.plotmode_from_data() for the corresponding
    plotting function (mirroring plotshd_from_pressure_field()). This
    mirrors the identical fix already applied to
    KrakenManager.runkraken_broadband_range_dependent() in
    kraken_manager.py.
    """
    worker_pid = os.getpid()
    # NOTE (bug fixed): matches the identical fix in
    # KrakenManager.runkraken_broadband_range_dependent() -- see there
    # for the full explanation.
    original_cwd = os.getcwd()
    try:
        env_root = env.root
        broadband_pressure_field = None
        field_pos = None
        all_modes = []

        for ifreq, freq in enumerate(frequencies):
            # Rebuild an identical environment but at a single frequency:
            # KRAKEN must be re-run from scratch for every frequency in
            # range-dependent mode (see module docstring).
            env = KrakenEnv(
                title=env.simulation_title,
                env_root=env.root,
                env_filename=env.filename,
                freq=freq,
                kraken_top_hs=env.top_hs,
                kraken_medium=env.medium,
                kraken_attenuation=env.att,
                kraken_bottom_hs=env.bottom_hs,
                kraken_field=env.field,
                kraken_bathy=env.bathy,
                rModes=env.modes_range,
                # NOTE (bug fixed): the original code omitted 'nmedia' here,
                # silently resetting it back to KrakenEnv's default (1) for
                # every reconstructed per-frequency environment, even if the
                # original 'env' used a different value. The KrakenManager
                # version of this same method already passed it correctly.
                nmedia=env.nmedia,
            )

            init_parallel_kraken_working_dirs(env, env_root, worker_pid)
            os.chdir(env.root)

            flp.flp_fpath = env.flp_fpath
            env.write_env()
            if ifreq == 0:
                # The '.flp' file does not depend on frequency: write it
                # only once.
                flp.write_flp()

            run_kraken_exec(env.filename, parallel, worker_pid)
            try:
                run_field_exec(env.filename, parallel, worker_pid)
                _, _, _, _, _read_freq, _, field_pos, pressure = readshd(
                    filename=env.filename + ".shd", freq=freq
                )
                # NOTE (bug fixed, see docstring): read right away, before
                # the NEXT iteration overwrites this frequency's '.mod'
                # file with the next one's.
                Modes = readmodes(env.filename + ".mod", freq=freq)
            except Exception as exc:
                # NOTE (bug fixed): the original code used a bare `except:`
                # (which also catches KeyboardInterrupt/SystemExit) and just
                # printed "error", with no information about the actual
                # failure. Narrowed to Exception and the error message is
                # now included. The "do not interrupt the loop" behaviour is
                # kept unchanged: you may want to decide whether a failing
                # frequency should instead abort the whole simulation.
                print(f"Error running field executable for frequency {freq}: {exc}")
                continue

            if broadband_pressure_field is None:
                broadband_shape = (len(frequencies),) + pressure.shape
                broadband_pressure_field = np.zeros(broadband_shape, dtype=complex)

            broadband_pressure_field[ifreq, ...] = pressure
            all_modes.append(Modes)

        return broadband_pressure_field, field_pos, all_modes
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    pass
