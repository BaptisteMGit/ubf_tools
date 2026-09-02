#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   kraken_manager.py
@Time    :   2025/04/07 11:53:05
@Author  :   Menetrier Baptiste
@Version :   1.1 (refactor)
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class orchestrating KRAKEN/FIELD (Fortran executables) runs:
             writing input files, launching (sequential or parallel),
             reading results.

This module does NOT change the public API of the original file (same
class/method/parameter names). It has been reorganized, documented, and
a large block of dead/duplicated code has been removed (see below).

------------------------------------------------------------------------
What runkraken() does, one sentence per case
------------------------------------------------------------------------
KRAKEN natively supports:
  - a flat-bottom environment (range-independent), over one or several
    frequencies (broadband);
  - a variable-bottom environment (range-dependent), but only at a
    SINGLE frequency at a time (a limitation of the FIELD version used
    here).

There is therefore no native "range-dependent + broadband" mode. The
runkraken_broadband_range_dependent() method works around this
limitation by re-running KRAKEN/FIELD once per frequency (optionally in
parallel, one process per frequency batch) and then merging the
resulting pressure fields into a single broadband array.

------------------------------------------------------------------------
IMPORTANT - code duplication found in the original file
------------------------------------------------------------------------
The original `readshd` and `readshd_bin` class methods reimplemented,
line for line (more than 150 lines), the content of the module already
imported at the top of the file:

    from propa.kraken_toolbox.read_shd import readshd

This 'read_shd.py' module is already used elsewhere in runkraken() /
runkraken_broadband_range_dependent(). Having two copies of the same
binary-parsing code is a maintenance risk (a bug fixed in one copy is
not necessarily fixed in the other). In this version, the class methods
have been kept (in case existing code calls them via
`KrakenManager.readshd(...)`) but turned into thin delegations to
'read_shd.py': the binary-parsing code now lives in a single place.

Similarly, this file duplicates a large part of the logic found in
`propa/kraken_toolbox/run_kraken.py` (procedural functions `runkraken`,
`assign_frequency_intervalls`, `run_exec`, etc., nearly identical to the
ones below). This has not been touched here (this file was not provided
as a target for the refactor) but deserves your attention: eventually,
only one of the two should probably remain.
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
from propa.kraken_toolbox.read_shd import readshd, readshd_bin
from propa.kraken_toolbox.read_modes import readmodes
from propa.kraken_toolbox.utils import find_optimal_intervals


class KrakenManager:
    """Orchestrates KRAKEN/FIELD runs: writing input files, launching
    (sequentially or in parallel depending on frequency and range
    dependence of the environment), reading the resulting pressure
    field.
    """

    def __init__(self, parallel=False, verbose=False, clear=True, n_workers=None):
        """
        Args:
            parallel (bool): enable parallelization over frequency
                batches (only useful for broadband + variable-bottom
                simulations, the only case that requires several
                independent KRAKEN/FIELD runs).
            verbose (bool): print progress messages.
            clear (bool): clean up parallel working directories before
                launching a parallel simulation.
            n_workers (int|None): number of parallel processes. Defaults
                to min(number of frequencies, number of CPU cores).
        """
        self.parallel = parallel
        self.verbose = verbose
        self.clear = clear
        self.n_workers = n_workers
        # Populated only after a broadband + range-dependent run (see
        # _run_broadband_range_dependent): a list of per-frequency
        # Modes dicts (as returned by read_modes.readmodes), in the
        # same order as 'frequencies'. None otherwise (including right
        # after construction, and after a native run -- in that case,
        # a single '.mod' file with every frequency already exists on
        # disk, so plot_utils.plotmode(mod_fpath, freq=...) can just
        # read it directly; no need for this side channel).
        self.last_modes = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def runkraken(self, env, flp, frequencies):
        """Write the '.env'/'.flp' files then run KRAKEN/FIELD, choosing
        automatically the strategy suited to the given environment.

        Args:
            env (KrakenEnv): environment (env.range_dependent_env and
                env.broadband_run drive which strategy is used).
            flp (KrakenFlp): field parameters associated with env.
            frequencies (array-like): simulation frequencies (Hz).

        Returns:
            tuple(pressure_field, field_pos): complex pressure field and
            receiver grid positions (see read_shd.readshd). For a
            broadband + range-dependent run specifically, also sets
            self.last_modes: a list of per-frequency Modes dicts (see
            read_modes.readmodes), in the same order as 'frequencies' --
            needed because that run strategy re-runs KRAKEN once per
            frequency, overwriting the same '.mod' file each time, so
            there is no single on-disk file left containing every
            frequency's modes to read back afterwards (mirrors why
            plot_utils.plotshd_from_pressure_field exists for the
            pressure field / '.shd' file). Use
            plot_utils.plotmode_from_data(manager.last_modes, freq=...)
            to plot mode shapes in that case; for every other case,
            self.last_modes is left at its initial value (None) and
            plot_utils.plotmode(mod_fpath, freq=...) works directly, as
            usual, against the single '.mod' file already on disk.
        """
        if self.verbose:
            print(f"Running Kraken  (parallel = {self.parallel})...")

        os.chdir(env.root)
        env.write_env()
        flp.write_flp()

        if env.range_dependent_env and env.broadband_run:
            return self._run_broadband_range_dependent(env, flp, frequencies)
        return self._run_native(env, flp, frequencies)

    def _run_broadband_range_dependent(self, env, flp, frequencies):
        """'Broadband + variable bottom' case: not natively supported by
        KRAKEN (see module docstring) -> re-run once per frequency and
        merge the results."""
        if self.parallel:
            pressure_field, field_pos, all_modes = (
                self._run_broadband_range_dependent_parallel(env, flp, frequencies)
            )
        else:
            if self.clear:
                self.clear_kraken_parallel_working_dir(root=env.root)

            pressure_field, field_pos, all_modes = (
                self.runkraken_broadband_range_dependent(
                    env=env, flp=flp, frequencies=frequencies
                )
            )
        # NOTE: see the module/README note on why this can't just be a
        # third return value of runkraken() itself -- runkraken()'s
        # public return signature (pressure_field, field_pos) must stay
        # unchanged for the native-run case, so the per-frequency mode
        # data collected here (only meaningful for THIS broadband +
        # range-dependent path) is instead exposed as an instance
        # attribute, read after the call: manager.last_modes.
        self.last_modes = all_modes

        if self.verbose:
            print("Broadband range dependent kraken simulation completed.")
        return pressure_field, field_pos

    def _run_broadband_range_dependent_parallel(self, env, flp, frequencies):
        """Distribute frequencies across several processes, each handling
        its own frequency range via runkraken_broadband_range_dependent(),
        then concatenate the resulting pressure fields (and mode lists,
        in the same frequency order -- each worker's frequency batch is
        contiguous and in order, see assign_frequency_intervalls)."""
        if self.clear:
            self.clear_kraken_parallel_working_dir(root=env.root)

        n_workers_requested = self.n_workers if self.n_workers is not None else N_CORES
        n_workers = min(len(frequencies), n_workers_requested)

        frequencies_intervalls, nb_used_workers = self.assign_frequency_intervalls(
            frequencies, n_workers, mode="optimal"
        )
        self.n_workers = nb_used_workers

        param_pool = [
            (env, flp, frequencies_intervalls[i], True)
            for i in range(len(frequencies_intervalls))
        ]

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            result = pool.starmap(
                self.runkraken_broadband_range_dependent, param_pool, chunksize=1
            )

        field_pos = result[0][1]
        pressure_field = np.concatenate([r[0] for r in result], axis=0)
        all_modes = [Modes for r in result for Modes in r[2]]
        return pressure_field, field_pos, all_modes

    def _run_native(self, env, flp, frequencies):
        """Case natively supported by KRAKEN: flat bottom (with or
        without broadband), or variable bottom at a single frequency. A
        single KRAKEN + FIELD run is enough; no parallelization here."""
        self.run_kraken_exec(env.filename)
        self.run_field_exec(env.filename)

        _, _, _, _, read_freq, _, field_pos, pressure_field = readshd(
            filename=env.filename + ".shd", freq=frequencies
        )

        if self.verbose:
            if env.broadband_run:
                print("Broadband range independent kraken simulation completed.")
            elif env.range_dependent_env:
                print("Single frequency range dependent kraken simulation completed.")
            else:
                print("Single frequency range independent kraken simulation completed.")

        return pressure_field, field_pos

    # ------------------------------------------------------------------
    # Frequency distribution across workers
    # ------------------------------------------------------------------
    @classmethod
    def assign_frequency_intervalls(
        cls, frequencies, n_workers, mode="equally_distributed"
    ):
        """Distribute the frequencies to process across 'n_workers' batches.

        Args:
            frequencies (array-like): frequencies to distribute (Hz).
            n_workers (int): desired number of batches.
            mode (str):
                - 'equally_distributed': same number of frequencies per
                  batch (give or take one), in order.
                - 'optimal': distributes frequencies to balance the
                  estimated CPU time across batches (see
                  find_optimal_intervals), since the cost of KRAKEN/FIELD
                  generally differs between low and high frequencies. If
                  there are fewer frequencies than workers, falls back to
                  'equally_distributed' (a batch cannot be empty).

        Returns:
            tuple(list[np.ndarray], int): the list of frequency batches
            (empty batches omitted) and the number of batches actually
            used (<= n_workers).
        """
        nf = len(frequencies)

        if mode == "equally_distributed":
            assigned_frequency_ranges = cls._split_equally(frequencies, n_workers, nf)
        elif mode == "optimal":
            if nf <= n_workers:
                # Fewer frequencies than workers: might as well assign
                # one per worker, no need to optimize.
                assigned_frequency_ranges, _ = cls.assign_frequency_intervalls(
                    frequencies=frequencies,
                    n_workers=n_workers,
                    mode="equally_distributed",
                )
            else:
                assigned_frequency_ranges = cls._split_optimal(
                    frequencies, n_workers, nf
                )
        else:
            raise ValueError(f"Mode {mode} not implemented.")

        nb_used_workers = len(assigned_frequency_ranges)
        return assigned_frequency_ranges, nb_used_workers

    @staticmethod
    def _split_equally(frequencies, n_workers, nf):
        """Split 'frequencies' into 'n_workers' contiguous chunks of
        sizes as equal as possible, omitting empty chunks (case where
        nf < n_workers)."""
        ranges = []
        for i in range(n_workers):
            lo = i * nf // n_workers
            hi = min((i + 1) * nf // n_workers, nf)
            chunk = frequencies[lo:hi]
            if len(chunk) > 0:
                ranges.append(chunk)
        return ranges

    @staticmethod
    def _split_optimal(frequencies, n_workers, nf):
        """Split 'frequencies' into frequency bounds chosen to balance
        the estimated CPU cost across workers (see
        find_optimal_intervals), then assign each worker the frequencies
        actually falling within its range."""
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

    # ------------------------------------------------------------------
    # Launching the Fortran executables
    # ------------------------------------------------------------------
    @classmethod
    def run_field_exec(cls, filename, parallel=False, worker_pid=None, silent=False):
        """Run the FIELD executable on '<filename>.flp'.

        Args:
            filename (str): '.flp' file name, without extension.
            parallel (bool): if True, look for the executable in the
                current worker's local 'bin' folder rather than in the
                PATH.
            worker_pid (int|None): worker process id (required if
                parallel=True).
            silent (bool): redirect stdout/stderr to /dev/null (or NUL
                on Windows).
        """
        cls.run_exec(
            exec="field",
            filename=filename,
            parallel=parallel,
            worker_pid=worker_pid,
            silent=silent,
        )

    @classmethod
    def run_kraken_exec(cls, filename, parallel=False, worker_pid=None, silent=True):
        """Run the KRAKEN executable on '<filename>.env'. Same
        parameters as run_field_exec."""
        cls.run_exec(
            exec="kraken",
            filename=filename,
            parallel=parallel,
            worker_pid=worker_pid,
            silent=silent,
        )

    @staticmethod
    def run_exec(exec, filename, parallel, worker_pid, silent):
        """Build and run the shell command launching the Fortran
        executable 'exec' (kraken/field) on 'filename'.

        On Windows, the '.exe' extension is added automatically. In
        parallel mode on Windows, each worker uses its own copy of the
        binaries (local 'bin' folder), because launching
        kraken.exe/field.exe simultaneously from several processes
        sharing the same executable causes issues on Windows (see
        init_parallel_kraken_working_dirs).
        """
        if os.name == "nt":
            ext = ".exe"
            silent_redirection = " >NUL 2>&1"
        else:
            ext = ""
            silent_redirection = " >/dev/null 2>&1"

        if parallel and (os.name == "nt"):
            if worker_pid is None:
                raise ValueError(
                    "worker_pid must be specified with parallel set to True."
                )
            subprocess_working_dir = os.path.join(os.getcwd(), "bin")
            cmd = os.path.join(subprocess_working_dir, exec)
        else:
            cmd = exec

        cmd += ext
        to_ex = f"{cmd} {filename}"
        if silent:
            to_ex += silent_redirection

        os.system(to_ex)

    # ------------------------------------------------------------------
    # Managing parallel working directories
    # ------------------------------------------------------------------
    @staticmethod
    def clear_kraken_parallel_working_dir(root):
        """Remove the working subdirectories created by previous
        parallel workers, under '<root>/parallel_working_dir/'."""
        parallel_dir = os.path.join(root, "parallel_working_dir")
        if not os.path.isdir(parallel_dir):
            return
        for entry in os.scandir(parallel_dir):
            if entry.is_dir():
                shutil.rmtree(entry.path)

    @staticmethod
    def get_subprocess_working_dir(env_root, worker_pid):
        """Return (creating it if needed) the working directory
        dedicated to process 'worker_pid', under
        '<env_root>/parallel_working_dir/child_process_<worker_pid>/'."""
        subprocess_working_dir = os.path.join(
            env_root, "parallel_working_dir", f"child_process_{worker_pid}"
        )
        os.makedirs(subprocess_working_dir, exist_ok=True)
        return subprocess_working_dir

    @classmethod
    def init_parallel_kraken_working_dirs(cls, env, env_root, worker_pid):
        """Prepare a parallel worker's working directory: create its
        dedicated folder, point env.root to it, and (Windows only)
        locally copy the KRAKEN/FIELD binaries and their DLLs, because
        launching several instances of the same shared executables from
        concurrent processes fails on Windows.
        """
        subprocess_working_dir = cls.get_subprocess_working_dir(env_root, worker_pid)
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

    # ------------------------------------------------------------------
    # Broadband + variable-bottom simulation (see module note above)
    # ------------------------------------------------------------------
    @classmethod
    def runkraken_broadband_range_dependent(cls, env, flp, frequencies, parallel=False):
        """Works around the lack of native "broadband + variable bottom"
        support by re-running KRAKEN/FIELD independently for each
        frequency in 'frequencies' (same range-dependent environment,
        different frequency each time), then stacking the resulting
        pressure fields along a new frequency axis.

        Args:
            env (KrakenEnv): reference range-dependent environment
                (serves as a template: halfspaces, medium, bathymetry...).
            flp (KrakenFlp): field parameters (written only once, they do
                not depend on frequency).
            frequencies (array-like): frequencies (Hz) to process.
            parallel (bool): passed to run_kraken_exec/run_field_exec so
                the worker's local binaries are used (see
                init_parallel_kraken_working_dirs).

        Returns:
            tuple(np.ndarray, dict, list): broadband pressure field of
            shape (n_freq, ...), receiver grid positions, and a list of
            per-frequency Modes dicts (see read_modes.readmodes), in the
            same order as 'frequencies' -- see the NOTE below for why
            this third value exists.

        NOTE (bug fixed): every frequency in this loop re-runs KRAKEN
        into the SAME working directory with the SAME '.env'/'.mod'/
        '.shd' filename (only the content differs) -- 'worker_pid' (and
        therefore the working directory returned by
        get_subprocess_working_dir) does not change across iterations
        in the serial case, and even in the parallel case each worker
        still reuses one directory across its own share of the
        frequencies. This is already handled correctly for the pressure
        field: it is read via readshd() and accumulated into
        'broadband_pressure_field' INSIDE this loop, before the next
        iteration overwrites the '.shd' file. Mode shapes were NOT
        collected the same way: a caller had no choice but to read the
        '.mod' file again afterwards, by which point it only ever
        contained the LAST frequency's modes -- confirmed to fail with
        `FileNotFoundError` for a caller expecting to find the
        top-level, original filename (the file that actually exists
        after this loop sits in a 'parallel_working_dir' subdirectory
        instead, and even there holds only the last frequency's data).
        Fixed the same way as the pressure field: read the '.mod' file
        with read_modes.readmodes() INSIDE the loop, right after it is
        written, and accumulate the per-frequency results into a list
        returned alongside the pressure field. See
        plot_utils.plotmode_from_data() for the corresponding plotting
        function (mirroring plotshd_from_pressure_field()).
        """
        worker_pid = os.getpid()
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
                nmedia=env.nmedia,
            )

            cls.init_parallel_kraken_working_dirs(env, env_root, worker_pid)
            os.chdir(env.root)

            flp.flp_fpath = env.flp_fpath
            env.write_env()
            if ifreq == 0:
                # The '.flp' file does not depend on frequency: write it
                # only once.
                flp.write_flp()

            cls.run_kraken_exec(env.filename, parallel, worker_pid)
            try:
                cls.run_field_exec(env.filename, parallel, worker_pid)
                _, _, _, _, _read_freq, _, field_pos, pressure = readshd(
                    filename=env.filename + ".shd", freq=freq
                )
                # NOTE (bug fixed, see docstring): read right away,
                # before the NEXT iteration overwrites this frequency's
                # '.mod' file with the next one's.
                Modes = readmodes(env.filename + ".mod", freq=freq)
            except Exception as exc:
                # NOTE: the original code used a bare `except:` (which
                # also catches KeyboardInterrupt/SystemExit) and just
                # printed a message. We narrow this to Exception and
                # include the error message, but keep the "do not
                # interrupt the loop" behaviour so as not to change
                # existing behaviour: you may want to decide whether a
                # failing frequency should instead abort the whole
                # simulation.
                print(f"Error running field executable for frequency {freq}: {exc}")
                continue

            if broadband_pressure_field is None:
                broadband_shape = (len(frequencies),) + pressure.shape
                broadband_pressure_field = np.zeros(broadband_shape, dtype=complex)

            broadband_pressure_field[ifreq, ...] = pressure
            all_modes.append(Modes)

        return broadband_pressure_field, field_pos, all_modes

    # ------------------------------------------------------------------
    # Reading results ('.shd')
    # ------------------------------------------------------------------
    # NOTE (duplication removed): these two methods now only delegate to
    # 'propa.kraken_toolbox.read_shd', which holds the single
    # implementation of the '.shd' binary parsing logic (see the module
    # docstring above). Signature and behaviour unchanged for any
    # existing code calling KrakenManager.readshd(...).
    @classmethod
    def readshd(cls, filename, xs=None, ys=None, freq=None):
        """Read a '.shd' file produced by FIELD.exe.
        Delegates to propa.kraken_toolbox.read_shd.readshd (see that
        function for the full description of the return values)."""
        return readshd(filename=filename, xs=xs, ys=ys, freq=freq)

    @staticmethod
    def readshd_bin(filename, xs=None, ys=None, freq=None):
        """Read the binary '.shd' file directly (no extension
        resolution). Delegates to
        propa.kraken_toolbox.read_shd.readshd_bin."""
        return readshd_bin(filename=filename, xs=xs, ys=ys, freq=freq)


if __name__ == "__main__":
    pass
