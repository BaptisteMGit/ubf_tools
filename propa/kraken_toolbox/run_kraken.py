#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   run_kraken.py
@Time    :   2024/02/21 15:34:27
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import os
import time

# import pathos
import psutil
import shutil
import multiprocessing
import numpy as np

# from tqdm import tqdm
from cst import BAR_FORMAT, N_CORES
from propa.kraken_toolbox.usefull_path import KRAKEN_BIN_DIRECTORY
from propa.kraken_toolbox.kraken_env import KrakenEnv, KrakenFlp
from propa.kraken_toolbox.read_shd import readshd
from propa.kraken_toolbox.utils import find_optimal_intervals


def runkraken(
    env, flp, frequencies, parallel=False, verbose=False, clear=True, n_workers=None
):
    if verbose:
        print(f"Running Kraken  (parallel = {parallel})...")

    # Change directory
    os.chdir(env.root)
    # Write env and flp files
    env.write_env()
    flp.write_flp()

    if (
        env.range_dependent_env and env.broadband_run
    ):  # Run broadband range dependent simulation

        # Clear working dirs
        if clear:
            clear_kraken_parallel_working_dir(root=env.root)

        # Run parallel
        if parallel:
            if n_workers is None:
                n_workers = min(len(frequencies), N_CORES)
            else:
                n_workers = min(len(frequencies), n_workers)

            # Get optimal frequencies intervals bounds
            frequencies_intervalls, nb_used_workers = assign_frequency_intervalls(
                frequencies, n_workers, mode="optimal"
            )
            n_workers = nb_used_workers

            # Build the parameter pool
            param_pool = [
                (
                    env,
                    flp,
                    frequencies_intervalls[i],
                    True,
                )
                for i in range(len(frequencies_intervalls))
            ]

            # t0 = time.time()
            # Spawn processes
            pool = multiprocessing.Pool(processes=n_workers)

            # Run parallel processes
            result = pool.starmap(
                runkraken_broadband_range_dependent, param_pool, chunksize=1
            )
            field_pos = result[0][1]
            pressure_field = np.concatenate([r[0] for r in result], axis=0)

            # Close pool
            pool.close()
            # Wait for all processes to finish
            pool.join()

            # cpu_time = time.time() - t0
            # print(f"CPU time (Map): {cpu_time:.2f} s")

        else:
            pressure_field, field_pos = runkraken_broadband_range_dependent(
                env=env, flp=flp, frequencies=frequencies
            )

        if verbose:
            print("Broadband range dependent kraken simulation completed.")

        return pressure_field, field_pos

    else:  # Run range independent simulation (no parallelization for now)

        # Run Fortran version of Kraken
        run_kraken(env.filename)
        # Run Fortran version of Field
        run_field(env.filename)

        # Read pressure field for the current frequency
        _, _, _, _, read_freq, _, field_pos, pressure_field = readshd(
            filename=env.filename + ".shd", freq=frequencies
        )
        if verbose and not env.range_dependent_env and env.broadband_run:
            print("Broadband range independent kraken simulation completed.")
        elif verbose and env.range_dependent_env and not env.broadband_run:
            print("Single frequency range dependent kraken simulation completed.")
        elif verbose and not env.range_dependent_env and not env.broadband_run:
            print("Single frequency range independent kraken simulation completed.")

        return pressure_field, field_pos


def assign_frequency_intervalls(frequencies, n_workers, mode="equally_distributed"):
    """
    Assign frequency intervals to workers.

    :param frequencies:
    :param n_workers:
    :return:
    """
    # Distribute frequencies to workers, ensuring decreasing subarray sizes
    nf = len(frequencies)

    if mode == "equally_distributed":
        assigned_frequency_ranges = [
            frequencies[slice(i * nf // n_workers, min((i + 1) * nf // n_workers, nf))]
            for i in range(n_workers)
            if len(
                frequencies[
                    slice(i * nf // n_workers, min((i + 1) * nf // n_workers, nf))
                ]
            )
            > 0  # Assert at least 1 freq falls into the interval
        ]

    elif mode == "optimal":
        if (
            nf <= n_workers
        ):  # If there is less freqs than workers the optimal choice is to assign one freq per worker
            assigned_frequency_ranges, _ = assign_frequency_intervalls(
                frequencies=frequencies, n_workers=n_workers, mode="equally_distributed"
            )
        else:
            assigned_frequency_ranges = []
            expected_cpu_time, f_bounds = find_optimal_intervals(
                fmin=frequencies.min(),
                fmax=frequencies.max(),
                nf=nf,
                n_workers=n_workers,
            )

            for i in range(n_workers):
                idx_freq = np.logical_and(
                    frequencies >= f_bounds[i], frequencies <= f_bounds[i + 1]
                )
                # Assert at least 1 freq falls into the interval
                if np.any(idx_freq):
                    assigned_frequency_ranges.append(frequencies[idx_freq])

    else:
        raise ValueError(f"Mode {mode} not implemented.")

    nb_used_workers = len(assigned_frequency_ranges)

    return assigned_frequency_ranges, nb_used_workers


def run_field(filename, parallel=False, worker_pid=None):
    if parallel:
        if worker_pid is not None:
            parallel_working_dir = os.getcwd()
            subprocess_working_dir = os.path.join(parallel_working_dir, "bin")
            fpath_to_kraken = os.path.join(subprocess_working_dir, "field.exe")
            cmd = f"{fpath_to_kraken} {filename}"
        else:
            raise ValueError(f"worker_pid must be specified with parallel set to True.")
    else:
        cmd = "field"

    # Run Fortran version of Field
    os.system(f"{cmd} {filename}")


def run_kraken(filename, parallel=False, worker_pid=None):
    if parallel and (os.name == "nt"):
        if worker_pid is not None:
            parallel_working_dir = os.getcwd()
            subprocess_working_dir = os.path.join(parallel_working_dir, "bin")
            fpath_to_kraken = os.path.join(subprocess_working_dir, "kraken.exe")
            cmd = f"{fpath_to_kraken} {filename}"
        else:
            raise ValueError(f"worker_pid must be specified with parallel set to True.")
    else:
        cmd = "kraken"

    # Run Fortran version of Kraken
    os.system(f"{cmd} {filename}")


def clear_kraken_parallel_working_dir(root):
    """
    Clear working directories.
    """
    root_parallel_folder = "parallel_working_dir"
    dir = os.path.join(root, root_parallel_folder)
    for root, dirs, files in os.walk(dir):
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def get_child_pids():
    """
    Get child process pid.

    :return:
    """
    parent_pid = multiprocessing.current_process().pid
    children = psutil.Process(parent_pid).children(recursive=True)
    return [child.pid for child in children]


def get_subprocess_working_dir(env_root, worker_pid):
    # Create folder dedicated to the worker_pid
    root_parallel_folder = "parallel_working_dir"
    parallel_folder = f"child_process_{worker_pid}"

    # Create folder dedicated to the worker_pid
    subprocess_working_dir = os.path.join(
        env_root, root_parallel_folder, parallel_folder
    )

    if not os.path.exists(subprocess_working_dir):
        os.makedirs(subprocess_working_dir)

    return subprocess_working_dir


def init_parallel_kraken_working_dirs(env, env_root, worker_pid):
    """
    Initialise working directory to be used by child processes for multiprocessing.

    :param root:
    :param worker_pid:
    :return:
    """

    subprocess_working_dir = get_subprocess_working_dir(env_root, worker_pid)
    env.root = subprocess_working_dir

    if os.name == "nt":  # Windows

        # Create bin folder
        bin_folder = os.path.join(subprocess_working_dir, "bin")
        if not os.path.exists(bin_folder):
            os.makedirs(bin_folder)

        # Copy bin files to subprocess working directory
        # (that's ugly but it works... calling kraken.exe and field.exe simultaneously from different process leads to errors)
        for bin in [
            "kraken.exe",
            "field.exe",
            "cyggcc_s-seh-1.dll",
            "cyggfortran-5.dll",
            "cygquadmath-0.dll",
            "cygwin1.dll",
        ]:
            f_path_src = os.path.join(KRAKEN_BIN_DIRECTORY, bin)
            f_path_dst = os.path.join(bin_folder, bin)
            if not os.path.exists(f_path_dst):
                shutil.copyfile(f_path_src, f_path_dst)


def runkraken_broadband_range_dependent(
    env,
    flp,
    frequencies,
    parallel=False,
):
    """KRAKEN is capable of running broadband simulations with range independent environments
    and single frequency simulations with range dependent environments. Yet, for some reason,
    it is not capable of running broadband simulations with range dependent environments.
    This function is a workaround to this issue. It runs KRAKEN with a range dependent environment
    for each frequency of the broadband simulation and then merge the results in a single pressure field.
    """
    # Root dir to share with subprocesses
    worker_pid = os.getpid()
    env_root = env.root

    # Loop over frequencies
    # for ifreq in tqdm(
    #     range(len(frequencies)),
    #     bar_format=BAR_FORMAT,
    #     desc=desc,
    # ):

    for ifreq in range(len(frequencies)):
        # Initialize environment with the current frequency and provided range dependent environment
        env = KrakenEnv(
            title=env.simulation_title,
            env_root=env.root,
            env_filename=env.filename,
            freq=frequencies[ifreq],
            kraken_top_hs=env.top_hs,
            kraken_medium=env.medium,
            kraken_attenuation=env.att,
            kraken_bottom_hs=env.bottom_hs,
            kraken_field=env.field,
            kraken_bathy=env.bathy,
            rModes=env.modes_range,
        )

        init_parallel_kraken_working_dirs(env, env_root, worker_pid)
        os.chdir(env.root)

        # Update flp_fpath with subprocess working directory
        flp.flp_fpath = env.flp_fpath

        # Write environment
        env.write_env()

        if ifreq == 0:
            # Write .flp file for the first frequency (independent from frequency)
            flp.write_flp()

        # Run Fortran version of Kraken
        run_kraken(env.filename, parallel, worker_pid)
        # Run Fortran version of Field
        run_field(env.filename, parallel, worker_pid)

        # Read pressure field for the current frequency
        _, _, _, _, read_freq, _, field_pos, pressure = readshd(
            filename=env.filename + ".shd", freq=frequencies[ifreq]
        )

        # Initialize broadband pressure field array
        if ifreq == 0:
            broadband_shape = (len(frequencies),) + pressure.shape
            broadband_pressure_field = np.zeros(broadband_shape, dtype=complex)

        broadband_pressure_field[ifreq, ...] = pressure

    # time.sleep(worker_pid / 10000)
    return broadband_pressure_field, field_pos


if __name__ == "__main__":
    pass
