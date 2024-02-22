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
import pathos
import psutil
import shutil
import multiprocess
import numpy as np

from tqdm import tqdm
from cst import BAR_FORMAT
from propa.kraken_toolbox.usefull_path import KRAKEN_BIN_DIRECTORY
from propa.kraken_toolbox.kraken_env import KrakenEnv, KrakenFlp
from propa.kraken_toolbox.read_shd import readshd

N_CORES = 2


def runkraken(env, flp, frequencies, parallel=False, verbose=False):
    if verbose:
        print(f"Running Kraken  (parallel = {parallel})...")

    if (
        env.range_dependent_env and env.broadband_run
    ):  # Run broadband range dependent simulation

        # Run parallel
        if parallel:
            # Clear working dirs
            clear_kraken_parallel_working_dir(root=env.root)

            # Spawn processes
            mp = (
                pathos.helpers.mp
            )  # Trick to get an equivalent of starmap using dill and not pickle for advanced serialization
            pool = mp.Pool(N_CORES)
            child_pids = get_child_pids()

            # Build the parameter pool
            nf = len(frequencies)
            param_pool = [
                (
                    env,
                    flp,
                    frequencies[
                        slice(i * nf // N_CORES, min((i + 1) * nf // N_CORES, nf))
                    ],
                    True,
                    child_pids[i],
                )
                for i in range(N_CORES)
                if len(
                    frequencies[
                        slice(i * nf // N_CORES, min((i + 1) * nf // N_CORES, nf))
                    ]
                )
                > 0
            ]

            # Run parallel processes
            result = pool.starmap(runkraken_broadband_range_dependent, param_pool)
            pressure_field = np.concatenate(result, axis=0)
        else:
            # Change directory
            os.chdir(env.root)
            # Write env and flp files
            env.write_env()
            flp.write_flp()

            pressure_field = runkraken_broadband_range_dependent(
                env=env, flp=flp, frequencies=frequencies
            )

        if verbose:
            print("Broadband range dependent kraken simulation completed.")

        return pressure_field

    else:  # Run range independent simulation (no parallelization for now)

        # Change directory
        os.chdir(env.root)

        # Write env and flp files
        env.write_env()
        flp.write_flp()

        # Run Fortran version of Kraken
        run_kraken(env.filename)
        # os.system(f"kraken {env.filename}")
        # Run Fortran version of Field
        # os.system(f"field {env.filename}")
        run_field(env.filename)

        # Read pressure field for the current frequency
        _, _, _, _, read_freq, _, _, pressure_field = readshd(
            filename=env.filename + ".shd", freq=frequencies
        )
        if verbose and not env.range_dependent_env and env.broadband_run:
            print("Broadband range independent kraken simulation completed.")
        else:
            print("Single frequency range dependent kraken simulation completed.")

        return pressure_field


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
    if parallel:
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
    parent_pid = multiprocess.current_process().pid
    children = psutil.Process(parent_pid).children(recursive=True)
    return [child.pid for child in children]


def init_parallel_kraken_working_dirs(env, env_root, worker_pid):
    """
    Initialise working directory to be used by child processes for multiprocessing.

    :param root:
    :param worker_pid:
    :return:
    """
    # Create folder dedicated to the worker_pid
    root_parallel_folder = "parallel_working_dir"
    parallel_folder = f"child_process_{worker_pid}"

    # Create folder dedicated to the worker_pid
    subprocess_working_dir = os.path.join(
        env_root, root_parallel_folder, parallel_folder
    )
    if not os.path.exists(subprocess_working_dir):
        os.makedirs(subprocess_working_dir)

    env.root = subprocess_working_dir

    # Create bin folder
    bin_folder = os.path.join(subprocess_working_dir, "bin")
    if not os.path.exists(bin_folder):
        os.makedirs(bin_folder)

    # Copy bin files to subprocess working directory
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
    env, flp, frequencies, parallel=False, worker_pid=None
):
    """KRAKEN is capable of running broadband simulations with range independent environments
    and single frequency simulations with range dependent environments. Yet, for some reason,
    it is not capable of running broadband simulations with range dependent environments.
    This function is a workaround to this issue. It runs KRAKEN with a range dependent environment
    for each frequency of the broadband simulation and then merge the results in a single pressure field.
    """
    # Root dir to share with subprocesses
    env_root = env.root

    # Loop over frequencies
    for ifreq in tqdm(
        range(len(frequencies)),
        bar_format=BAR_FORMAT,
        desc="Computing broadband pressure field for range dependent environment",
    ):
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

        if parallel:
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
        # os.system(f"kraken {env.filename}")
        # Run Fortran version of Field
        run_field(env.filename, parallel, worker_pid)
        # os.system(f"field {env.filename}")

        # Read pressure field for the current frequency
        _, _, _, _, read_freq, _, _, pressure = readshd(
            filename=env.filename + ".shd", freq=frequencies[ifreq]
        )

        # Initialize broadband pressure field array
        if ifreq == 0:
            broadband_shape = (len(frequencies),) + pressure.shape

            broadband_pressure_field = np.zeros(broadband_shape, dtype=complex)

        broadband_pressure_field[ifreq, ...] = pressure

    return broadband_pressure_field


if __name__ == "__main__":
    pass
