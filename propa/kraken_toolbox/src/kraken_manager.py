#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   kraken_manager.py
@Time    :   2025/04/07 11:53:05
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to handle Kraken simulations
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
from propa.kraken_toolbox.utils import find_optimal_intervals


class KrakenManager:
    """
    Class to handle Kraken simulations
    """

    def __init__(
        self,
        parallel=False,
        verbose=False,
        clear=True,
        n_workers=None,
    ):
        """
        Constructor
        """

        self.parallel = parallel
        self.verbose = verbose
        self.clear = clear
        self.n_workers = n_workers

    def runkraken(self, env, flp, frequencies):
        if self.verbose:
            print(f"Running Kraken  (parallel = {self.parallel})...")

        # Change directory
        os.chdir(env.root)
        # Write env and flp files
        env.write_env()
        flp.write_flp()
        if (
            env.range_dependent_env and env.broadband_run
        ):  # Run broadband range dependent simulation

            # Run parallel
            if self.parallel:

                # Clear working dirs
                if self.clear:
                    self.clear_kraken_parallel_working_dir(root=env.root)

                if self.n_workers is None:
                    self.n_workers = min(len(frequencies), N_CORES)
                else:
                    self.n_workers = min(len(frequencies), self.n_workers)

                # Get optimal frequencies intervals bounds
                frequencies_intervalls, nb_used_workers = (
                    self.assign_frequency_intervalls(
                        frequencies, self.n_workers, mode="optimal"
                    )
                )
                self.n_workers = nb_used_workers

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
                # pool = multiprocessing.Pool(processes=self.n_workers)

                # # Run parallel processes
                # result = pool.starmap(
                #     self.runkraken_broadband_range_dependent, param_pool, chunksize=1
                # )
                # field_pos = result[0][1]
                # pressure_field = np.concatenate([r[0] for r in result], axis=0)
                # # Close pool
                # pool.close()
                # # Wait for all processes to finish
                # pool.join()

                with multiprocessing.Pool(processes=self.n_workers) as pool:
                    result = pool.starmap(self.runkraken_broadband_range_dependent, param_pool, chunksize=1)
                    field_pos = result[0][1]
                    pressure_field = np.concatenate([r[0] for r in result], axis=0)


                # cpu_time = time.time() - t0
                # print(f"CPU time (Map): {cpu_time:.2f} s")

            else:
                pressure_field, field_pos = self.runkraken_broadband_range_dependent(
                    env=env, flp=flp, frequencies=frequencies
                )

            if self.verbose:
                print("Broadband range dependent kraken simulation completed.")

            return pressure_field, field_pos

        else:  # Run range independent simulation (no parallelization for now)

            # Run Fortran version of Kraken
            self.run_kraken_exec(env.filename)
            # Run Fortran version of Field
            self.run_field_exec(env.filename)

            # Read pressure field for the current frequency
            _, _, _, _, read_freq, _, field_pos, pressure_field = readshd(
                filename=env.filename + ".shd", freq=frequencies
            )
            if self.verbose and not env.range_dependent_env and env.broadband_run:
                print("Broadband range independent kraken simulation completed.")
            elif self.verbose and env.range_dependent_env and not env.broadband_run:
                print("Single frequency range dependent kraken simulation completed.")
            elif self.verbose and not env.range_dependent_env and not env.broadband_run:
                print("Single frequency range independent kraken simulation completed.")

            return pressure_field, field_pos

    @classmethod
    def assign_frequency_intervalls(
        cls, frequencies, n_workers, mode="equally_distributed"
    ):
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
                frequencies[
                    slice(i * nf // n_workers, min((i + 1) * nf // n_workers, nf))
                ]
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
                assigned_frequency_ranges, _ = cls.assign_frequency_intervalls(
                    frequencies=frequencies,
                    n_workers=n_workers,
                    mode="equally_distributed",
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

    @classmethod
    def run_field_exec(cls, filename, parallel=False, worker_pid=None, silent=False):
        """
        Wrapper for field executable.

        :param filename: (str) Flp filename without the .flp extension
        :param parallel: (bool) Run in parallel mode (default: False)
        :param worker_pid: (int) Worker process id
        :param silent: (bool) Silent mode (default: True)
        :return:

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
        """
        Wrapper for kraken executable.

        :param filename: (str) Env filename without the .env extension
        :param parallel: (bool) Run in parallel mode (default: False
        :param worker_pid: (int) Worker process id
        :param silent: (bool) Silent mode (default: True)
        :return:

        """
        cls.run_exec(
            exec="kraken",
            filename=filename,
            parallel=parallel,
            worker_pid=worker_pid,
            silent=silent,
        )

    @staticmethod
    def run_exec(exec, filename, parallel, worker_pid, silent):
        """
        Run executable (field / kraken).

        :param exec: (str) Executable name
        :param filename: (str) Env filename without the .env extension
        :param parallel: (bool) Run in parallel mode
        :param worker_pid: (int) Worker process id
        :param silent: (bool) Silent mode
        :return:

        """
        if os.name == "nt":
            ext = ".exe"
            silent_redirection = " >NUL 2>&1"
        else:
            ext = ""
            silent_redirection = " >/dev/null 2>&1"

        if parallel and (os.name == "nt"):
            if worker_pid is not None:
                parallel_working_dir = os.getcwd()
                subprocess_working_dir = os.path.join(parallel_working_dir, "bin")
                cmd = os.path.join(subprocess_working_dir, exec)
                # cmd = f"{fpath_to_exec} {filename}"
            else:
                raise ValueError(
                    f"worker_pid must be specified with parallel set to True."
                )
        else:
            cmd = exec

        cmd += ext
        to_ex = f"{cmd} {filename}"
        if silent:
            # Silent to avoid warning
            to_ex += silent_redirection

        # Run Fortran version of Kraken
        os.system(to_ex)

    @staticmethod
    def clear_kraken_parallel_working_dir(root):
        """
        Clear working directories.
        """
        root_parallel_folder = "parallel_working_dir"
        dir = os.path.join(root, root_parallel_folder)
        for root, dirs, files in os.walk(dir):
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    @staticmethod
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

    @classmethod
    def init_parallel_kraken_working_dirs(cls, env, env_root, worker_pid):
        """
        Initialise working directory to be used by child processes for multiprocessing.

        :param root:
        :param worker_pid:
        :return:
        """

        subprocess_working_dir = cls.get_subprocess_working_dir(env_root, worker_pid)
        env.root = subprocess_working_dir

        if os.name == "nt":  # Windows

            # Create bin folder
            bin_folder = os.path.join(subprocess_working_dir, "bin")
            if not os.path.exists(bin_folder):
                os.makedirs(bin_folder)

            # Copy bin files to subprocess working directory
            # (that's ugly but it works... calling kraken.exe and field.exe simultaneously from different process fails on Windows OS)
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

    @classmethod
    def runkraken_broadband_range_dependent(
        cls,
        env,
        flp,
        frequencies,
        parallel=False,
    ):
        """KRAKEN can run broadband simulations with range dependent environments yet it seems version of field we are using is not capable of
        computing the broadband and range-independent pressure field.
        This function is a workaround to this issue. It runs KRAKEN with a range dependent environment
        for each frequency of the broadband simulation and then merge the results in a single pressure field.
        """
        # Root dir to share with subprocesses
        worker_pid = os.getpid()
        env_root = env.root

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
                nmedia=env.nmedia,
            )

            cls.init_parallel_kraken_working_dirs(env, env_root, worker_pid)
            os.chdir(env.root)

            # Update flp_fpath with subprocess working directory
            flp.flp_fpath = env.flp_fpath

            # Write environment
            env.write_env()

            if ifreq == 0:
                # Write .flp file for the first frequency (independent from frequency)
                flp.write_flp()

            # Run Fortran version of Kraken
            cls.run_kraken_exec(env.filename, parallel, worker_pid)
            # Run Fortran version of Field
            try:
                cls.run_field_exec(env.filename, parallel, worker_pid)

                # Read pressure field for the current frequency
                _, _, _, _, read_freq, _, field_pos, pressure = readshd(
                    filename=env.filename + ".shd", freq=frequencies[ifreq]
                )
            except:
                print("Error running field executable.")

            # Initialize broadband pressure field array
            if ifreq == 0:
                broadband_shape = (len(frequencies),) + pressure.shape
                broadband_pressure_field = np.zeros(broadband_shape, dtype=complex)

            broadband_pressure_field[ifreq, ...] = pressure

        # time.sleep(worker_pid / 10000)
        return broadband_pressure_field, field_pos

    @classmethod
    def readshd(cls, filename, xs=None, ys=None, freq=None):
        """Read a shade file produce by FIELD.exe and return the data in a dictionary.

        Usage : PlotTitle, PlotType, freqVec, freq0, read_freq, atten, Pos, pressure = read_shd(filename, xs, ys, freq)

        Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter https://oalib.hlsresearch.com/AcousticsToolbox/

        """

        if freq is None:
            if xs is None:
                (
                    PlotTitle,
                    PlotType,
                    freqVec,
                    freq0,
                    read_freq,
                    atten,
                    Pos,
                    pressure,
                ) = cls.readshd_bin(filename=filename)
            else:
                (
                    PlotTitle,
                    PlotType,
                    freqVec,
                    freq0,
                    read_freq,
                    atten,
                    Pos,
                    pressure,
                ) = cls.readshd_bin(filename=filename, xs=xs, ys=ys)
        else:
            (
                PlotTitle,
                PlotType,
                freqVec,
                freq0,
                read_freq,
                atten,
                Pos,
                pressure,
            ) = cls.readshd_bin(filename=filename, freq=freq)

        # else:
        #     raise ValueError("Unrecognized file extension")

        return PlotTitle, PlotType, freqVec, freq0, read_freq, atten, Pos, pressure

    def readshd_bin(filename, xs=None, ys=None, freq=None):
        """Read a '.shd' binary file.

        Adapted from the original Matlab Acoustics Toolbox by Michael B. Porter https://oalib.hlsresearch.com/AcousticsToolbox/

        """

        try:
            fid = open(filename, "rb")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"readshd_bin.py: No shade file with the name {filename} exists"
            )

        recl = int(np.fromfile(fid, dtype=np.int32, count=1))  # record length in bytes
        title = fid.read(80).decode("utf-8").strip()  # read and decode the title

        fid.seek(4 * recl)  # reposition to end of first record
        PlotType = fid.read(10).decode("utf-8").strip()  # read and decode the PlotType

        fid.seek(2 * 4 * recl)  # reposition to end of second record
        Nfreq = int(np.fromfile(fid, dtype=np.int32, count=1))
        Ntheta = int(np.fromfile(fid, dtype=np.int32, count=1))
        Nsx = int(np.fromfile(fid, dtype=np.int32, count=1))
        Nsy = int(np.fromfile(fid, dtype=np.int32, count=1))
        Nsz = int(np.fromfile(fid, dtype=np.int32, count=1))
        Nrz = int(np.fromfile(fid, dtype=np.int32, count=1))
        Nrr = int(np.fromfile(fid, dtype=np.int32, count=1))
        freq0 = float(np.fromfile(fid, dtype=np.float64, count=1))
        atten = float(np.fromfile(fid, dtype=np.float64, count=1))

        fid.seek(3 * 4 * recl)  # reposition to end of record 3
        freqVec = np.fromfile(fid, dtype=np.float64, count=Nfreq)

        fid.seek(4 * 4 * recl)  # reposition to end of record 4
        Pos = {}
        Pos["theta"] = np.fromfile(fid, dtype=np.float64, count=Ntheta)

        if PlotType.strip() != "TL":
            fid.seek(5 * 4 * recl)  # reposition to end of record 5
            Pos["s"] = {}
            Pos["s"]["x"] = np.fromfile(fid, dtype=np.float64, count=Nsx)

            fid.seek(6 * 4 * recl)  # reposition to end of record 6
            Pos["s"]["y"] = np.fromfile(fid, dtype=np.float64, count=Nsy)
        else:
            fid.seek(5 * 4 * recl)  # reposition to end of record 5
            Pos["s"] = {}
            Pos["s"]["x"] = np.fromfile(fid, dtype=np.float64, count=2)
            Pos["s"]["x"] = np.linspace(Pos["s"]["x"][0], Pos["s"]["x"][1], Nsx)

            fid.seek(6 * 4 * recl)  # reposition to end of record 6
            Pos["s"]["y"] = np.fromfile(fid, dtype=np.float64, count=2)
            Pos["s"]["y"] = np.linspace(Pos["s"]["y"][0], Pos["s"]["y"][1], Nsy)

        fid.seek(7 * 4 * recl)  # reposition to end of record 7
        Pos["s"]["z"] = np.fromfile(fid, dtype=np.float32, count=Nsz)

        fid.seek(8 * 4 * recl)  # reposition to end of record 8
        Pos["r"] = {}
        Pos["r"]["z"] = np.fromfile(fid, dtype=np.float32, count=Nrz)

        fid.seek(9 * 4 * recl)  # reposition to end of record 9
        Pos["r"]["r"] = np.fromfile(fid, dtype=np.float64, count=Nrr)

        if PlotType == "rectilin  ":
            Nrcvrs_per_range = Nrz
        elif PlotType == "irregular ":
            Nrcvrs_per_range = 1
        else:
            Nrcvrs_per_range = Nrz

        if freq is None:
            nread_freq = 1
        else:
            freq = np.array(freq)  # Ensure freq is a np array
            freq = np.reshape(
                freq, (freq.size,)
            )  # Ensure freq as one dimension (to avoid issue when freq is given as a scalar)
            nread_freq = freq.size

        pressure = np.zeros(
            (nread_freq, Ntheta, Nsz, Nrcvrs_per_range, Nrr), dtype=complex
        )

        if xs is None or ys is None:
            if freq is not None:
                # Old version (prior 15/12/2023)
                # freqdiff = np.abs(freqVec - freq)
                # freq_idx = np.argmin(freqdiff)

                # Updated version to handle multiple frequencies reading at the same time
                freq_idx = np.array([np.argmin(np.abs(freqVec - f)) for f in freq])
            else:
                freq_idx = np.array([0])
            read_freq = freqVec[freq_idx]

            for idx_f_pressure, ifreq in enumerate(freq_idx):
                for itheta in range(Ntheta):
                    for isz in range(Nsz):
                        for irz in range(Nrcvrs_per_range):
                            recnum = (
                                10
                                + ifreq * Ntheta * Nsz * Nrcvrs_per_range
                                + itheta * Nsz * Nrcvrs_per_range
                                + isz * Nrcvrs_per_range
                                + irz
                            )

                            status = fid.seek(recnum * 4 * recl)
                            if status == -1:
                                raise ValueError(
                                    "Seek to specified record failed in readshd_bin"
                                )

                            temp = np.fromfile(fid, dtype=np.float32, count=2 * Nrr)
                            pressure[idx_f_pressure, itheta, isz, irz, :] = (
                                temp[0::2] + 1j * temp[1::2]
                            )
            # Get rid of the useless first dimension in case of single frequency (mainly for coherence with other functions like plotshd ...)
            if nread_freq == 1:
                pressure = pressure[0, ...]

        else:
            # Note : this part of the function is inherited from the MATLAB function from AT and might not work anymore
            # TODO : check if this part of the function is still working
            read_freq = None
            xdiff = np.abs(Pos["s"]["x"] - xs * 1000)
            idxX = np.argmin(xdiff)
            ydiff = np.abs(Pos["s"]["y"] - ys * 1000)
            idxY = np.argmin(ydiff)

            for itheta in range(Ntheta):
                for isz in range(Nsz):
                    for irz in range(Nrcvrs_per_range):
                        recnum = (
                            10
                            + idxX * Nsy * Ntheta * Nsz * Nrz
                            + idxY * Ntheta * Nsz * Nrz
                            + itheta * Nsz * Nrz
                            + isz * Nrz
                            + irz
                        )

                        status = fid.seek(recnum * 4 * recl)
                        if status == -1:
                            raise ValueError(
                                "Seek to specified record failed in readshd_bin"
                            )

                        temp = np.fromfile(fid, dtype=np.float32, count=2 * Nrr)
                        pressure[itheta, isz, irz, :] = temp[0::2] + 1j * temp[1::2]

        fid.close()

        return title, PlotType, freqVec, freq0, read_freq, atten, Pos, pressure


if __name__ == "__main__":
    pass
