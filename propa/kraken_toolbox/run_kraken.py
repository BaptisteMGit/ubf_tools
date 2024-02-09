import os
import numpy as np

from tqdm import tqdm
from cst import BAR_FORMAT
from propa.kraken_toolbox.kraken_env import KrakenEnv, KrakenFlp
from propa.kraken_toolbox.read_shd import readshd


def runkraken(env, flp, frequencies, verbose=False):
    if verbose:
        print("Running Kraken...")

    # Change directory
    os.chdir(env.root)

    # Write env and flp files
    env.write_env()
    flp.write_flp()

    if (
        env.range_dependent_env and env.broadband_run
    ):  # Run broadband range dependent simulation
        pressure_field = runkraken_broadband_range_dependent(
            env=env, flp=flp, frequencies=frequencies
        )
        if verbose:
            print("Broadband range dependent kraken simulation completed.")
        return pressure_field

    else:  # Run range independent simulation
        # Run Fortran version of Kraken
        os.system(f"kraken {env.filename}")
        # Run Fortran version of Field
        os.system(f"field {env.filename}")

        # Read pressure field for the current frequency
        _, _, _, _, read_freq, _, _, pressure_field = readshd(
            filename=env.filename + ".shd", freq=frequencies
        )
        if verbose and not env.range_dependent_env and env.broadband_run:
            print("Broadband range independent kraken simulation completed.")
        else:
            print("Single frequency range dependent kraken simulation completed.")

        return pressure_field


def runfield(filename):
    # Run Fortran version of Field
    os.system(f"field {filename}")


def runkraken_broadband_range_dependent(env, flp, frequencies):
    """KRAKEN is capable of running broadband simulations with range independent environments
    and single frequency simulations with range dependent environments. Yet, for some reason,
    it is not capable of running broadband simulations with range dependent environments.
    This function is a workaround to this issue. It runs KRAKEN with a range dependent environment
    for each frequency of the broadband simulation and then merge the results in a single pressure field.
    """

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

        # Write environment
        env.write_env()

        if ifreq == 0:
            # Write .flp file for the first frequency (independent from frequency)
            flp.write_flp()

        # Run Fortran version of Kraken
        os.system(f"kraken {env.filename}")
        # Run Fortran version of Field
        os.system(f"field {env.filename}")

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
