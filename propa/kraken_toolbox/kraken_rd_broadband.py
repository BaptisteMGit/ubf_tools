import numpy as np

from tqdm import tqdm
from cst import BAR_FORMAT
from propa.kraken_toolbox.kraken_env import KrakenEnv, KrakenFlp
from propa.kraken_toolbox.read_shd import readshd
from propa.kraken_toolbox.utils import runkraken


def runkraken_broadband_range_dependent(range_dependent_env, flp, frequencies):
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
            title=range_dependent_env.simulation_title,
            env_root=range_dependent_env.root,
            env_filename=range_dependent_env.filename,
            freq=frequencies[ifreq],
            kraken_top_hs=range_dependent_env.top_hs,
            kraken_medium=range_dependent_env.medium,
            kraken_attenuation=range_dependent_env.att,
            kraken_bottom_hs=range_dependent_env.bottom_hs,
            kraken_field=range_dependent_env.field,
            kraken_bathy=range_dependent_env.bathy,
            rModes=range_dependent_env.modes_range,
        )

        # Write environment
        env.write_env()

        if ifreq == 0:
            # Write .flp file for the first frequency (independent from frequency)
            flp.write_flp()

        # Run KRAKEN
        runkraken(env.filename)

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
