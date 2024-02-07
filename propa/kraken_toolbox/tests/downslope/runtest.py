""" Run kraken with range dependent downslope environment. 
Defining a single media with an semi-infinite halfspace bottom leads to crash. 
To fix it we need to define a 2 layer environement with the seconde layer being the sediment. """

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from propa.kraken_toolbox.run_kraken import runkraken
from propa.kraken_toolbox.read_shd import readshd
from propa.kraken_toolbox.plot_utils import plotshd
from propa.kraken_toolbox.kraken_env import (
    KrakenTopHalfspace,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
    KrakenEnv,
    KrakenFlp,
    Bathymetry,
)
from propa.kraken_toolbox.utils import default_nb_rcv_z

working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\downslope"
kraken_dir = os.path.join(working_dir, "env_kraken")


def define_test_env(f):
    top_hs = KrakenTopHalfspace()

    ssp_data = pd.read_csv(
        os.path.join(working_dir, "data", "ssp_data.csv"), sep=",", header=None
    )
    z_ssp = ssp_data[0].values
    cp_ssp = ssp_data[1].values

    medium = KrakenMedium(
        ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp, nmesh=5000
    )

    medium.plot_medium()
    plt.savefig(os.path.join(working_dir, "result", "medium.png"))
    plt.close()

    bott_hs_properties = {
        "rho": 1.5,
        "c_p": 1600.0,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
        "a_p": 0.5,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
    }

    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(halfspace_properties=bott_hs_properties, fmin=100)

    att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)

    # Range dependent bathymetry
    bathy = Bathymetry(
        data_file=os.path.join(working_dir, "data", "bathy_data.csv"),
        interpolation_method="linear",
        units="km",
    )

    plt.plot(bathy.bathy_range, bathy.bathy_depth)
    plt.xlabel("Range (km)")
    plt.ylabel("Depth (m)")
    plt.title("Bathymetry")
    plt.savefig(os.path.join(working_dir, "result", "bathy.png"))
    plt.close()

    n_rcv_z = default_nb_rcv_z(f, 3000, n_per_l=15)
    bott_hs.derive_sedim_layer_max_depth(z_max=bathy.bathy_depth.max())

    field = KrakenField(
        src_depth=18,
        phase_speed_limits=[0, 20000],
        n_rcv_z=n_rcv_z,
        rcv_z_max=bott_hs.sedim_layer_max_depth,
        # rcv_z_max=bathy.bathy_depth.max(),
    )

    env_filename = "test_kraken_downslope"

    return (top_hs, medium, att, bott_hs, field, bathy, env_filename)


def range_dependent_test(freq):
    """
    Test with range dependent bathymetry
    Test case configuration is inspired from stepK test in the Acoustics Toolbox
    """

    top_hs, medium, att, bott_hs, field, bathy, env_filename = define_test_env(freq)

    env = KrakenEnv(
        title="Test downslope environment",
        env_root=kraken_dir,
        env_filename=env_filename,
        freq=[freq],
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
        kraken_bathy=bathy,
        rModes=np.arange(1, 10, 1),
    )
    env.write_env()
    flp = KrakenFlp(
        env=env,
        src_depth=18,
        mode_theory="coupled",
        rcv_r_max=10,
        rcv_z_max=3000,
    )
    flp.write_flp()

    runkraken(env, flp, frequencies=[100], verbose=True)
    plotshd(
        env_filename + ".shd",
        title=f"Downslope test - f={freq}Hz",
        bathy=bathy,
    )

    plt.savefig(os.path.join(working_dir, "result", "kraken_downslope.png"))


if __name__ == "__main__":
    range_dependent_test(freq=100)
