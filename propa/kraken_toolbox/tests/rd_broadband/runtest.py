import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cst import SAND_PROPERTIES, TICKS_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE
from propa.kraken_toolbox.kraken_rd_broadband import runkraken_broadband_range_dependent
from propa.kraken_toolbox.plot_utils import plotshd_from_pressure_field
from propa.kraken_toolbox.kraken_env import (
    KrakenTopHalfspace,
    KrakenMedium,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
    KrakenEnv,
    Bathymetry,
)

from propa.kraken_toolbox.post_process import postprocess_received_signal
from signals import pulse
from localisation.verlinden.AcousticComponent import AcousticSource


def define_test_env():
    top_hs = KrakenTopHalfspace()

    ssp_data = pd.read_csv("ssp_data.csv", sep=",", header=None)
    z_ssp = ssp_data[0].values
    cp_ssp = ssp_data[1].values
    medium = KrakenMedium(
        ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp, nmesh=5000
    )

    bott_hs_properties = {
        "rho": 1.5,
        "c_p": 1506.50,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
        "a_p": 0.5,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
    }  # stepK properties

    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(
        halfspace_properties=bott_hs_properties,
    )

    att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)
    field = KrakenField(
        src_depth=18,
        phase_speed_limits=[1400, 2000],
        n_rcv_z=5001,
        rcv_z_max=3000,
    )

    # Range dependent bathymetry
    bathy = Bathymetry(
        data_file=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd_broadband\bathy_data.csv",
        interpolation_method="linear",
        units="km",
    )

    env_filename = "test_kraken_rd_broadband"

    return top_hs, medium, att, bott_hs, field, bathy, env_filename


def test():
    top_hs, medium, att, bott_hs, field, bathy, env_filename = define_test_env()

    env = KrakenEnv(
        title="Test de la classe KrakenEnv",
        env_root=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd_broadband",
        env_filename=env_filename,
        freq=100,
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
        kraken_bathy=bathy,
        rModes=np.arange(1, 30, 0.1),
    )

    freqs = np.arange(20, 200, 40)
    src_depth = 18
    rcv_r_max = 30
    rcv_z_max = 3000

    broadband_pressure_field = runkraken_broadband_range_dependent(
        range_dependent_env=env,
        frequencies=freqs,
        src_depth=src_depth,
        rcv_r_max=rcv_r_max,
        rcv_z_max=rcv_z_max,
    )
    for ifreq in range(len(freqs)):
        pressure_field_to_plot = broadband_pressure_field[ifreq, ...]
        plotshd_from_pressure_field(
            env_filename + ".shd",
            pressure_field=pressure_field_to_plot,
            tl_min=60,
            tl_max=110,
            title=f"Step K - f = {freqs[ifreq]} Hz",
            freq=freqs[ifreq],
        )
    plt.show()

    # time_vector, s_at_rcv_pos, Pos = postprocess_received_signal(
    #     shd_fpath=os.path.join(working_dir, env_filename + ".shd"),
    #     source=source,
    #     rcv_range=rcv_range,
    #     rcv_depth=rcv_depth,
    #     apply_delay=True,
    # )

    # pd.DataFrame({"time": time_vector, "s_received": s_at_rcv_pos}).to_csv(
    #     f"propagated_signal_dr_{dr}.csv", index=False
    # )

    # plt.figure()
    # plt.plot(time_vector, s_at_rcv_pos, "o-")
    # plt.xlabel("Time [s]", fontsize=LABEL_FONTSIZE)
    # plt.ylabel("Pressure [Pa]", fontsize=LABEL_FONTSIZE)
    # plt.xticks(fontsize=TICKS_FONTSIZE)
    # plt.yticks(fontsize=TICKS_FONTSIZE)
    # plt.title(
    #     f"Received signal at {rcv_range} m - dr{dr*1e3}m", fontsize=TITLE_FONTSIZE
    # )


if __name__ == "__main__":
    working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd_broadband"
    os.chdir(working_dir)

    test()
