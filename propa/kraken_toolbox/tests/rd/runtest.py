import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cst import SAND_PROPERTIES, TICKS_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE
from propa.kraken_toolbox.run_kraken import run_kraken_exec
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

from propa.kraken_toolbox.post_process import postprocess_received_signal
from signals import pulse
from localisation.verlinden.misc.AcousticComponent import AcousticSource


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
        "c_p": 1600.0,  # P-wave celerity (m/s)
        "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
        "a_p": 0.5,  # Compression wave attenuation (dB/wavelength)
        "a_s": 0.0,  # Shear wave attenuation (dB/wavelength)
    }  # stepK properties

    bott_hs_properties["z"] = z_ssp.max()
    bott_hs = KrakenBottomHalfspace(
        halfspace_properties=bott_hs_properties,
    )

    att = KrakenAttenuation(units="dB_per_wavelength", use_volume_attenuation=False)

    # Range dependent bathymetry
    bathy = Bathymetry(
        data_file=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd\bathy_data.csv",
        interpolation_method="linear",
        units="km",
    )

    field = KrakenField(
        src_depth=18,
        phase_speed_limits=[0, 20000],
        n_rcv_z=5001,
        rcv_z_max=bathy.bathy_depth.max(),
    )

    env_filename = "test_kraken_rd"

    return top_hs, medium, att, bott_hs, field, bathy, env_filename


def range_dependent_test():
    """
    Test with range dependent bathymetry
    Test case configuration is inspired from stepK test in the Acoustics Toolbox
    """

    top_hs, medium, att, bott_hs, field, bathy, env_filename = define_test_env()

    env = KrakenEnv(
        title="Test de la classe KrakenEnv",
        env_root=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd",
        env_filename=env_filename,
        freq=[100],
        kraken_top_hs=top_hs,
        kraken_medium=medium,
        kraken_attenuation=att,
        kraken_bottom_hs=bott_hs,
        kraken_field=field,
        kraken_bathy=bathy,
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

    run_kraken_exec(env_filename)
    plotshd(env_filename + ".shd", title="Step K", tl_max=110, tl_min=60, bathy=bathy)

    PlotTitle, _, _, _, read_freq, _, field_pos, pressure = readshd(
        filename=env_filename + ".shd",
    )

    pressure = np.squeeze(pressure, axis=(0, 1))

    rcv_range = field_pos["r"]["r"]
    rcv_depth = [20]
    rcv_pos_idx_r = [
        np.argmin(np.abs(field_pos["r"]["r"] - rcv_r)) for rcv_r in rcv_range
    ]
    rcv_pos_idx_z = [
        np.argmin(np.abs(field_pos["r"]["z"] - rcv_z)) for rcv_z in rcv_depth
    ]
    rr, zz = np.meshgrid(rcv_pos_idx_r, rcv_pos_idx_z)
    tl = pressure[zz, rr].flatten()
    tlt = np.abs(tl).astype(float)
    # Remove infinities and nan values
    tlt[np.isnan(tlt)] = 1e-6
    tlt[np.isinf(tlt)] = 1e-6

    values_counting = tlt > 1e-37
    tlt[~values_counting] = 1e-37
    tlt = -20.0 * np.log10(tlt)

    pd.DataFrame({"range": rcv_range, "tl": tlt}).to_csv(
        "tl_along_range_semiflat_5000.csv"
    )

    plt.show()


def cpu_time_test():
    cpu_time = []
    dr_list = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10]
    # ndr
    for dr in dr_list:
        t0 = time.time()
        top_hs, medium, att, bott_hs, field, bathy, env_filename = define_test_env()

        env = KrakenEnv(
            title="Test de la classe KrakenEnv",
            env_root=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd",
            env_filename=env_filename,
            freq=250,
            kraken_top_hs=top_hs,
            kraken_medium=medium,
            kraken_attenuation=att,
            kraken_bottom_hs=bott_hs,
            kraken_field=field,
            kraken_bathy=bathy,
            rModes=np.arange(1, bathy.bathy_range.max(), dr),
        )

        env.write_env()
        flp = KrakenFlp(
            env=env,
            src_depth=18,
            mode_theory="coupled",
            rcv_r_max=bathy.bathy_range.max(),
            rcv_z_max=bathy.bathy_depth.max(),
        )
        flp.write_flp()

        run_kraken_exec(env_filename)

        delta_t = time.time() - t0
        cpu_time.append(delta_t)

    pd.DataFrame({"dr": dr_list, "cpu_time": cpu_time}).to_csv(
        "cpu_time_vs_dr.csv", index=False
    )

    plt.figure()
    plt.plot(dr_list, cpu_time, "o-")
    plt.xlabel("dr [km]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("CPU time [s]", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.title("CPU time vs dr", fontsize=TITLE_FONTSIZE)
    plt.savefig("cpu_time_vs_dr.png", dpi=300)
    plt.show()


def postprocess_accuracy_test():
    # dr_list = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10]
    dr_list = [10, 5]

    rcv_range = np.array([30000])
    # rcv_range = np.arange(1000, 30000, 10)
    rcv_depth = [20]

    fc = 50
    T = 10
    s, t = pulse(T=1, f=fc, fs=200)

    window = np.hanning(s.size)
    s *= window
    source = AcousticSource(s, t)

    for dr in dr_list:
        top_hs, medium, att, bott_hs, field, bathy, env_filename = define_test_env()

        env = KrakenEnv(
            title="Test de la classe KrakenEnv",
            env_root=r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd",
            env_filename=env_filename,
            freq=source.kraken_freq[0:2],
            kraken_top_hs=top_hs,
            kraken_medium=medium,
            kraken_attenuation=att,
            kraken_bottom_hs=bott_hs,
            kraken_field=field,
            kraken_bathy=bathy,
            rModes=np.arange(1, 30, dr),
        )

        # env.write_env()
        flp = KrakenFlp(
            env=env, src_depth=18, mode_theory="coupled", rcv_r_max=30, rcv_z_max=3000
        )
        flp.write_flp()

        run_kraken_exec(env_filename)

        plotshd(env_filename + ".shd", tl_min=60, tl_max=110, title="Step K", freq=20)
        plt.show()

        time_vector, s_at_rcv_pos, Pos = postprocess_received_signal(
            shd_fpath=os.path.join(working_dir, env_filename + ".shd"),
            source=source,
            rcv_range=rcv_range,
            rcv_depth=rcv_depth,
            apply_delay=True,
        )

        pd.DataFrame({"time": time_vector, "s_received": s_at_rcv_pos}).to_csv(
            f"propagated_signal_dr_{dr}.csv", index=False
        )

        plt.figure()
        plt.plot(time_vector, s_at_rcv_pos, "o-")
        plt.xlabel("Time [s]", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Pressure [Pa]", fontsize=LABEL_FONTSIZE)
        plt.xticks(fontsize=TICKS_FONTSIZE)
        plt.yticks(fontsize=TICKS_FONTSIZE)
        plt.title(
            f"Received signal at {rcv_range} m - dr{dr*1e3}m", fontsize=TITLE_FONTSIZE
        )

        # plt.savefig("rc.png", dpi=300)
        plt.show()


def analyse_cpu_test():
    data = pd.read_csv("cpu_time_vs_dr.csv", sep=",", header=0)
    cpu_t = data["cpu_time"].values
    dr = data["dr"].values
    ndr = [np.arange(1, 30, d_r).size for d_r in dr]

    plt.figure()
    plt.plot(np.log(ndr), cpu_t, "o-")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("ndr", fontsize=LABEL_FONTSIZE)
    plt.ylabel("CPU time [s]", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.title("CPU time vs ndr", fontsize=TITLE_FONTSIZE)
    plt.savefig("cpu_time_vs_ndr.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    working_dir = (
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd"
    )
    os.chdir(working_dir)
    # postprocess_accuracy_test()
    range_dependent_test()

    # cpu_time_test()
    # analyse_cpu_test()
