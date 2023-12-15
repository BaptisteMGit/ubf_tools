import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from localisation.verlinden.AcousticComponent import AcousticSource
from propa.kraken_toolbox.plot_utils import plotshd
from propa.kraken_toolbox.post_process import (
    fourier_synthesis_kraken,
    postprocess,
    process_broadband,
)
from propa.kraken_toolbox.utils import runkraken
from signals import pulse


if __name__ == "__main__":
    """ "
    Run test to check the Fourier synthesis method. Test case based on figures presented in Jensen et al. (2000) p.638
    """

    working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\test_synthesis"
    # template_env = "MunkK"
    # template_env = "CalibSynthesis_bis"
    template_env = "CalibSynthesis"

    os.chdir(working_dir)

    # Define the pulse
    fc = 50
    s, t = pulse(T=1, f=fc, fs=200)

    # Boat signal generated with the code from Samuel Pinson
    # data = np.loadtxt(
    #     r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\test_synthesis\sig_samuel.txt",
    #     skiprows=1,
    #     usecols=[0, 1],
    #     dtype=np.float64,
    #     delimiter=",",
    # )

    # tmax = 10 * 60
    # t, s = data[:, 0], data[:, 1]
    # dt = t[1] - t[0]
    # T = 0.10 * 60
    # nt = int(T / dt)
    # idx_cpa = t.size // 2
    # t = t[idx_cpa - nt // 2 : idx_cpa + nt // 2]
    # t -= t.min()
    # s = s[idx_cpa - nt // 2 : idx_cpa + nt // 2]
    source = AcousticSource(s, t)
    print(source.spectrum_df)

    print(f"Signal duration {source.time[-1]} s")
    print(f"Signal sampling frequency {source.fs} Hz")
    print(f"Signal number of samples {source.ns}")
    print(f"deltaf = fs / nfft = {source.fs / source.nfft} Hz")
    print(f"Signal spectrum df {source.spectrum_df} Hz")
    print(f"df = 1 / T = {1 / source.time[-1]} Hz")
    print(f"T_tot = 1 / df = {1 / source.spectrum_df} s")

    # source.display_source()
    # plt.show()
    # source.display_source()
    # harmonic = [5, 10, 15, 20, 25, 30]
    # f_vrec = []
    # for h in harmonic:
    #     sub_freq_vec = [h - 2, h - 1, h, h + 2, h + 1]
    #     f_vrec += sub_freq_vec
    # source.kraken_freq = np.array(f_vrec)

    # source.kraken_freq = np.arange(3, 30, 0.5)
    # plt.show()

    # Receiver position
    rcv_range = np.array([5000, 5500, 10000, 15000])
    rcv_depth = [20]

    # process_broadband(fname=template_env, source=source, max_depth=100)

    delay = list([rcv_range[0] / 1500]) * len(rcv_range)

    # runkraken(template_env)

    time_vector, s_at_rcv_pos, Pos = postprocess(
        shd_fpath=os.path.join(working_dir, template_env + ".shd"),
        source=source,
        rcv_range=rcv_range,
        rcv_depth=rcv_depth,
        apply_delay=False,
        delay=delay,
    )

    # rcv = AcousticSource(s_at_rcv_pos[:, 0, 0], time_vector)
    # rcv.display_source()

    # # Plot received signal
    fig, ax = plt.subplots(1 + len(rcv_range), 1, sharex=True, sharey=True)
    # source.plot_signal(ax=ax[0])
    # ax[0].set_ylabel("")

    for ir, rcv_r in enumerate(rcv_range):
        ax[ir + 1].plot(time_vector, s_at_rcv_pos[:, 0, ir])

        # ax[ir + 1].axvline()
        # ax[ir + 1].set_xlabel("Time (s)")
        # ax[ir + 1].set_ylabel("Pressure (Pa)")
        # ax[ir + 1].set_title("Transmitted signal - r = {} m".format(rcv_r))
    plt.tight_layout()
    fig.supxlabel("Reduced time  t - r/c0 (s)")
    fig.supylabel("Pressure (Pa)")

    src_energy = np.sum(np.abs(source.signal) ** 2)
    rcv_energy = np.sum(np.abs(s_at_rcv_pos[:, 0, 0]) ** 2)
    print(f"Energy ratio: {rcv_energy/src_energy}")
    plt.show()

    # # Plot spectrogram of the received signal
    # plt.figure()
    # f, t, Sxx = signal.spectrogram(
    #     s_at_rcv_pos, source.fs, nperseg=50, noverlap=25, nfft=1024
    # )
    # plt.pcolormesh(t, f, Sxx)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")

    # # Plot pressure field
    # list_r_idx = [np.argmin(np.abs(Pos["r"]["r"] - r)) for r in rcv_range]
    # plotshd(template_env + ".shd", 150)
    # plt.scatter(
    #     Pos["r"]["r"][list_r_idx],
    #     [5] * len(rcv_range),
    #     marker="+",
    #     color="red",
    #     s=5,
    #     label="Receiver position",
    # )
    # plt.legend()

    # plt.show()

    # Doppler

    # # Define source signal
    # f0 = 50
    # fs = 20 * f0
    # T = 30
    # t = np.arange(0, T, 1 / fs)
    # s = np.sin(2 * np.pi * f0 * t)
    # dt = t[1] - t[0]
    # source = AcousticSource(s, t)

    # source.display_source()
    # source.kraken_freq = np.array([f0])
    # plt.show()

    # # Define ship trajecory
    # x_ship_begin = -25000
    # y_ship_begin = 1000
    # x_ship_end = 25000
    # y_ship_end = 0

    # z_ship = 5
    # source.z_src = z_ship

    # v_ship = 50 / 3.6

    # Dtot = np.sqrt((x_ship_begin - x_ship_end) ** 2 + (y_ship_begin - y_ship_end) ** 2)

    # vx = v_ship * (x_ship_end - x_ship_begin) / Dtot
    # vy = v_ship * (y_ship_end - y_ship_begin) / Dtot

    # Ttot = Dtot / v_ship + 3
    # # t_ship = np.arange(0, Ttot - 1 / source.fs, 1 / source.fs)
    # t_ship = np.arange(0, Ttot - T, T)

    # x_ship_t = x_ship_begin + vx * t_ship
    # y_ship_t = y_ship_begin + vy * t_ship

    # # OBS position
    # x_obs = -1000
    # y_obs = 0

    # r_ship_t = np.sqrt((x_ship_t - x_obs) ** 2 + (y_ship_t - y_obs) ** 2)

    # # process_broadband(fname=template_env, source=source, max_depth=1000)

    # # Plot pressure field
    # plotshd(template_env + ".shd", 30)
    # plt.scatter(
    #     r_ship_t,
    #     np.ones(r_ship_t.shape) * z_ship,
    #     marker=">",
    #     color="red",
    #     s=20,
    #     label="Receiver positions",
    # )
    # plt.legend()

    # plt.figure()
    # plt.plot(t_ship, r_ship_t)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ship distance from receiver")

    # # Receiver position
    # rcv_range = r_ship_t
    # rcv_depth = [source.z_src]

    # # Re run kraken
    # # runkraken(template_env)

    # time_vector, s_at_rcv_pos, Pos = postprocess(
    #     fname=template_env, source=source, rcv_range=rcv_range, rcv_depth=rcv_depth
    # )

    # # Stack signals
    # full_t = time_vector
    # full_s = s_at_rcv_pos[..., 0]
    # for ir in range(1, len(rcv_range)):
    #     new_t = time_vector + max(full_t) + dt
    #     new_s = s_at_rcv_pos[..., ir]
    #     full_t = np.append(full_t, new_t)
    #     full_s = np.append(full_s, new_s)

    # # Plot full signal
    # plt.figure()
    # plt.plot(full_t, full_s)
    # plt.xlabel("Time (s)")

    # # Plot spectrogram of the received signal
    # nperseg = 512 * 2
    # overlap_window = 3 / 4
    # noverlap = int(nperseg * overlap_window)

    # f, t, Sxx = signal.spectrogram(
    #     full_s, source.fs, nperseg=nperseg, noverlap=noverlap, window="hamming"
    # )

    # plt.figure()
    # plt.pcolormesh(t, f, 20 * np.log10(np.abs(Sxx)))
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")

    # plt.show()
