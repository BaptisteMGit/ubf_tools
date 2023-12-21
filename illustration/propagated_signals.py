import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from localisation.verlinden.AcousticComponent import AcousticSource
from propa.kraken_toolbox.post_process import (
    postprocess_received_signal,
    process_broadband,
)

from signals import pulse, pulse_train, ship_noise

img_path = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\generation_signaux"
)
figsize = (16, 8)


def ideal_waveguide_propa():
    working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\test_synthesis"
    template_env = "CalibSynthesis"

    os.chdir(working_dir)

    T = 1
    fc = 50
    fs = 200
    window = "hanning"

    # Receiver position
    rcv_range = np.array([10000, 13000])
    rcv_depth = [20]
    delays = rcv_range / 1500

    # Need to reprocess kraken for differents frequency content of the source

    delay = list([rcv_range[0] / 1500]) * len(rcv_range)

    for src in ["pulse", "pulse_train", "ship"]:
        if src == "pulse":
            s, t = pulse(T=T, f=fc, fs=fs, t0=0.5 * T)
        elif src == "pulse_train":
            s, t = pulse_train(T=T, f=fc, fs=fs, interpulse_delay=0.1)
        elif src == "ship":
            s, t = ship_noise(T=T)

        source = AcousticSource(s, t, name="source", waveguide_depth=100, window=window)
        process_broadband(fname=template_env, source=source, max_depth=100)

        for apply_delay in [True, False]:
            time_vector, s_at_rcv_pos, Pos = postprocess_received_signal(
                shd_fpath=os.path.join(working_dir, template_env + ".shd"),
                source=source,
                rcv_range=rcv_range,
                rcv_depth=rcv_depth,
                apply_delay=apply_delay,
                delay=delay,
            )

            th = 0.5 * 1e-8
            end_s2 = np.where(np.abs(s_at_rcv_pos[:, 0, 1]) > th)[0].max()

            plt.figure(figsize=figsize)

            plt.plot(time_vector, s_at_rcv_pos[:, 0, 0], color="b", label=r"$s_1(t)$")
            plt.plot(time_vector, s_at_rcv_pos[:, 0, 1], color="r", label=r"$s_2(t)$")

            if not apply_delay:
                fname = f"source_{src}_no_delay.png"
                plt.axvline(delays[0], label=r"$\tau_1$", color="b", linestyle="--")
                plt.axvline(delays[1], label=r"$\tau_2$", color="r", linestyle="--")
                plt.axvline(
                    time_vector[end_s2],
                    label=r"$t_{max}= \tau_{s_1} + \Delta \tau + \tau_{s_2}$",
                    color="k",
                    linestyle="--",
                )

            else:
                fname = f"source_{src}_delay.png"
                plt.axvline(
                    delays[1] - delays[0],
                    label=r"$\Delta \tau = \tau_2 - \tau_1$",
                    color="r",
                    linestyle="--",
                )
                plt.axvline(
                    time_vector[end_s2],
                    label=r"$t_{max}= \Delta \tau + \tau_{s_2}$",
                    color="k",
                    linestyle="--",
                )

            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Received signal")
            plt.tight_layout()
            plt.savefig(os.path.join(img_path, fname))
            # plt.show()


def noisy_signal_verlinden_process(ds_library):
    rcv_sig = ds_library.rcv_signal_library.isel(x=20, y=20)

    plt.figure()
    rcv_sig.sel(idx_obs=0).plot(color="b", label=r"$s_1(t)$")
    rcv_sig.sel(idx_obs=1).plot(color="r", label=r"$s_2(t)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, "noisy_signal_verlinden_process.png"))


if __name__ == "__main__":
    # ideal_waveguide_propa()

    ds_library = xr.open_dataset(
        r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\localisation\verlinden\verlinden_process_populated_library\verlinden_1_test_case\pulse_train\populated_snr0dB.nc"
    )
    noisy_signal_verlinden_process(ds_library)
