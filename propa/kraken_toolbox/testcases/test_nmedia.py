import matplotlib.pyplot as plt

# from propa.kraken_toolbox.plot_utils import plotmode
from propa.kraken_toolbox.read_modes import readmodes


def plot_mode_shapes(mod_fpath, freq, n_modes=4):
    """Plot the real part of the first 'n_modes' mode shapes as a
    function of depth."""
    Modes = readmodes(mod_fpath, freq=freq)
    n_modes = min(n_modes, Modes["M"])

    fig, ax = plt.subplots(figsize=(6, 8))
    for i in range(n_modes):
        ax.plot(Modes["phi"][:, i].real, Modes["z"], label=f"Mode {i + 1}")
        # ax.plot(
        #     Modes["phi"][:, i].imag, Modes["z"], label=f"Mode {i + 1}", linestyle="--"
        # )

    ax.invert_yaxis()
    ax.set_xlabel("Mode amplitude (real part)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Mode shapes at {freq} Hz  (Number of modes = {Modes['M']})")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\examples\case_01_pekeris_range_independent_single_freq\case_01_pekeris_ri_single_freq.mod"
plot_mode_shapes(fpath, 100, n_modes=8)


# fnmedia1 = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\testcases\test_nmedia_1\io_files\test_nmedia_1.mod"
# fnmedia2 = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\testcases\test_nmedia_2\io_files\test_nmedia_2.mod"
# plot_mode_shapes(fnmedia1, 100, n_modes=8)
# plot_mode_shapes(fnmedia2, 100, n_modes=8)

# fnmedia1_bis = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\testcases\test_nmedia_1\io_files\test_nmedia_1_bis.mod"
# plot_mode_shapes(fnmedia1_bis, 100, n_modes=8)


# fcalib_k = r"C:\Users\baptiste.menetrier\Desktop\ressource\AcousticToolbox\at_2023_5_18\at\tests\calib\calibK.mod"
# fcalib_k = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\testcases\test_calibK\test_calibK.mod"
# plot_mode_shapes(fcalib_k, 100, n_modes=8)

# fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\testcases\test_kraken_cpu_time\io_files\test_kraken_cpu_time.mod"
# plot_mode_shapes(fpath, 100, n_modes=8)

plt.show()
