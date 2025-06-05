import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from time import time
import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from propa.rtf.rtf_localisation.uace_testcase.src.acoustic_source import ZcallInterferer
from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna
from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.rtf.rtf_localisation.uace_testcase.src.data_builder import DataBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.feature_builder import FeatureBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.localization_processor import (
    LocalizationProcessor,
)
from propa.rtf.rtf_localisation.uace_testcase.src.testcase_builder import (
    DeepWaterRealEnv,
)
from publication.publication_figure import PubFigure


def preprocess_msr_dr(root_wgn):
    # Load msr and dr data for wgn_testcase
    msr = pd.read_csv(os.path.join(root_wgn, "msr_snr_s1_s2_s3_s4_s5_s6.txt"), sep=" ")
    dr = pd.read_csv(
        os.path.join(root_wgn, "dr_pos_snr_s1_s2_s3_s4_s5_s6.txt"), sep=" "
    )

    # Compute mean and std of msr for each snr
    msr_mean = msr.groupby("snr").mean()
    msr_std = msr.groupby("snr").std()

    msr_pd = pd.DataFrame([], index=msr_std.index, columns=["rtf_mean", "rtf_std"])
    msr_pd["rtf_mean"] = msr_mean["d_rtf"].values
    msr_pd["rtf_std"] = msr_std["d_rtf"].values

    # Compute mean and std of position error for each snr
    dr_mean = dr.groupby("snr").mean()
    dr_std = dr.groupby("snr").std()
    dr_pd = pd.DataFrame([], index=msr_std.index, columns=["rtf_mean", "rtf_std"])

    dr_pd["rtf_mean"] = dr_mean["dr_rtf"].values
    dr_pd["rtf_std"] = dr_std["dr_rtf"].values

    rmse_pd = pd.DataFrame([], index=msr_std.index, columns=["rtf"])
    dr["dr_rtf"] = dr["dr_rtf"] ** 2
    mse = dr.groupby("snr").mean()["dr_rtf"]
    rmse_pd["rtf"] = np.sqrt(mse)

    return msr_pd, dr_pd, rmse_pd


def plot_results_figure(mode="demo"):
    """
    Plot the final figure used to present result in the UACE 2025 article
    """

    root_data_publi = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\uace_testcase\data\data_publi_uace2025"
    wgn_tc = "wgn_testcase"
    interf_tc = "interferer_testcase"

    root_wgn = os.path.join(root_data_publi, wgn_tc, mode)
    root_interf = os.path.join(root_data_publi, interf_tc, mode)

    # True target position
    event_ship_x = 30000
    event_ship_y = 23000

    # Zcall interferer position
    if mode == "demo":
        x_abw = 25180
        y_abw = 11860
    else:
        x_abw = 30280
        y_abw = 22860

    # Load msr and rmse
    msr, _, rmse = preprocess_msr_dr(root_wgn)

    print("WGN Testcase Results:")
    print(f"RMSE : {rmse['rtf']}")
    print(f"MSR : {msr['rtf_mean']}")

    if mode == "demo":
        sir = 0
        snr = 5
    else:
        sir = -5  # Signal to Interference Ratio in dB
        snr = 0  # Signal to Noise Ratio in dB

    # Load interferer testcase ambiguity surface data
    fpath = os.path.join(
        root_interf, f"loc_s1_s2_s3_s4_s5_s6_wgn_snr_{snr}dB_z_call_sir_{sir}dB.nc"
    )
    ds_interf = xr.open_dataset(fpath)
    msr_interf, _, rmse_interf = preprocess_msr_dr(root_interf)

    print("Interferer Testcase Results:")
    print(f"RMSE : {rmse_interf['rtf']}")
    print(f"MSR : {msr_interf['rtf_mean']}")

    # Load wgn testcase (only wgn) ambiguity surface
    fpath = os.path.join(root_wgn, f"loc_s1_s2_s3_s4_s5_s6_wgn_snr_{snr}dB.nc")
    ds_wgn = xr.open_dataset(fpath)

    # Build results subplots
    vmin = -5
    vmax = 0
    target_pos_circle_size = 180
    abw_pos_star_size = 400
    target_pos_linewidth = 3
    # pfig = PubFigure(
    #     label_fontsize=20,
    #     ticks_fontsize=18,
    #     labelpad=12,
    #     title_fontsize=16,
    # )
    pfig = PubFigure(
        label_fontsize=32,
        ticks_fontsize=30,
        title_fontsize=32,
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        gridspec_kw={"width_ratios": [2, 1, 1]},
    )

    # Plot perf vs snr in first subplot -> Combined plot
    # Plot RMSE
    rmse_rtf = ax1.plot(
        rmse.index,
        rmse["rtf"],
        "-",
        color="k",
        markersize=3,
    )
    ax1.set_xlabel("SNR [dB]")
    ax1.set_ylabel("RMSE [m]", color="k")
    ax1.tick_params(axis="y", labelcolor="k")

    # Create a second y-axis for MSR
    ax1_bis = ax1.twinx()
    msr_rtf = ax1_bis.plot(
        msr.index,
        -msr.rtf_mean,
        "-",
        color="red",
        markersize=3,
    )

    ax1_bis.set_ylabel("MSR [dB]", color="red")
    ax1_bis.tick_params(axis="y", labelcolor="red")
    ax1.set_title("(a)")

    # Plot ambiguity surface for snr = 5dB without interference
    ds_wgn.d_rtf.plot(
        ax=ax2,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        rasterized=False,
        # cbar_kwargs={"label": r"$\theta [°]$"},
    )
    # Add true target position
    ax2.scatter(
        event_ship_x,
        event_ship_y,
        marker="o",
        facecolors="none",
        s=target_pos_circle_size,
        linewidths=target_pos_linewidth,
        color="k",
    )
    ax2.set_title("(b)")
    ax2.tick_params(direction="out", pad=15)

    # Plot ambiguity surface for snr = 5dB with interference
    ds_interf.d_rtf.plot(
        ax=ax3,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        # cbar_kwargs={"label": r"$\theta [°]$"},
        cbar_kwargs={"label": "[dB]"},
        rasterized=False,
    )
    # Add true target position and interferer position
    ax3.scatter(
        event_ship_x,
        event_ship_y,
        marker="o",
        facecolors="none",
        s=target_pos_circle_size,
        linewidths=target_pos_linewidth,
        color="k",
    )
    ax3.scatter(
        x_abw,
        y_abw,
        marker="*",
        facecolors="w",
        s=abw_pos_star_size,
        linewidths=1,
        color="k",
    )

    ax3.set_ylabel("")
    ax3.set_yticklabels([])
    ax3.tick_params(direction="out", pad=15)
    ax3.set_title("(c)")

    fpath = os.path.join(p.root_img_publi, "uace_2025_paper_results", "results")
    plt.savefig(fpath + ".png", dpi=300)
    plt.savefig(fpath + ".pdf", dpi=300)


if __name__ == "__main__":

    mode = "publi"
    plot_results_figure(mode=mode)
