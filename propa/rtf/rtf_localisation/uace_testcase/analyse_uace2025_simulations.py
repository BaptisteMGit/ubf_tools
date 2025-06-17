import os
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt

from propa.rtf.rtf_utils import D_hermitian_angle_fast
import propa.rtf.rtf_localisation.uace_testcase.src.params as p
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


def plot_results_figure_presentation(mode="demo"):
    """
    Plot the final figure used to present result in the UACE 2025 oral presentation
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

    """ Plot spectrogram of the received signal + interference """
    fpath = os.path.join(root_interf, "sig_noise_snr_0.0dB_sir-5dB.nc")
    ds_sig_noise = xr.open_dataset(fpath)

    # Check at event position
    ircv_check = 0
    s_l = ds_sig_noise.s_l.sel(
        idx_rcv=ircv_check, x=event_ship_x, y=event_ship_y, method="nearest"
    )
    s_l = s_l.values / np.max(np.abs(s_l.values))
    fs = 1 / ds_sig_noise.t.diff("t").values[0]

    # Define fig
    pfig = PubFigure(
        label_fontsize=32,
        ticks_fontsize=30,
        title_fontsize=32,
        legend_fontsize=16,
    )
    f, axs = plt.subplots(1, 2, figsize=(12, 6), width_ratios=[1, 2], sharey=True)

    # Z-call characteristic
    f_unit_A = 26.3
    f_unit_B = 18.6

    # PSD
    f, Pxx = sp.welch(s_l, fs=fs, nperseg=2**7, noverlap=int(2**7 * 3 / 4))
    axs[0].axhline(
        y=f_unit_A, color="r", linestyle="--", label=f"Unit A ({f_unit_A:.1f} Hz)"
    )
    axs[0].axhline(
        y=f_unit_B, color="b", linestyle="--", label=f"Unit B ({f_unit_B:.1f} Hz)"
    )
    axs[0].plot(10 * np.log10(Pxx), f, color="k")
    axs[0].legend(loc="upper right")
    axs[0].set_xlabel("PSD [dB]")
    axs[0].set_ylabel("Frequency [Hz]")
    axs[0].set_title("(a)")
    # axs[0].set_xlim(-100, 0)
    axs[0].set_ylim(5, 50)

    # STFT
    ff, tt, sxx = sp.stft(s_l, fs=fs, nperseg=2**7, noverlap=int(2**7 * 3 / 4))

    # Compute energy in the z call band
    idx_f_band = np.logical_and((ff <= f_unit_A), (ff >= f_unit_B))
    sxx_band = sxx[idx_f_band, :]
    e_band = np.sum(np.abs(sxx_band) ** 2, axis=0)
    plt.figure()
    plt.plot(tt, e_band)
    fpath = os.path.join(
        p.root_img_publi,
        "uace_2025_presentation_results",
        "lib_sig_interf_stft_zcall_e",
    )

    plt.savefig(fpath + ".png", dpi=300)

    # abs_sxx = np.abs(sxx) / np.max(np.abs(sxx))
    im = axs[1].pcolormesh(
        tt,
        ff,
        10 * np.log10(np.abs(sxx)),
        shading="gouraud",
        cmap="jet",
        vmin=-40,
        vmax=0,
        rasterized=True,
    )
    axs[1].set_xlabel("Time [s]")
    axs[1].set_title("(b)")
    axs[1].axhline(
        y=f_unit_A, color="r", linestyle="--", label=f"Unit A ({f_unit_A:.1f} Hz)"
    )
    axs[1].axhline(
        y=f_unit_B, color="b", linestyle="--", label=f"Unit B ({f_unit_B:.1f} Hz)"
    )
    # plt.colorbar(im)

    # plt.suptitle(f"SNR (WGN) = 0 dB and SIR (Z-call) = -5 dB")
    fpath = os.path.join(
        p.root_img_publi,
        "uace_2025_presentation_results",
        "lib_sig_interf_stft",
    )

    plt.savefig(fpath + ".png", dpi=300)
    plt.savefig(fpath + ".pdf", dpi=300)

    plt.close("all")

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

    # Convert coord to km
    to_km = True
    if to_km:
        ds_wgn = ds_wgn.assign_coords(x=("x", ds_wgn.x.values * 1e-3))
        ds_wgn = ds_wgn.assign_coords(y=("y", ds_wgn.y.values * 1e-3))
        ds_interf = ds_interf.assign_coords(x=("x", ds_interf.x.values * 1e-3))
        ds_interf = ds_interf.assign_coords(y=("y", ds_interf.y.values * 1e-3))
        event_ship_x *= 1e-3
        event_ship_y *= 1e-3
        x_abw *= 1e-3
        y_abw *= 1e-3
    # Build results subplots
    vmin = -5
    vmax = 0
    target_pos_circle_size = 180
    abw_pos_star_size = 400
    target_pos_linewidth = 3
    pfig = PubFigure(
        label_fontsize=32,
        # label_fontsize=22,
        ticks_fontsize=30,
        title_fontsize=32,
        legend_fontsize=16,
    )

    """ White Gaussian Noise Testcase Plotting """

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [1.5, 1]},
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
    ax1.set_title("Performance vs SNR")

    # Plot ambiguity surface for snr = 5dB without interference
    ds_wgn.d_rtf.plot(
        ax=ax2,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        rasterized=False,
        cbar_kwargs={"label": "[dB]"},
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
    ax2.set_title("SNR = 0 dB")
    ax2.set_xlabel("X [km]")
    ax2.set_ylabel("Y [km]")
    ax2.tick_params(direction="out", pad=15)

    fpath = os.path.join(
        p.root_img_publi, "uace_2025_presentation_results", "results_wgn"
    )
    plt.savefig(fpath + ".png", dpi=300)
    plt.savefig(fpath + ".pdf", dpi=300)

    """ Interferer Testcase Plotting """

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        gridspec_kw={"width_ratios": [2, 1, 1]},
    )

    # Plot theta vs frequency for wgn and interferer dataset
    root_tf = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\uace_testcase\data\backups\interfer_testcase_28052025_1200\data"
    ds_tf_true = xr.open_dataset(os.path.join(root_tf, "tf_grid_dx20m_dy20m.nc"))
    tf_true = ds_tf_true.tf_real + 1j * ds_tf_true.tf_imag

    ds_rtf_hat_interf = xr.open_dataset(
        os.path.join(root_interf, "features_dx20m_dy20m_snr_0.0dB.nc")
    )
    ds_rtf_hat_wgn = xr.open_dataset(
        os.path.join(root_wgn, "features_dx20m_dy20m_snr_0.0dB.nc")
    )

    rtf_hat_interf = ds_rtf_hat_interf.rtf_real + 1j * ds_rtf_hat_interf.rtf_imag
    rtf_hat_wgn = ds_rtf_hat_wgn.rtf_real + 1j * ds_rtf_hat_wgn.rtf_imag

    i_rcv_ref = 0
    tf_ref = tf_true.sel(idx_rcv=i_rcv_ref)
    rtf_true = tf_true / tf_ref
    rtf_hat_interf = rtf_hat_interf.sel(idx_rcv_ref=i_rcv_ref)
    rtf_hat_wgn = rtf_hat_wgn.sel(idx_rcv_ref=i_rcv_ref)

    ax_rcv = 0
    ax_f = 1
    dist_func = D_hermitian_angle_fast
    dist_kwargs = {
        "ax_rcv": ax_rcv,
        "unit": "deg",
        "apply_mean": False,
        "weights": None,
        "ax_f": ax_f,
    }
    # Interpolate at common frequencies
    rtf_true_interf = rtf_true.sel(f=rtf_hat_interf.f_rtf, method="nearest")
    theta_interf = dist_func(
        rtf_hat_interf.values, rtf_true_interf.values, **dist_kwargs
    )
    rtf_true_wgn = rtf_true.sel(f=rtf_hat_wgn.f_rtf, method="nearest")
    theta_wgn = dist_func(rtf_hat_wgn.values, rtf_true_wgn.values, **dist_kwargs)

    # Fuse x,y dims
    theta_interf = theta_interf.reshape(
        theta_interf.shape[0], theta_interf.shape[1] * theta_interf.shape[2]
    )
    theta_wgn = theta_wgn.reshape(
        theta_wgn.shape[0], theta_wgn.shape[1] * theta_wgn.shape[2]
    )
    # theta_mean_over_entire_grid = np.mean(theta, axis=1)
    theta_mean_over_entire_grid_interf = np.mean(theta_interf, axis=1)
    theta_mean_over_entire_grid_wgn = np.mean(theta_wgn, axis=1)

    # Roll
    roll_w = 7
    theta_interf = np.convolve(
        theta_mean_over_entire_grid_interf, np.ones(roll_w) / roll_w, mode="same"
    )
    theta_wgn = np.convolve(
        theta_mean_over_entire_grid_wgn, np.ones(roll_w) / roll_w, mode="same"
    )
    delta_theta = theta_interf - theta_wgn

    # Plot Theta
    ax1.plot(
        rtf_hat_interf.f_rtf.values,
        delta_theta,
        "-",
        color="k",
        markersize=3,
        label=r"$\Delta \theta$",
    )
    f_unit_A = 18.6
    f_unit_B = 26.3
    ax1.axvline(f_unit_A, label="Unit A", color="r", linestyle="--")
    ax1.axvline(f_unit_B, label="Unit B", color="b", linestyle="--")

    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel(r"$\theta$ [°]", color="k")
    ax1.tick_params(axis="y", labelcolor="k")
    ax1.set_title("(a)")
    ax1.legend(ncol=2, loc="lower right")

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
    ax2.set_title("SNR = 0 dB")
    ax2.set_xlabel("X [km]")
    ax2.set_ylabel("Y [km]")
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
    ax3.set_xlabel("X [km]")
    ax3.set_yticklabels([])
    ax3.tick_params(direction="out", pad=15)
    ax3.set_title("SIR = -5 dB")

    fpath = os.path.join(
        p.root_img_publi,
        "uace_2025_presentation_results",
        "results_interferer_delta_theta",
    )
    plt.savefig(fpath + ".png", dpi=300)
    plt.savefig(fpath + ".pdf", dpi=300)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        gridspec_kw={"width_ratios": [2, 1, 1]},
    )

    ax1.plot(
        rtf_hat_interf.f_rtf.values,
        theta_interf,
        "-",
        color="k",
        markersize=3,
        label="WGN + Zcall",
    )
    ax1.plot(
        rtf_hat_wgn.f_rtf.values,
        theta_wgn,
        "-",
        color="red",
        markersize=3,
        label="WGN",
    )
    f_unit_A = 18.6
    f_unit_B = 26.3
    ax1.axvline(f_unit_A, label="Unit A", color="r", linestyle="--")
    ax1.axvline(f_unit_B, label="Unit B", color="b", linestyle="--")

    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel(r"$\theta$ [°]", color="k")
    ax1.tick_params(axis="y", labelcolor="k")
    ax1.set_title("(a)")
    ax1.legend(ncol=2, loc="lower right")

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
    ax2.set_title("SNR = 0 dB")
    ax2.set_xlabel("X [km]")
    ax2.set_ylabel("Y [km]")
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
    ax3.set_xlabel("X [km]")
    ax3.set_yticklabels([])
    ax3.tick_params(direction="out", pad=15)
    ax3.set_title("SIR = -5 dB")

    fpath = os.path.join(
        p.root_img_publi, "uace_2025_presentation_results", "results_interferer_theta"
    )
    plt.savefig(fpath + ".png", dpi=300)
    plt.savefig(fpath + ".pdf", dpi=300)

    fig, (ax2, ax3) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [1, 1]},
    )

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
    ax2.set_title("SNR = 0 dB")
    ax2.set_xlabel("X [km]")
    ax2.set_ylabel("Y [km]")
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
    ax3.set_xlabel("X [km]")
    ax3.set_yticklabels([])
    ax3.tick_params(direction="out", pad=15)
    ax3.set_title("SIR = -5 dB")

    fpath = os.path.join(
        p.root_img_publi, "uace_2025_presentation_results", "results_interferer"
    )
    plt.savefig(fpath + ".png", dpi=300)
    plt.savefig(fpath + ".pdf", dpi=300)


if __name__ == "__main__":

    mode = "publi"
    # plot_results_figure(mode=mode)
    plot_results_figure_presentation(mode=mode)
