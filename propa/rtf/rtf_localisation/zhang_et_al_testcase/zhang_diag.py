#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   zhang_diag.py
@Time    :   2025/02/13 19:51:28
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Diagnostics to run after zhang testcase 
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from propa.rtf.rtf_utils import D_hermitian_angle_fast, normalize_metric_contrast
from propa.rtf.rtf_localisation.zhang_et_al_testcase.zhang_misc import params

ROOT_DATA = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_localisation\zhang_et_al_testcase\data"
ROOT_IMG = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\zhang_et_al_2023\diagnostic"


def diag_hermitian_angle_vs_snr(
    ref_to_use="kraken", rtf_to_compare="cs", antenna_type="zhang", debug=False
):

    # Ensure img folder exists
    if not os.path.exists(ROOT_IMG):
        os.makedirs(ROOT_IMG)

    _, _, source, grid, _, _ = params(debug=debug, antenna_type=antenna_type)

    # Load gridded dataset
    fname = f"zhang_output_fullsimu_dx{grid['dx']}m_dy{grid['dy']}m.nc"
    fpath = os.path.join(ROOT_DATA, fname)
    ds_rtf = xr.open_dataset(fpath)

    # Define reference receiver to use
    i_ref = 0
    # Extract data corresponding to the two-sensor pair rcv_cpl
    ds_fa_rtf = ds_rtf.sel(idx_rcv_ref=i_ref)

    rtf_grid = ds_fa_rtf.rtf_real + 1j * ds_fa_rtf.rtf_imag
    rtf_true_pos = rtf_grid.sel(x=source["x"], y=source["y"], method="nearest")

    # rtf_event = ds_fa_rtf.rtf_event_real.values + 1j * ds_fa_rtf.rtf_event_imag.values

    dist_func = D_hermitian_angle_fast
    dist_kwargs = {
        "ax_rcv": 1,
        "unit": "deg",
        "apply_mean": True,
    }

    # List file in root_data folder
    files = os.listdir(ROOT_DATA)
    # Select the files with following format zhang_output_from_signal_dx20m_dy20m_snr0dB.nc
    files = [f for f in files if "zhang_output_from_signal" in f]

    theta_x = []
    theta_y = []
    theta_srcpos = []

    npos_x = 10
    npos_y = 10

    # Ensure npos_x and npos_y are odd numbers
    if npos_x % 2 == 0:
        npos_x += 1
    if npos_y % 2 == 0:
        npos_y += 1

    snrs = []
    # Iterate over selected files
    for i_f, f in enumerate(files):

        theta_x_snri = []
        theta_y_snri = []

        # Get snr information from file name
        snr = int(f.split("_")[-1].split("snr")[1].split("dB")[0])
        snrs.append(snr)
        # Load dataset
        ds_rtf_cs = xr.open_dataset(os.path.join(ROOT_DATA, f))
        ds_rtf_cs = ds_rtf_cs.sel(idx_rcv_ref=i_ref)

        # Extract tf between fmin and fmax from ds_rtf_cs
        # tf = tf.sel(f=slice(ds_rtf_cs.f_rtf.min(), ds_rtf_cs.f_rtf.max()))

        # Select the reference RTF to use for comparison (ie the RTF at source position)
        if ref_to_use == "event":
            rtf_cs_e = ds_rtf_cs.rtf_event_real + 1j * ds_rtf_cs.rtf_event_imag
            rtf_true_pos_interp = rtf_cs_e
        elif ref_to_use == "kraken":
            # Interp rtf_true at frequencies
            if i_f == 0:
                rtf_true_pos_real = []
                rtf_true_pos_imag = []
                for i_rcv in rtf_true_pos.idx_rcv:
                    rtf_true_pos_real_i = np.interp(
                        ds_rtf_cs.f_rtf.values,
                        rtf_true_pos.f.values,
                        rtf_true_pos.sel(idx_rcv=i_rcv).real.values,
                    )
                    rtf_true_pos_imag_i = np.interp(
                        ds_rtf_cs.f_rtf.values,
                        rtf_true_pos.f.values,
                        rtf_true_pos.sel(idx_rcv=i_rcv).imag.values,
                    )
                    rtf_true_pos_real.append(rtf_true_pos_real_i)
                    rtf_true_pos_imag.append(rtf_true_pos_imag_i)

                rtf_true_pos_real_interp = np.array(rtf_true_pos_real).T
                rtf_true_pos_imag_interp = np.array(rtf_true_pos_imag).T

                rtf_true_pos_interp = (
                    rtf_true_pos_real_interp + 1j * rtf_true_pos_imag_interp
                )

                # Plot interpolation to ensure it is correct
                for i_rcv in rtf_true_pos.idx_rcv.values:
                    plt.figure()

                    rtf_true_pos.sel(idx_rcv=i_rcv).imag.plot(
                        label="imag-true",
                        linewidth=2,
                        color="r",
                        linestyle="--",
                    )
                    plt.plot(
                        ds_rtf_cs.f_rtf,
                        rtf_true_pos_imag_interp[:, i_rcv],
                        label="imag-interp",
                        linewidth=1,
                        linestyle="-",
                        color="r",
                        marker="o",
                        markersize=4,
                    )
                    rtf_true_pos.sel(idx_rcv=i_rcv).real.plot(
                        label="real-true",
                        linewidth=2,
                        color="b",
                        linestyle="--",
                    )
                    plt.plot(
                        ds_rtf_cs.f_rtf,
                        rtf_true_pos_real_interp[:, i_rcv],
                        label="real-interp",
                        linewidth=1,
                        linestyle="-",
                        color="b",
                        marker="o",
                        markersize=4,
                    )
                    plt.legend()

                    plt.savefig(
                        os.path.join(ROOT_IMG, f"rtf_interp_ircv{i_rcv}.png"), dpi=300
                    )
            plt.close("all")

        # Select the RTF to compare with the reference RTF
        if rtf_to_compare == "cs":
            rtf_cs = (
                ds_rtf_cs.rtf_real + 1j * ds_rtf_cs.rtf_imag
            )  # Estimated RTF from the CS method
        elif (
            rtf_to_compare == "kraken"
        ):  # Compare kraken rtf vectors to show the theoritical variations of the rtf around the source
            rtf_cs = rtf_grid.rename({"f": "f_rtf"})
            rtf_true_pos_interp = rtf_grid.rename({"f": "f_rtf"}).sel(
                x=source["x"], y=source["y"], method="nearest"
            )  # True RTF at source position does not need to be interpolated

        # Iterate over positions close to the source
        if i_f == 0:
            idx_pos_src_x = np.argmin(np.abs(ds_rtf_cs.x.values - source["x"]))
            range_pos_idx_x = np.arange(
                max(idx_pos_src_x - npos_x // 2, 0),
                min(idx_pos_src_x + npos_x // 2 + 1, ds_rtf_cs.x.size),
            )
            idx_pos_src_y = np.argmin(np.abs(ds_rtf_cs.y.values - source["y"]))
            range_pos_idx_y = np.arange(
                max(idx_pos_src_y - npos_y // 2, 0),
                min(idx_pos_src_y + npos_y // 2 + 1, ds_rtf_cs.y.size),
            )

        # Along x axis
        for i_x in range_pos_idx_x:

            # Extract data at required position
            rtf_cs_pos = rtf_cs.isel(x=i_x).sel(y=source["y"], method="nearest")
            dtheta = dist_func(rtf_cs_pos, rtf_true_pos_interp, **dist_kwargs)
            theta_x_snri.append(dtheta)

        # Along y axis
        for i_y in range_pos_idx_y:
            # Extract data at required position
            rtf_cs_pos = rtf_cs.isel(y=i_y).sel(x=source["x"], method="nearest")
            # Derive distance
            dtheta = dist_func(rtf_cs_pos, rtf_true_pos_interp, **dist_kwargs)
            theta_y_snri.append(dtheta)

        theta_x.append(theta_x_snri)
        theta_y.append(theta_y_snri)

        rtf_cs_src_pos = rtf_cs.sel(x=source["x"], y=source["y"], method="nearest")
        dtheta_srcpos = dist_func(rtf_cs_src_pos, rtf_true_pos_interp, **dist_kwargs)
        theta_srcpos.append(dtheta_srcpos)

    theta_x = np.array(theta_x)  # (n_snr, n_pos_x)
    theta_y = np.array(theta_y)  # (n_snr, n_pos_y)
    theta_srcpos = np.array(theta_srcpos)  # (n_snr,)

    x_pos = ds_rtf_cs.x.isel(x=range_pos_idx_x)
    range_to_src_pos_x = (
        # x_pos - ds_rtf_cs.x.isel(x=idx_pos_src_x).values
        x_pos
        - source["x"]
    )  # -> negative to positive

    y_pos = ds_rtf_cs.y.isel(y=range_pos_idx_y)
    # range_to_src_pos_y = y_pos - ds_rtf_cs.y.isel(y=idx_pos_src_y).values
    range_to_src_pos_y = y_pos - source["y"]

    # Iterate over snrs
    snrs = np.array(snrs)
    # Order snrs
    snrs = np.sort(snrs)
    # Order theta_x and theta_y according to snrs order
    theta_x = theta_x[np.argsort(snrs)]
    theta_y = theta_y[np.argsort(snrs)]
    theta_srcpos = theta_srcpos[np.argsort(snrs)]

    # Same plots but with the contrast metrics
    q_x = []
    q_y = []
    for i_snr, snr in enumerate(snrs):
        q_x.append(normalize_metric_contrast(theta_x[i_snr]))
        q_y.append(normalize_metric_contrast(theta_y[i_snr]))
    q_x = np.array(q_x)
    q_y = np.array(q_y)

    for metric in ["theta", "q"]:
        if metric == "theta":
            data_x = theta_x
            data_y = theta_y
            data_srcpos = theta_srcpos
            if rtf_to_compare == "cs":
                metric_label = r"$\theta_{CS}$" + " [deg]"
            elif rtf_to_compare == "kraken":
                metric_label = r"$\theta$" + " [deg]"

        elif metric == "q":
            data_x = q_x
            data_y = q_y
            data_srcpos = normalize_metric_contrast(theta_srcpos)
            metric_label = "q"

        # Plot data_x and theta_y vs range to source as lines for each snr
        if len(snrs) <= 5:

            # Plot along x axis
            plt.figure()
            for i_snr, snr in enumerate(snrs):
                data_x_i = data_x[i_snr]
                plt.plot(
                    range_to_src_pos_x.values,
                    data_x_i,
                    label=f"SNR = {snr} dB",
                )
            plt.xlabel("Range to source [m]")
            plt.ylabel(metric_label)
            plt.title(f"Variation along x-axis")
            plt.legend()
            plt.savefig(
                os.path.join(
                    ROOT_IMG, f"{metric}_x_{rtf_to_compare}_vs_{ref_to_use}.png"
                ),
                dpi=300,
            )

            # Plot along y axis
            plt.figure()
            for i_snr, snr in enumerate(snrs):
                data_y_i = data_y[i_snr]
                plt.plot(
                    range_to_src_pos_y.values,
                    data_y_i,
                    label=f"SNR = {snr} dB",
                )
            plt.xlabel("Range to source [m]")
            plt.ylabel(metric_label)
            plt.title("Variation along y-axis")
            plt.legend()
            plt.savefig(
                os.path.join(
                    ROOT_IMG, f"{metric}_y_{rtf_to_compare}_vs_{ref_to_use}.png"
                ),
                dpi=300,
            )

        # Plot a map of data_x and data_y (range to source, snr) for a better visualization
        else:
            plt.figure()
            im = plt.pcolormesh(
                range_to_src_pos_x.values,
                snrs,
                data_x,
                shading="auto",
                cmap="jet",
            )
            plt.colorbar(im, label=metric_label)
            plt.xlabel("Range to source [m]")
            plt.ylabel("SNR [dB]")
            plt.title(f"Variation along x-axis")
            plt.savefig(
                os.path.join(
                    ROOT_IMG, f"{metric}_x_{rtf_to_compare}_vs_{ref_to_use}.png"
                ),
                dpi=300,
            )

            plt.figure()
            im = plt.pcolormesh(
                range_to_src_pos_y.values,
                snrs,
                data_y,
                shading="auto",
                cmap="jet",
            )
            plt.colorbar(im, label=metric_label)
            plt.xlabel("Range to source [m]")
            plt.ylabel("SNR [dB]")
            plt.title("Variation along y-axis")
            plt.savefig(
                os.path.join(
                    ROOT_IMG, f"{metric}_y_{rtf_to_compare}_vs_{ref_to_use}.png"
                ),
                dpi=300,
            )

        # Plot distance vs snr at the source position
        plt.figure()
        plt.plot(snrs, data_srcpos)
        plt.xlabel("SNR [dB]")
        plt.ylabel(r"$\theta_{CS}$" + " [deg]")
        plt.title("Variation at source position")
        plt.savefig(
            os.path.join(
                ROOT_IMG, f"{metric}_src_{rtf_to_compare}_vs_{ref_to_use}.png"
            ),
            dpi=300,
        )

    print("Done")


if __name__ == "__main__":
    diag_hermitian_angle_vs_snr(ref_to_use="kraken", rtf_to_compare="kraken")
