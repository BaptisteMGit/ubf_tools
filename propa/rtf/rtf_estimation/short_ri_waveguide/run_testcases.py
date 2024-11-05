#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   run_testcases.py
@Time    :   2024/11/04 14:21:41
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt

from misc import *

from propa.rtf.rtf_utils import (
    D_frobenius,
    D_hermitian_angle_fast,
    true_rtf,
    interp_true_rtf,
)
from propa.rtf.ideal_waveguide import *
from propa.rtf.rtf_estimation_const import *
from propa.rtf.rtf_estimation.rtf_estimation_utils import *
from propa.rtf.rtf_estimation.rtf_estimation_plot_tools import *
from propa.rtf.rtf_estimation.short_ri_waveguide.rtf_short_ri_kraken import *
from propa.rtf.rtf_estimation.short_ri_waveguide.rtf_short_ri_consts import *
from propa.rtf.rtf_estimation.short_ri_waveguide.rtf_short_ri_testcases import *


def dist_versus_snr(snrs, testcase=1, dist="frobenius"):

    # Select dist function to apply
    if dist == "frobenius":
        dist_func = D_frobenius
        dist_kwargs = {}
    elif dist == "hermitian_angle":
        dist_func = D_hermitian_angle_fast
        dist_kwargs = {
            "unit": "deg",
            "apply_mean": True,
        }

    # Derive results for each snr
    rtf_cs = []
    rtf_cw = []
    for i_snr, snr_dB in enumerate(snrs):
        plot = False
        # plot = i_snr % 100 == 0
        # print(f"i = {i_snr}, snr = {snr_dB}, plot = {plot}")
        if testcase == 1:
            # plot = False
            res_snr = testcase_1_unpropagated_whitenoise(snr_dB=snr_dB, plot=plot)
        elif testcase == 2:
            res_snr = testcase_2_propagated_whitenoise(snr_dB=snr_dB, plot=plot)
        elif testcase == 3:
            res_snr = testcase_3_propagated_interference(
                snr_dB=snr_dB, plot=plot, interference_type="z_call"
            )

        # Save rtfs into dedicated list
        rtf_cs.append(res_snr["cs"]["rtf"])
        rtf_cw.append(res_snr["cw"]["rtf"])

    f_cs = res_snr["cs"]["f"]
    f_cw = res_snr["cw"]["f"]
    f = f_cs

    # Load true RTF
    kraken_data = load_data()
    _, rtf_true_interp = interp_true_rtf(kraken_data, f)

    dist_cs = []
    dist_cw = []
    dist_cs_band = []
    dist_cw_band = []
    dist_cs_band_smooth = []
    dist_cw_band_smooth = []
    for i in range(len(snrs)):
        rtf_cs_i = rtf_cs[i]
        rtf_cw_i = rtf_cw[i]
        # Derive distance between estimated rtf and true rtf
        d_cs = dist_func(rtf_true_interp, rtf_cs_i, **dist_kwargs)
        d_cw = dist_func(rtf_true_interp, rtf_cw_i, **dist_kwargs)

        # Append to list
        dist_cs.append(d_cs)
        dist_cw.append(d_cw)

        # Same distance derivation for a restricted frequency band
        fmin_rtf = 5
        fmax_rtf = 50

        rtf_cs_i_band = rtf_cs_i[(f_cs >= fmin_rtf) & (f_cs <= fmax_rtf)]
        rtf_cw_i_band = rtf_cw_i[(f_cw >= fmin_rtf) & (f_cw <= fmax_rtf)]
        rtf_true_interp_band = rtf_true_interp[(f >= fmin_rtf) & (f <= fmax_rtf)]

        d_cs_band = dist_func(rtf_true_interp_band, rtf_cs_i_band, **dist_kwargs)
        d_cw_band = dist_func(rtf_true_interp_band, rtf_cw_i_band, **dist_kwargs)

        # Append to list
        dist_cs_band.append(d_cs_band)
        dist_cw_band.append(d_cw_band)

        # Distance for smoothed rtf
        window = 5
        rtf_cs_i_band_smooth = np.zeros_like(rtf_cs_i_band)
        rtf_cw_i_band_smooth = np.zeros_like(rtf_cw_i_band)
        for i in range(kraken_data["n_rcv"]):
            rtf_cs_i_band_smooth[:, i] = np.convolve(
                np.abs(rtf_cs_i_band[:, i]), np.ones(window) / window, mode="same"
            )
            rtf_cw_i_band_smooth[:, i] = np.convolve(
                np.abs(rtf_cw_i_band[:, i]), np.ones(window) / window, mode="same"
            )

        d_cs_band_smooth = dist_func(
            rtf_true_interp_band, rtf_cs_i_band_smooth, **dist_kwargs
        )
        d_cw_band_smooth = dist_func(
            rtf_true_interp_band, rtf_cw_i_band_smooth, **dist_kwargs
        )

        # Append to list
        dist_cs_band_smooth.append(d_cs_band_smooth)
        dist_cw_band_smooth.append(d_cw_band_smooth)

    # Plot distance versus snr
    props = res_snr["props"]
    title = (
        r"$\textrm{"
        + res_snr["tc_name"]
        + r"}\,"
        + " - "
        + " ["
        + f"{0}, {50}"
        + r"] \, \textrm{Hz}$"
        + f"\n({csdm_info_line(props)})"
    )
    fpath = os.path.join(ROOT_IMG, res_snr["tc_label"], "Df.png")
    plot_dist_vs_snr(
        snrs, dist_cs, dist_cw, title=title, dist_type=dist, savepath=fpath
    )
    # plt.figure()
    # plt.plot(snrs, 10 * np.log10(dist_cs), marker=".", label=r"$\mathcal{D}_F^{(CS)}$")
    # plt.plot(snrs, 10 * np.log10(dist_cw), marker=".", label=r"$\mathcal{D}_F^{(CW)}$")
    # plt.ylabel(r"$\mathcal{D}_F\, \textrm{[dB]}$")
    # plt.xlabel(r"$\textrm{snr} \, \textrm{[dB]}$")
    # plt.legend()
    # plt.grid()
    # plt.title(title)

    # # Save
    # plt.savefig(fpath)

    title = (
        r"$\textrm{"
        + res_snr["tc_name"]
        + r"}\,"
        + " - "
        + " ["
        + f"{fmin_rtf}, {fmax_rtf}"
        + r"] \, \textrm{Hz}$"
        + f"\n({csdm_info_line(props)})"
    )
    fpath = os.path.join(
        ROOT_IMG, res_snr["tc_label"], f"Df_band_{fmin_rtf}_{fmax_rtf}.png"
    )
    plot_dist_vs_snr(
        snrs, dist_cs_band, dist_cw_band, title=title, dist_type=dist, savepath=fpath
    )
    # plt.figure()
    # plt.plot(
    #     snrs, 10 * np.log10(dist_cs_band), marker=".", label=r"$\mathcal{D}_F^{(CS)}$"
    # )
    # plt.plot(
    #     snrs, 10 * np.log10(dist_cw_band), marker=".", label=r"$\mathcal{D}_F^{(CW)}$"
    # )
    # plt.ylabel(r"$\mathcal{D}_F\, \textrm{[dB]}$")  # TODO check unity  \textrm{[dB]}
    # plt.xlabel(r"$\textrm{snr} \, \textrm{[dB]}$")
    # plt.title(title)
    # plt.legend()
    # plt.grid()

    # Save
    # plt.savefig(fpath)

    title = (
        r"$\textrm{"
        + res_snr["tc_name"]
        + r"}\,"
        + " - "
        + " ["
        + f"{fmin_rtf}, {fmax_rtf}"
        + r"] \, \textrm{Hz}"
        + " - "
        + r"\textrm{smooth} \,"
        + f"(n = {window})$"
        + f"\n({csdm_info_line(props)})"
    )
    fpath = os.path.join(
        ROOT_IMG, res_snr["tc_label"], f"Df_band_{fmin_rtf}_{fmax_rtf}_smooth.png"
    )
    plot_dist_vs_snr(
        snrs,
        dist_cs_band_smooth,
        dist_cw_band_smooth,
        title=title,
        dist_type=dist,
        savepath=fpath,
    )
    # plt.figure()
    # plt.plot(
    #     snrs,
    #     10 * np.log10(dist_cs_band_smooth),
    #     marker=".",
    #     label=r"$\mathcal{D}_F^{(CS)}$",
    # )
    # plt.plot(
    #     snrs,
    #     10 * np.log10(dist_cw_band_smooth),
    #     marker=".",
    #     label=r"$\mathcal{D}_F^{(CW)}$",
    # )
    # plt.ylabel(r"$\mathcal{D}_F\, \textrm{[dB]}$")  # TODO check unity  \textrm{[dB]}
    # plt.xlabel(r"$\textrm{snr} \, \textrm{[dB]}$")
    # plt.title(title)
    # plt.legend()
    # plt.grid()

    # # Save

    # plt.savefig(fpath)


if __name__ == "__main__":
    # derive_kraken_tf()
    # derive_kraken_tf_surface_noise()
    # derive_kraken_tf_loose_grid()
    # kraken_data = load_data()
    # plot_ir(kraken_data, shift_ir=True)
    # plot_tf(kraken_data)

    # xr_surfnoise = xr.open_dataset(
    #     r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide\data\kraken_tf_surface_noise.nc"
    # )
    # xr_surfnoise_rcv = xr.open_dataset(
    #     r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\rtf\rtf_estimation\short_ri_waveguide\data\kraken_tf_noise_rcv0.nc"
    # )

    # rcv_sig = derive_received_signal()
    # plot_signal(tau_ir=TAU_IR)
    # snrs = [-25, 25]
    # for snr_dB in snrs:
    #     testcase_1_unpropagated_whitenoise(snr_dB=snr_dB)

    res = testcase_1_unpropagated_whitenoise(snr_dB=0)
    # compare_rtf_vs_received_spectrum(
    #     res["props"],
    #     res["cs"]["f"],
    #     res["cs"]["rtf"],
    #     res["cw"]["f"],
    #     res["cw"]["rtf"],
    #     rcv_signal=res["signal"],
    # )

    testcase_2_propagated_whitenoise(snr_dB=0)
    # testcase_3_propagated_interference(plot=True, interference_type="dirac")
    testcase_3_propagated_interference(snr_dB=0, plot=True, interference_type="z_call")
    # testcase_3_propagated_interference(snr_dB=10, plot=True, interference_type="z_call")
    # testcase_3_propagated_interference(plot=True, interference_type="ricker_pulse")

    # check_interp()
    # snrs = [0, 10]
    # snrs = np.round(np.arange(-50, 50, 0.1), 1)
    # snrs = np.arange(-5, 5, 1)

    # dist_versus_snr(snrs, testcase=1, dist="hermitian_angle")
    # dist_versus_snr(snrs, testcase=2, dist="hermitian_angle")
    # dist_versus_snr(snrs, testcase=3, dist="hermitian_angle")

    # dist_versus_snr(snrs, testcase=2)

    plt.show()
