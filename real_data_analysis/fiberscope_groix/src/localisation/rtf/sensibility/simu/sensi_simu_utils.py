#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sensi_simu_utils.py
@Time    :   2026/04/10 11:11:44
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import sys
import numpy as np
import xarray as xr
import scipy.signal as sp
import matplotlib.pyplot as plt

from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

from misc import mult_along_axis
from propa.ideal_waveguide import (
    kr,
    psi,
    psi_normalised,
    h,
    field,
    plot_tl,
    nb_propagating_modes,
    intensity,
    intensity_1,
)

from misc import cast_matrix_to_target_shape
from propa.rtf.rtf_utils import D_hermitian_angle_fast


import source.global_constants as g
from publication.publication_figure import (
    PubFigure,
    LargeFigure,
    SmallFigure,
    color,
    set_subfigures_abc_labels,
)

pfig = PubFigure()


# ======================================================================================================================
# Plot utils
# ======================================================================================================================


def plot_tl_pi_frequency_range_plan(
    intensity_1_dB, intensity_2_dB, rtf2_mod, x_coord="r", y_coord="f"
):
    fig, axs = plt.subplots(1, 3, figsize=(18, 12), sharex=True, sharey=True)

    vmin = max(np.percentile(intensity_1_dB, 5), np.percentile(intensity_2_dB, 5))
    vmax = min(np.percentile(intensity_1_dB, 95), np.percentile(intensity_2_dB, 95))

    intensity_1_dB.plot(
        ax=axs[0],
        x=x_coord,
        y=y_coord,
        cmap="magma",
        add_colorbar=True,
        cbar_kwargs={
            # "label": r"$\text{TL}(r, f)$ [dB]",
            "label": "[dB]",
        },
        vmin=vmin,
        vmax=vmax,
    )
    axs[0].set_title(
        r"$\text{TL}_1(r, f) = -10 \log_{10}(\lvert 4 \pi H_1(f, r) \rvert^2)$",
        fontsize=16,
    )
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")

    intensity_2_dB.plot(
        ax=axs[1],
        x=x_coord,
        y=y_coord,
        cmap="magma",
        add_colorbar=True,
        cbar_kwargs={
            # "label": r"$\text{TL}(r, f)$ [dB]",
            "label": "[dB]",
        },
        vmin=vmin,
        vmax=vmax,
    )
    axs[1].set_title(
        r"$\text{TL}_2(r, f) = -10 \log_{10}(\lvert 4 \pi H_2(f, r) \rvert^2)$",
        fontsize=16,
    )
    axs[1].set_xlabel("")
    axs[1].set_ylabel("")

    rtf2_mod_2_log = 10 * np.log10(rtf2_mod**2)
    vmin = np.percentile(rtf2_mod_2_log, 5)
    vmax = np.percentile(rtf2_mod_2_log, 95)
    # d_tl = I_1_dB - I_2_dB
    rtf2_mod_2_log.plot(
        ax=axs[2],
        x=x_coord,
        y=y_coord,
        cmap="magma",
        add_colorbar=True,
        cbar_kwargs={
            # "label": r"$\lvert \Pi_2(f, r) \rvert$",
            "label": "[dB]",
        },
        vmin=vmin,
        vmax=vmax,
    )
    axs[2].set_title(
        # r"$10 \log_{10} \left( \lvert \Pi_2(f, r) \rvert^2 \right) = \left \lvert \frac{H_2(f, r)}{H_1(f, r)}  \right \rvert$"
        r"$10 \log_{10} \left( \lvert \Pi_2(f, r) \rvert^2 \right) =  \text{TL}_1(r, f) - \text{TL}_2(r, f) $",
        fontsize=16,
    )
    axs[2].set_xlabel("")
    axs[2].set_ylabel("")
    if x_coord == "r":
        fig.supxlabel("Range [m]")
        fig.supylabel("Frequency [Hz]")
    elif x_coord == "f":
        fig.supylabel("Range [m]")
        fig.supxlabel("Frequency [Hz]")

    # axs[0].set_xlim([0, 3000])
    # plt.ylim([21.5, 26])

    set_subfigures_abc_labels(axs, x_pos=0.005, y_pos=1.1, fontsize=14)

    return fig, axs


def plot_intensity(intensity_dB, fmin, fmax, step_f=0.5):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    f_plot = np.arange(fmin, fmax, step_f)
    for i_f, f in enumerate(f_plot):
        I_f = intensity_dB.sel(f=f, method="nearest").copy()
        I_f -= i_f * 2
        I_f.plot(
            ax=axs[0],
            label=f"{f} Hz",
        )

    # Add lines to show the trajectory of maximas of the intensity ?

    # Reverse y axis
    axs[0].invert_yaxis()
    axs[0].set_xlim([1, 3000])
    axs[0].set_ylim([80, 0])
    axs[0].legend(ncols=3)
    axs[0].set_xlabel("")
    axs[0].set_ylabel(
        r"$\text{TL}(r)$ [dB]",
    )
    axs[0].set_title("")

    # vmin, vmax = 20, 70
    vmin = np.percentile(intensity_dB, 10)
    vmax = np.percentile(intensity_dB, 90)
    im = intensity_dB.plot(
        ax=axs[1],
        x="r",
        y="f",
        cmap="jet",
        add_colorbar=True,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = im.colorbar
    cbar.set_label(
        r"$\text{TL}(r, f) = -10 \log_{10}(\lvert 4 \pi H(f, r) \rvert^2)$ [dB]",
        fontsize=14,
    )

    set_subfigures_abc_labels(axs, x_pos=0.01, y_pos=1.06, fontsize=14)


# ======================================================================================================================
# Functions
# ======================================================================================================================


# Compute the dataset
def calc_pi_12(
    f, r_grid1, r_grid2, z_source, z_antenna, depth, bottom_bc, n_modes=None
):

    ff, rr, zz, p_field1 = field(
        f=f,
        z_src=z_antenna,  # Reciprocity: source is at antenna depth
        z=z_source,  # Reciprocity: receiver is at source depth
        r=r_grid1,
        depth=depth,
        bottom_bc=bottom_bc,
        n=n_modes,
    )
    ff, _, _, p_field2 = field(
        f=f,
        z_src=z_antenna,  # Reciprocity: source is at antenna depth
        z=z_source,  # Reciprocity: receiver is at source depth
        r=r_grid2,
        depth=depth,
        bottom_bc=bottom_bc,
        n=n_modes,
    )
    p_field2[(p_field2 == 0) | np.isnan(p_field2)] = 1e-20
    p_field1[(p_field1 == 0) | np.isnan(p_field1)] = 1e-20
    pi_12 = p_field2 / p_field1

    return ff, rr, zz, p_field1, p_field2, pi_12


def compute_rtf_dataset(
    f,
    r_grid1,
    r_grid2,
    z_s,
    z_a,
    waveguide_depth,
    bottom_bc,
    r_ref,
    d12,
    n_modes=None,
):

    # Compute RTF on the whole grid for a different number of modes
    ff, rr, zz, p_field1, p_field2, pi_12 = calc_pi_12(
        f=f,
        r_grid1=r_grid1,
        r_grid2=r_grid2,
        z_source=z_s,
        z_antenna=z_a,
        depth=waveguide_depth,
        bottom_bc=bottom_bc,
        n_modes=n_modes,
    )
    # Shape pi_12 = (nf, 1, nr)

    # Compute RTF at reference range
    ff, rr_ref, zz, p_field1_ref, p_field2_ref, pi_12_ref = calc_pi_12(
        f=f,
        r_grid1=np.array([r_ref]),
        r_grid2=np.array([r_ref + d12]),
        z_source=z_s,
        z_antenna=z_a,
        depth=waveguide_depth,
        bottom_bc=bottom_bc,
        n_modes=n_modes,
    )
    # Shape pi_12_ref = (nf, 1, 1)

    n_propa_modes_vs_f = np.array(
        [
            nb_propagating_modes(
                f=fi, c=g.c0, depth=waveguide_depth, bottom_bc=bottom_bc
            )
            for fi in ff
        ]
    )

    # Add one dimension to fit the usual shape of RTF vectors (array of one for H_ref/H_ref)
    # Target shape is (n_rcv, nf, nr)
    p_field1 = p_field1.squeeze(axis=1)  # (nf, nr)
    p_field2 = p_field2.squeeze(axis=1)  # (nf, nr)
    pi_12 = pi_12.squeeze(axis=1)  # (nf, nr)
    pi_11 = np.ones_like(pi_12).astype(np.complex64)
    pi_12_full = np.array([pi_11, pi_12])  # (nrcv, nf, nr)

    # Same thing for pi_ref -> (nrcv, nf)
    pi_12_ref = pi_12_ref.squeeze(axis=(1, 2))
    pi_11_ref = np.ones_like(pi_12_ref).astype(np.complex64)
    pi_12_ref_full = np.array([pi_11_ref, pi_12_ref])

    ds = xr.Dataset(
        data_vars=dict(
            H1=(["f", "r"], p_field1),
            H2=(["f", "r"], p_field2),
            data=(["h_index", "f", "r"], pi_12_full),
            phase=(["h_index", "f", "r"], np.angle(pi_12_full)),
            module=(["h_index", "f", "r"], np.abs(pi_12_full)),
            data_ref=(["h_index", "f"], pi_12_ref_full),
            nb_modes=(["f"], n_propa_modes_vs_f),
        ),
        coords=dict(
            f=ff,
            r=rr.squeeze(),
            h_index=[1, 2],
        ),
        attrs=dict(
            description=f"RTF dataset for an ideal waveguide. The antenna is composed of two receivers separated by d12={d12}m. The source is at z_s={z_s}m and the antenna at z_a={z_a}m in a waveguide of depth {waveguide_depth}m.",
            bottom_bc=bottom_bc,
            waveguide_depth=waveguide_depth,
            z_s=z_s,
            z_a=z_a,
            r_ref=r_ref,
            d12=d12,
            h_index_ref=1,
        ),
    )

    # Add usefull attrs
    ds.f.attrs = {"unit": "Hz", "long_name": "Frequency"}
    ds.r.attrs = {"unit": "m", "long_name": "Range"}
    ds.H1.attrs = {"long_name": r"$H_1$", "units": "1"}
    ds.H2.attrs = {"long_name": r"$H_2$", "units": "1"}

    print("Dataset computed")

    return ds


def plot_tls(ds, fmin, fmax, root_fig):
    # Plot TL at fmin, fmax and center frequency
    lfig = LargeFigure(legend_fontsize=12)
    frequencies_to_plot = [fmin, (fmin + fmax) / 2, fmax]
    plt.figure()
    for freq in frequencies_to_plot:
        # idx_freq = np.argmin(np.abs(ff - freq))

        # tl1 = -20 * np.log10(4 * np.pi * np.abs(p_field1[idx_freq, :]) + 1e-20)
        # tl2 = -20 * np.log10(4 * np.pi * np.abs(p_field2[idx_freq, :]) + 1e-20)

        tl1 = -20 * np.log10(
            4 * np.pi * np.abs(ds.H1.sel(f=freq, method="nearest")) + 1e-20
        )
        tl2 = -20 * np.log10(
            4 * np.pi * np.abs(ds.H2.sel(f=freq, method="nearest")) + 1e-20
        )

        tl1.plot(label=f"Receiver 1 at {freq} Hz", linestyle="-")
        tl2.plot(label=f"Receiver 2 at {freq} Hz", linestyle="--")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.title("")
    plt.ylabel("TL [dB]")

    fname = f"TL_fmin_fmax_fmid.png"
    fpath = os.path.join(root_fig, fname)
    plt.savefig(fpath)


def plot_pi_fr_plan(ds, root_fig, rmin=None, rmax=None):
    lfig = LargeFigure(legend_fontsize=14, size=(16, 8))
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax1, ax2, ax3 = axs

    rtf_mod = 10 * np.log10(ds.module)

    rtf_mod.sel(h_index=2).plot(
        x="f",
        cmap="magma",
        vmin=np.percentile(rtf_mod, 5),
        vmax=np.percentile(rtf_mod, 95),
        ax=ax1,
        cbar_kwargs={"label": r"$\lvert \Pi\rvert$ [dB]"},
    )

    rtf_phase = ds.phase
    rtf_phase.sel(h_index=2).plot(
        x="f",
        cmap="bwr",
        vmin=np.percentile(rtf_phase, 5),
        vmax=np.percentile(rtf_phase, 95),
        ax=ax2,
        cbar_kwargs={"label": r"$\Phi$ [rad]"},
    )

    ds.nb_modes.plot(ax=ax3)
    ax3.set_ylabel("Modes")

    ax1.set_title("")
    ax2.set_title("")
    ax3.set_title("")

    if rmin is not None and rmax is not None:

        ax1.set_ylim([rmin, rmax])
        ax2.set_ylim([rmin, rmax])
        fname = f"rtf_module_phase_f_r_plan_zoom_rmin{rmin}_rmax{rmax}.png"

    else:
        fname = "rtf_module_phase_f_r_plan.png"

    fpath = os.path.join(root_fig, fname)
    plt.savefig(fpath)
    # plt.xscale("log")


def get_dist(ds, distances="all"):

    rtf = ds.data.values[..., np.newaxis]  # (nrcv, nf, nr, 1)
    rtf_ref = ds.data_ref.values  # (nrcv, nf)
    rtf_ref_expanded = cast_matrix_to_target_shape(
        rtf_ref, rtf.shape
    )  # (nrcv, nf, nr, 1)

    # Compute dist
    dist_output = {}

    if "hermitian_angle" in distances or distances == "all":
        dist_kwargs = {
            "ax_rcv": 0,
            "ax_f": 1,
            "apply_mean": False,
            "apply_median": True,
            "data_space": "complex",
        }
        theta = D_hermitian_angle_fast(
            rtf_ref=rtf_ref,
            rtf=rtf,
            **dist_kwargs,
        )

        dist_output["hermitian_angle"] = {
            "d": theta,
            "name": r"$\theta$",
            "unit": "[°]",
        }

    if "hermitian_angle_module" in distances or distances == "all":
        dist_kwargs = {
            "ax_rcv": 0,
            "ax_f": 1,
            "apply_mean": False,
            "apply_median": True,
            "data_space": "real",
        }
        x = np.abs(rtf_ref)
        y = np.abs(rtf)

        # Apply rolling average along f axis to smooth the RTF and make the distance more robust to small variations
        n_roll_win = 15
        x = np.apply_along_axis(
            lambda v: np.convolve(v, np.ones(n_roll_win) / n_roll_win, mode="same"),
            axis=1,
            arr=x,
        )
        y = np.apply_along_axis(
            lambda v: np.convolve(v, np.ones(n_roll_win) / n_roll_win, mode="same"),
            axis=1,
            arr=y,
        )

        theta_mod = D_hermitian_angle_fast(
            rtf_ref=x,
            rtf=y,
            **dist_kwargs,
        )

        dist_output["hermitian_angle_module"] = {
            "d": theta_mod,
            "name": r"$\theta_{\text{mod}}$",
            "unit": "[°]",
        }

    if "norm_L1" in distances or distances == "all":
        x = rtf_ref_expanded
        y = rtf

        d_L1 = np.sum(np.abs(x - y), axis=0)
        d_L1 = np.median(d_L1, axis=0).squeeze()  # Median along f axis

        dist_output["norm_L1"] = {
            "d": d_L1,
            "name": r"$\lVert \rVert_{1}$",
            "unit": "[-]",
        }

    if "norm_L1_module" in distances or distances == "all":
        x = np.abs(rtf_ref_expanded)
        y = np.abs(rtf)

        d_L1_mod = np.sum(np.abs(x - y), axis=0)
        d_L1_mod = np.median(d_L1_mod, axis=0).squeeze()

        dist_output["norm_L1_module"] = {
            "d": d_L1_mod,
            "name": r"$\lVert \rVert_{1_{\text{mod}}}$",
            "unit": "[-]",
        }

    if "norm_L2" in distances or distances == "all":
        x = rtf_ref_expanded
        y = rtf

        d_L2 = np.sqrt(np.sum(np.abs(x - y) ** 2, axis=0))
        d_L2 = np.median(d_L2, axis=0).squeeze()

        dist_output["norm_L2"] = {
            "d": d_L2,
            "name": r"$\lVert \rVert_{2}$",
            "unit": "[-]",
        }

    if "norm_L2_module" in distances or distances == "all":
        x = np.abs(rtf_ref_expanded)
        y = np.abs(rtf)

        d_L2_mod = np.sqrt(np.sum(np.abs(x - y) ** 2, axis=0))
        d_L2_mod = np.median(d_L2_mod, axis=0).squeeze()

        dist_output["norm_L2_module"] = {
            "d": d_L2_mod,
            "name": r"$\lVert \rVert_{2_{\text{mod}}}$",
            "unit": "[-]",
        }

    if "norm_L2_normalized" in distances or distances == "all":
        x = rtf_ref_expanded
        y = rtf

        d_L2_normalized = np.sqrt(np.sum(np.abs(x - y) ** 2, axis=0)) / np.sqrt(
            np.sum(np.abs(x) ** 2, axis=0)
        )
        d_L2_normalized = np.median(d_L2_normalized, axis=0).squeeze()

        dist_output["norm_L2_normalized"] = {
            "d": d_L2_normalized,
            "name": r"$\lVert \rVert_{2} \, \text{(normalized)}$",
            "unit": "[-]",
        }

    if "norm_L2_module_normalized" in distances or distances == "all":
        x = np.abs(rtf_ref_expanded)
        y = np.abs(rtf)

        d_L2_mod_normalized = np.sqrt(np.sum(np.abs(x - y) ** 2, axis=0)) / np.sqrt(
            np.sum(np.abs(x) ** 2, axis=0)
        )
        d_L2_mod_normalized = np.median(d_L2_mod_normalized, axis=0).squeeze()

        dist_output["norm_L2_module_normalized"] = {
            "d": d_L2_mod_normalized,
            "name": r"$\lVert \rVert_{2} \, \text{(normalized)}_{\text{mod}}$",
            "unit": "[-]",
        }

    if "intercorr_max" in distances or distances == "all":
        x = np.abs(rtf_ref_expanded)
        y = np.abs(rtf)

        x = x - np.mean(x, axis=1, keepdims=True)
        y = y - np.mean(y, axis=1, keepdims=True)

        x_fft = np.fft.fft(x, axis=1)
        y_fft = np.fft.fft(y, axis=1)
        # df = ds.f.values[1] - ds.f.values[0]
        s_xy = x_fft * np.conj(y_fft)
        c_xy = np.fft.ifft(s_xy, axis=1)
        d_intercorr = np.fft.fftshift(np.real(c_xy), axes=1)
        d_intercorr_max = np.max(d_intercorr, axis=1)  # Max along delta_f axis
        # d_intercorr_max = np.sum(d_intercorr_max, axis=0)  # Sum along rcv axis
        d_intercorr_max = d_intercorr_max[1]

        dist_output["intercorr_max"] = {
            "d": d_intercorr_max,
            "name": r"$\max_{\delta_f} C_{xy}(\delta_f)$",
            "unit": "[-]",
        }

        # # EUCLIDEAN distance
    # d_rcv_1 = np.linalg.norm(
    #     np.abs(test_replica_rtf.sel(h_index=1).values)[:, np.newaxis]
    #     - np.abs(bootstrap_replica_rtfs[0, ...]), axis=0
    # )
    # d_rcv_2 = np.linalg.norm(
    #     np.abs(test_replica_rtf.sel(h_index=2).values)[:, np.newaxis]
    #     - np.abs(bootstrap_replica_rtfs[1, ...]),
    #     axis=0,
    # )
    # d_rcv_3 = np.linalg.norm(
    #     np.abs(test_replica_rtf.sel(h_index=3).values)[:, np.newaxis]
    #     - np.abs(bootstrap_replica_rtfs[2, ...]),
    #     axis=0,
    # )
    # dist = 1 / 3 * (d_rcv_1 + d_rcv_2 + d_rcv_3)

    # # # Compute dist
    # dist_kwargs = {"ax_rcv": 0, "ax_f": 1, "apply_mean": True}
    # dist = D_euclidian(
    #     rtf_ref=np.abs(test_replica_rtf.isel(h_index=slice(0, 1)).values),
    #     rtf=np.abs(bootstrap_replica_rtfs[0:1, ...]),
    #     **dist_kwargs,
    # )

    # theta_phi = D_hermitian_angle_fast(
    #     rtf_ref=np.angle(ds.data_ref.values),
    #     rtf=np.angle(rtf),
    #     **dist_kwargs,
    # )

    return dist_output


def plot_distances_vs_range(r_grid_km, dist, root_fig, rmin=None, rmax=None):

    plt.figure()
    for i, d_type in enumerate(dist.keys()):
        # Scale to 0, 1 for comparison
        d = dist[d_type]["d"]
        d = (d - d.min()) / (d.max() - d.min())
        plt.plot(r_grid_km, d, label=dist[d_type]["name"], color=color(i))

    plt.xlabel("Range [km]")
    plt.ylabel("d")
    plt.legend()

    if rmin is not None and rmax is not None:
        plt.xlim([rmin, rmax])
        fname = f"distances_vs_r_zoom_rmin{rmin}_rmax{rmax}.png"

    else:
        fname = "distances_vs_r.png"

    fpath = os.path.join(root_fig, fname)
    plt.savefig(fpath)

    ncols = 4
    nrows = int(np.ceil(len(dist.keys()) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)
    axs = axs.flatten()

    for i, d_type in enumerate(dist.keys()):
        # No scaling
        d = dist[d_type]["d"]
        axs[i].plot(r_grid_km, d, color=color(i))

        # axs[i].set_xlabel("Range [km]")
        axs[i].set_ylabel(f"{dist[d_type]['name']} {dist[d_type]['unit']}")
    # plt.legend()
    fig.supxlabel("Range [km]")

    if rmin is not None and rmax is not None:
        plt.xlim([rmin, rmax])
        fname = f"distances_vs_r_panels_zoom_rmin{rmin}_rmax{rmax}.png"

    else:
        fname = "distances_vs_r_panels.png"

    fpath = os.path.join(root_fig, fname)
    plt.savefig(fpath)


def plot_pi_close_ref_pos(ds, dist, root_fig, save_rtf=False, root_rtf=None):

    # Extract r around r_ref
    r = ds.r.values
    r_minus_rref = r - ds.r_ref
    dr_pos_idx = r_minus_rref >= 0
    r_pos = r[dr_pos_idx]

    # Plot module and phase at close pos
    for i, d_type in enumerate(dist.keys()):

        # theta_thresholds = [20, 30, 40]
        # Get dist
        d = dist[d_type]["d"]
        d_name = dist[d_type]["name"]
        d_unit = dist[d_type]["unit"]

        # Define thresholds as Q1, Q2, Q3 quartiles
        d_th = [np.percentile(d, q=qi) for qi in [0.1, 1, 5]]

        d_r_pos = d[dr_pos_idx]

        lfig = LargeFigure(legend_fontsize=14, size=(16, 10))
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False)
        ax1, ax2, ax3 = axs

        # At ref pos
        rtf_mod_at_th_pos = ds.module.sel(h_index=2).sel(r=ds.r_ref, method="nearest")
        rtf_mod_at_th_pos.plot(ax=ax2, label=f"Ref", color="k")

        rtf_phase_at_th_pos = ds.phase.sel(h_index=2).sel(r=ds.r_ref, method="nearest")
        rtf_phase_at_th_pos.plot(ax=ax3, label=f"Ref", color="k")

        ax1.plot(r, d, color="k")
        ax1.set_ylabel(d_name + f" {d_unit}")

        for i, th in enumerate(d_th):
            idx = np.where(d_r_pos >= th)[0][0]
            r_th = r_pos[idx]

            rtf_mod_at_th_pos = ds.module.sel(h_index=2).sel(r=r_th, method="nearest")
            rtf_mod_at_th_pos.plot(
                ax=ax2, label=f"r = {rtf_mod_at_th_pos.r.values:.1f}m", color=color(i)
            )

            rtf_phase_at_th_pos = ds.phase.sel(h_index=2).sel(r=r_th, method="nearest")
            rtf_phase_at_th_pos.plot(
                ax=ax3, label=f"r = {rtf_mod_at_th_pos.r.values:.1f}m", color=color(i)
            )

            label = (
                d_name
                + f" = {th:.1f}{d_unit[1]} (r = {rtf_mod_at_th_pos.r.values:.1f}m)"
            )
            ax1.scatter(
                r_th,
                d_r_pos[idx],
                color=color(i),
                label=label,
                # label=r"$\theta$ = "
                # + str(th)
                # + f"° (r = {rtf_mod_at_th_pos.r.values:.1f}m)",
            )

        ax2.set_yscale("log")
        ax2.set_ylabel(r"$\lvert \Pi \rvert $")

        ax3.set_ylabel(r"$\Phi$")

        ax1.set_title("")
        ax2.set_title("")
        ax3.set_title("")
        ax1.legend()
        ax2.legend()
        ax3.legend()

        fname = f"rtf_at_selected_pos_{d_type}.png"
        fpath = os.path.join(root_fig, fname)
        plt.savefig(fpath)
        plt.close("all")

        if save_rtf:
            # sel_pos = [ds.r_ref] + [r_pos[np.where(d_r_pos >= th)[0][0]] for th in d_th]
            # ds_sel_pos = ds.sel(r=sel_pos, method="nearest")
            # ds_sel_pos.to_netcdf(fpath)

            # Extract for Abdel 12/04/2026
            data_dict = {
                "f": ds.f.values,
                "rtf_ref_pos": ds.data_ref.sel(h_index=2).values,
            }
            sel_pos = [r_pos[np.where(d_r_pos >= th)[0][0]] for th in d_th]
            for i, s_pos in enumerate(sel_pos):
                ds_s_pos = ds.sel(r=s_pos, method="nearest")
                ds_s_pos_array = ds_s_pos.sel(h_index=2).data.values
                data_dict[f"rtf_pos_{i+1}"] = ds_s_pos_array

            from scipy.io import savemat

            fpath = os.path.join(root_rtf, "rtf_ideal_waveguide_extract_12042026.mat")
            savemat(fpath, data_dict)


def run_full_study(
    bottom_bc="perfectly_rigid",
    waveguide_depth=1000,
    d_inter_rcv=1000,
    z_rcv=999,
    src_ref_range=2 * 1e3,
    z_src=5,
    nrgrid=1001,
    delta_r=1 * 1e3,
    fmin=10,
    fmax=20,
    nfgrid=500,
    root_img=None,
    distances="all",
    save_rtf=False,
    root_data=None,
    n_modes=None,
):
    # Compute grid
    rmin = src_ref_range - delta_r
    rmax = src_ref_range + delta_r
    r_grid = np.linspace(rmin, rmax, nrgrid)
    r_grid_km = r_grid / 1e3
    r_grid1 = r_grid  # Range from source to receiver 1
    r_grid2 = r_grid + d_inter_rcv  # Range from source to receiver 2
    # r_grid2 = np.abs(r_grid - d_inter_rcv)  # Range from source to receiver 2

    f = np.linspace(fmin, fmax, nfgrid)  # Frequency vector

    n_propagating_modes_fmin = nb_propagating_modes(
        f=fmin, c=g.c0, depth=waveguide_depth, bottom_bc=bottom_bc
    )
    print(
        f"Number of propagating modes at fmin = {fmin} Hz : {n_propagating_modes_fmin}"
    )
    n_propagating_modes_fmax = nb_propagating_modes(
        f=fmax, c=g.c0, depth=waveguide_depth, bottom_bc=bottom_bc
    )
    print(
        f"Number of propagating modes at fmax = {fmax} Hz : {n_propagating_modes_fmax}"
    )

    if n_modes is not None:
        print(
            f"Computing RTF using  M = {n_modes} modes (instead of all the propagating modes which are {n_propagating_modes_fmax} at fmax)"
        )

    # Compute RTF dataset
    ds = compute_rtf_dataset(
        f=f,
        r_grid1=r_grid1,
        r_grid2=r_grid2,
        z_s=z_src,
        z_a=z_rcv,
        waveguide_depth=waveguide_depth,
        bottom_bc=bottom_bc,
        r_ref=src_ref_range,
        d12=d_inter_rcv,
        n_modes=n_modes,
    )

    # Define root to store img
    if root_img is not None:
        root_fig = os.path.join(
            root_img,
            f"Dw_{waveguide_depth:.1f}m_fmin_{fmin:.1f}Hz_fmax_{fmax:.1f}Hz",
        )
        os.makedirs(root_fig, exist_ok=True)
    else:
        root_fig = None

    if root_data is not None:
        root_rtf = os.path.join(
            root_data,
            f"Dw_{waveguide_depth:.1f}m_fmin_{fmin:.1f}Hz_fmax_{fmax:.1f}Hz",
        )
        os.makedirs(root_rtf, exist_ok=True)
    else:
        root_rtf = None

    # Plot TL
    plot_tls(ds, fmin=fmin, fmax=fmax, root_fig=root_fig)

    # Plot Pi in f, r plan
    plot_pi_fr_plan(ds, root_fig=root_fig)

    # Plot zoom
    dr_zoom = 50
    plot_pi_fr_plan(
        ds,
        rmin=src_ref_range - dr_zoom,
        rmax=src_ref_range + dr_zoom,
        root_fig=root_fig,
    )

    # Compute theta
    dist = get_dist(ds, distances=distances)

    # Plot theta vs range
    plot_distances_vs_range(r_grid_km, dist, root_fig=root_fig)

    # Plot zoom
    plot_distances_vs_range(
        r_grid_km,
        dist,
        rmin=(src_ref_range - dr_zoom) * 1e-3,
        rmax=(src_ref_range + dr_zoom) * 1e-3,
        root_fig=root_fig,
    )

    plot_pi_close_ref_pos(
        ds, dist=dist, root_fig=root_fig, save_rtf=save_rtf, root_rtf=root_rtf
    )

    # plt.close("all")

    return ds


def study_intercorr(
    bottom_bc="perfectly_rigid",
    waveguide_depth=1000,
    d_inter_rcv=1000,
    z_rcv=999,
    src_ref_range=2 * 1e3,
    z_src=5,
    nrgrid=1001,
    delta_r=1 * 1e3,
    fmin=10,
    fmax=20,
    nfgrid=500,
    root_img=None,
    distances="all",
):
    # Compute grid
    rmin = src_ref_range - delta_r
    rmax = src_ref_range + delta_r
    r_grid = np.linspace(rmin, rmax, nrgrid)
    r_grid_km = r_grid / 1e3
    r_grid1 = r_grid  # Range from source to receiver 1
    r_grid2 = r_grid + d_inter_rcv  # Range from source to receiver 2

    f = np.linspace(fmin, fmax, nfgrid)  # Frequency vector

    n_propagating_modes_fmin = nb_propagating_modes(
        f=fmin, c=g.c0, depth=waveguide_depth, bottom_bc=bottom_bc
    )
    print(
        f"Number of propagating modes at fmin = {fmin} Hz : {n_propagating_modes_fmin}"
    )
    n_propagating_modes_fmax = nb_propagating_modes(
        f=fmax, c=g.c0, depth=waveguide_depth, bottom_bc=bottom_bc
    )
    print(
        f"Number of propagating modes at fmax = {fmax} Hz : {n_propagating_modes_fmax}"
    )

    # Compute RTF dataset
    ds = compute_rtf_dataset(
        f=f,
        r_grid1=r_grid1,
        r_grid2=r_grid2,
        z_s=z_src,
        z_a=z_rcv,
        waveguide_depth=waveguide_depth,
        bottom_bc=bottom_bc,
        r_ref=src_ref_range,
        d12=d_inter_rcv,
    )

    # Define root to store img
    if root_img is not None:
        root_fig = os.path.join(
            root_img,
            f"Dw_{waveguide_depth:.1f}m_fmin_{fmin:.1f}Hz_fmax_{fmax:.1f}Hz",
        )
        os.makedirs(root_fig, exist_ok=True)
    else:
        root_fig = None

    rtf = ds.data.values[..., np.newaxis]  # (nrcv, nf, nr, 1)
    rtf_ref = ds.data_ref.values  # (nrcv, nf)
    rtf_ref_expanded = cast_matrix_to_target_shape(
        rtf_ref, rtf.shape
    )  # (nrcv, nf, nr, 1)

    df = ds.f.values[1] - ds.f.values[0]
    # Compute dist
    x = np.abs(rtf_ref_expanded)
    y = np.abs(rtf)

    # x = x - np.mean(x, axis=1, keepdims=True)
    # y = y - np.mean(y, axis=1, keepdims=True)

    # x_fft = np.fft.fft(x, axis=1)
    # y_fft = np.fft.fft(y, axis=1)
    # s_xy = x_fft * np.conj(y_fft)
    # c_xy = np.fft.ifft(s_xy, axis=1)
    # d_intercorr = np.fft.fftshift(np.real(c_xy), axes=1)

    delta_f = sp.correlation_lags(x.shape[1], y.shape[1], mode="full") * df

    n = x.shape[1]
    nfft = 2 * n - 1  # Zero pad to compute full correlation

    x_fft = np.fft.fft(x, n=nfft, axis=1)
    y_fft = np.fft.fft(y, n=nfft, axis=1)

    c_xy = np.fft.ifft(x_fft * np.conj(y_fft), axis=1)
    c_xy = np.real(c_xy)
    d_intercorr = np.fft.fftshift(c_xy, axes=1)

    # c_xx = np.fft.ifft(x_fft * np.conj(x_fft), axis=1)
    # c_yy = np.fft.ifft(y_fft * np.conj(y_fft), axis=1)
    # c_xx_0 = c_xx[:, 0:1, :, :]
    # c_yy_0 = c_yy[:, 0:1, :, :]
    # c_xy_normalized = c_xy / np.sqrt(c_xx_0 * c_yy_0)
    # d_intercorr = np.fft.fftshift(np.real(c_xy_normalized), axes=1)

    # plt.figure()
    # plt.plot(
    #     delta_f,
    #     d_intercorr[1, :, 10, 0],
    #     label=f"Receiver 2 at r={ds.r.values[10]:.1f}m",
    #     color=color(0),
    # )
    # plt.plot(
    #     delta_f,
    #     d_intercorr[1, :, 9, 0],
    #     label=f"Receiver 2 at r={ds.r.values[9]:.1f}m",
    #     color=color(1),
    # )
    # plt.plot(
    #     delta_f,
    #     d_intercorr[1, :, 11, 0],
    #     label=f"Receiver 2 at r={ds.r.values[11]:.1f}m",
    #     color=color(2),
    # )
    # plt.legend()
    # # d_intercorr_max = np.max(d_intercorr, axis=1)  # Max along delta_f axis
    # # d_intercorr_max = np.sum(d_intercorr_max, axis=0)  # Sum along rcv axis
    # plt.show()

    d_intercorr_mat = d_intercorr[1, :, :, 0]  # (nf, nr)

    plt.figure()
    vmin = np.percentile(d_intercorr_mat, 10)
    vmax = np.percentile(d_intercorr_mat, 95)
    im = plt.pcolormesh(
        delta_f,
        ds.r.values,
        d_intercorr_mat.T,
        vmin=vmin,
        vmax=vmax,
        cmap="jet",
    )
    plt.colorbar(im)
    # plt.legend()
    plt.xlabel(r"$\delta_f$")
    plt.ylabel("Range [m]")
    plt.title("Intercorrelation matrix computed with FFT")

    # Comparaison avec np.correlate sur tous les points
    # d_intercorr_mat_np = np.zeros_like(d_intercorr_mat)
    delta_f = sp.correlation_lags(x.shape[1], y.shape[1], mode="full") * df
    d_intercorr_mat_np = np.zeros((delta_f.shape[0], d_intercorr_mat.shape[1]))
    for i in range(d_intercorr_mat.shape[1]):
        x = ds.data_ref.values
        x = x[1, :]  # Receiver 2
        x = np.abs(x)
        y = ds.data.isel(r=i)  # (nrcv, nf)
        y = y[1, :]  # Receiver 2
        y = np.abs(y)
        # c_xy = np.correlate(x, y, mode="same")
        c_xy = sp.correlate(x, y, mode="full", method="fft")
        d_intercorr_mat_np[:, i] = c_xy

    plt.figure()
    vmin = np.percentile(d_intercorr_mat_np, 10)
    vmax = np.percentile(d_intercorr_mat_np, 95)
    im = plt.pcolormesh(
        delta_f,
        ds.r.values,
        d_intercorr_mat_np.T,
        vmin=vmin,
        vmax=vmax,
        cmap="jet",
    )
    plt.colorbar(im)
    plt.xlabel(r"$\delta_f$")
    plt.ylabel("Range [m]")
    plt.title("Intercorrelation matrix computed with np.correlate")

    # Comparaison avec np.correlate sur un point
    plt.figure()
    # A la main avec des fft
    plt.plot(
        delta_f,
        d_intercorr_mat[:, 108],
        label=f"Calcul FFT (r = {ds.r.values[108]:.1f}m)",
        color=color(0),
    )

    # Avec np correlate
    x = ds.data_ref.values  # (nrcv, nf)
    x = x[1, :]  # Receiver 2
    x = np.abs(x)

    y = ds.data.sel(r=2016, method="nearest")  # (nrcv, nf)
    r_plot = y.r.values
    y = y.values[1, :]  # Receiver 2
    y = np.abs(y)
    c_xy = sp.correlate(x, y, mode="same")
    delta_f = sp.correlation_lags(x.shape[0], y.shape[0], mode="same") * df
    plt.plot(
        delta_f,
        np.abs(c_xy),
        label=f"Fonction np.correlate (r = {r_plot:.1f}m)",
        color=color(1),
    )
    plt.xlabel(r"$\delta_f$")
    plt.ylabel(r"$ \lvert C_{xy} \rvert $")
    plt.legend()
    plt.show()

    # # Extract for Abdel 15/04/2026
    # from scipy.io import savemat

    # data_dict = {
    #     "f": ds.f.values,
    #     "r": ds.r.values,
    #     "rtf_ref_pos": ds.data_ref.sel(h_index=2).values,
    #     "rtf_mat": ds.data.sel(h_index=2).values,
    # }

    # root_rtf = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data\rtf_mfp\sensibility\ideal_wg\Dw_30.0m_fmin_100.0Hz_fmax_200.0Hz"
    # fpath = os.path.join(root_rtf, "rtf_ideal_waveguide_extract_15042026.mat")
    # savemat(fpath, data_dict)


if __name__ == "__main__":
    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\img\rtf_mfp\sensibility\ideal_wg"
    root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data\rtf_mfp\sensibility\ideal_wg"
    # distances = [
    #     "hermitian_angle",
    #     "hermitian_angle_module",
    #     "norm_L1",
    #     "norm_L1_module",
    #     "norm_L2",
    #     "norm_L2_module",
    # ]
    # distances = "all"

    distances = [
        "hermitian_angle",
        "hermitian_angle_module",
    ]
    # Shallow water, very high frequency
    study_intercorr(
        bottom_bc="perfectly_rigid",
        waveguide_depth=30,
        d_inter_rcv=1000,
        z_rcv=29,
        src_ref_range=2 * 1e3,
        z_src=5,
        nrgrid=201,
        delta_r=0.2 * 1e3,
        fmin=100,
        fmax=200,
        nfgrid=500,
        root_img=root_img,
        distances=distances,
    )
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=30,
    #     d_inter_rcv=1000,
    #     z_rcv=29,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=0.5 * 1e3,
    #     fmin=100,
    #     fmax=200,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )

    # # Deep water, ultra low frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=1000,
    #     d_inter_rcv=1000,
    #     z_rcv=999,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=1,
    #     fmax=10,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )

    # # Deep water, low frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=1000,
    #     d_inter_rcv=1000,
    #     z_rcv=999,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=10,
    #     fmax=20,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )

    # # Deep water, high frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=1000,
    #     d_inter_rcv=1000,
    #     z_rcv=999,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=100,
    #     fmax=110,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )

    # # Shallow water, low frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=30,
    #     d_inter_rcv=1000,
    #     z_rcv=29,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=10,
    #     fmax=20,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )

    # # Shallow water, high frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=30,
    #     d_inter_rcv=1000,
    #     z_rcv=29,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=100,
    #     fmax=110,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )

    # # Shallow water, very high frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=30,
    #     d_inter_rcv=1000,
    #     z_rcv=29,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=100,
    #     fmax=200,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    #     save_rtf=False,
    # )

    # # Shallow water, ultra high frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=30,
    #     d_inter_rcv=1000,
    #     z_rcv=29,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=400,
    #     fmax=800,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )

    # # Shallow water, very high frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=30,
    #     d_inter_rcv=1000,
    #     z_rcv=29,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=145,
    #     fmax=155,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )

    # # Shallow water, very high frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=30,
    #     d_inter_rcv=1000,
    #     z_rcv=29,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=140,
    #     fmax=160,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )

    # # Shallow water, very high frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=30,
    #     d_inter_rcv=1000,
    #     z_rcv=29,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=130,
    #     fmax=170,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )

    # # Shallow water, very high frequency
    # run_full_study(
    #     bottom_bc="perfectly_rigid",
    #     waveguide_depth=30,
    #     d_inter_rcv=1000,
    #     z_rcv=29,
    #     src_ref_range=2 * 1e3,
    #     z_src=5,
    #     nrgrid=5001,
    #     delta_r=1 * 1e3,
    #     fmin=110,
    #     fmax=190,
    #     nfgrid=500,
    #     root_img=root_img,
    #     distances=distances,
    # )
