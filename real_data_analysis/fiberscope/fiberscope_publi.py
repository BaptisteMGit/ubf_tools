#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   Untitled-1
@Time    :   2025/02/27 16:08:50
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from real_data_analysis.fiberscope.fiberscope_utils import *
from publication.PublicationFigure import PubFigure
from propa.rtf.rtf_utils import normalize_metric_contrast
from matplotlib.animation import FuncAnimation, PillowWriter

PubFigure()

# Distance from P1 in meters for each position
dict_th_pos = {
    "P1": 0,
    "P2": 10,
    "P3": 20,
    "P4": 25,
    "P5": 15,
    "P6": 5,
}

recording_names_N1 = [
    "09-10-2024T10-34-58-394627_P1_N1_Sweep_34",
    "09-10-2024T16-51-22-900122_P2_N1_Sweep_93",
    "10-10-2024T09-43-06-620681_P3_N1_Sweep_151",
    "10-10-2024T12-03-02-201689_P4_N1_Sweep_211",
    "10-10-2024T14-42-01-833325_P5_N2_Sweep_267",  # Name N2 but actually N1
    "10-10-2024T15-54-25-737795_P6_N1_Sweep_323",
    # "11-10-2024T10-51-56-563968_P3_N1_Sweep_385",  # P7
    # "11-10-2024T12-08-20-091131_P1_N1_Sweep_437",  # P8
]

recording_names_N3 = [
    "09-10-2024T10-37-04-088817_P1_N3_Sweep_36",
    "09-10-2024T16-53-16-681510_P2_N3_Sweep_95",
    "10-10-2024T09-45-50-516056_P3_N3_Sweep_153",
    "10-10-2024T12-04-46-610661_P4_N3_Sweep_213",
    "10-10-2024T14-43-47-603375_P5_N3_Sweep_269",
    "10-10-2024T15-56-16-837150_P6_N3_Sweep_325",
]

recording_names_N5 = [
    "09-10-2024T10-39-11-308093_P1_N5_Sweep_38",
    "09-10-2024T16-55-08-243011_P2_N5_Sweep_97",
    "10-10-2024T09-47-33-438942_P3_N5_Sweep_155",
    "10-10-2024T12-06-31-200643_P4_N5_Sweep_215",
    "10-10-2024T14-45-21-047719_P5_N5_Sweep_271",
    "10-10-2024T15-57-57-549910_P6_N5_Sweep_327",
]

t_interp_pulse = 1  # Inter sweep period
t_pulse = 100 * 1e-3  # Single sweep duration
t_ir = 1  # Approximated impulse response duration (simple value to ensure no energy is received after this time)
n_sweep = 10  # Number of sweep emitted
f0 = 8e3  # Start frequency
f1 = 15e3  # End frequency

recording_props = {
    "t_interp_pulse": t_interp_pulse,
    "t_pulse": t_pulse,
    "t_ir": t_ir,
    "n_em": n_sweep,
    "f0": f0,
    "f1": f1,
}

processing_props = {
    "hydro_to_process": None,
    "ref_hydro": 1,
    "method": "cs",
    "alpha_th": 0.001 * 1e-2,
    "split_method": "band_energy",
}


sweep_1 = {
    "recording_props": recording_props,
    "processing_props": processing_props,
    "recording_names": {
        "N1": recording_names_N1,
        "N3": recording_names_N3,
        "N5": recording_names_N5,
    },
}

t_interp_pulse = 1  # Inter sweep period
t_pulse = 100 * 1e-3  # Single sweep duration
t_ir = 1  # Approximated impulse response duration (simple value to ensure no energy is received after this time)
n_sweep = 10  # Number of sweep emitted
f0 = 10e3  # Start frequency
f1 = 13e3  # End frequency

recording_props = {
    "t_interp_pulse": t_interp_pulse,
    "t_pulse": t_pulse,
    "t_ir": t_ir,
    "n_em": n_sweep,
    "f0": f0,
    "f1": f1,
}

processing_props = {
    "hydro_to_process": None,
    "ref_hydro": 1,
    "method": "cs",
    "alpha_th": 0.001 * 1e-2,
    "split_method": "band_energy",
}

sweep_2 = {
    "recording_props": recording_props,
    "processing_props": processing_props,
}


# # Source localisation using dynamic source library
# Recording from the moving source : speed = 0.1 m/s
dynamic_recording = "10-10-2024T16-53-43-200271_PR_N1_346"

recording_props = sweep_2["recording_props"]
recording_props["n_em"] = 10
recording_props["src_end_pos"] = "P4"
recording_props["src_start_pos"] = "P1"
recording_props["dynamic_recording_name"] = dynamic_recording

n_sweep = 3
processing_props = sweep_2["processing_props"]
processing_props["time_step"] = n_sweep * sweep_2["recording_props"]["t_interp_pulse"]

## Analyse dynamic recording
# ### Split recording into shorter sections to be analysed

# Unpack usefull props
src_speed = recording_props.get("src_speed", 0.1)  # Source speed in m/s
src_start_pos = recording_props.get("src_start_pos", "P1")  # Source start position id
dynamic_recording_name = recording_props.get("dynamic_recording_name", "")
src_end_pos = recording_props.get("src_end_pos", "P4")  # Source end position id

time_step = processing_props.get(
    "time_step", 10
)  # Time step to use to devide the recording into

n_records = 90
displacement_from_start_pos = [
    ((i + 1) * time_step - time_step / 2) * src_speed for i in range(n_records)
]
recording_names_dynamic = [
    f"{dynamic_recording_name}_{src_start_pos}_r{np.round(dr, 2)}m_{src_end_pos}"
    for dr in displacement_from_start_pos
]


# ### Run analysis on each section (needs to be run only once)


def process_dyn_analysis(recording_names_dynamic):
    for recording_name in recording_names_dynamic:
        run_analysis(
            recording_name,
            recording_props,
            processing_props,
            plot_rtf_estimation=True,
            verbose=True,
        )


# ## Localise


def process_dyn_loc(records_to_loc, gcc_methods=["scot"], run_a=False):
    dict_loc_dynamic = {}
    for recording_name_to_loc in records_to_loc:

        if run_a:
            run_analysis(
                recording_name_to_loc,
                recording_props,
                processing_props,
                plot_rtf_estimation=True,
                gcc_methods=gcc_methods,
                verbose=True,
            )

        # Localizing static records
        pos_id = recording_name_to_loc.split("_")[1]
        th_pos = dict_th_pos[pos_id]
        # Localizing section from dynamic record
        # th_pos = float(recording_name_to_loc.split("_")[-2][1:-1])

        range_from_p1, dist, _, dist_gcc = localise(
            recording_names=recording_names_dynamic,
            recording_name_to_loc=recording_name_to_loc,
            recording_props=recording_props,
            processing_props=processing_props,
            th_pos=th_pos,
            gcc_methods=gcc_methods,
        )

        d_gcc = dist_gcc["mean_dist"]["scot"]

        r_hat = range_from_p1[np.argmin(dist)]
        dict_loc_dynamic[recording_name_to_loc] = {
            "d_rtf": dist,
            "d_gcc": d_gcc,
            "r": range_from_p1,
            "r_th": th_pos,
            "r_hat": r_hat,
        }

    return dict_loc_dynamic


# ## Trajectography


def plot_theta(theta):
    plt.figure()
    plt.imshow(theta, aspect="auto", cmap="jet_r")
    plt.colorbar(
        label=r"$\theta \, \textrm{[°]}$"
    )  # = \frac{\theta_{max} - \theta}{\theta_{max} - \theta_{min}}
    # plt.scatter(t_th, np.arange(len(t_th)), c="r", s=250, marker="x")
    plt.gca().invert_yaxis()  # Reverse the y axis
    plt.xticks(np.arange(0, theta.shape[1], 10), np.round(t[::10], 2))
    plt.yticks(np.arange(0, theta.shape[0]), ordered_pos)
    # plt.pcolormesh(t, p, loc_mat, shading="nearest", cmap="jet")
    plt.xlabel(r"$\textrm{Time [s]}$")
    plt.ylabel(r"$\textrm{Position}$")
    plt.title(r"$\textrm{" + f"{ordered_records[0].split('_')[-3]}" + r"}$")
    # Save the figure
    # plt.savefig(os.path.join(img_path, f"localisation_matrix_{em_lvl_to_loc}_theta.png"))
    plt.savefig(f"localisation_matrix_{em_lvl_to_loc}_theta.png")


# plt.show()


def plot_dyn_q(q, em_lvl_to_loc):
    plt.figure()
    plt.imshow(q, aspect="auto", cmap="jet")
    plt.colorbar(label=r"$q$")
    plt.gca().invert_yaxis()  # Reverse the y axis
    plt.xticks(np.arange(0, q.shape[1], 10), np.round(t[::10], 2))
    plt.yticks(np.arange(0, q.shape[0]), ordered_pos)
    plt.xlabel(r"$\textrm{Time [s]}$")
    plt.ylabel(r"$\textrm{Position}$")
    plt.title(r"$\textrm{" + f"{em_lvl_to_loc}" + r"}$")

    # Save the figure
    # plt.savefig(os.path.join(img_path, f"localisation_matrix_{em_lvl_to_loc}_q.png"))
    plt.savefig(f"localisation_matrix_{em_lvl_to_loc}_q.png")


def plot_dyn_q_dB(q_dB, em_lvl_to_loc, method, pos_names):

    plt.figure()
    plt.imshow(q_dB, aspect="auto", cmap="jet", vmin=-8, rasterized=False)
    plt.colorbar(label=r"$q\, \textrm{[dB]}$")
    plt.gca().invert_yaxis()  # Reverse the y axis
    plt.xticks(np.arange(0, q_dB.shape[1], 10), np.round(t[::10], 2))
    plt.yticks(np.arange(0, q_dB.shape[0]), pos_names)
    plt.xlabel(r"$\textrm{Time [s]}$")
    plt.ylabel(r"$\textrm{Position}$")
    # plt.title(r"$\textrm{" + f"{em_lvl_to_loc}" + r"}$")
    # Save the figure
    # plt.savefig(os.path.join(img_path, f"localisation_matrix_{em_lvl_to_loc}_q.png"))
    fname = f"localisation_matrix_{em_lvl_to_loc}_{method}_q_dB"
    plt.savefig(os.path.join(ROOT_IMG_PUBLI, f"{fname}.png"), dpi=300)
    plt.savefig(os.path.join(ROOT_IMG_PUBLI, f"{fname}.eps"), dpi=300)
    # plt.savefig(f"localisation_matrix_{em_lvl_to_loc}_{method}_q_dB.eps", dpi=300)


def msr_time_dyn(t, q_dB, method):
    # MSR for each time
    main_lobe_idx = np.argmax(q_dB, axis=0)
    main_lobe_mask = np.array(
        [np.arange(q_dB.shape[0]) == main_lobe_idx[i] for i in range(q_dB.shape[1])]
    ).T
    main_lobe = np.max(q_dB, axis=0)
    q_masked = np.array(
        [q_dB[:, i][~main_lobe_mask[:, i]] for i in range(q_dB.shape[1])]
    ).T
    side_lobe = np.max(q_masked, axis=0)

    msr = -(main_lobe - side_lobe)  # MSR = mainlobe_dB - side_lobe_dB

    plt.figure()
    plt.plot(t, msr)
    plt.xlabel(r"Time [s]")
    plt.ylabel(r"MSR [dB]")

    # Find peaks corresponding to
    # Annotate each position
    fname = f"msr_t_{em_lvl_to_loc}_{method}"
    plt.savefig(os.path.join(ROOT_IMG_PUBLI, f"{fname}.png"), dpi=300)
    plt.savefig(os.path.join(ROOT_IMG_PUBLI, f"{fname}.eps"), dpi=300)

    return t, msr


def msr_position_dyn(pos_names, q_dB, method):
    # MSR for each position
    main_lobe_idx = np.argmax(q_dB, axis=1)
    main_lobe_mask = np.array(
        [np.arange(q_dB.shape[1]) == main_lobe_idx[i] for i in range(q_dB.shape[0])]
    )
    main_lobe = np.max(q_dB, axis=1)
    q_masked = np.array([q_dB[i][~main_lobe_mask[i]] for i in range(q_dB.shape[0])])
    side_lobe = np.max(q_masked, axis=1)

    msr_pos = -(main_lobe - side_lobe)  # MSR = mainlobe_dB - side_lobe_dB

    plt.figure()
    plt.plot(msr_pos)
    plt.xlabel(r"Position")
    plt.ylabel(r"MSR [dB]")
    plt.xticks(np.arange(0, q.shape[0]), pos_names)

    fname = f"msr_pos_{em_lvl_to_loc}_{method}"
    plt.savefig(os.path.join(ROOT_IMG_PUBLI, f"{fname}.png"), dpi=300)
    plt.savefig(os.path.join(ROOT_IMG_PUBLI, f"{fname}.eps"), dpi=300)

    return pos_names, msr_pos
    # plt.savefig(f"msr_{em_lvl_to_loc}_{method}.png")
    # plt.savefig(f"msr_{em_lvl_to_loc}_{method}.eps", dpi=300)


def plot_dyn_loc_probability(mu, em_lvl_to_loc, method):
    # Check sum equal one for each time step
    print(np.sum(mu, axis=0))

    plt.figure()
    plt.imshow(mu, aspect="auto", cmap="jet", vmax=0.5)
    plt.colorbar(
        label=r"$\mu$"
    )  # = \frac{\theta_{max} - \theta}{\theta_{max} - \theta_{min}}
    # plt.scatter(t_th, np.arange(len(t_th)), c="r", s=250, marker="x")
    plt.gca().invert_yaxis()  # Reverse the y axis
    plt.xticks(np.arange(0, mu.shape[1], 10), np.round(t[::10], 2))
    plt.yticks(np.arange(0, mu.shape[0]), ordered_pos)
    # plt.pcolormesh(t, p, loc_mat, shading="nearest", cmap="jet")
    plt.xlabel(r"$\textrm{Time [s]}$")
    plt.ylabel(r"$\textrm{Position}$")
    plt.title(r"$\textrm{" + f"{em_lvl_to_loc}" + r"}$")
    # Save the figure
    # plt.savefig(os.path.join(img_path, f"localisation_matrix_{em_lvl_to_loc}_q.png"))
    fname = f"localisation_matrix_{em_lvl_to_loc}_{method}_mu"
    plt.savefig(os.path.join(ROOT_IMG_PUBLI, f"{fname}.png"), dpi=300)


def plot_anim_fiberscope_results(d):
    time_step
    pubfig = PubFigure(label_fontsize=55)

    # Paramètres du graphe
    speed_factor = 15
    fig, ax = plt.subplots()
    # x_labels = ["P1", "P2", "P3", "P4", "P5", "P6"]
    # x_labels = [f"${ordered_pos[i][0]}_{ordered_pos[i][1]}$" for i in range(len(ordered_pos))]
    x_labels = [f"$P_{i+1}$" for i in range(len(ordered_pos))]

    bar_width = 0.5
    bars = ax.bar(x_labels, d[:, 0], width=bar_width, color=plt.cm.hot(d[:, 0]))

    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$q$")
    # ax.set_title("Évolution des probabilités au cours du temps")

    nt = d.shape[1]

    # Mise à jour du graphe à chaque frame
    def update(frame):
        heights = d[:, frame]
        for i, bar in enumerate(bars):
            bar.set_height(heights[i])
            # bar.set_color(plt.cm.jet(heights[i]))  # Mise à jour de la couleur
            bar.set_color(plt.cm.binary(heights[i]))  # Mise à jour de la couleur

        ax.set_title(f"t = {frame}")

    # # Animation
    ani = FuncAnimation(fig, update, frames=nt, interval=100)

    # writer = FFMpegWriter(fps=10, metadata=dict(artist="Baptiste"), bitrate=1800)
    # ani.save("evolution_probabilites.mp4", writer=writer)

    # Sauvegarde de l'animation en GIF
    fps_video = 24
    nf_video = 408
    t_video = nf_video / fps_video
    fps = nt / t_video
    writer = PillowWriter(fps=fps)

    # writer = PillowWriter(fps=1/time_step*speed_factor)
    fname = f"evolution_probabilites.gif"
    ani.save(os.path.join(ROOT_IMG_PUBLI, fname), writer=writer)

    plt.show()


def plot_anim_fiberscope_experiment(nt):
    """Animation du dispositif expérimental"""

    # Initialisation de la figure
    nframes = nt
    # nframes = 10
    fig, ax = plt.subplots()
    ax.set_xlim(0, 11)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$z$")

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Dessin des interfaces surface et fond
    ax.plot([0, 11], [4, 4], color="black", linewidth=2)  # Surface
    ax.plot(
        [0, 8.5, 8.5, 10.5, 10.5, 11],
        [0, 0, -4, -4, 0, 0.5],
        color="black",
        linewidth=2,
    )  # Fond

    # Coloration du fond
    ax.fill_between(
        [0, 8.5, 8.5, 10.5, 10.5, 11],
        -5,
        [0, 0, -4, -4, 0, 0.5],
        color="rosybrown",
        alpha=0.5,
    )

    # Dessin de la poutre (rectangle gris)
    ax.fill_between([0.5, 5.5], 0, 0.49, color="gray", alpha=0.3)
    ax.plot([0.5, 0.5, 5.5, 5.5], [0, 0.49, 0.49, 0], color="black")

    # Positions des hydrophones H1 à H4 (sur la poutre)
    h_pos = [1.5, 2.5, 3.5, 4.5]
    for i, x in enumerate(h_pos):
        ax.plot(x, 0.5, "ko")
        ax.text(x, 0.7, f"$H_{{{i+1}}}$", ha="right", fontsize=18)

    # Hydrophone H5 (au-dessus de la poutre)
    ax.plot(1.5, 2.5, "ko")
    ax.text(1.5, 2.7, "$H_{5}$", ha="right", fontsize=18)

    # Positions initiales des sources P1 à P6
    p_r = np.array([4, 5, 6, 7, 8, 9])
    p_z = np.full_like(p_r, 2.5)
    ax.scatter(p_r, p_z, color="r", s=70, alpha=0.4)  # Positions fixes des sources

    # Légendes des positions P1 à P6
    for i, (x, y) in enumerate(zip(p_r, p_z)):
        # ax.text(x, y + 0.2, f"${ordered_pos[i][0]}_{ordered_pos[i][1]}$", ha="right", fontsize=18)
        ax.text(x, y + 0.2, f"$P_{i+1}$", ha="right", fontsize=18)

    # Dessin de la plateforme mobile (rectangle gris)
    platform_r = [3.65, 4.65]
    platform_z = [4, 4.8]
    # Plateforme
    platform = ax.plot(
        [platform_r[0], platform_r[0], platform_r[1], platform_r[1]],
        [platform_z[0], platform_z[1], platform_z[1], platform_z[0]],
        color="black",
    )[0]
    ax.fill_between(
        [platform_r[0], platform_r[1]],
        [platform_z[0], platform_z[0]],
        [platform_z[1], platform_z[1]],
        color="gray",
        alpha=0.3,
    )

    # Roue
    wheal_r = platform_r[0] + 0.5
    wheal_z = platform_z[0] + 0.2
    platform_wheal = ax.plot(
        wheal_r,
        wheal_z,
        "ko",
        markersize=15,
    )[
        0
    ]  # facecolors=None, edgecolors="k", s=200

    # Support source
    source_vline = ax.plot(
        [platform_r[0] + 0.5, platform_r[0] + 0.5], [p_z[0], 4], "k"
    )[0]
    source_hline = ax.plot([p_r[0], platform_r[0] + 0.5], [p_z[0], p_z[0]], "k")[0]

    # Cercle mobile représentant la source active
    source_active = ax.plot(p_r[0], p_z[0], "ro", markersize=7)[0]

    src_dx = (p_r[-1] - p_r[0] + 0.5) / nframes

    # Fonction de mise à jour de l'animation
    def update(frame):
        # Update de la position de la source
        new_x = platform_r[0] + src_dx * frame
        new_x_src = p_r[0] + src_dx * frame
        source_vline.set_xdata(
            [new_x + 0.5, new_x + 0.5]
        )  # Deux points pour la ligne verticale
        source_hline.set_xdata(
            [new_x_src, new_x + 0.5]
        )  # Deux points pour la ligne horizontale
        source_active.set_xdata([new_x_src])  # Source

        # Update de la position de la plateforme
        platform.set_xdata([new_x, new_x, new_x + 1, new_x + 1])
        ax.fill_between(
            [new_x, new_x + 1],
            [platform_z[0], platform_z[0]],
            [platform_z[1], platform_z[1]],
            color="gray",
            alpha=0.3,
        )

        platform_wheal.set_xdata([wheal_r + src_dx * frame])

        ax.set_title(f"t = {frame}")

    # Animation (déplacement de la source)
    ani = FuncAnimation(fig, update, frames=nframes, interval=100)

    # Sauvegarde de l'animation en GIF
    # writer = PillowWriter(fps=1 / time_step * speed_factor)
    fps_video = 24
    nf_video = 408
    t_video = nf_video / fps_video
    fps = nt / t_video
    writer = PillowWriter(fps=fps)

    fname = f"schema_plateforme_dynamique.gif"
    ani.save(os.path.join(ROOT_IMG_PUBLI, fname), writer=writer)

    # plt.show()


if __name__ == "__main__":

    ROOT_IMG_PUBLI = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\imgs"

    # records_to_loc = ["10-10-2024T16-53-43-200271_PR_N1_346_P1_r20.25m_P4"]
    em_lvl_to_loc = "N5"
    records_to_loc = sweep_1["recording_names"][em_lvl_to_loc]
    gcc_methods = ["scot"]

    # Process dynamic recording
    dict_loc_dynamic = process_dyn_loc(records_to_loc, gcc_methods, run_a=True)

    # Build matrix containing the localization results
    for loc_method in ["d_gcc", "d_rtf"]:
        method = loc_method[2:]
        ordered_pos, ordered_records = re_order_recordings(
            list(dict_loc_dynamic.keys())
        )
        d = np.array([dict_loc_dynamic[key][loc_method] for key in ordered_records])
        if method == "rtf":
            d *= -1

        x_th = [dict_th_pos[key] for key in ordered_pos]
        t_th = np.array(list(x_th)) / src_speed / time_step
        # Time vector
        t = np.arange(0, d.shape[1]) * processing_props["time_step"]
        p = list(dict_th_pos.keys())

        axis_norm = None
        if axis_norm is None:
            d_max = np.max(d, axis=axis_norm) * np.ones_like(d)
            d_min = np.min(d, axis=axis_norm) * np.ones_like(d)
            norm_label = "norm_over_entire_surface"
        else:
            d_max = np.tile(
                np.max(d, axis=axis_norm), (d.shape[axis_norm], 1)
            )  # Cast to d shape
            d_min = np.tile(np.min(d, axis=axis_norm), (d.shape[axis_norm], 1))
            if axis_norm == 1:
                norm_label = f"norm_along_time_axis"
            elif axis_norm == 0:
                norm_label = f"norm_along_position_axis"

        if axis_norm == 1:
            d_max = d_max.T
            d_min = d_min.T

        # Normalized distance for each time step
        q = (d - d_min) / (d_max - d_min)

        # In dB
        q[q == 0] = 1e-6
        q_dB = 10 * np.log10(q)

        # Build proba measure
        norm = np.tile(np.sum(q, axis=0), (d.shape[0], 1))
        mu = q / norm

        # Selection publi
        method = f"{method}_{norm_label}"
        # pos_names = ordered_pos     # Real order p1 p6 p2 etc
        pos_names = [f"$P_{i}$" for i in range(1, 7)]  # Easier to understand order
        plot_dyn_q_dB(
            q_dB, em_lvl_to_loc=em_lvl_to_loc, method=method, pos_names=pos_names
        )

        t, msr_t = msr_time_dyn(t, q_dB, method=method)
        p_names, msr_p = msr_position_dyn(pos_names, q_dB, method=method)

        # Save msr
        root_publi_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope\publi_data"

        # Save msr_t to .dat
        data = np.c_[t, msr_t]
        fpath = os.path.join(root_publi_data, f"msr_time_{method}.dat")
        np.savetxt(fname=fpath, X=data, fmt="%.3f %.3f", header="t msr", comments="")

        # Save msr_pos to .dat
        pos_idx = np.arange(1, 7)
        data = np.c_[pos_idx, msr_p]
        fpath = os.path.join(root_publi_data, f"msr_pos_{method}.dat")
        np.savetxt(
            fname=fpath, X=data, fmt="%.0f %.3f", header="pos_idx msr", comments=""
        )

    # plot_dyn_q(
    #     q, em_lvl_to_loc=em_lvl_to_loc
    # )
    # plot_dyn_loc_probability(mu, em_lvl_to_loc=em_lvl_to_loc)

    # # Plot results animations
    # plot_anim_fiberscope_results(mu)
    # plot_anim_fiberscope_experiment(nt=mu.shape[1])
