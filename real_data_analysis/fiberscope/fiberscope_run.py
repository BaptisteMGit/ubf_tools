#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   fiberscope_run.py
@Time    :   2024/11/18 11:43:18
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
from real_data_analysis.fiberscope.fiberscope_utils import *

from publication.PublicationFigure import PubFigure

PubFigure()

# ======================================================================================================================
# Test 1 : Sweep 8 - 15 kHz
# Sweep length = 100 ms
# Inter sweep period = 1s
# Number of sweep = 10
# ======================================================================================================================
# Define general parameters
t_interp_pulse = 1  # Inter sweep period
t_pulse = 100 * 1e-3  # Single sweep duration
t_ir = 1  # Approximated impulse response duration (simple value to ensure no energy is received after this time)
n_sweep = 10  # Number of sweep emitted
f0 = 8e3  # Start frequency
f1 = 15e3  # End frequency

recording_props_1 = {
    "t_interp_pulse": t_interp_pulse,
    "t_pulse": t_pulse,
    "t_ir": t_ir,
    "n_em": n_sweep,
    "f0": f0,
    "f1": f1,
}

processing_props_1 = {
    "hydro_to_process": None,
    "ref_hydro": 1,
    "method": "cs",
    "alpha_th": 0.001 * 1e-2,
    "split_method": "band_energy",
}

recording_name_1 = "09-10-2024T10-34-58-394627_P1_N1_Sweep_34"

# process_recording(recording_name_1, recording_props_1, processing_props_1)
# derive_rtf(recording_name_1, recording_props_1, processing_props_1)

# ======================================================================================================================
# Test 2 : Sweep 2 - 20 kHz
# Sweep length = 800 ms
# Inter sweep period = 2s
# Number of sweep = 10
# ======================================================================================================================

t_interp_pulse = 2  # Inter sweep period
t_pulse = 800 * 1e-3  # Single sweep duration
f0 = 2e3  # Start frequency
f1 = 20e3  # End frequency
t_ir = 1

recording_props_2 = {
    "t_interp_pulse": t_interp_pulse,
    "t_pulse": t_pulse,
    "t_ir": t_ir,
    "n_em": n_sweep,
    "f0": f0,
    "f1": f1,
}

processing_props_2 = {
    "hydro_to_process": None,
    "ref_hydro": 1,
    "method": "cs",
    "alpha_th": 0.01 * 1e-2,
    "split_method": "band_energy",
}

recording_name_2 = "09-10-2024T11-03-11-806485_P1_N1_Sweep_49"

# process_recording(recording_name_2, recording_props_2, processing_props_2)
# derive_rtf(recording_name_2, recording_props_2, processing_props_2)

# ======================================================================================================================
# Run analysis for a series of positions
# ======================================================================================================================
t_interp_pulse = 1  # Inter sweep period
t_pulse = 100 * 1e-3  # Single sweep duration
t_ir = 1  # Approximated impulse response duration (simple value to ensure no energy is received after this time)
# n_sweep = 10  # Number of sweep emitted
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
    # "method": "both",
    "alpha_th": 0.001 * 1e-2,
    "split_method": "band_energy",
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

# for recording_name in recording_names_N1:
#     run_analysis(
#         recording_name, recording_props, processing_props, plot_rtf_estimation=True
#     )

# recording_name_to_loc = "10-10-2024T12-04-46-610661_P4_N3_Sweep_213"
# recording_name_to_loc = "09-10-2024T11-03-11-806485_P1_N1_Sweep_49"
# recording_name_to_loc = (
#     "11-10-2024T10-51-56-563968_P3_N1_Sweep_385"  # Position P7, chirp1
# )
# recording_name_to_loc = "11-10-2024T12-08-20-091131_P1_N1_Sweep_437"  # P8, chirp1
# recording_name_to_loc = "11-10-2024T15-31-53-479056_P3_N1_Sweep_472"  # P11 chirp1

recording_name_to_loc = "10-10-2024T09-45-50-516056_P3_N3_Sweep_153"
# recording_name = "09-10-2024T10-34-58-394627_P1_N1_Sweep_34"
# run_analysis(
#     recording_name_to_loc, recording_props, processing_props, plot_rtf_estimation=True
# )

# localise(
#     recording_names=recording_names_N1,
#     recording_name_to_loc=recording_name_to_loc,
#     recording_props=recording_props,
#     processing_props=processing_props,
#     # pos_ids=["P1", "P2", "P3", "P4", "P5", "P6", "P8"],
# )

### Same batch of test with lower snr ###

# N3
recording_names_N3 = [
    "09-10-2024T10-37-04-088817_P1_N3_Sweep_36",
    "09-10-2024T16-53-16-681510_P2_N3_Sweep_95",
    "10-10-2024T09-45-50-516056_P3_N3_Sweep_153",
    "10-10-2024T12-04-46-610661_P4_N3_Sweep_213",
    "10-10-2024T14-43-47-603375_P5_N3_Sweep_269",
    "10-10-2024T15-56-16-837150_P6_N3_Sweep_325",
]

# for recording_name in recording_names_N3:
#     run_analysis(recording_name, recording_props, processing_props)

# recording_name_to_loc = "10-10-2024T12-04-46-610661_P4_N3_Sweep_213"
# recording_name_to_loc = "09-10-2024T11-03-11-806485_P1_N1_Sweep_49"


# localise(
#     recording_names=recording_names_N3,
#     recording_name_to_loc=recording_name_to_loc,
#     recording_props=recording_props,
#     processing_props=processing_props
# )

# N5
recording_names_N5 = [
    "09-10-2024T10-39-11-308093_P1_N5_Sweep_38",
    # "09-10-2024T11-09-10-352993_P1_N5_Sweep_53",
    "09-10-2024T16-55-08-243011_P2_N5_Sweep_97",
    "10-10-2024T09-47-33-438942_P3_N5_Sweep_155",
    "10-10-2024T12-06-31-200643_P4_N5_Sweep_215",
    "10-10-2024T14-45-21-047719_P5_N5_Sweep_271",
    "10-10-2024T15-57-57-549910_P6_N5_Sweep_327",
]

# for recording_name in recording_names_N5:
#     run_analysis(recording_name, recording_props, processing_props)

# recording_name_to_loc = "10-10-2024T12-03-02-201689_P4_N1_Sweep_211"
# recording_name_to_loc = "09-10-2024T11-09-10-352993_P1_N5_Sweep_53"
# run_analysis(recording_name_to_loc, recording_props_2, processing_props_2)
# recording_name_to_loc = "09-10-2024T11-03-11-806485_P1_N1_Sweep_49"

# localise(
#     recording_names=recording_names_N5,
#     recording_name_to_loc=recording_name_to_loc,
#     recording_props=recording_props,
#     processing_props=processing_props
# )


# Load and plot dynamic recording

recording_name = "10-10-2024T16-53-43-200271_PR_N1_346"
# recording_name = "10-10-2024T17-13-59-861860_PR_N1_Sweep_350"
# recording_name = "10-10-2024T17-05-55-728169_PR_bruit_348"

date = recording_name.split("T")[0]
data_path = os.path.join(data_root, f"Campagne_{date}")
file_name = f"{recording_name}.tdms"
file_path = os.path.join(data_path, file_name)

img_path = os.path.join(img_root, recording_name)
if not os.path.exists(img_path):
    os.makedirs(img_path)

data = load_fiberscope_data(file_path)
data = data.drop_vars(
    [
        "ff",
        "tt",
        "stft_amp",
        "stft_phase",
    ]
)

data = data.sel(time=slice(0, 40))
data.signal.sel(h_index=2).plot()
plt.show()

t_interp_pulse = 1  # Inter sweep period
t_pulse = 100 * 1e-3  # Single sweep duration
t_ir = 1  # Approximated impulse response duration (simple value to ensure no energy is received after this time)
n_sweep = 3  # Number of sweep emitted
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

recording_props["src_start_pos"] = "P1"
recording_props["src_end_pos"] = "P4"
recording_props["dynamic_recording_name"] = recording_name
processing_props["time_step"] = n_sweep * t_interp_pulse
# split_dynamic_recording(data, recording_props, processing_props)

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
# print(recording_names_dynamic)

# for recording_name in recording_names_dynamic:
#     run_analysis(
#         recording_name, recording_props, processing_props, plot_rtf_estimation=True
#     )

recording_props["n_em"] = 10
# recording_name_to_loc = "10-10-2024T09-43-06-620681_P3_N1_Sweep_151"
# recording_name_to_loc = "09-10-2024T16-51-22-900122_P2_N1_Sweep_93"
# recording_name_to_loc = "09-10-2024T10-34-58-394627_P1_N1_Sweep_34"
# recording_name_to_loc = "10-10-2024T12-03-02-201689_P4_N1_Sweep_211"

root_loc = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\fiberscope"
img_path = os.path.join(root_loc, recording_props["dynamic_recording_name"])
if not os.path.exists(img_path):
    os.makedirs(img_path)

dict_th_pos = {
    "P1": 0,
    "P2": 10,
    "P3": 20,
    "P4": 25,
    "P5": 15,
    "P6": 5,
}


records_to_loc = [
    # P1
    "09-10-2024T10-34-58-394627_P1_N1_Sweep_34",
    # "09-10-2024T10-36-04-231393_P1_N2_Sweep_35",
    # "09-10-2024T10-37-04-088817_P1_N3_Sweep_36",
    # "09-10-2024T10-38-08-970528_P1_N4_Sweep_37",
    # "09-10-2024T10-39-11-308093_P1_N5_Sweep_38",
    # P2
    "09-10-2024T16-51-22-900122_P2_N1_Sweep_93",
    # P3
    "10-10-2024T09-43-06-620681_P3_N1_Sweep_151",
    "10-10-2024T09-44-55-048883_P3_N2_Sweep_152",
    "10-10-2024T09-45-50-516056_P3_N3_Sweep_153",
    "10-10-2024T09-46-45-046175_P3_N4_Sweep_154",
    "10-10-2024T09-47-33-438942_P3_N5_Sweep_155",
    # P4
    "10-10-2024T12-03-02-201689_P4_N1_Sweep_211",
    "10-10-2024T14-42-01-833325_P5_N2_Sweep_267",
    "10-10-2024T15-54-25-737795_P6_N1_Sweep_323",
]


# record_names_snr_test = [
#     "10-10-2024T09-43-06-620681_P3_N1_Sweep_151",
#     "10-10-2024T09-44-55-048883_P3_N2_Sweep_152",
#     "10-10-2024T09-45-50-516056_P3_N3_Sweep_153",
#     "10-10-2024T09-46-45-046175_P3_N4_Sweep_154",
#     "10-10-2024T09-47-33-438942_P3_N5_Sweep_155",
# ]
# record_names_snr_test = [
#     "09-10-2024T10-34-58-394627_P1_N1_Sweep_34",
#     "09-10-2024T10-36-04-231393_P1_N2_Sweep_35",
#     "09-10-2024T10-37-04-088817_P1_N3_Sweep_36",
#     "09-10-2024T10-38-08-970528_P1_N4_Sweep_37",
#     "09-10-2024T10-39-11-308093_P1_N5_Sweep_38",
# ]

# process records
# for recording_name in record_names_snr_test:
#     run_analysis(
#         recording_name, recording_props, processing_props, plot_rtf_estimation=True
#     )

# dict_loc = {}

# for recording_name_to_loc in record_names_snr_test:
#     pos_id = recording_name_to_loc.split("_")[1]
#     th_pos = dict_th_pos[pos_id]
#     range_from_p1, dist, snrs = localise(
#         recording_names=recording_names_dynamic,
#         recording_name_to_loc=recording_name_to_loc,
#         recording_props=recording_props,
#         processing_props=processing_props,
#         th_pos=th_pos,
#     )
#     r_hat = range_from_p1[np.argmin(dist)]
#     dict_loc[recording_name_to_loc] = {
#         "d": dist,
#         "r": range_from_p1,
#         "r_th": th_pos,
#         "r_hat": r_hat,
#         "snr_h5": snrs[-1],
#     }

# abs_err = np.abs(
#     np.array([d["r_hat"] for d in dict_loc.values()])
#     - np.array([d["r_th"] for d in dict_loc.values()])
# )
# snrs = np.array([d["snr_h5"] for d in dict_loc.values()])
# lvls = [r"$N_1$", r"$N_2$", r"$N_3$", r"$N_4$", r"$N_5$"]
# plt.figure()
# plt.plot(snrs, abs_err, color="k", marker="o")
# # Annote levels
# for i, txt in enumerate(lvls):
#     plt.annotate(txt, (snrs[i] + 0.2, abs_err[i] + 0.0001), fontsize=16)

# plt.ylabel(r"$\epsilon_r = |r_{th} - \hat{r}|\, \textrm{[m]}$")
# plt.xlabel(r"$H_5 \,\textrm{SNR [dB]}$")
# filepath = os.path.join(img_path, "abs_err_vs_snr_P1.png")
# plt.savefig(filepath)

records_to_loc = ["10-10-2024T16-53-43-200271_PR_N1_346_P1_r20.25m_P4"]
dict_loc = {}

for recording_name_to_loc in records_to_loc:
    # pos_id = recording_name_to_loc.split("_")[-1]
    # th_pos = dict_th_pos[pos_id]
    th_pos = float(recording_name_to_loc.split("_")[-2][1:-1])
    range_from_p1, dist, _ = localise(
        recording_names=recording_names_dynamic,
        recording_name_to_loc=recording_name_to_loc,
        recording_props=recording_props,
        processing_props=processing_props,
        th_pos=th_pos,
    )
    r_hat = range_from_p1[np.argmin(dist)]
    dict_loc[recording_name_to_loc] = {
        "d": dist,
        "r": range_from_p1,
        "r_th": th_pos,
        "r_hat": r_hat,
    }

# # Derive absolute error between theoretical and estimated range
# abs_err = np.abs(
#     np.array([d["r_hat"] for d in dict_loc.values()])
#     - np.array([d["r_th"] for d in dict_loc.values()])
# )
# plt.figure()
# plt.plot(dict_th_pos.keys(), abs_err, color="k", marker="o")
# plt.ylabel(r"$\epsilon_r = |r_{th} - \hat{r}|\, \textrm{[m]}$")
# plt.xlabel(r"$\textrm{Position id}$")

# root_loc = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\rtf\rtf_localisation\fiberscope"
# img_path = os.path.join(root_loc, recording_props["dynamic_recording_name"])
# if not os.path.exists(img_path):
#     os.makedirs(img_path)


# filepath = os.path.join(img_path, "abs_err.png")
# plt.savefig(filepath)

# run_analysis(recording_name_to_loc, recording_props, processing_props)

# # recording_name_to_loc = "09-10-2024T11-03-11-806485_P1_N1_Sweep_49"

### Positions with different immersions ###
# Position P7
# recording_name_to_loc = "11-10-2024T10-51-56-563968_P3_N1_Sweep_385"    # Chirp1
# recording_name_to_loc = "11-10-2024T10-57-39-705595_P3_N1_Sweep_390"  # Chirp2
# run_analysis(recording_name_to_loc, recording_props, processing_props)
# localise(
#     recording_names=recording_names_dynamic,
#     recording_name_to_loc=recording_name_to_loc,
#     recording_props=recording_props,
#     processing_props=processing_props
# )


# # Position P8
# recording_name_to_loc = "11-10-2024T12-08-20-091131_P1_N1_Sweep_437"  # Chirp 1
# # run_analysis(recording_name_to_loc, recording_props, processing_props)
# localise(
#     recording_names=recording_names_dynamic,
#     recording_name_to_loc=recording_name_to_loc,
#     recording_props=recording_props,
# )

# recording_name_to_loc = "10-10-2024T15-54-25-737795_P6_N1_Sweep_323"
# # recording_name_to_loc = "10-10-2024T14-42-01-833325_P5_N2_Sweep_267"

# pos_id = recording_name_to_loc.split("_")[1]
# th_pos = dict_th_pos[pos_id]
# localise(
#     recording_names=recording_names_dynamic,
#     recording_name_to_loc=recording_name_to_loc,
#     recording_props=recording_props,
#     processing_props=processing_props,
#     th_pos=th_pos,
# )

# Load recording from dynamic split corresponding to P3
recording_name = "10-10-2024T09-43-06-620681_P3_N1_Sweep_151"
recording_name_dyn = "10-10-2024T16-53-43-200271_PR_N1_346_P1_r20.25m_P4"

data_ref = xr.open_dataset(
    os.path.join(processed_data_path, f"{recording_name}_rtf.nc")
)
data_dyn = xr.open_dataset(
    os.path.join(processed_data_path, f"{recording_name_dyn}_rtf.nc")
)

f0 = 10e3
f1 = 13e3
# plt.figure()
# data_ref.sel(f_rtf=slice(f0, f1)).rtf_amp_hat.sel(h_index=3).plot()
# data_dyn.sel(f_rtf=slice(f0, f1)).rtf_amp_hat.sel(h_index=3).plot()
# plt.yscale("log")
# plt.show()


# plt.figure()
# data.signal.sel(h_index=1).plot()
# plt.show()
# plt.savefig("test.png")
# print()
