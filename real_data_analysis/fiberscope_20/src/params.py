#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   params.py
@Time    :   2025/04/30 16:28:27
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Define Fiberscope parameters
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os

# ======================================================================================================================
# General parameters
# ======================================================================================================================
# Distance from P1 in meters for each position -> True positions (not in the ascent order)
dict_th_pos = {
    "P1": 0,
    "P2": 10,
    "P3": 20,
    "P4": 25,
    "P5": 15,
    "P6": 5,
}

# tau_ir = 0.5  # Until 18/04/2025
tau_ir = 0.5
alpha_overlap = 0.75  # Overlap factor defining STFT snapshots

# Subsampling factor to reduce memory charge and cpu time
subsampling_factor = 5  # -> fs = 40 kHz

# Hydrophone index (real id of the hydrophone)
h_index_ref = 1

phd_folder = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd"
root = os.path.join(phd_folder, "real_data_analysis", "fiberscope_20")
root_img = os.path.join(root, "img")
root_data = os.path.join(root, "data")
root_tdms_data = os.path.join(phd_folder, "data", "Fiberscope_campagne_oct_2024")

# ======================================================================================================================
# Static recordings - Sweep 1 (8 - 15 kHz)
# ======================================================================================================================

# Sweep 1 properties
t_interp_pulse = 1  # Inter sweep period
t_pulse = 100 * 1e-3  # Single sweep duration
t_ir = 1  # Approximated impulse response duration (simple value to ensure no energy is received after this time)
n_sweep = 10  # Number of sweep emitted
f0 = 8e3  # Start frequency
f1 = 15e3  # End frequency

recording_names_N1_sweep1 = [
    "09-10-2024T10-34-58-394627_P1_N1_Sweep_34",
    "09-10-2024T16-51-22-900122_P2_N1_Sweep_93",
    "10-10-2024T09-43-06-620681_P3_N1_Sweep_151",
    "10-10-2024T12-03-02-201689_P4_N1_Sweep_211",
    "10-10-2024T14-42-01-833325_P5_N2_Sweep_267",  # Name N2 but actually N1
    "10-10-2024T15-54-25-737795_P6_N1_Sweep_323",
    # "11-10-2024T10-51-56-563968_P3_N1_Sweep_385",  # P7
    # "11-10-2024T12-08-20-091131_P1_N1_Sweep_437",  # P8
]

recording_names_N3_sweep1 = [
    "09-10-2024T10-37-04-088817_P1_N3_Sweep_36",
    "09-10-2024T16-53-16-681510_P2_N3_Sweep_95",
    "10-10-2024T09-45-50-516056_P3_N3_Sweep_153",
    "10-10-2024T12-04-46-610661_P4_N3_Sweep_213",
    "10-10-2024T14-43-47-603375_P5_N3_Sweep_269",
    "10-10-2024T15-56-16-837150_P6_N3_Sweep_325",
]

recording_names_N5_sweep1 = [
    "09-10-2024T10-39-11-308093_P1_N5_Sweep_38",
    "09-10-2024T16-55-08-243011_P2_N5_Sweep_97",
    "10-10-2024T09-47-33-438942_P3_N5_Sweep_155",
    "10-10-2024T12-06-31-200643_P4_N5_Sweep_215",
    "10-10-2024T14-45-21-047719_P5_N5_Sweep_271",
    "10-10-2024T15-57-57-549910_P6_N5_Sweep_327",
]

signal_props_sweep1 = {
    "t_interp_pulse": t_interp_pulse,
    "t_pulse": t_pulse,
    "t_ir": t_ir,
    "n_em": n_sweep,
    "f0": f0,
    "f1": f1,
}

processing_props_sweep1 = {
    "hydro_to_process": None,
    "ref_hydro": 1,
    "method": "cs",
    "alpha_th": 0.001 * 1e-2,
    "split_method": "band_energy",
}


sweep_1 = {
    "signal_props": signal_props_sweep1,
    "processing_props": processing_props_sweep1,
    "recording_names": {
        "N1": recording_names_N1_sweep1,
        "N3": recording_names_N3_sweep1,
        "N5": recording_names_N5_sweep1,
    },
}


# ======================================================================================================================
# Static recordings - Sweep 2 (10 - 13 kHz)
# ======================================================================================================================

t_interp_pulse = 1  # Inter sweep period
t_pulse = 100 * 1e-3  # Single sweep duration
t_ir = 1  # Approximated impulse response duration (simple value to ensure no energy is received after this time)
n_sweep = 10  # Number of sweep emitted
f0 = 10e3  # Start frequency
f1 = 13e3  # End frequency

signal_props_sweep2 = {
    "t_interp_pulse": t_interp_pulse,
    "t_pulse": t_pulse,
    "t_ir": t_ir,
    "n_em": n_sweep,
    "f0": f0,
    "f1": f1,
}

processing_props_sweep2 = {
    "hydro_to_process": None,
    "ref_hydro": 1,
    "method": "cs",
    "alpha_th": 0.001 * 1e-2,
    "split_method": "band_energy",
}

# Recording names are not provided yet (not used for static analysis so far)
sweep_2 = {
    "signal_props": signal_props_sweep2,
    "processing_props": processing_props_sweep2,
    "recording_names": {
        "N1": [],
        "N3": [],
        "N5": [],
    },
}


# ======================================================================================================================
# Dynamic recordings - Sweep 2 (10 - 13 kHz)
# ======================================================================================================================
# Recording from the moving source : speed = 0.1 m/s
dynamic_recording = "10-10-2024T16-53-43-200271_PR_N1_346"
src_speed = 0.1  # Source speed in m/s
src_start_pos = "P1"  # Source start position id
src_end_pos = "P4"  # Source last position id


# n_sweep = 3
# dynamic_recording_props = sweep_2["recording_props"]
# dynamic_recording_props["n_em"] = n_sweep
# dynamic_recording_props["src_speed"] = 0.1  # Source speed in m/s
# dynamic_recording_props["src_end_pos"] = "P4"  # Source start position id
# dynamic_recording_props["src_start_pos"] = "P1"  # Source last position id
# dynamic_recording_props["recording_name"] = dynamic_recording

# dynamic_processing_props = sweep_2["processing_props"]
# dynamic_processing_props["time_step"] = (
#     n_sweep * sweep_2["recording_props"]["t_interp_pulse"]
# )

# dynamic_sweep = {
#     "recording_props": dynamic_recording_props,
#     "processing_props": dynamic_processing_props,
# }
