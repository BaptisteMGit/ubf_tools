#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   fiberscope_recording.py
@Time    :   2025/05/01 11:10:06
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to handle recording informations
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import real_data_analysis.fiberscope_20.src.params as p


class FiberscopeRecording:
    """
    Class to manage Fiberscope recording information
    """

    def __init__(
        self,
        records_N1: list = [],
        records_N3: list = [],
        records_N5: list = [],
        signal_props: dict = {},
    ):
        """
        Constructor
        :param root_data: Root data folder
        """
        # Names of the recrordings
        self.records_N1 = records_N1  # List of recording names for N1 emission level
        self.records_N3 = records_N3  # List of recording names for N3 emission level
        self.records_N5 = records_N5  # List of recording names for N5 emission level

        # Signal properties
        self.interp_pulse_period = signal_props["t_interp_pulse"]
        self.pulse_length = signal_props["t_pulse"]
        self.ir_duration = signal_props["t_ir"]
        self.t_pulse = signal_props["t_pulse"]
        self.n_sweep = signal_props["n_em"]
        self.fmin = signal_props["f0"]
        self.fmax = signal_props["f1"]

        # Other usefull attributes for processing
        records_folder = None  # Full path to the folder used to store netcdf files associated with the differents records


class FiberscopeSweep1(FiberscopeRecording):

    def __init__(self):

        records_N1 = p.sweep_1["recording_names"]["N1"]
        records_N3 = p.sweep_1["recording_names"]["N3"]
        records_N5 = p.sweep_1["recording_names"]["N5"]
        signal_props = p.sweep_1["signal_props"]

        super().__init__(
            records_N1=records_N1,
            records_N3=records_N3,
            records_N5=records_N5,
            signal_props=signal_props,
        )


class FiberscopeSweep2(FiberscopeRecording):

    def __init__(self):

        records_N1 = p.sweep_2["recording_names"]["N1"]
        records_N3 = p.sweep_2["recording_names"]["N3"]
        records_N5 = p.sweep_2["recording_names"]["N5"]
        signal_props = p.sweep_2["signal_props"]

        super().__init__(
            records_N1=records_N1,
            records_N3=records_N3,
            records_N5=records_N5,
            signal_props=signal_props,
        )


class FiberscopeDynamicRecording:

    def __init__(
        self,
        recording_name: str = p.dynamic_recording,
        src_speed: float = p.src_speed,
        src_start_pos: str = p.src_start_pos,
        src_end_pos: str = p.src_end_pos,
    ):
        """ """

        # Sweep propertes - sweep2
        self.signal = FiberscopeSweep2()

        # Dynamic properties
        self.recording_name = recording_name
        self.src_speed = src_speed
        self.src_start_pos = src_start_pos
        self.src_end_pos = src_end_pos

        # Init list to store splited recording names
        self.splitted_records_names = []
        self.splitted_records_folder = None

        # Processing properties
        self.time_step = None


if __name__ == "__main__":
    sp1_rec = FiberscopeSweep1()
    print(sp1_rec.records_N1)
