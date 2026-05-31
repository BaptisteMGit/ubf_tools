#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rtf_mfp_tests.py
@Time    :   2026/05/18 13:51:21
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
from datetime import datetime
from source.rtf_mfp.rtf_mfp import RTF_MFP_Processor
from source.rtf_mfp.rtf_mfp_feature_manager import BandFilter

# =====================================================================================================================
# Test
# =====================================================================================================================


def test_Fiberscope():
    """
    Test function
    """
    root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data"
    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\img\rtf_mfp"
    ref_rcv_id = 2
    rtf_estimator = "cs-evd"

    tau_ir_hat = 0.2  # estimated impulse response duration from sequence 144
    tau_ir_hat *= 2  # To ensure we include the entire response
    fsm_active_kwargs = {
        "bandfilter": BandFilter(order=4, lowcut=100, highcut=900),
        "tau_ir": tau_ir_hat,
        "process_pulse_one_by_one": True,
        "estimate_ir_duration": False,
    }
    fsm_passive_kwargs = {
        "analysis_segment_duration": 5,
        "analysis_segment_alpha_overlap": 0.75,
    }

    rtf_mfp_processor = RTF_MFP_Processor(
        root_data=root_data,
        root_img=root_img,
        reference_receiver_id=ref_rcv_id,
        rtf_estimator=rtf_estimator,
        fsm_active_kwargs=fsm_active_kwargs,
        fsm_passive_kwargs=fsm_passive_kwargs,
        mode="overwrite",
        plot_replicas_features=False,
        verbose=True,
    )

    ###########################
    # Library computation
    ###########################

    # Populate library
    active_replicas_args = {
        "replica_sequence_ids": [],
        "replica_pulse_slice": [],
        "load_precomputed_feature": True,
    }

    passive_replicas_args = {
        "start_datetimes": [
            datetime(year=2025, month=10, day=14, hour=1, minute=42, second=00),  # OK
        ],
        "end_datetimes": [
            datetime(year=2025, month=10, day=14, hour=1, minute=48, second=00),  # OK
        ],
        "load_precomputed_feature": True,
    }

    rtf_mfp_processor.compute_library(
        active_feature_args=active_replicas_args,
        passive_feature_args=passive_replicas_args,
        id=0,
    )

    ###########################
    # Event computation
    ###########################

    # Derive event
    active_feature_args = {
        "replica_sequence_ids": [],
        "load_precomputed_feature": True,
    }

    target_mmsi = None
    passive_feature_args = {
        "start_datetimes": [
            datetime(year=2025, month=10, day=14, hour=16, minute=40, second=30),
        ],
        "end_datetimes": [
            datetime(year=2025, month=10, day=14, hour=16, minute=50, second=30),
        ],
        "load_precomputed_feature": True,
    }

    rtf_mfp_processor.compute_event(
        active_feature_args=active_feature_args,
        passive_feature_args=passive_feature_args,
        id=0,
        target_mmsi=target_mmsi,
    )

    ###########################
    # Matching library and event features
    ###########################

    rtf_mfp_processor.match(id_library=0, id_event=0)


def test_reu_08042026():
    """
    Cas test pour illustrer les pbs rencontrés pour la réunion du 09/04/2026 et du 10/04/2026
    """
    root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\data"
    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\fiberscope_groix\img\rtf_mfp"
    ref_rcv_id = 2
    rtf_estimator = "cs-evd"

    tau_ir_hat = 0.2  # estimated impulse response duration from sequence 144
    tau_ir_hat *= 2  # To ensure we include the entire response
    fsm_active_kwargs = {
        "bandfilter": BandFilter(order=4, lowcut=100, highcut=900),
        "tau_ir": tau_ir_hat,
        "process_pulse_one_by_one": True,
        "estimate_ir_duration": False,
    }
    fsm_passive_kwargs = {
        "analysis_segment_duration": 10,
        "analysis_segment_alpha_overlap": 0.75,
    }
    fsm_props = {
        "fs": 2000,
        # From the sensibility
        "tau_rtf_analysis": 3.0,
        "alpha_overlap": 0.9,
    }

    rtf_mfp_processor = RTF_MFP_Processor(
        root_data=root_data,
        root_img=root_img,
        reference_receiver_id=ref_rcv_id,
        rtf_estimator=rtf_estimator,
        fsm_props=fsm_props,
        fsm_active_kwargs=fsm_active_kwargs,
        fsm_passive_kwargs=fsm_passive_kwargs,
        mode="overwrite",
        plot_replicas_features=False,
        verbose=True,
    )

    ###########################
    # Library computation
    ###########################

    # Populate library
    active_replicas_args = {
        "replica_sequence_ids": [144],
        "replica_pulse_slice": [(40, 180)],
        "load_precomputed_feature": True,
    }

    passive_replicas_args = {
        "start_datetimes": [],
        "end_datetimes": [],
        "load_precomputed_feature": True,
    }

    rtf_mfp_processor.compute_library(
        active_feature_args=active_replicas_args,
        passive_feature_args=passive_replicas_args,
        id=200,
    )

    ###########################
    # Event computation
    ###########################

    # Derive event
    active_feature_args = {
        "replica_sequence_ids": [],
        "load_precomputed_feature": True,
    }

    target_mmsi = None
    passive_feature_args = {
        "start_datetimes": [
            datetime(year=2025, month=10, day=16, hour=20, minute=55, second=00),  # OK
        ],
        "end_datetimes": [
            datetime(year=2025, month=10, day=16, hour=21, minute=10, second=00),  # OK
        ],
        "load_precomputed_feature": False,
    }

    rtf_mfp_processor.compute_event(
        active_feature_args=active_feature_args,
        passive_feature_args=passive_feature_args,
        id=300,
        target_mmsi=target_mmsi,
    )

    ###########################
    # Matching library and event features
    ###########################

    dist_args = {
        "fmin": 400,
        "fmax": 800,
        "dist_type": "hermitian_angle",
        "use_weighted_mean": False,
    }

    rtf_mfp_processor.match(id_library=200, id_event=300, dist_args=dist_args)


def test_9R():
    root_data = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\reseau_9R\data"
    root_img = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\real_data_analysis\reseau_9R\img\rtf_mfp"

    ref_rcv_id = 19
    rtf_estimator = "cs-evd"
    fm_passive_kwargs = {
        "analysis_segment_duration": 30,
        "analysis_segment_alpha_overlap": 0.9,
    }
    fm_props = {
        "fs": 100,
        "tau_rtf_analysis": 20,
        "alpha_overlap": 0.75,
    }

    # TODO Get this from rtf_mfp_processor
    receiver_ids = [
        1,
        # 2,
        3,
        # 5,
        6,
        # 7,
        8,
        # 9,
        10,
        # 11,
        12,
        # 13,
        14,
        # 15,
        16,
        # 17,
        18,
        # 19,
        21,
        # 22,
        23,
        # 24,
        25,
        # 26,
        28,
        # 29,
        30,
        # 31,
        33,
        # 34,
        35,
        # 37,
        38,
        # 39,
    ]

    rtf_mfp_processor = RTF_MFP_Processor(
        root_data=root_data,
        root_img=root_img,
        receiver_ids=receiver_ids,
        reference_receiver_id=ref_rcv_id,
        rtf_estimator=rtf_estimator,
        fm_props=fm_props,
        fm_passive_kwargs=fm_passive_kwargs,
        mode="overwrite",
        plot_replicas_features=True,
        wav_dataset_filename="channel_EDH_wav.nc",
        ais_dataset_filename="ais.nc",
        bathy_dataset_filename="bathy.nc",
        verbose=True,
    )

    id_library = 0

    passive_replicas_args = {
        "start_datetimes": [
            # datetime(year=2023, month=1, day=2, hour=5, minute=30, second=00),  # 1
            # datetime(year=2023, month=5, day=1, hour=1, minute=0, second=00),  # 2
            datetime(year=2023, month=5, day=3, hour=23, minute=40, second=00),  # 3
        ],
        "end_datetimes": [
            # datetime(year=2023, month=1, day=2, hour=6, minute=30, second=00),  # 1
            # datetime(year=2023, month=5, day=1, hour=2, minute=0, second=00),  # 2
            datetime(year=2023, month=5, day=4, hour=0, minute=20, second=00),  # 3
        ],
        "load_precomputed_feature": False,
    }

    rtf_mfp_processor.compute_library(
        passive_feature_args=passive_replicas_args,
        id=id_library,
    )

    # passive_feature_args = {
    #     "start_datetimes": [
    #         datetime(year=2025, month=10, day=16, hour=20, minute=55, second=00),  # OK
    #     ],
    #     "end_datetimes": [
    #         datetime(year=2025, month=10, day=16, hour=21, minute=10, second=00),  # OK
    #     ],
    #     "load_precomputed_feature": False,
    # }


if __name__ == "__main__":
    test_9R()
