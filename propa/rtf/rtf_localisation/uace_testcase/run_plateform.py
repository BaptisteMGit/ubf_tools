import os
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client

import propa.rtf.rtf_localisation.uace_testcase.src.params as p
from propa.rtf.rtf_localisation.uace_testcase.src.antenna import SparseAntenna
from propa.rtf.rtf_localisation.uace_testcase.src.simulation import Simulation
from propa.rtf.rtf_localisation.uace_testcase.src.data_builder import DataBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.feature_builder import FeatureBuilder
from propa.rtf.rtf_localisation.uace_testcase.src.localization_processor import (
    LocalizationProcessor,
)
from propa.rtf.rtf_localisation.uace_testcase.src.testcase_builder import (
    DeepWaterPekerisMunk,
    DeepWaterPekerisRhumrumSSP,
    DeepWaterRealEnv,
)
from publication.publication_figure import PubFigure

if __name__ == "__main__":

    antenna = SparseAntenna(
        name="Test_sparse_antenna", n_elements=6, random_radius=5e3, rng_seed=42
    )
    # antenna.plot_antenna()
    # plt.savefig("antenna")

    ### Simulation 1 

    name = "dw_real_env"
    n_mc = 100
    search_area_length = 1 * 1e3

    # Window properties set to the best properties according to the results from window_props_study.py
    nperseg = 2**10
    alpha_overlap = 0.5

    # Library ships used
    library_ship = p.library_ship
    # Plot library ships
    nperseg_plot = 2**9
    noverlap_plot = 2**8
    for library_ship_i in library_ship:
        library_ship_i.plot_signal(tmax=2)
        library_ship_i.plot_spectrum()
        library_ship_i.plot_psd()
        library_ship_i.plot_stft(nperseg=nperseg_plot, noverlap=noverlap_plot)
        plt.close("all")

    # Flags
    check = False
    debug = False
    verbose = True
    use_weighted_rtf = True

    simu = Simulation(
        name=name,
        debug=debug,
        antenna=antenna,
        check_features=check,
        monte_carlo_iterations=n_mc,
        feature_nperseg=nperseg,
        feature_overlap_ratio=alpha_overlap,
        use_weighted_rtf=use_weighted_rtf,
        search_area_length=search_area_length,
        verbose=verbose,
    )
    test_case = DeepWaterRealEnv(simulation=simu, mode="run", name=name)

    # db = DataBuilder(simulation=simu)
    # db.build_tf_dataset()
    # # # print("Grid dataset")
    # db.grid_dataset()
    # db.build_signal()


    # snrs = [-6, -4, -2, 0]
    # snrs = [50]
    snrs = np.arange(-10, 16, 1)[::-1]
    print(f"Processing snrs : {snrs}")
    lp = LocalizationProcessor(simulation=simu, use_dask=False)
    lp.process_multiple_snrs(snrs=snrs, run_mode="w")

    # # # TODO : remove this
    # import xarray as xr
    # ship_dist = xr.open_dataset(os.path.join(simu.data_folder, "library_ship_distribution.nc"))
    # amb_surf_fa = xr.open_dataset(os.path.join(simu.data_folder, "from_signal_dx20m_dy20m",
    #                                            "snr_50.0dB", "localization_dx20m_dy20m_fullarray_s1_s2_s3_s4_s5_s6.nc"))

    # fig, axs = plt.subplots(nrows=2, ncols=3, sharey=True)
    # axs = axs.flatten()
    # ships_idx = np.unique(ship_dist.library_ship_id.values)
    # for iship in ships_idx:
    #     mask_i = np.where(ship_dist.library_ship_id.values == iship, 1, np.nan)
    #     # mask_i = (ship_dist.library_ship_id.values == iship).astype(int)
    #     amb_surf_fa_i = amb_surf_fa.d_rtf * mask_i
    #     amb_surf_fa_i.plot(x="x", y="y", cmap="jet", ax=axs[iship], vmin=-6, vmax=0)
    #     axs[iship].set_title(f"ship n°{iship}")

    # amb_surf_fa.d_rtf.plot(x="x", y="y", cmap="jet", ax=axs[-1],  vmin=-6, vmax=0)
    # axs[-1].set_title("Ambiguity surface")
    # fpath = os.path.join(simu.img_folder, "q_rtf_masked.png")
    # plt.savefig(fpath)
