#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for propa/kraken_toolbox/sensibility.py.

Run with either:
    python -m unittest discover -s propa/kraken_toolbox/tests
or (if pytest is installed in your environment):
    pytest propa/kraken_toolbox/tests
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import matplotlib

matplotlib.use("Agg")

from propa.kraken_toolbox import sensibility as sb
from propa.kraken_toolbox.src.kraken_manager import KrakenManager
from propa.kraken_toolbox.read_shd import readshd

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
REAL_SHD_PATH = os.path.join(FIXTURES_DIR, "real_field.shd")


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="sensibility_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        import matplotlib.pyplot as plt
        plt.close("all")


# ======================================================================
# Baseline parameter dicts
# ======================================================================
class TestBaselineParams(unittest.TestCase):
    def test_env_param_has_expected_keys(self):
        params = sb.baseline_env_param()
        self.assertEqual(set(params.keys()), {"c1", "c2", "rho1", "rho2", "attn2", "depth"})

    def test_freq_param_has_expected_keys(self):
        self.assertEqual(set(sb.baseline_freq_param().keys()), {"freq"})

    def test_src_rcv_param_has_expected_keys(self):
        self.assertEqual(
            set(sb.baseline_src_rcv_param().keys()), {"z_s", "z_rcv", "r0", "r_rcv"}
        )

    def test_arg_dict_merges_everything(self):
        all_args = sb.baseline_arg_dict()
        expected_keys = {"c1", "c2", "rho1", "rho2", "attn2", "depth",
                          "freq", "z_s", "z_rcv", "r0", "r_rcv"}
        self.assertEqual(set(all_args.keys()), expected_keys)

    def test_src_rcv_stay_within_shallowest_recommended_depth(self):
        # NOTE: z_s/z_rcv must remain valid (< water depth) across the
        # documented depth-sweep range (as low as 10 m) -- see
        # baseline_src_rcv_param()'s docstring.
        params = sb.baseline_src_rcv_param()
        self.assertLess(params["z_s"], 10.0)
        self.assertLess(params["z_rcv"], 10.0)


# ======================================================================
# build_kraken_env
# ======================================================================
class TestBuildKrakenEnv(TempDirTestCase):
    def test_builds_a_valid_pekeris_environment(self):
        all_args = sb.baseline_arg_dict()
        env, flp = sb.build_kraken_env(all_args, root=self.tmp_dir, filename="test")
        self.assertEqual(env.nmedia, 1)  # direct half-space, no buffer
        self.assertFalse(env.range_dependent_env)  # not written yet, but no bathy given
        self.assertFalse(env.bathy.use_bathy)
        env.write_env()
        self.assertFalse(env.range_dependent_env)
        flp.write_flp()
        self.assertTrue(os.path.exists(env.env_fpath))
        self.assertTrue(os.path.exists(flp.flp_fpath))

    def test_water_depth_override_is_reflected(self):
        all_args = sb.baseline_arg_dict()
        all_args["depth"] = 500.0
        env, flp = sb.build_kraken_env(all_args, root=self.tmp_dir, filename="test")
        self.assertAlmostEqual(env.medium.z_ssp.max(), 500.0)
        self.assertAlmostEqual(env.field.rcv_depth_max, 500.0)

    def test_bottom_properties_match_input(self):
        all_args = sb.baseline_arg_dict()
        all_args["c2"] = 1700.0
        all_args["rho2"] = 1.9 * 1e3
        all_args["attn2"] = 0.8
        env, flp = sb.build_kraken_env(all_args, root=self.tmp_dir, filename="test")
        self.assertAlmostEqual(env.bottom_hs.cp_bot_halfspace, 1700.0)
        self.assertAlmostEqual(env.bottom_hs.rhobot_halfspace, 1.9)  # kg/m3 -> g/cm3
        self.assertAlmostEqual(env.bottom_hs.apbot_halfspace, 0.8)

    def test_single_receiver_depth_in_flp(self):
        all_args = sb.baseline_arg_dict()
        env, flp = sb.build_kraken_env(all_args, root=self.tmp_dir, filename="test")
        self.assertEqual(flp.n_rcv_z, 1)
        self.assertAlmostEqual(flp.rcv_z_min, all_args["z_rcv"])
        self.assertAlmostEqual(flp.rcv_z_max, all_args["z_rcv"])

    def test_receiver_range_grid_matches_input(self):
        all_args = sb.baseline_arg_dict()
        env, flp = sb.build_kraken_env(all_args, root=self.tmp_dir, filename="test")
        self.assertEqual(flp.n_rcv_r, all_args["r_rcv"].size)
        self.assertAlmostEqual(flp.rcv_r_max, all_args["r_rcv"].max() / 1000.0)


# ======================================================================
# compute_qoi_green_level_at_reference_range
# ======================================================================
class TestComputeQoi(unittest.TestCase):
    def test_matches_manual_formula(self):
        field_pos = {"r": {"r": np.array([100.0, 200.0, 300.0])}}
        g_fr = np.array([0.1 + 0.0j, 0.05 + 0.0j, 0.01 + 0.0j])
        qoi = sb.compute_qoi_green_level_at_reference_range(g_fr, field_pos, r0=200.0)
        self.assertAlmostEqual(qoi, 20 * np.log10(0.05))

    def test_uses_nearest_available_range(self):
        field_pos = {"r": {"r": np.array([100.0, 200.0, 300.0])}}
        g_fr = np.array([0.1 + 0.0j, 0.05 + 0.0j, 0.01 + 0.0j])
        qoi = sb.compute_qoi_green_level_at_reference_range(g_fr, field_pos, r0=190.0)
        self.assertAlmostEqual(qoi, 20 * np.log10(0.05))

    def test_does_not_crash_on_zero_amplitude(self):
        field_pos = {"r": {"r": np.array([100.0])}}
        g_fr = np.array([0.0 + 0.0j])
        qoi = sb.compute_qoi_green_level_at_reference_range(g_fr, field_pos, r0=100.0)
        self.assertTrue(np.isfinite(qoi))


# ======================================================================
# run_sensibility_study -- dry run (no KRAKEN)
# ======================================================================
class TestRunSensibilityStudyDryRun(TempDirTestCase):
    def test_dry_run_writes_only_last_value_and_returns_none(self):
        all_args = sb.baseline_arg_dict()
        computation_root = os.path.join(self.tmp_dir, "io_files")
        result_root = os.path.join(self.tmp_dir, "data", "sensibility")

        result = sb.run_sensibility_study(
            "depth", np.array([10.0, 100.0, 1000.0]), all_args,
            computation_root=computation_root, result_root=result_root,
            run_kraken=False, verbose=False,
        )
        self.assertIsNone(result)

        work_dir = os.path.join(computation_root, "depth")
        self.assertTrue(os.path.exists(os.path.join(work_dir, "pekeris_depth.env")))
        # No results file should have been written in a dry run.
        self.assertFalse(os.path.exists(os.path.join(result_root, "depth.csv")))

    def test_dry_run_uses_the_last_requested_value(self):
        all_args = sb.baseline_arg_dict()
        computation_root = os.path.join(self.tmp_dir, "io_files")
        result_root = os.path.join(self.tmp_dir, "data", "sensibility")

        sb.run_sensibility_study(
            "depth", np.array([10.0, 100.0, 999.0]), all_args,
            computation_root=computation_root, result_root=result_root,
            run_kraken=False, verbose=False,
        )
        env_path = os.path.join(computation_root, "depth", "pekeris_depth.env")
        with open(env_path) as f:
            content = f.read()
        self.assertIn("999.0", content)


# ======================================================================
# run_sensibility_study -- full run (mocked KrakenManager.runkraken)
# ======================================================================
@unittest.skipUnless(os.path.exists(REAL_SHD_PATH), "real_field.shd fixture not present")
class TestRunSensibilityStudyFullRun(TempDirTestCase):
    def _fake_pressure_for(self, n_r):
        _, _, _, _, _, _, field_pos, pressure = readshd(REAL_SHD_PATH, freq=30)
        pressure = np.squeeze(pressure)
        row = pressure[0] if pressure.ndim > 1 else pressure
        # Tile/truncate to the requested number of ranges so this works
        # regardless of how many ranges the sensibility study's own grid
        # asks for.
        reps = int(np.ceil(n_r / row.size))
        row = np.tile(row, reps)[:n_r]
        return row.reshape(1, 1, 1, -1), field_pos

    def test_reuses_a_single_working_directory_across_values(self):
        all_args = sb.baseline_arg_dict()
        computation_root = os.path.join(self.tmp_dir, "io_files")
        result_root = os.path.join(self.tmp_dir, "data", "sensibility")
        values = np.linspace(10.0, 5000.0, 4)

        def fake_runkraken(self_, env, flp, frequencies):
            return self._fake_pressure_for(int(flp.n_rcv_r))

        with mock.patch.object(KrakenManager, "runkraken", fake_runkraken):
            results = sb.run_sensibility_study(
                "depth", values, all_args,
                computation_root=computation_root, result_root=result_root,
                verbose=False,
            )

        self.assertEqual(results.shape, (4, 2))
        np.testing.assert_allclose(results[:, 0], values)

        # Exactly one subdirectory for the whole sweep.
        self.assertEqual(os.listdir(computation_root), ["depth"])

    def test_writes_one_csv_with_a_row_per_value(self):
        all_args = sb.baseline_arg_dict()
        computation_root = os.path.join(self.tmp_dir, "io_files")
        result_root = os.path.join(self.tmp_dir, "data", "sensibility")
        values = np.array([1500.0, 1550.0, 1600.0])  # e.g. sweeping c1

        def fake_runkraken(self_, env, flp, frequencies):
            return self._fake_pressure_for(int(flp.n_rcv_r))

        with mock.patch.object(KrakenManager, "runkraken", fake_runkraken):
            sb.run_sensibility_study(
                "c1", values, all_args,
                computation_root=computation_root, result_root=result_root,
                verbose=False,
            )

        csv_path = os.path.join(result_root, "c1.csv")
        self.assertTrue(os.path.exists(csv_path))
        reloaded = sb.load_sensibility_result(result_root, "c1")
        np.testing.assert_allclose(reloaded[:, 0], values)

    def test_custom_qoi_function_is_used(self):
        all_args = sb.baseline_arg_dict()
        computation_root = os.path.join(self.tmp_dir, "io_files")
        result_root = os.path.join(self.tmp_dir, "data", "sensibility")
        values = np.array([100.0, 200.0])

        def fake_runkraken(self_, env, flp, frequencies):
            return self._fake_pressure_for(int(flp.n_rcv_r))

        def constant_qoi(g_fr, field_pos, r0):
            return 42.0

        with mock.patch.object(KrakenManager, "runkraken", fake_runkraken):
            results = sb.run_sensibility_study(
                "depth", values, all_args,
                computation_root=computation_root, result_root=result_root,
                compute_qoi=constant_qoi, verbose=False,
            )

        np.testing.assert_allclose(results[:, 1], [42.0, 42.0])

    def test_baseline_dict_is_not_mutated(self):
        all_args = sb.baseline_arg_dict()
        original_depth = all_args["depth"]
        computation_root = os.path.join(self.tmp_dir, "io_files")
        result_root = os.path.join(self.tmp_dir, "data", "sensibility")

        def fake_runkraken(self_, env, flp, frequencies):
            return self._fake_pressure_for(int(flp.n_rcv_r))

        with mock.patch.object(KrakenManager, "runkraken", fake_runkraken):
            sb.run_sensibility_study(
                "depth", np.array([10.0, 5000.0]), all_args,
                computation_root=computation_root, result_root=result_root,
                verbose=False,
            )

        self.assertEqual(all_args["depth"], original_depth)


# ======================================================================
# plot_sensibility_result
# ======================================================================
class TestPlotSensibilityResult(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_basic_plot_does_not_crash(self):
        results = np.array([[10.0, -60.0], [100.0, -70.0], [1000.0, -85.0]])
        fig = sb.plot_sensibility_result(results, "depth", param_units="m")
        ax = fig.axes[0]
        np.testing.assert_allclose(ax.lines[0].get_xdata(), results[:, 0])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), results[:, 1])
        self.assertIn("m", ax.get_xlabel())


if __name__ == "__main__":
    unittest.main()
