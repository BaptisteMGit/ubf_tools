#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for propa/kraken_toolbox/utils.py.

Run with either:
    python -m unittest discover -s propa/kraken_toolbox/tests
or (if pytest is installed in your environment):
    pytest propa/kraken_toolbox/tests
"""
import os
import tempfile
import shutil
import unittest
from unittest import mock

import numpy as np

from propa.kraken_toolbox.utils import (
    get_component,
    align_var_description,
    default_nb_rcv_z,
    waveguide_cutoff_freq,
    get_rcv_pos_idx,
    find_optimal_intervals,
    g,
)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="utils_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


# ======================================================================
# get_component - regression tests for the fixed medium-boundary bug
# ======================================================================
class TestGetComponent(unittest.TestCase):
    def test_unknown_component_raises(self):
        Modes = {"phi": np.zeros((1, 1))}
        with self.assertRaises(Exception):
            get_component(Modes, "X")

    def test_single_acoustic_medium(self):
        # 1 medium, 3 ACOUSTIC points, 1 mode -> trivial 1-to-1 mapping
        Modes = {
            "Nmedia": 1,
            "N": np.array([3]),
            "Mater": np.array(["ACOUSTIC"], dtype=object),
            "z": np.array([0.0, 1.0, 2.0]),
            "phi": np.arange(3).reshape(3, 1).astype(complex),
        }
        phi = get_component(Modes, "H")
        np.testing.assert_array_equal(phi.real.flatten(), [0, 1, 2])

    def test_two_media_acoustic_then_elastic(self):
        # NOTE: regression test for the fixed bug -- the original code
        # used len(Modes["z"]) (the TOTAL number of points) as the
        # bound of the inner per-medium loop instead of
        # Modes["N"][medium] (that medium's own point count). With a
        # mixed ACOUSTIC + ELASTIC setup, this misattributed the
        # boundary point's material and overwrote earlier results.
        # Confirmed failure mode: the original code returned
        # [3, 1, 2] instead of [0, 1, 2] for this exact scenario.
        Modes = {
            "Nmedia": 2,
            "N": np.array([2, 1]),  # medium0: 2 ACOUSTIC pts, medium1: 1 ELASTIC pt
            "Mater": np.array(["ACOUSTIC", "ELASTIC"], dtype=object),
            "z": np.array([0.0, 1.0, 2.0]),
            # storage: 2 ACOUSTIC rows (1 each) + 4 ELASTIC rows (H,V,T,N) = 6 rows
            "phi": np.arange(6).reshape(6, 1).astype(complex),
        }
        phi_h = get_component(Modes, "H")
        np.testing.assert_array_equal(phi_h.real.flatten(), [0, 1, 2])

        # component index for 'V' is 1 -> point 2 (ELASTIC) should read
        # raw row (k=2) + 1 = row 3
        phi_v = get_component(Modes, "V")
        np.testing.assert_array_equal(phi_v.real.flatten(), [0, 1, 3])

    def test_stops_gracefully_if_phi_storage_runs_out(self):
        # Fewer raw phi rows than the declared point/medium counts would
        # need: must return early rather than raising an IndexError.
        Modes = {
            "Nmedia": 1,
            "N": np.array([5]),
            "Mater": np.array(["ACOUSTIC"], dtype=object),
            "z": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "phi": np.arange(2).reshape(2, 1).astype(complex),  # only 2 rows available
        }
        phi = get_component(Modes, "H")
        self.assertEqual(phi.shape, (5, 1))
        np.testing.assert_array_equal(phi.real.flatten()[:2], [0, 1])


# ======================================================================
# align_var_description
# ======================================================================
class TestAlignVarDescription(unittest.TestCase):
    def test_basic_formatting(self):
        line = align_var_description("50.0", "Nominal frequency (Hz)")
        self.assertTrue(line.startswith("50.0"))
        self.assertIn("! Nominal frequency (Hz)", line)
        self.assertTrue(line.endswith("\n"))

    def test_long_value_still_gets_minimum_spacing(self):
        long_value = "x" * 100
        line = align_var_description(long_value, "desc")
        # Even for a value longer than the alignment column, at least a
        # minimal gap must separate it from the description (3 spaces
        # from the padding, plus the 1 space in " ! {desc}").
        self.assertIn(long_value + "    ! desc", line)


# ======================================================================
# default_nb_rcv_z
# ======================================================================
class TestDefaultNbRcvZ(unittest.TestCase):
    def test_basic_computation_is_positive_int(self):
        nz = default_nb_rcv_z(fmax=50, max_depth=1000)
        self.assertIsInstance(nz, int)
        self.assertGreater(nz, 0)

    def test_n_per_l_floor_is_5(self):
        nz_low = default_nb_rcv_z(fmax=50, max_depth=1000, n_per_l=1)
        nz_default = default_nb_rcv_z(fmax=50, max_depth=1000, n_per_l=5)
        self.assertEqual(nz_low, nz_default)

    def test_zero_or_negative_fmax_raises(self):
        with self.assertRaises(ValueError):
            default_nb_rcv_z(fmax=0, max_depth=1000)
        with self.assertRaises(ValueError):
            default_nb_rcv_z(fmax=-10, max_depth=1000)


# ======================================================================
# waveguide_cutoff_freq
# ======================================================================
class TestWaveguideCutoffFreq(unittest.TestCase):
    def test_deep_waveguide_gives_low_cutoff_floored_at_minimum(self):
        fc = waveguide_cutoff_freq(waveguide_depth=100_000)
        self.assertEqual(fc, 0.15)

    def test_shallow_waveguide_gives_higher_cutoff(self):
        fc_shallow = waveguide_cutoff_freq(waveguide_depth=10)
        fc_deep = waveguide_cutoff_freq(waveguide_depth=1000)
        self.assertGreater(fc_shallow, fc_deep)

    def test_zero_or_negative_depth_raises(self):
        with self.assertRaises(ValueError):
            waveguide_cutoff_freq(waveguide_depth=0)


# ======================================================================
# get_rcv_pos_idx - regression tests for the fixed partial-args bug
# ======================================================================
class TestGetRcvPosIdx(unittest.TestCase):
    def test_both_grids_given_directly(self):
        kraken_range = np.array([0.0, 1.0, 2.0, 3.0])
        kraken_depth = np.array([0.0, 50.0, 100.0])
        rr, zz, field_pos = get_rcv_pos_idx(
            kraken_range=kraken_range, kraken_depth=kraken_depth,
            rcv_range=[2.0], rcv_depth=[50.0],
        )
        self.assertIsNone(field_pos)
        self.assertEqual(rr.flatten().tolist(), [2])  # index of range 2.0
        self.assertEqual(zz.flatten().tolist(), [1])  # index of depth 50.0

    def test_full_grid_used_when_rcv_positions_not_given(self):
        kraken_range = np.array([0.0, 1.0, 2.0])
        kraken_depth = np.array([0.0, 50.0])
        rr, zz, _ = get_rcv_pos_idx(kraken_range=kraken_range, kraken_depth=kraken_depth)
        self.assertEqual(rr.shape, (2, 3))  # meshgrid(range, depth) -> (nz, nr)

    def test_only_one_grid_given_raises_clear_error(self):
        # NOTE: regression test for the fixed bug -- the original code
        # only checked `kraken_range is None and kraken_depth is None`,
        # so supplying just one of the two fell through to
        # `kraken_depth.size` on a None value, raising a confusing
        # AttributeError. Now raises an explicit, actionable ValueError.
        with self.assertRaises(ValueError):
            get_rcv_pos_idx(kraken_range=np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            get_rcv_pos_idx(kraken_depth=np.array([0.0, 1.0]))

    def test_neither_grid_nor_shd_fpath_raises_clear_error(self):
        with self.assertRaises(ValueError):
            get_rcv_pos_idx()

    def test_reads_grid_from_shd_file_when_neither_grid_given(self):
        sentinel_field_pos = {"r": {"r": np.array([0.0, 1.0]), "z": np.array([0.0, 10.0])}}
        fake_pressure = np.zeros((1, 1, 2, 2))  # last two dims: (z, r)
        with mock.patch(
            "propa.kraken_toolbox.utils.readshd",
            return_value=(None, None, None, None, None, None, sentinel_field_pos, fake_pressure),
        ) as mocked:
            rr, zz, field_pos = get_rcv_pos_idx(shd_fpath="fake.shd")
        mocked.assert_called_once()
        self.assertIs(field_pos, sentinel_field_pos)
        self.assertEqual(rr.shape, (2, 2))


# ======================================================================
# find_optimal_intervals / g - CPU-time balancing model
# ======================================================================
class TestFindOptimalIntervals(unittest.TestCase):
    def test_returns_n_workers_plus_one_bounds(self):
        _t, bounds = find_optimal_intervals(fmin=10, fmax=100, nf=50, n_workers=4)
        self.assertEqual(len(bounds), 5)
        self.assertAlmostEqual(bounds[0], 10)
        self.assertAlmostEqual(bounds[-1], 100)

    def test_bounds_are_monotonically_increasing(self):
        _t, bounds = find_optimal_intervals(fmin=10, fmax=100, nf=50, n_workers=4)
        for a, b in zip(bounds[:-1], bounds[1:]):
            self.assertLessEqual(a, b)

    def test_single_worker_returns_full_range(self):
        _t, bounds = find_optimal_intervals(fmin=10, fmax=100, nf=20, n_workers=1)
        self.assertEqual(len(bounds), 2)
        self.assertAlmostEqual(bounds[0], 10)
        self.assertAlmostEqual(bounds[1], 100)

    def test_g_matches_direct_polynomial_integral(self):
        # Sanity check of g() against a hand-computed integral of
        # a*f^2 + b*f + c over [f0, f1], for a trivial single-band case.
        z = [1.0, 2.0, 3.0]  # a, b, c
        fmin, fmax = 0.0, 10.0
        alpha = 0.0
        expected = (
            z[0] / 3 * (fmax**3 - fmin**3)
            + z[1] / 2 * (fmax**2 - fmin**2)
            + z[2] * (fmax - fmin)
        )
        got = g(fi=[], alpha=alpha, k=0, z=z, fmin=fmin, fmax=fmax)
        self.assertAlmostEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
