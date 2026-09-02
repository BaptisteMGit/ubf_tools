#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for propa/kraken_toolbox/src/kraken_env.py.

These tests target the pure Python logic (parameter validation, letter
code selection, numeric computations, file writing) and do NOT require
the actual kraken.exe / field.exe binaries: running those is out of
scope for a unit test suite and belongs in an integration/acceptance
test run on a machine where KRAKEN is installed.

Run with either:
    python -m unittest discover -s propa/kraken_toolbox/tests
or (if pytest is installed in your environment):
    pytest propa/kraken_toolbox/tests
"""
import os
import tempfile
import shutil
import unittest

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no display needed / available in CI or this sandbox

from propa.kraken_toolbox.src.kraken_env import (
    KrakenMedium,
    KrakenTopHalfspace,
    KrakenBottomHalfspace,
    KrakenAttenuation,
    KrakenField,
    Bathymetry,
    KrakenEnv,
    KrakenFlp,
)


def _make_bathy_csv(path, rows):
    """Write a small (range, depth) CSV file with no header, as expected
    by Bathymetry."""
    with open(path, "w") as f:
        for r, h in rows:
            f.write(f"{r},{h}\n")


class TempDirTestCase(unittest.TestCase):
    """Base class creating/cleaning a temporary directory per test."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="kraken_env_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


# ======================================================================
# KrakenMedium
# ======================================================================
class TestKrakenMedium(unittest.TestCase):
    def test_default_interpolation_code(self):
        medium = KrakenMedium()
        self.assertEqual(medium.interp_code, "C")

    def test_all_interpolation_codes(self):
        expected = {
            "C_linear": "C",
            "N2_linear": "N",
            "cubic_spline": "S",
        }
        for method, code in expected.items():
            medium = KrakenMedium(ssp_interpolation_method=method)
            self.assertEqual(medium.interp_code, code)

    def test_analytic_interpolation_warns(self):
        with self.assertWarns(UserWarning):
            medium = KrakenMedium(ssp_interpolation_method="analytic")
        self.assertEqual(medium.interp_code, "A")

    def test_unknown_interpolation_raises(self):
        with self.assertRaises(ValueError):
            KrakenMedium(ssp_interpolation_method="not_a_real_method")

    def test_size_mismatch_raises_on_write(self):
        # z_ssp has 3 points, c_p only 2 -> inconsistent, not a scalar
        medium = KrakenMedium(z_ssp=[0, 50, 100], c_p=[1500, 1510])
        with self.assertRaises(ValueError):
            medium.write_lines()

    def test_scalar_properties_are_broadcast(self):
        # rho/a_p/a_s are scalars: write_lines should not raise, and
        # every SSP line should be written (one line per depth + header)
        medium = KrakenMedium(z_ssp=[0, 50, 100], c_p=[1500, 1505, 1510])
        medium.write_lines()
        # header line + 3 depth lines (no bottom_hs supplied -> no sediment block)
        self.assertEqual(len(medium.lines), 1 + 3)

    def test_write_lines_with_sediment_layer_block(self):
        bott_hs = KrakenBottomHalfspace()  # acousto_elastic by default -> write_sedim_layer_bloc=True
        bott_hs.derive_sedim_layer_max_depth(z_max=100)
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        medium.write_lines(bottom_hs=bott_hs)
        # header + 2 depth lines + 3 sediment lines
        self.assertEqual(len(medium.lines), 1 + 2 + 3)

    def test_set_default_resets_medium(self):
        medium = KrakenMedium(z_ssp=[0, 50, 100], c_p=[1500, 1505, 1510])
        medium.set_default()
        np.testing.assert_array_equal(medium.z_ssp, [0.0, 100.0])
        np.testing.assert_array_equal(medium.cp_ssp, [1500.0, 1500.0])


# ======================================================================
# KrakenTopHalfspace
# ======================================================================
class TestKrakenTopHalfspace(unittest.TestCase):
    def test_vacuum_default(self):
        top = KrakenTopHalfspace()
        self.assertEqual(top.boundary_code, "V")

    def test_acousto_elastic_requires_properties(self):
        with self.assertRaises(ValueError):
            KrakenTopHalfspace(boundary_condition="acousto_elastic")

    def test_acousto_elastic_with_properties(self):
        props = {"z": 0.0, "c_p": 1600.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.5, "a_s": 0.0}
        top = KrakenTopHalfspace(boundary_condition="acousto_elastic", halfspace_properties=props)
        self.assertEqual(top.boundary_code, "A")
        self.assertEqual(top.cp_top_halfspace, 1600.0)

    def test_twersky_scatter_requires_properties(self):
        with self.assertRaises(ValueError):
            KrakenTopHalfspace(boundary_condition="soft_boss_Twersky_scatter")

    def test_reflection_coefficient_warns(self):
        with self.assertWarns(UserWarning):
            KrakenTopHalfspace(boundary_condition="reflection_coefficient")

    def test_unknown_condition_raises(self):
        with self.assertRaises(ValueError):
            KrakenTopHalfspace(boundary_condition="not_a_real_condition")

    def test_write_lines_broadband_and_slow_rootfinder_flags(self):
        top = KrakenTopHalfspace()
        medium = KrakenMedium()
        att = KrakenAttenuation()
        top.write_lines(medium, att, slow_rootfinder=True, broadband_run=True)
        line = top.lines[0]
        # code string is quoted: 'C' + 'V' + unitscode + thorp_code + '.' + 'B'
        self.assertIn(".", line)
        self.assertIn("B", line)


# ======================================================================
# KrakenBottomHalfspace
# ======================================================================
class TestKrakenBottomHalfspace(unittest.TestCase):
    def test_vacuum_has_no_sediment_layer(self):
        bott = KrakenBottomHalfspace(boundary_condition="vacuum")
        self.assertEqual(bott.boundary_code, "V")
        self.assertEqual(bott.sedim_layer_depth, 0)
        self.assertFalse(bott.write_sedim_layer_bloc)

    def test_acousto_elastic_default_has_sediment_layer(self):
        bott = KrakenBottomHalfspace()  # default: acousto_elastic
        self.assertEqual(bott.boundary_code, "A")
        self.assertTrue(bott.write_sedim_layer_bloc)
        self.assertGreater(bott.sedim_layer_depth, 0)

    def test_unknown_condition_raises(self):
        with self.assertRaises(ValueError):
            KrakenBottomHalfspace(boundary_condition="not_a_real_condition")

    def test_derive_sedim_layer_max_depth_rounding(self):
        bott = KrakenBottomHalfspace(fmin=10, alpha_wavelength=10)
        # sedim_layer_depth = 10 * c0 / 10 = c0 (1500 by default stub)
        bott.derive_sedim_layer_max_depth(z_max=1234)
        # Result must be a multiple of 100 (rounded) and >= z_max
        self.assertEqual(bott.sedim_layer_max_depth % 100, 0)
        self.assertGreaterEqual(bott.sedim_layer_max_depth, 1234)

    def test_derive_sedim_layer_max_depth_is_capped(self):
        bott = KrakenBottomHalfspace()
        bott.sedim_layer_max_z = 500
        bott.derive_sedim_layer_max_depth(z_max=10_000)
        self.assertLessEqual(bott.sedim_layer_max_depth, 10_000 + bott.sedim_layer_depth)
        self.assertLessEqual(bott.sedim_layer_max_depth, 500 + 100)  # capped, +100 rounding slack

    def test_bathymetry_code_toggle(self):
        bott = KrakenBottomHalfspace()
        bott.set_bathymetry_code(True)
        self.assertEqual(bott.bathymetry_code, "~")
        bott.set_bathymetry_code(False)
        self.assertEqual(bott.bathymetry_code, "")

    # ------------------------------------------------------------------
    # Regression tests for the fixed nmedia / sediment-buffer bug (see
    # KrakenEnv's docstring): a range-dependent, single-medium
    # environment with a direct acousto-elastic bottom (no buffer) was
    # confirmed, via a real KRAKEN/FIELD run, to work correctly with
    # nmedia=1 -- so add_sediment_buffer_layer=False must produce
    # exactly that: no automatic second medium block, and a half-space
    # line written at the water column's own local depth.
    # ------------------------------------------------------------------
    def test_no_buffer_layer_when_disabled(self):
        bott = KrakenBottomHalfspace(add_sediment_buffer_layer=False)
        self.assertFalse(bott.write_sedim_layer_bloc)
        self.assertEqual(bott.sedim_layer_depth, 0)

    def test_buffer_layer_enabled_by_default(self):
        bott = KrakenBottomHalfspace()
        self.assertTrue(bott.add_sediment_buffer_layer)
        self.assertTrue(bott.write_sedim_layer_bloc)

    def test_derive_sedim_layer_max_depth_without_buffer_uses_z_max_directly(self):
        bott = KrakenBottomHalfspace(add_sediment_buffer_layer=False)
        bott.derive_sedim_layer_max_depth(z_max=3000.0)
        # No buffer -> no extension, no rounding: exactly z_max.
        self.assertEqual(bott.sedim_layer_max_depth, 3000.0)

    def test_write_lines_without_buffer_uses_explicit_halfspace_depth(self):
        bott = KrakenBottomHalfspace(
            halfspace_properties={"z": 0, "c_p": 1600.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.5, "a_s": 0.0},
            add_sediment_buffer_layer=False,
        )
        bott.derive_sedim_layer_max_depth(z_max=3000.0)
        bott.write_lines(use_bathymetry=True, halfspace_depth=2500.0)
        # The explicit override (2500.0, e.g. a range-dependent profile's
        # own local depth) must be used, NOT self.sedim_layer_max_depth
        # (3000.0).
        self.assertIn("2500.00", bott.lines[1])
        self.assertNotIn("3000.00", bott.lines[1])

    def test_sediment_top_properties_produces_a_gradient(self):
        # sediment_top_properties lets the buffer sediment layer use
        # DIFFERENT properties at its top vs. its bottom (a simple
        # linear gradient, connected by KRAKEN's C-linear SSP
        # interpolation), instead of the isovelocity default (same
        # properties at both ends).
        bott = KrakenBottomHalfspace(
            halfspace_properties={"c_p": 1800.0, "c_s": 0.0, "rho": 1.9, "a_p": 0.3, "a_s": 0.0},
            sediment_top_properties={"c_p": 1650.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.8, "a_s": 0.0},
        )
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        bott.derive_sedim_layer_max_depth(z_max=100.0)
        medium.write_lines(bottom_hs=bott)
        # medium.lines: [header, ssp(0), ssp(100), sedim_header, sedim_top, sedim_bottom]
        self.assertIn("1650.00", medium.lines[4])  # sediment top uses sediment_top_properties
        self.assertNotIn("1650.00", medium.lines[5])
        self.assertIn("1800.00", medium.lines[5])  # sediment bottom uses halfspace_properties
        self.assertNotIn("1800.00", medium.lines[4])

    def test_sediment_top_properties_none_keeps_isovelocity_buffer(self):
        # Default (no sediment_top_properties): both ends of the buffer
        # use the same halfspace_properties -- unchanged behaviour.
        bott = KrakenBottomHalfspace(
            halfspace_properties={"c_p": 1650.0, "c_s": 0.0, "rho": 1.9, "a_p": 0.8, "a_s": 0.0},
        )
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        bott.derive_sedim_layer_max_depth(z_max=100.0)
        medium.write_lines(bottom_hs=bott)
        self.assertIn("1650.00", medium.lines[4])
        self.assertIn("1650.00", medium.lines[5])


# ======================================================================
# KrakenAttenuation
# ======================================================================
class TestKrakenAttenuation(unittest.TestCase):
    def test_default_units_code(self):
        att = KrakenAttenuation()
        self.assertEqual(att.unitscode, "W")
        self.assertEqual(att.thorp_code, " ")

    def test_volume_attenuation_sets_thorp_code(self):
        att = KrakenAttenuation(use_volume_attenuation=True)
        self.assertEqual(att.thorp_code, "T")

    def test_unknown_units_raises(self):
        with self.assertRaises(ValueError):
            KrakenAttenuation(units="not_a_real_unit")

    def test_set_default(self):
        att = KrakenAttenuation(units="dB_per_m", use_volume_attenuation=True)
        att.set_default()
        self.assertEqual(att.units, "dB_per_wavelength")
        self.assertFalse(att.volume_attenuation)


# ======================================================================
# KrakenField
# ======================================================================
class TestKrakenField(unittest.TestCase):
    def test_default_phase_speed_limits_are_applied(self):
        # NOTE: regression test for the fixed bug -- in the original
        # code, this raised IndexError because the default value was
        # discarded (see kraken_env.py module docstring).
        field = KrakenField()
        np.testing.assert_array_equal(field.phase_speed_limits, [0.0, 2000.0])
        field.write_lines()  # must not raise
        self.assertIn("0.0 2000.0", field.lines[0])

    def test_explicit_phase_speed_limits(self):
        field = KrakenField(phase_speed_limits=[1000, 20000])
        np.testing.assert_array_equal(field.phase_speed_limits, [1000, 20000])

    def test_src_depth_atleast_1d(self):
        field = KrakenField(src_depth=50)
        self.assertEqual(field.src_depth.size, 1)
        field2 = KrakenField(src_depth=[10, 20, 30])
        self.assertEqual(field2.src_depth.size, 3)

    def test_write_lines_line_count(self):
        field = KrakenField()
        field.write_lines()
        self.assertEqual(len(field.lines), 6)


# ======================================================================
# Bathymetry
# ======================================================================
class TestBathymetry(TempDirTestCase):
    def test_no_data_file_means_no_bathy(self):
        bathy = Bathymetry()
        self.assertFalse(bathy.use_bathy)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            Bathymetry(data_file=os.path.join(self.tmp_dir, "does_not_exist.csv"))

    def test_flat_bathymetry_detected(self):
        path = os.path.join(self.tmp_dir, "flat.csv")
        _make_bathy_csv(path, [(0, 100), (5, 100), (10, 100)])
        bathy = Bathymetry(data_file=path, units="km")
        self.assertFalse(bathy.use_bathy)

    def test_variable_bathymetry_detected(self):
        path = os.path.join(self.tmp_dir, "variable.csv")
        _make_bathy_csv(path, [(0, 100), (5, 150), (10, 200)])
        bathy = Bathymetry(data_file=path, units="km")
        self.assertTrue(bathy.use_bathy)
        self.assertTrue(hasattr(bathy, "interpolator"))

    def test_units_conversion_m_to_km(self):
        path = os.path.join(self.tmp_dir, "in_meters.csv")
        _make_bathy_csv(path, [(0, 100), (5000, 150)])
        bathy = Bathymetry(data_file=path, units="m")
        np.testing.assert_allclose(bathy.bathy_range, [0.0, 5.0])

    def test_unknown_units_raises(self):
        path = os.path.join(self.tmp_dir, "data.csv")
        _make_bathy_csv(path, [(0, 100), (5, 150)])
        with self.assertRaises(ValueError):
            Bathymetry(data_file=path, units="miles")


# ======================================================================
# KrakenEnv - regression tests for the fixed bugs + core behaviour
# ======================================================================
class TestKrakenEnvFlatBottom(TempDirTestCase):
    def _build_env(self, freq=50.0):
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        return KrakenEnv(
            title="unit test",
            env_root=self.tmp_dir,
            env_filename="testenv",
            freq=freq,
            kraken_medium=medium,
        )

    def test_env_without_bathymetry_does_not_raise(self):
        # NOTE: regression test -- the original code raised
        # AttributeError here (self.bathy.bathy_depth.max() on a
        # Bathymetry() with no data). See kraken_env.py module docstring.
        env = self._build_env()
        self.assertFalse(env.bathy.use_bathy)
        self.assertIsNotNone(env.bottom_hs.sedim_layer_max_depth)

    def test_single_frequency_is_not_broadband(self):
        env = self._build_env(freq=50.0)
        self.assertFalse(env.broadband_run)
        self.assertEqual(env.nominal_frequency, 50.0)

    def test_single_frequency_as_array_does_not_crash(self):
        # NOTE: regression test -- float(self.freq) used to raise
        # TypeError with numpy>=1.25 for a size-1 array. See module
        # docstring.
        env = self._build_env(freq=[50.0])
        self.assertEqual(env.nominal_frequency, 50.0)
        self.assertFalse(env.broadband_run)

    def test_broadband_frequencies_sorted_and_deduped(self):
        env = self._build_env(freq=[50, 10, 50, 30])
        np.testing.assert_array_equal(env.freq, [10.0, 30.0, 50.0])
        self.assertTrue(env.broadband_run)
        self.assertEqual(env.nominal_frequency, 10.0)

    def test_write_env_creates_file_and_sets_flag(self):
        env = self._build_env()
        env.write_env()
        self.assertTrue(os.path.exists(env.env_fpath))
        self.assertFalse(env.range_dependent_env)
        with open(env.env_fpath) as f:
            content = f.read()
        self.assertIn("unit test", content)
        self.assertIn("Nominal frequency", content)

    def test_root_setter_updates_all_paths(self):
        env = self._build_env()
        new_root = os.path.join(self.tmp_dir, "sub")
        os.makedirs(new_root, exist_ok=True)
        env.root = new_root
        self.assertEqual(env.env_fpath, os.path.join(new_root, "testenv.env"))
        self.assertEqual(env.flp_fpath, os.path.join(new_root, "testenv.flp"))
        self.assertEqual(env.shd_fpath, os.path.join(new_root, "testenv.shd"))


class TestKrakenEnvRangeDependent(TempDirTestCase):
    def _build_env(self):
        bathy_path = os.path.join(self.tmp_dir, "bathy.csv")
        _make_bathy_csv(bathy_path, [(0, 100), (5, 150), (10, 200)])
        bathy = Bathymetry(data_file=bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        # NOTE: rcv_z_max is set comfortably beyond the default buffered
        # bottom's sedim_layer_max_depth (~1700 m here, with the default
        # fmin=10 Hz) -- see KrakenEnv.write_range_dependent_lines's
        # docstring for why FIELD.exe requires this for a coupled-mode,
        # buffered-bottom, range-dependent run. Unrelated to what these
        # tests actually check; just needed for write_env() to succeed.
        field = KrakenField(rcv_z_max=2000.0)
        return KrakenEnv(
            title="range dependent test",
            env_root=self.tmp_dir,
            env_filename="rdenv",
            freq=50.0,
            kraken_medium=medium,
            kraken_bathy=bathy,
            kraken_field=field,
        )

    def test_modes_range_sorted_and_includes_zero(self):
        # NOTE: regression test -- in-place .sort() used to raise
        # ValueError with pandas Copy-on-Write. See module docstring.
        env = self._build_env()
        np.testing.assert_array_equal(env.modes_range, [0.0, 5.0, 10.0])

    def test_write_env_produces_one_profile_per_range(self):
        env = self._build_env()
        env.write_env()
        self.assertTrue(env.range_dependent_env)
        with open(env.env_fpath) as f:
            content = f.read()
        self.assertEqual(content.count("range dependent test"), 3)

    # ------------------------------------------------------------------
    # Regression tests for the fixed nmedia / sediment-buffer bug.
    #
    # Root cause: KrakenBottomHalfspace, with the default acousto-elastic
    # bottom, always used to add an extra "buffer sediment layer" medium
    # block, regardless of what 'nmedia' the user declared on KrakenEnv.
    # nmedia=1 (a completely reasonable choice for "one water column, one
    # bottom") then produced a '.env' file DECLARING 1 medium while
    # actually WRITING 2 -- corrupting the file for KRAKEN's Fortran
    # reader. Confirmed against a real KRAKEN/FIELD run that nmedia=1
    # with NO buffer (a direct acousto-elastic half-space right below
    # the water column) is a valid, working configuration.
    # ------------------------------------------------------------------
    def test_nmedia_auto_derived_with_default_buffered_bottom(self):
        env = self._build_env()  # default bottom_hs: acousto_elastic + buffer
        self.assertEqual(env.nmedia, 2)

    def test_nmedia_auto_derived_without_buffer(self):
        bathy_path = os.path.join(self.tmp_dir, "bathy_nobuffer.csv")
        _make_bathy_csv(bathy_path, [(0, 100), (5, 150), (10, 200)])
        bathy = Bathymetry(data_file=bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        bottom_hs = KrakenBottomHalfspace(
            halfspace_properties={"z": 0, "c_p": 1600.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.5, "a_s": 0.0},
            add_sediment_buffer_layer=False,
        )
        env = KrakenEnv(
            title="no buffer", env_root=self.tmp_dir, env_filename="nobuffer",
            freq=100.0, kraken_medium=medium, kraken_bottom_hs=bottom_hs, kraken_bathy=bathy,
        )
        self.assertEqual(env.nmedia, 1)

    def test_mismatched_explicit_nmedia_raises_clear_error(self):
        # NOTE: regression test -- this exact call (nmedia=1 with the
        # default, buffered acousto-elastic bottom) used to silently
        # produce a corrupted '.env' file. It must now fail loudly and
        # explicitly, at construction time, before any file is written.
        bathy_path = os.path.join(self.tmp_dir, "bathy_mismatch.csv")
        _make_bathy_csv(bathy_path, [(0, 100), (5, 150), (10, 200)])
        bathy = Bathymetry(data_file=bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        with self.assertRaises(ValueError):
            KrakenEnv(
                title="mismatch", env_root=self.tmp_dir, env_filename="mismatch",
                freq=100.0, kraken_medium=medium, kraken_bathy=bathy, nmedia=1,
            )

    def test_no_buffer_halfspace_depth_follows_local_profile_depth(self):
        # Reproduces the user-provided reference case: bathymetry with a
        # varying depth (3000 / 3000 / 2500 m at r=0/4/10 km), no buffer
        # sediment layer -> the acousto-elastic half-space line in each
        # profile must sit at THAT profile's own local depth, not a
        # single value shared by every profile (which is instead the
        # correct behaviour when a buffer IS used -- see
        # test_write_env_range_dependent in TestKrakenEnvFlatBottom-style
        # buffered tests above).
        bathy_path = os.path.join(self.tmp_dir, "bathy_variable_depth.csv")
        _make_bathy_csv(bathy_path, [(0, 3000), (4, 3000), (10, 2500)])
        bathy = Bathymetry(data_file=bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 3000], c_p=[1500, 1500])
        bottom_hs = KrakenBottomHalfspace(
            halfspace_properties={"z": 0, "c_p": 1600.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.5, "a_s": 0.0},
            add_sediment_buffer_layer=False,
        )
        env = KrakenEnv(
            title="variable depth", env_root=self.tmp_dir, env_filename="variabledepth",
            freq=100.0, kraken_medium=medium, kraken_bottom_hs=bottom_hs, kraken_bathy=bathy,
        )
        env.write_env()
        with open(env.env_fpath) as f:
            content = f.read()

        # Exactly one medium block per profile (no buffer): 3 profiles,
        # so "Number of mesh points" must appear exactly 3 times.
        self.assertEqual(content.count("Number of mesh points"), 3)
        # Each profile's half-space line uses ITS OWN local depth.
        self.assertIn("3000.00 1600.00", content)  # r=0 and r=4 km
        self.assertIn("2500.00 1600.00", content)  # r=10 km

    def test_medium_truncated_to_local_depth(self):
        env = self._build_env()
        medium_at_5km = env._medium_truncated_to_depth(150.0)
        self.assertAlmostEqual(medium_at_5km.z_ssp.max(), 150.0)
        # cp should be interpolated (constant profile here -> stays 1500)
        self.assertAlmostEqual(medium_at_5km.cp_ssp[-1], 1500.0)

    # ------------------------------------------------------------------
    # Regression tests for the fixed "deepest point not at r=0" bug.
    #
    # Root cause: with a DIRECT half-space bottom
    # (add_sediment_buffer_layer=False), FIELD.exe crashes with a
    # cryptic Fortran runtime error ('Non-existing record number') --
    # or, depending on version, "Fatal Error: modes must be tabulated
    # throughout the ocean and sediment to compute the coupling coefs."
    # -- whenever the FIRST profile (r=0) is not the single deepest
    # point along the whole bathymetry. Confirmed with a real
    # KRAKEN/FIELD run (a failing and a working reproduction of the
    # exact same environment, differing only in how deep the first
    # profile's tabulation reaches). A buffer sediment layer
    # (add_sediment_buffer_layer=True) avoids this entirely: its
    # thickness is derived from the bathymetry's GLOBAL maximum depth
    # (see __init__), not each profile's own local depth.
    # ------------------------------------------------------------------
    def test_direct_halfspace_raises_when_deepest_point_is_not_at_r0(self):
        bathy_path = os.path.join(self.tmp_dir, "bathy_undulating.csv")
        _make_bathy_csv(bathy_path, [(0, 100), (5, 80), (10, 110), (15, 120)])
        bathy = Bathymetry(data_file=bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 120], c_p=[1500, 1500])
        bottom_hs = KrakenBottomHalfspace(
            halfspace_properties={"z": 0, "c_p": 1650.0, "c_s": 0.0, "rho": 1.8, "a_p": 0.8, "a_s": 0.0},
            add_sediment_buffer_layer=False,
        )
        env = KrakenEnv(
            title="undulating", env_root=self.tmp_dir, env_filename="undulating",
            freq=300.0, kraken_medium=medium, kraken_bottom_hs=bottom_hs, kraken_bathy=bathy,
        )
        with self.assertRaises(ValueError) as ctx:
            env.write_env()
        self.assertIn("deepest point at r=0", str(ctx.exception))
        self.assertIn("add_sediment_buffer_layer=True", str(ctx.exception))

    def test_direct_halfspace_does_not_raise_when_deepest_point_is_at_r0(self):
        # Same shape of bathymetry as above, but monotonically
        # decreasing (deepest point genuinely at r=0) -- matches this
        # project's other range-dependent "wedge"-style examples, which
        # never triggered this bug.
        bathy_path = os.path.join(self.tmp_dir, "bathy_wedge.csv")
        _make_bathy_csv(bathy_path, [(0, 200), (5, 200), (10, 50)])
        bathy = Bathymetry(data_file=bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 200], c_p=[1500, 1500])
        bottom_hs = KrakenBottomHalfspace(
            halfspace_properties={"z": 0, "c_p": 1700.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.5, "a_s": 0.0},
            add_sediment_buffer_layer=False,
        )
        env = KrakenEnv(
            title="wedge", env_root=self.tmp_dir, env_filename="wedge",
            freq=25.0, kraken_medium=medium, kraken_bottom_hs=bottom_hs, kraken_bathy=bathy,
        )
        env.write_env()  # must not raise

    def test_buffered_bottom_does_not_raise_regardless_of_bathymetry_order(self):
        # The exact bathymetry that raises above must NOT raise when a
        # buffer sediment layer is used instead -- this is the
        # documented fix.
        bathy_path = os.path.join(self.tmp_dir, "bathy_undulating2.csv")
        _make_bathy_csv(bathy_path, [(0, 100), (5, 80), (10, 110), (15, 120)])
        bathy = Bathymetry(data_file=bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 120], c_p=[1500, 1500])
        bottom_hs = KrakenBottomHalfspace(
            halfspace_properties={"z": 0, "c_p": 1650.0, "c_s": 0.0, "rho": 1.8, "a_p": 0.8, "a_s": 0.0},
            add_sediment_buffer_layer=True,
            fmin=100.0,
        )
        env = KrakenEnv(
            title="undulating buffered", env_root=self.tmp_dir, env_filename="undulating_buffered",
            freq=300.0, kraken_medium=medium, kraken_bottom_hs=bottom_hs, kraken_bathy=bathy,
        )
        env.write_env()  # must not raise

        # And the resulting buffer must reach past the GLOBAL max depth
        # (120 m), not just the first profile's own local depth (100 m).
        self.assertGreater(bottom_hs.sedim_layer_max_depth, 120.0)

    # ------------------------------------------------------------------
    # Regression tests for the fixed "receivers must reach the bottom
    # of medium 2" bug.
    #
    # Root cause: FIELD.exe requires the receiver depth grid (both the
    # one written into each profile block of the '.env', via
    # KrakenField, and the one in the '.flp', via KrakenFlp) to extend
    # all the way down to the bottom of the buffer sediment layer when
    # one is used, or it fails with "Fatal Error: modes must be
    # tabulated throughout the ocean and sediment to compute the
    # coupling coefs." Confirmed with a real KRAKEN/FIELD run and its
    # field.prt output, which cites the exact mismatched depths.
    # ------------------------------------------------------------------
    def _build_buffered_env(self, rcv_z_max):
        bathy_path = os.path.join(self.tmp_dir, "bathy_buffered.csv")
        _make_bathy_csv(bathy_path, [(0, 100), (5, 80), (10, 110), (15, 120)])
        bathy = Bathymetry(data_file=bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 120], c_p=[1500, 1500])
        bottom_hs = KrakenBottomHalfspace(
            halfspace_properties={"z": 0, "c_p": 1650.0, "c_s": 0.0, "rho": 1.8, "a_p": 0.8, "a_s": 0.0},
            add_sediment_buffer_layer=True,
            fmin=100.0,  # -> sedim_layer_max_depth = 300.0 m (120 + 150, rounded)
        )
        field = KrakenField(rcv_z_max=rcv_z_max)
        env = KrakenEnv(
            title="buffered rcv depth test", env_root=self.tmp_dir, env_filename="rcvdepth",
            freq=300.0, kraken_medium=medium, kraken_bottom_hs=bottom_hs, kraken_bathy=bathy,
            kraken_field=field,
        )
        return env, bottom_hs

    def test_env_write_auto_extends_receivers_to_sediment_bottom(self):
        # NOTE: behaviour changed from "raise ValueError" to "auto-fix"
        # -- see KrakenEnv.write_range_dependent_lines's docstring/NOTE.
        env, bottom_hs = self._build_buffered_env(rcv_z_max=120.0)  # water depth only
        env.write_env()  # must NOT raise -- auto-extended instead
        self.assertEqual(env.field.rcv_depth_max, bottom_hs.sedim_layer_max_depth)
        with open(env.env_fpath) as f:
            content = f.read()
        self.assertIn(f"0.0 {bottom_hs.sedim_layer_max_depth:.1f}", content)

    def test_env_write_leaves_sufficient_rcv_z_max_untouched(self):
        env, bottom_hs = self._build_buffered_env(rcv_z_max=300.0)
        env.write_env()  # must not raise
        self.assertEqual(env.field.rcv_depth_max, 300.0)

    def test_flp_never_touches_rcv_z_max(self):
        # NOTE: regression test for a CORRECTION to an earlier version
        # of this fix, which used to also auto-extend KrakenFlp's own
        # 'rcv_z_max' (mirroring KrakenEnv's). Confirmed against a real
        # KRAKEN/FIELD run that this was an overreach: the '.flp' file's
        # receiver depth grid is the PRESSURE-FIELD OUTPUT grid (where
        # FIELD.exe reports pressure), entirely independent of the
        # '.env' file's mode-tabulation depth that the coupling-
        # coefficient computation actually needs -- the user is free to
        # request pressure at a single, shallow (r, z) point regardless
        # of how deep the sediment buffer extends. KrakenFlp must leave
        # 'rcv_z_max' exactly as given, in every case.
        env, bottom_hs = self._build_buffered_env(rcv_z_max=300.0)
        env.write_env()
        for theory in ("coupled", "adiabatic"):
            flp = KrakenFlp(env=env, src_depth=20.0, mode_theory=theory, rcv_z_max=120.0)
            self.assertEqual(flp.rcv_z_max, 120.0)

    def test_flp_never_touches_rcv_z_max_even_at_a_single_shallow_point(self):
        # The extreme case explicitly called out above: a single,
        # shallow receiver depth, far shallower than the sediment
        # buffer -- must be accepted as-is.
        env, bottom_hs = self._build_buffered_env(rcv_z_max=300.0)
        env.write_env()
        flp = KrakenFlp(
            env=env, src_depth=20.0, mode_theory="coupled",
            rcv_z_min=10.0, rcv_z_max=10.0, n_rcv_z=1,
        )
        self.assertEqual(flp.rcv_z_max, 10.0)

    def test_flat_bottom_never_triggers_the_check(self):
        # No bathymetry at all (flat bottom, range-independent by
        # definition) -> self.bathy.use_bathy is False -> the new check
        # must be a complete no-op, regardless of add_sediment_buffer_layer.
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        bottom_hs = KrakenBottomHalfspace(
            halfspace_properties={"z": 0, "c_p": 1650.0, "c_s": 0.0, "rho": 1.8, "a_p": 0.8, "a_s": 0.0},
            add_sediment_buffer_layer=False,
        )
        env = KrakenEnv(
            title="flat", env_root=self.tmp_dir, env_filename="flat",
            freq=100.0, kraken_medium=medium, kraken_bottom_hs=bottom_hs,
        )
        env.write_env()  # must not raise; also not range-dependent at all
        self.assertFalse(env.range_dependent_env)


# ======================================================================
# KrakenFlp
# ======================================================================
class TestKrakenFlp(TempDirTestCase):
    def _build_flat_env(self):
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        env = KrakenEnv(
            title="flp test", env_root=self.tmp_dir, env_filename="flpenv",
            freq=50.0, kraken_medium=medium,
        )
        env.write_env()
        return env

    def test_default_codes(self):
        env = self._build_flat_env()
        flp = KrakenFlp(env=env, src_depth=50)
        self.assertEqual(flp.src_code, "R")
        self.assertEqual(flp.th_code, "A")
        self.assertEqual(flp.add_code, "C")

    def test_line_source_and_coupled_incoherent(self):
        env = self._build_flat_env()
        flp = KrakenFlp(
            env=env, src_depth=50, src_type="line_source",
            mode_theory="coupled", mode_addition="incoherent",
        )
        self.assertEqual(flp.src_code, "X")
        self.assertEqual(flp.th_code, "C")
        self.assertEqual(flp.add_code, "I")

    def test_unknown_src_type_raises(self):
        env = self._build_flat_env()
        with self.assertRaises(ValueError):
            KrakenFlp(env=env, src_type="not_a_real_source")

    def test_flat_env_has_single_profile(self):
        env = self._build_flat_env()
        flp = KrakenFlp(env=env, src_depth=50)
        self.assertEqual(flp.n_profiles, 1)
        np.testing.assert_array_equal(flp.profiles_ranges, [0.0])

    def test_range_dependent_env_has_multiple_profiles(self):
        bathy_path = os.path.join(self.tmp_dir, "bathy.csv")
        _make_bathy_csv(bathy_path, [(0, 100), (5, 150), (10, 200)])
        bathy = Bathymetry(data_file=bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        field = KrakenField(rcv_z_max=2000.0)  # see TestKrakenEnvRangeDependent._build_env
        env = KrakenEnv(
            title="rd", env_root=self.tmp_dir, env_filename="rdflp",
            freq=50.0, kraken_medium=medium, kraken_bathy=bathy, kraken_field=field,
        )
        env.write_env()
        flp = KrakenFlp(env=env, src_depth=50, rcv_z_max=2000.0)
        self.assertEqual(flp.n_profiles, 3)

    def test_write_flp_creates_file(self):
        env = self._build_flat_env()
        flp = KrakenFlp(env=env, src_depth=50)
        flp.write_flp()
        self.assertTrue(os.path.exists(flp.flp_fpath))
        with open(flp.flp_fpath) as f:
            content = f.read()
        self.assertIn("flp test", content)


# ======================================================================
# plot_env / plot_medium / plot_bottom_halfspace -- sediment gradient fix
# ======================================================================
class TestPlotEnvSedimentGradient(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def _build_env_with_gradient(self):
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        bottom_hs = KrakenBottomHalfspace(
            halfspace_properties={"c_p": 1800.0, "c_s": 0.0, "rho": 1.9, "a_p": 0.3, "a_s": 0.0},
            sediment_top_properties={"c_p": 1600.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.8, "a_s": 0.0},
            fmin=100.0, alpha_wavelength=10,
        )
        env = KrakenEnv(
            title="gradient test", env_root="/tmp", env_filename="gradient_plot_test",
            freq=100.0, kraken_medium=medium, kraken_bottom_hs=bottom_hs,
        )
        return env, bottom_hs

    def test_plot_env_reflects_sediment_gradient(self):
        # NOTE: regression test for the fixed bug -- plot_env() used to
        # broadcast the SAME terminal halfspace value (cp_bot_halfspace)
        # to both ends of the sediment layer, always drawing a flat
        # (isovelocity) bottom even when a genuine gradient was
        # configured via 'sediment_top_properties' -- which the '.env'
        # file itself already wrote correctly. The plotted celerity
        # curve's two bottom points must now show the actual top/bottom
        # values (1600 -> 1800), not the same value twice.
        env, bottom_hs = self._build_env_with_gradient()
        fig = env.plot_env()
        cwave_line = next(ln for ln in fig.axes[0].get_lines() if ln.get_label() == "C-wave")
        cp_values = cwave_line.get_xdata()
        # Last two points are the sediment layer's top and bottom.
        self.assertAlmostEqual(cp_values[-2], 1600.0)
        self.assertAlmostEqual(cp_values[-1], 1800.0)

    def test_plot_env_density_reflects_sediment_gradient(self):
        env, bottom_hs = self._build_env_with_gradient()
        fig = env.plot_env()
        rho_line = fig.axes[2].get_lines()[0]
        rho_values = rho_line.get_xdata()
        self.assertAlmostEqual(rho_values[-2], 1.5)
        self.assertAlmostEqual(rho_values[-1], 1.9)

    def test_plot_env_flat_bottom_when_no_gradient_configured(self):
        # No regression for the common (non-gradient) case: both ends
        # of the sediment layer must still show the SAME value.
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        bottom_hs = KrakenBottomHalfspace(
            halfspace_properties={"c_p": 1650.0, "c_s": 0.0, "rho": 1.8, "a_p": 0.8, "a_s": 0.0},
        )
        env = KrakenEnv(
            title="flat test", env_root="/tmp", env_filename="flat_plot_test",
            freq=100.0, kraken_medium=medium, kraken_bottom_hs=bottom_hs,
        )
        fig = env.plot_env()
        cwave_line = next(ln for ln in fig.axes[0].get_lines() if ln.get_label() == "C-wave")
        cp_values = cwave_line.get_xdata()
        self.assertAlmostEqual(cp_values[-2], cp_values[-1])

    def test_plot_env_direct_halfspace_still_flat(self):
        # The genuinely semi-infinite, no-buffer case (see
        # KrakenBottomHalfspace's docstring) has no real gradient
        # concept -- must still render as a flat symbolic extension.
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        bottom_hs = KrakenBottomHalfspace(
            halfspace_properties={"z": 0, "c_p": 1700.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.5, "a_s": 0.0},
            add_sediment_buffer_layer=False,
        )
        env = KrakenEnv(
            title="direct test", env_root="/tmp", env_filename="direct_plot_test",
            freq=100.0, kraken_medium=medium, kraken_bottom_hs=bottom_hs,
        )
        fig = env.plot_env()  # must not raise
        cwave_line = next(ln for ln in fig.axes[0].get_lines() if ln.get_label() == "C-wave")
        cp_values = cwave_line.get_xdata()
        self.assertAlmostEqual(cp_values[-2], cp_values[-1])

    def test_plot_bottom_halfspace_reflects_sediment_gradient(self):
        _, bottom_hs = self._build_env_with_gradient()
        fig = bottom_hs.plot_bottom_halfspace()
        cwave_line = next(ln for ln in fig.axes[0].get_lines() if ln.get_label() == "C-wave")
        cp_values = cwave_line.get_xdata()
        self.assertAlmostEqual(cp_values[0], 1600.0)
        self.assertAlmostEqual(cp_values[1], 1800.0)


if __name__ == "__main__":
    unittest.main()
