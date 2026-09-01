#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for propa/kraken_toolbox/src/kraken_testcase.py.

These tests focus on:
  - the mutable-default-argument bug fix (the main motivation for this
    file's refactor): creating several KrakenTestCase/KrakenProperties
    instances without explicit properties must NOT let one instance's
    mutations leak into another;
  - the DomainProperties/ReceiverProperties unit conversion and
    validation;
  - the directory tree creation and bathymetry handling logic.

They do NOT invoke kraken.exe / field.exe (KrakenTestCase.run() is not
exercised here).

Run with either:
    python -m unittest discover -s propa/kraken_toolbox/tests
or (if pytest is installed in your environment):
    pytest propa/kraken_toolbox/tests
"""
import os
import shutil
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

from propa.kraken_toolbox.src.kraken_env import Bathymetry, KrakenBottomHalfspace
from propa.kraken_toolbox.src.kraken_testcase import (
    DomainProperties,
    ReceiverProperties,
    SourceProperties,
    KrakenProperties,
    KrakenTestCase,
)


def _make_bathy_csv(path, rows):
    with open(path, "w") as f:
        for r, h in rows:
            f.write(f"{r},{h}\n")


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="kraken_testcase_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


# ======================================================================
# DomainProperties / ReceiverProperties
# ======================================================================
class TestDomainProperties(unittest.TestCase):
    def test_units_in_meters(self):
        d = DomainProperties(zmin=0, zmax=500, rmin=0, rmax=2000, unit="m")
        self.assertEqual(d.zmax_m, 500)
        self.assertEqual(d.rmax_km, 2.0)

    def test_units_in_km(self):
        d = DomainProperties(zmin=0, zmax=1, rmin=0, rmax=5, unit="km")
        self.assertEqual(d.zmax_m, 1000)
        self.assertEqual(d.rmax_km, 5)

    def test_unknown_unit_raises(self):
        # NOTE: regression test -- the original code raised an
        # UnboundLocalError here (see module docstring).
        with self.assertRaises(ValueError):
            DomainProperties(unit="miles")


class TestReceiverProperties(unittest.TestCase):
    def test_same_behaviour_as_domain_properties(self):
        r = ReceiverProperties(zmin=0, zmax=800, rmin=0, rmax=3000, unit="m")
        self.assertEqual(r.zmax_m, 800)
        self.assertEqual(r.rmax_km, 3.0)

    def test_unknown_unit_raises(self):
        with self.assertRaises(ValueError):
            ReceiverProperties(unit="miles")


# ======================================================================
# KrakenProperties - mutable default argument regression tests
# ======================================================================
class TestKrakenPropertiesMutableDefaults(unittest.TestCase):
    def test_two_default_instances_do_not_share_field(self):
        # NOTE: regression test for the fixed bug -- in the original
        # code, all KrakenProperties() built without an explicit 'field'
        # argument shared and could mutate the SAME KrakenField instance.
        kp1 = KrakenProperties()
        kp2 = KrakenProperties()
        self.assertIsNot(kp1.field, kp2.field)
        self.assertIsNot(kp1.medium, kp2.medium)
        self.assertIsNot(kp1.bott_hs, kp2.bott_hs)
        self.assertIsNot(kp1.top_hs, kp2.top_hs)
        self.assertIsNot(kp1.att, kp2.att)

    def test_mutating_one_instance_does_not_affect_the_other(self):
        kp1 = KrakenProperties()
        kp2 = KrakenProperties()
        kp1.field.n_rcv_z = 999
        self.assertNotEqual(kp2.field.n_rcv_z, 999)

    def test_default_nmedia_is_none_and_derived_by_krakenenv(self):
        # NOTE: regression test tied to the nmedia/sediment-buffer fix
        # in kraken_env.py -- KrakenProperties used to hard-code
        # nmedia=2 (matching the default bott_hs's automatic buffer
        # layer only by coincidence). If a caller swapped in a
        # different bott_hs (e.g. one with
        # add_sediment_buffer_layer=False, or a vacuum bottom) without
        # remembering to also update this hard-coded value, KrakenEnv
        # would silently write a corrupted '.env' file. Defaulting to
        # None lets KrakenEnv derive the correct value automatically,
        # regardless of which bott_hs is used.
        kp = KrakenProperties()
        self.assertIsNone(kp.nmedia)


# ======================================================================
# KrakenTestCase - mutable default argument regression tests
# ======================================================================
class TestKrakenTestCaseMutableDefaults(TempDirTestCase):
    def test_two_default_testcases_do_not_share_properties(self):
        # NOTE: regression test for the fixed bug -- see module
        # docstring. In the original code,
        # `tc1.kraken is tc2.kraken` was True.
        tc1 = KrakenTestCase(name="tc1", root_dir=self.tmp_dir)
        tc2 = KrakenTestCase(name="tc2", root_dir=self.tmp_dir)
        self.assertIsNot(tc1.kraken, tc2.kraken)
        self.assertIsNot(tc1.domain, tc2.domain)
        self.assertIsNot(tc1.src, tc2.src)
        self.assertIsNot(tc1.rcv, tc2.rcv)

    def test_variable_bathymetry_does_not_leak_into_next_flat_testcase(self):
        # This reproduces the exact scenario that silently corrupted
        # results in the original code: a testcase with a variable
        # bathymetry (which mutates kraken.field.n_rcv_z /
        # rcv_depth_max) followed by a plain flat-bottom testcase built
        # with no explicit properties.
        bathy_path = os.path.join(self.tmp_dir, "bathy_in.csv")
        _make_bathy_csv(bathy_path, [(0, 500), (5, 800), (10, 1200)])
        bathy = Bathymetry(bathy_path, units="km")

        tc_variable = KrakenTestCase(name="tc_variable", root_dir=self.tmp_dir, bathy=bathy)
        tc_flat = KrakenTestCase(name="tc_flat", root_dir=self.tmp_dir)

        # The flat testcase must use the untouched default field
        # settings, independent of what tc_variable computed.
        default_field = KrakenProperties().field
        self.assertEqual(tc_flat.kraken.field.n_rcv_z, default_field.n_rcv_z)
        self.assertEqual(tc_flat.kraken.field.rcv_depth_max, default_field.rcv_depth_max)
        # Sanity check: tc_variable's field WAS actually updated (bathymetry active)
        self.assertNotEqual(tc_variable.kraken.field.rcv_depth_max, default_field.rcv_depth_max)


# ======================================================================
# Directory tree / bathymetry / env+flp wiring
# ======================================================================
class TestKrakenTestCaseDirsAndFiles(TempDirTestCase):
    def test_directory_tree_created(self):
        tc = KrakenTestCase(name="tc1", root_dir=self.tmp_dir)
        for d in (tc.testcase_directory, tc.io_files_dir, tc.imgs_dir, tc.imgs_env_dir, tc.imgs_outputs_dir):
            self.assertTrue(os.path.isdir(d))

    def test_flat_bottom_by_default(self):
        tc = KrakenTestCase(name="tc1", root_dir=self.tmp_dir)
        self.assertFalse(tc.env.range_dependent_env)

    def test_env_and_flp_files_written(self):
        tc = KrakenTestCase(name="tc1", root_dir=self.tmp_dir)
        self.assertTrue(os.path.exists(tc.env.env_fpath))
        self.assertTrue(os.path.exists(tc.flp.flp_fpath))

    def test_bathy_csv_written_alongside_io_files(self):
        tc = KrakenTestCase(name="tc1", root_dir=self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(tc.io_files_dir, "bathy.csv")))

    def test_variable_bathymetry_truncated_to_domain(self):
        bathy_path = os.path.join(self.tmp_dir, "bathy_in.csv")
        # One point beyond the default domain rmax (10 km)
        _make_bathy_csv(bathy_path, [(0, 500), (5, 800), (20, 5000)])
        bathy = Bathymetry(bathy_path, units="km")

        tc = KrakenTestCase(name="tc1", root_dir=self.tmp_dir, bathy=bathy)

        self.assertTrue(max(tc.bathy.bathy_range) <= tc.domain.rmax_km)
        # zmax_m must reflect the truncated (not the raw) max depth
        self.assertEqual(tc.domain.zmax_m, 800)

    def test_run_method_delegates_to_kraken_manager(self):
        from unittest import mock
        tc = KrakenTestCase(name="tc1", root_dir=self.tmp_dir)
        with mock.patch(
            "propa.kraken_toolbox.src.kraken_testcase.KrakenManager"
        ) as MockManager:
            tc.run()
            MockManager.assert_called_once_with()
            MockManager.return_value.runkraken.assert_called_once_with(
                env=tc.env, flp=tc.flp, frequencies=tc.src.freq
            )

    def test_testcase_with_no_buffer_bottom_and_variable_bathymetry(self):
        # End-to-end reproduction, through the high-level KrakenTestCase
        # API, of the user-reported scenario: a range-dependent
        # environment with a single water-column medium and a direct
        # acousto-elastic bottom (no buffer sediment layer). Before the
        # fix, this required manually overriding 'nmedia' in
        # KrakenProperties to a value matching the (undocumented, always
        # added) buffer layer; now it just works by passing a bott_hs
        # with add_sediment_buffer_layer=False and leaving nmedia=None.
        bathy_path = os.path.join(self.tmp_dir, "bathy_variable.csv")
        _make_bathy_csv(bathy_path, [(0, 3000), (4, 3000), (10, 2500)])
        bathy = Bathymetry(bathy_path, units="km")

        bott_hs = KrakenBottomHalfspace(
            halfspace_properties={"z": 0, "c_p": 1600.0, "c_s": 0.0, "rho": 1.5, "a_p": 0.5, "a_s": 0.0},
            add_sediment_buffer_layer=False,
        )
        kraken_properties = KrakenProperties(bott_hs=bott_hs)  # nmedia stays None -> derived

        tc = KrakenTestCase(
            name="tc_no_buffer", root_dir=self.tmp_dir, bathy=bathy,
            kraken_properties=kraken_properties,
        )

        self.assertEqual(tc.env.nmedia, 1)
        self.assertTrue(tc.env.range_dependent_env)
        with open(tc.env.env_fpath) as f:
            content = f.read()
        # One medium block per profile (3 profiles, no buffer).
        self.assertEqual(content.count("Number of mesh points"), 3)


if __name__ == "__main__":
    unittest.main()
