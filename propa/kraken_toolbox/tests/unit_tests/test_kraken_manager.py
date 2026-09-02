#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for propa/kraken_toolbox/src/kraken_manager.py.

These tests exercise the pure orchestration logic (frequency
distribution, shell command construction, working directory management,
delegation to read_shd) using mocks. They deliberately do NOT invoke the
real kraken.exe / field.exe binaries.

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

from propa.kraken_toolbox.src.kraken_manager import KrakenManager


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="kraken_manager_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


# ======================================================================
# Frequency distribution across workers
# ======================================================================
class TestAssignFrequencyIntervalls(unittest.TestCase):
    def test_equally_distributed_basic_split(self):
        freqs = np.array([10, 20, 30, 40, 50, 60, 70])
        ranges, n_used = KrakenManager.assign_frequency_intervalls(
            freqs, 3, mode="equally_distributed"
        )
        self.assertEqual(n_used, 3)
        # Every frequency must appear exactly once, across all batches
        all_freqs = np.concatenate(ranges)
        np.testing.assert_array_equal(np.sort(all_freqs), np.sort(freqs))

    def test_equally_distributed_more_workers_than_freqs_drops_empty_batches(self):
        freqs = np.array([10, 20, 30])
        ranges, n_used = KrakenManager.assign_frequency_intervalls(
            freqs, 10, mode="equally_distributed"
        )
        # No empty batch should be returned
        self.assertTrue(all(len(r) > 0 for r in ranges))
        self.assertEqual(n_used, len(ranges))
        self.assertLessEqual(n_used, 10)

    def test_optimal_mode_falls_back_when_fewer_freqs_than_workers(self):
        freqs = np.array([10, 20, 30])
        ranges, n_used = KrakenManager.assign_frequency_intervalls(
            freqs, 10, mode="optimal"
        )
        self.assertEqual(n_used, 3)
        all_freqs = np.concatenate(ranges)
        np.testing.assert_array_equal(np.sort(all_freqs), np.sort(freqs))

    def test_optimal_mode_covers_all_frequencies(self):
        freqs = np.array([10, 20, 30, 40, 50, 60, 70])
        ranges, n_used = KrakenManager.assign_frequency_intervalls(
            freqs, 3, mode="optimal"
        )
        all_freqs = np.concatenate(ranges)
        np.testing.assert_array_equal(np.sort(all_freqs), np.sort(freqs))
        self.assertLessEqual(n_used, 3)

    def test_unknown_mode_raises(self):
        freqs = np.array([10, 20, 30])
        with self.assertRaises(ValueError):
            KrakenManager.assign_frequency_intervalls(freqs, 2, mode="not_a_real_mode")


# ======================================================================
# Shell command construction (run_exec) - os.system is mocked, no real
# process is ever launched. os.name is explicitly patched in every test
# so the expected command does not depend on the OS actually running the
# test suite (these tests must pass identically on Linux, macOS, and
# Windows CI/dev machines).
# ======================================================================
class TestRunExec(unittest.TestCase):
    def test_kraken_exec_silent_by_default_posix(self):
        calls = []
        with mock.patch("os.name", "posix"):
            with mock.patch("os.system", side_effect=lambda c: calls.append(c)):
                KrakenManager.run_kraken_exec("myenv")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("kraken myenv"))
        self.assertNotIn(".exe", calls[0])
        # silent=True by default for kraken -> redirection present
        self.assertIn(">/dev/null", calls[0])

    def test_kraken_exec_silent_by_default_windows(self):
        calls = []
        with mock.patch("os.name", "nt"):
            with mock.patch("os.system", side_effect=lambda c: calls.append(c)):
                KrakenManager.run_kraken_exec("myenv")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("kraken.exe myenv"))
        self.assertIn(">NUL", calls[0])

    def test_field_exec_not_silent_by_default_posix(self):
        calls = []
        with mock.patch("os.name", "posix"):
            with mock.patch("os.system", side_effect=lambda c: calls.append(c)):
                KrakenManager.run_field_exec("myenv")
        self.assertEqual(calls, ["field myenv"])

    def test_field_exec_not_silent_by_default_windows(self):
        calls = []
        with mock.patch("os.name", "nt"):
            with mock.patch("os.system", side_effect=lambda c: calls.append(c)):
                KrakenManager.run_field_exec("myenv")
        self.assertEqual(calls, ["field.exe myenv"])

    def test_parallel_without_worker_pid_raises_on_windows(self):
        with mock.patch("os.name", "nt"):
            with self.assertRaises(ValueError):
                KrakenManager.run_exec(
                    exec="kraken", filename="myenv", parallel=True,
                    worker_pid=None, silent=True,
                )

    def test_parallel_ignored_on_posix(self):
        # 'parallel' only changes the command on Windows (see
        # run_exec docstring): on posix it is a no-op, so no
        # worker_pid is required even when parallel=True.
        calls = []
        with mock.patch("os.name", "posix"):
            with mock.patch("os.system", side_effect=lambda c: calls.append(c)):
                KrakenManager.run_exec(
                    exec="kraken", filename="myenv", parallel=True,
                    worker_pid=None, silent=False,
                )
        self.assertEqual(calls, ["kraken myenv"])

    def test_parallel_windows_uses_local_bin_folder(self):
        calls = []
        with mock.patch("os.name", "nt"), mock.patch("os.getcwd", return_value=r"C:\work"):
            with mock.patch("os.system", side_effect=lambda c: calls.append(c)):
                KrakenManager.run_exec(
                    exec="kraken", filename="myenv", parallel=True,
                    worker_pid=1234, silent=False,
                )
        self.assertEqual(len(calls), 1)
        self.assertIn("bin", calls[0])
        self.assertTrue(calls[0].endswith(".exe myenv"))


# ======================================================================
# Parallel working directory management
# ======================================================================
class TestWorkingDirs(TempDirTestCase):
    def test_get_subprocess_working_dir_creates_directory(self):
        path = KrakenManager.get_subprocess_working_dir(self.tmp_dir, worker_pid=42)
        self.assertTrue(os.path.isdir(path))
        self.assertIn("child_process_42", path)

    def test_clear_kraken_parallel_working_dir_removes_subdirs(self):
        sub1 = KrakenManager.get_subprocess_working_dir(self.tmp_dir, worker_pid=1)
        sub2 = KrakenManager.get_subprocess_working_dir(self.tmp_dir, worker_pid=2)
        self.assertTrue(os.path.isdir(sub1))
        self.assertTrue(os.path.isdir(sub2))

        KrakenManager.clear_kraken_parallel_working_dir(root=self.tmp_dir)

        self.assertFalse(os.path.isdir(sub1))
        self.assertFalse(os.path.isdir(sub2))

    def test_clear_kraken_parallel_working_dir_no_op_if_missing(self):
        # Must not raise even if 'parallel_working_dir' does not exist yet
        KrakenManager.clear_kraken_parallel_working_dir(root=self.tmp_dir)


# ======================================================================
# readshd / readshd_bin delegation (no duplicated parsing logic anymore)
# ======================================================================
class TestReadshdDelegation(unittest.TestCase):
    def test_readshd_delegates_to_read_shd_module(self):
        sentinel = object()
        with mock.patch(
            "propa.kraken_toolbox.src.kraken_manager.readshd", return_value=sentinel
        ) as mocked:
            result = KrakenManager.readshd("file.shd", freq=50)
        mocked.assert_called_once_with(filename="file.shd", xs=None, ys=None, freq=50)
        self.assertIs(result, sentinel)

    def test_readshd_bin_delegates_to_read_shd_module(self):
        sentinel = object()
        with mock.patch(
            "propa.kraken_toolbox.src.kraken_manager.readshd_bin", return_value=sentinel
        ) as mocked:
            result = KrakenManager.readshd_bin("file.shd", freq=50)
        mocked.assert_called_once_with(filename="file.shd", xs=None, ys=None, freq=50)
        self.assertIs(result, sentinel)


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
REAL_MOD_PATH = os.path.join(FIXTURES_DIR, "real_kraken.mod")


# ======================================================================
# runkraken_broadband_range_dependent -- per-frequency mode collection
# ======================================================================
@unittest.skipUnless(os.path.exists(REAL_MOD_PATH), "real_kraken.mod fixture not present")
class TestRunkrakenBroadbandRangeDependentModesCollection(TempDirTestCase):
    def _build_range_dependent_env(self, env_filename):
        from propa.kraken_toolbox.src.kraken_env import (
            KrakenEnv, KrakenMedium, KrakenField, Bathymetry,
        )

        bathy_path = os.path.join(self.tmp_dir, "bathy.csv")
        with open(bathy_path, "w") as f:
            f.write("0,100\n5,150\n")
        bathy = Bathymetry(bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        # NOTE: rcv_z_max is set comfortably beyond the default buffered
        # bottom's sedim_layer_max_depth -- see
        # KrakenEnv.write_range_dependent_lines's docstring for why
        # FIELD.exe requires this for a coupled-mode, buffered-bottom,
        # range-dependent run. Unrelated to what this test actually
        # checks; just needed for write_env() to succeed.
        field = KrakenField(rcv_z_max=2000.0)
        env = KrakenEnv(
            title="t", env_root=self.tmp_dir, env_filename=env_filename,
            freq=[10.0, 20.0], kraken_medium=medium, kraken_bathy=bathy, kraken_field=field,
        )
        env.write_env()
        return env

    def test_modes_are_collected_per_frequency_before_being_overwritten(self):
        # NOTE: regression test for the fixed bug -- each frequency's
        # '.mod' file used to be silently overwritten by the next
        # iteration, with no attempt to read it in the meantime: a
        # caller reading the '.mod' file AFTER this loop only ever saw
        # the last frequency's modes (or hit a FileNotFoundError,
        # since the file that does exist afterwards sits in a
        # different, per-worker subdirectory than a naive caller would
        # expect). readmodes() must now be called once per frequency,
        # INSIDE the loop, right after that frequency's '.mod' file is
        # written, with the results collected in order.
        env_filename = "rdenv"
        env = self._build_range_dependent_env(env_filename)

        class DummyFlp:
            def __init__(self, flp_fpath):
                self.flp_fpath = flp_fpath

            def write_flp(self):
                with open(self.flp_fpath, "w") as f:
                    f.write("dummy flp\n")

        flp = DummyFlp(os.path.join(self.tmp_dir, f"{env_filename}.flp"))
        frequencies = np.array([10.0, 20.0])

        # Every frequency's KRAKEN/FIELD run is faked: run_kraken_exec is
        # a no-op, and run_field_exec copies the SAME real fixture
        # '.mod' file into place under the CURRENT working filename --
        # exactly mimicking each iteration overwriting the same on-disk
        # filename with (what would be) new content.
        def fake_run_field_exec(filename, *args, **kwargs):
            shutil.copyfile(REAL_MOD_PATH, os.path.join(os.getcwd(), f"{filename}.mod"))

        fake_pressure = np.zeros((1, 1, 1, 1), dtype=complex)

        with mock.patch.object(KrakenManager, "run_kraken_exec"), \
             mock.patch.object(KrakenManager, "run_field_exec", side_effect=fake_run_field_exec), \
             mock.patch(
                 "propa.kraken_toolbox.src.kraken_manager.readshd",
                 return_value=(None, None, None, None, None, None, {}, fake_pressure),
             ):
            pressure, field_pos, all_modes = KrakenManager.runkraken_broadband_range_dependent(
                env=env, flp=flp, frequencies=frequencies
            )

        # One real (not mocked) readmodes() result per frequency,
        # collected before the next iteration overwrote the file.
        self.assertEqual(len(all_modes), 2)
        for Modes in all_modes:
            self.assertIn("phi", Modes)
            self.assertGreater(Modes["M"], 0)

    def test_restores_the_original_working_directory(self):
        # NOTE: regression test for the fixed bug -- os.chdir(env.root)
        # used to never be restored, leaking a global process-wide side
        # effect past the end of this call. Confirmed to break OTHER,
        # unrelated code afterwards: a test that deletes its own temp
        # directory in tearDown() after this method chdir'd into it left
        # the process's cwd pointing at a now-deleted directory,
        # breaking any later relative-path filesystem operation
        # (anywhere else in the same test run) with a confusing
        # FileNotFoundError.
        env_filename = "rdenv_cwd"
        env = self._build_range_dependent_env(env_filename)

        class DummyFlp:
            def __init__(self, flp_fpath):
                self.flp_fpath = flp_fpath

            def write_flp(self):
                with open(self.flp_fpath, "w") as f:
                    f.write("dummy flp\n")

        flp = DummyFlp(os.path.join(self.tmp_dir, f"{env_filename}.flp"))
        fake_pressure = np.zeros((1, 1, 1, 1), dtype=complex)

        def fake_run_field_exec(filename, *args, **kwargs):
            # No real '.mod' needed here (readmodes is mocked below too).
            open(os.path.join(os.getcwd(), f"{filename}.mod"), "w").close()

        cwd_before = os.getcwd()
        with mock.patch.object(KrakenManager, "run_kraken_exec"), \
             mock.patch.object(KrakenManager, "run_field_exec", side_effect=fake_run_field_exec), \
             mock.patch(
                 "propa.kraken_toolbox.src.kraken_manager.readshd",
                 return_value=(None, None, None, None, None, None, {}, fake_pressure),
             ), \
             mock.patch(
                 "propa.kraken_toolbox.src.kraken_manager.readmodes",
                 return_value={"M": 1, "phi": np.zeros((1, 1))},
             ):
            KrakenManager.runkraken_broadband_range_dependent(
                env=env, flp=flp, frequencies=np.array([10.0, 20.0])
            )
        self.assertEqual(os.getcwd(), cwd_before)


if __name__ == "__main__":
    unittest.main()
