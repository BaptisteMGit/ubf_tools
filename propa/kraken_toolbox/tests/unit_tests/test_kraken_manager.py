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


if __name__ == "__main__":
    unittest.main()
