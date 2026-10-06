#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for propa/kraken_toolbox/run_kraken.py.

Mirrors test_kraken_manager.py where the logic is shared (frequency
distribution, shell command construction, working directory
management), plus tests targeting the two bugs specific to this file
(missing 'nmedia', missing Pool context manager). os.system and
multiprocessing.Pool are mocked -- no real process is ever launched.

Run with either:
    python -m unittest discover -s propa/kraken_toolbox/tests
or (if pytest is installed in your environment):
    pytest propa/kraken_toolbox/tests
"""
import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from propa.kraken_toolbox import run_kraken
from propa.kraken_toolbox.src.kraken_env import KrakenEnv, KrakenMedium


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="run_kraken_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


# ======================================================================
# Frequency distribution across workers (shared logic with KrakenManager)
# ======================================================================
class TestAssignFrequencyIntervalls(unittest.TestCase):
    def test_equally_distributed_covers_all_frequencies(self):
        freqs = np.array([10, 20, 30, 40, 50, 60, 70])
        ranges, n_used = run_kraken.assign_frequency_intervalls(
            freqs, 3, mode="equally_distributed"
        )
        self.assertEqual(n_used, 3)
        np.testing.assert_array_equal(np.sort(np.concatenate(ranges)), np.sort(freqs))

    def test_optimal_mode_covers_all_frequencies(self):
        freqs = np.array([10, 20, 30, 40, 50, 60, 70])
        ranges, n_used = run_kraken.assign_frequency_intervalls(freqs, 3, mode="optimal")
        np.testing.assert_array_equal(np.sort(np.concatenate(ranges)), np.sort(freqs))
        self.assertLessEqual(n_used, 3)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            run_kraken.assign_frequency_intervalls(np.array([10, 20]), 2, mode="bogus")


# ======================================================================
# Shell command construction (run_exec) - os.name explicitly patched so
# expectations do not depend on the OS running the tests.
# ======================================================================
class TestRunExec(unittest.TestCase):
    def test_kraken_exec_posix(self):
        calls = []
        with mock.patch("os.name", "posix"), mock.patch("os.system", side_effect=lambda c: calls.append(c)):
            run_kraken.run_kraken_exec("myenv")
        self.assertEqual(calls, ["kraken myenv >/dev/null 2>&1"])

    def test_kraken_exec_windows(self):
        calls = []
        with mock.patch("os.name", "nt"), mock.patch("os.system", side_effect=lambda c: calls.append(c)):
            run_kraken.run_kraken_exec("myenv")
        self.assertEqual(calls, ["kraken.exe myenv >NUL 2>&1"])

    def test_field_exec_posix(self):
        calls = []
        with mock.patch("os.name", "posix"), mock.patch("os.system", side_effect=lambda c: calls.append(c)):
            run_kraken.run_field_exec("myenv")
        self.assertEqual(calls, ["field myenv"])

    def test_parallel_without_worker_pid_raises_on_windows(self):
        with mock.patch("os.name", "nt"):
            with self.assertRaises(ValueError):
                run_kraken.run_exec(
                    exec="kraken", filename="myenv", parallel=True, worker_pid=None, silent=True
                )


# ======================================================================
# Parallel working directory management
# ======================================================================
class TestWorkingDirs(TempDirTestCase):
    def test_get_subprocess_working_dir_creates_directory(self):
        path = run_kraken.get_subprocess_working_dir(self.tmp_dir, worker_pid=7)
        self.assertTrue(os.path.isdir(path))
        self.assertIn("child_process_7", path)

    def test_clear_kraken_parallel_working_dir_removes_subdirs(self):
        sub = run_kraken.get_subprocess_working_dir(self.tmp_dir, worker_pid=1)
        self.assertTrue(os.path.isdir(sub))
        run_kraken.clear_kraken_parallel_working_dir(root=self.tmp_dir)
        self.assertFalse(os.path.isdir(sub))

    def test_clear_kraken_parallel_working_dir_no_op_if_missing(self):
        run_kraken.clear_kraken_parallel_working_dir(root=self.tmp_dir)  # must not raise


# ======================================================================
# runkraken_broadband_range_dependent - regression test for the missing
# 'nmedia' bug
# ======================================================================
class TestRunkrakenBroadbandRangeDependent(TempDirTestCase):
    def _build_range_dependent_env(self, nmedia=2):
        from propa.kraken_toolbox.src.kraken_env import Bathymetry, KrakenField

        bathy_path = os.path.join(self.tmp_dir, "bathy.csv")
        with open(bathy_path, "w") as f:
            f.write("0,100\n5,150\n")
        bathy = Bathymetry(bathy_path, units="km")
        medium = KrakenMedium(z_ssp=[0, 100], c_p=[1500, 1500])
        # NOTE: rcv_z_max is set comfortably beyond the default buffered
        # bottom's sedim_layer_max_depth -- see
        # KrakenEnv.write_range_dependent_lines's docstring for why
        # FIELD.exe requires this for a coupled-mode, buffered-bottom,
        # range-dependent run. Unrelated to what these tests actually
        # check; just needed for write_env() to succeed.
        field = KrakenField(rcv_z_max=2000.0)
        env = KrakenEnv(
            title="t", env_root=self.tmp_dir, env_filename="rdenv",
            freq=[10.0, 20.0], kraken_medium=medium, kraken_bathy=bathy, nmedia=nmedia,
            kraken_field=field,
        )
        env.write_env()
        return env

    def test_nmedia_is_preserved_across_per_frequency_reconstruction(self):
        # NOTE: regression test for the fixed bug -- the original code
        # rebuilt a KrakenEnv per frequency without passing
        # 'nmedia=env.nmedia', silently resetting it to KrakenEnv's
        # default (1). We intercept the KrakenEnv constructor call
        # inside the loop to check what 'nmedia' it actually receives.
        env = self._build_range_dependent_env(nmedia=2)

        class DummyFlp:
            def __init__(self, flp_fpath):
                self.flp_fpath = flp_fpath

            def write_flp(self):
                with open(self.flp_fpath, "w") as f:
                    f.write("dummy flp\n")

        flp = DummyFlp(os.path.join(self.tmp_dir, "rdenv.flp"))
        seen_nmedia = []

        real_kraken_env_init = KrakenEnv.__init__

        def spy_init(self_env, *args, **kwargs):
            seen_nmedia.append(kwargs.get("nmedia"))
            return real_kraken_env_init(self_env, *args, **kwargs)

        with mock.patch.object(KrakenEnv, "__init__", spy_init):
            with mock.patch.object(run_kraken, "run_kraken_exec"), \
                 mock.patch.object(run_kraken, "run_field_exec"), \
                 mock.patch.object(
                     run_kraken, "readshd",
                     return_value=(None, None, None, None, None, None, {}, np.zeros((1, 1, 1, 1))),
                 ), \
                 mock.patch.object(
                     run_kraken, "readmodes",
                     return_value={"M": 0, "z": np.array([]), "phi": np.zeros((0, 0))},
                 ):
                run_kraken.runkraken_broadband_range_dependent(
                    env=env, flp=flp, frequencies=np.array([10.0, 20.0])
                )

        # One reconstruction per frequency, all must carry nmedia=2
        self.assertEqual(len(seen_nmedia), 2)
        self.assertTrue(all(n == 2 for n in seen_nmedia))

    def test_modes_are_collected_per_frequency_before_being_overwritten(self):
        # NOTE: regression test for the fixed bug -- mirrors
        # test_kraken_manager.py's
        # TestRunkrakenBroadbandRangeDependentModesCollection. Each
        # frequency's '.mod' file used to be silently overwritten by
        # the next iteration, with no attempt to read it in the
        # meantime: a caller reading the '.mod' file AFTER this loop
        # only ever saw the last frequency's modes (or hit a
        # FileNotFoundError). readmodes() must now be called once per
        # frequency, INSIDE the loop, with the results collected in
        # order.
        real_mod_path = os.path.join(
            os.path.dirname(__file__), "fixtures", "real_kraken.mod"
        )
        if not os.path.exists(real_mod_path):
            self.skipTest("real_kraken.mod fixture not present")

        env = self._build_range_dependent_env(nmedia=2)

        class DummyFlp:
            def __init__(self, flp_fpath):
                self.flp_fpath = flp_fpath

            def write_flp(self):
                with open(self.flp_fpath, "w") as f:
                    f.write("dummy flp\n")

        flp = DummyFlp(os.path.join(self.tmp_dir, "rdenv.flp"))

        def fake_run_field_exec(filename, *args, **kwargs):
            shutil.copyfile(real_mod_path, os.path.join(os.getcwd(), f"{filename}.mod"))

        fake_pressure = np.zeros((1, 1, 1, 1), dtype=complex)

        with mock.patch.object(run_kraken, "run_kraken_exec"), \
             mock.patch.object(run_kraken, "run_field_exec", side_effect=fake_run_field_exec), \
             mock.patch.object(
                 run_kraken, "readshd",
                 return_value=(None, None, None, None, None, None, {}, fake_pressure),
             ):
            pressure, field_pos, all_modes = run_kraken.runkraken_broadband_range_dependent(
                env=env, flp=flp, frequencies=np.array([10.0, 20.0])
            )

        self.assertEqual(len(all_modes), 2)
        for Modes in all_modes:
            self.assertIn("phi", Modes)
            self.assertGreater(Modes["M"], 0)

    def test_field_error_for_one_frequency_does_not_abort_the_loop(self):
        env = self._build_range_dependent_env(nmedia=2)

        class DummyFlp:
            def __init__(self, flp_fpath):
                self.flp_fpath = flp_fpath

            def write_flp(self):
                with open(self.flp_fpath, "w") as f:
                    f.write("dummy flp\n")

        flp = DummyFlp(os.path.join(self.tmp_dir, "rdenv.flp"))

        def failing_run_field_exec(*args, **kwargs):
            raise RuntimeError("boom")

        # NOTE: the code under test intentionally prints
        # "Error running field executable for frequency ...: boom" for
        # each failing frequency (see the NOTE in
        # runkraken_broadband_range_dependent's except-block) instead of
        # raising. That is expected here -- redirect stdout so this
        # test's console output only shows pass/fail, not that expected
        # noise.
        with contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(run_kraken, "run_kraken_exec"), \
                 mock.patch.object(run_kraken, "run_field_exec", side_effect=failing_run_field_exec):
                # Must not raise: errors for individual frequencies are
                # caught and logged, matching the original (bare-except)
                # behaviour, just with a narrower exception type.
                pressure, field_pos, all_modes = run_kraken.runkraken_broadband_range_dependent(
                    env=env, flp=flp, frequencies=np.array([10.0, 20.0])
                )
        self.assertIsNone(pressure)
        self.assertIsNone(field_pos)
        self.assertEqual(all_modes, [])

    def test_restores_the_original_working_directory(self):
        # NOTE: regression test for the fixed bug -- matches
        # test_kraken_manager.py's identical test for
        # KrakenManager.runkraken_broadband_range_dependent().
        env = self._build_range_dependent_env(nmedia=2)

        class DummyFlp:
            def __init__(self, flp_fpath):
                self.flp_fpath = flp_fpath

            def write_flp(self):
                with open(self.flp_fpath, "w") as f:
                    f.write("dummy flp\n")

        flp = DummyFlp(os.path.join(self.tmp_dir, "rdenv_cwd.flp"))

        def fake_run_field_exec(filename, *args, **kwargs):
            open(os.path.join(os.getcwd(), f"{filename}.mod"), "w").close()

        fake_pressure = np.zeros((1, 1, 1, 1), dtype=complex)

        cwd_before = os.getcwd()
        with mock.patch.object(run_kraken, "run_kraken_exec"), \
             mock.patch.object(run_kraken, "run_field_exec", side_effect=fake_run_field_exec), \
             mock.patch.object(
                 run_kraken, "readshd",
                 return_value=(None, None, None, None, None, None, {}, fake_pressure),
             ), \
             mock.patch.object(
                 run_kraken, "readmodes",
                 return_value={"M": 1, "phi": np.zeros((1, 1))},
             ):
            run_kraken.runkraken_broadband_range_dependent(
                env=env, flp=flp, frequencies=np.array([10.0, 20.0])
            )
        self.assertEqual(os.getcwd(), cwd_before)


if __name__ == "__main__":
    unittest.main()
