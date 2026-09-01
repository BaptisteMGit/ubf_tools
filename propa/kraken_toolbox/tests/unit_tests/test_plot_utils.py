#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Unit tests for propa/kraken_toolbox/plot_utils.py.

These tests focus on the bugs found and fixed (see plot_utils.py's
module docstring), using the real KRAKEN/FIELD fixture files already in
propa/kraken_toolbox/tests/fixtures/ (see test_real_kraken_files.py)
wherever real mode-count variation across frequencies matters -- no
synthetic construction can substitute for that.

Run with either:
    python -m unittest discover -s propa/kraken_toolbox/tests
or (if pytest is installed in your environment):
    pytest propa/kraken_toolbox/tests
"""
import os
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from propa.kraken_toolbox import plot_utils as pu

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
MOD_PATH = os.path.join(FIXTURES_DIR, "real_kraken.mod")
SHD_PATH = os.path.join(FIXTURES_DIR, "real_field.shd")


def _close_all_figures():
    plt.close("all")


# ======================================================================
# plotmode / plotmode_several_freqs
# ======================================================================
@unittest.skipUnless(os.path.exists(MOD_PATH), "real_kraken.mod fixture not present")
class TestPlotmode(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_single_mode_does_not_crash(self):
        # NOTE: regression test for the fixed bug -- plt.subplots(1, 1)
        # returns a single Axes (not an array), so `ax[0]` used to
        # raise `TypeError: 'Axes' object is not subscriptable`.
        # Confirmed on real data: 10 Hz has exactly 1 mode in this file.
        fig = pu.plotmode(MOD_PATH, freq=10)
        self.assertIsNotNone(fig)

    def test_multiple_modes_does_not_crash(self):
        fig = pu.plotmode(MOD_PATH, freq=50)  # 7 modes at 50 Hz
        self.assertIsNotNone(fig)

    def test_title_is_a_string_not_a_list(self):
        # NOTE: regression test -- fig.suptitle([...]) used to render a
        # literal "['title', 'Freq = ... Hz']" string.
        fig = pu.plotmode(MOD_PATH, freq=10)
        title_obj = fig._suptitle
        self.assertIsInstance(title_obj.get_text(), str)
        self.assertNotIn("[", title_obj.get_text())

    def test_no_modes_raises(self):
        with mock.patch.object(pu, "readmodes", return_value={"M": 0}):
            with self.assertRaises(Exception):
                pu.plotmode(MOD_PATH, freq=10)


@unittest.skipUnless(os.path.exists(MOD_PATH), "real_kraken.mod fixture not present")
class TestPlotmodeSeveralFreqs(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_varying_mode_counts_across_frequencies_does_not_crash(self):
        # NOTE: regression test for the fixed bug -- the subplot grid
        # used to be sized to the FIRST frequency's mode count only.
        # This file has 1, 3, 4, 5, 7 modes at 10/20/30/40/50 Hz
        # respectively: starting from the smallest (10 Hz, 1 mode) and
        # growing used to raise an IndexError on `ax[iplot]` for any
        # later, richer frequency.
        fig = pu.plotmode_several_freqs(MOD_PATH, freq=np.array([10, 20, 30, 40, 50]))
        self.assertIsNotNone(fig)
        # The grid must have been sized to the LARGEST mode count (7 at 50 Hz).
        self.assertEqual(len(fig.axes), 7)

    def test_starting_from_the_richest_frequency_also_works(self):
        # Same fixture, reversed order -- also exercises the case where
        # the FIRST frequency alone would have been enough (already
        # worked before the fix); included for symmetry.
        fig = pu.plotmode_several_freqs(MOD_PATH, freq=np.array([50, 40, 30, 20, 10]))
        self.assertEqual(len(fig.axes), 7)

    def test_title_lists_every_frequency(self):
        fig = pu.plotmode_several_freqs(MOD_PATH, freq=np.array([10, 20]))
        title_text = fig._suptitle.get_text()
        self.assertIn("10", title_text)
        self.assertIn("20", title_text)


# ======================================================================
# plotshd
# ======================================================================
@unittest.skipUnless(os.path.exists(SHD_PATH), "real_field.shd fixture not present")
class TestPlotshd(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_does_not_alter_filename_case(self):
        # NOTE: regression test for the fixed bug -- the original code
        # did `filename = filename.lower()` unconditionally, which
        # breaks on case-sensitive filesystems for any real path
        # containing uppercase characters. We can't rely on the actual
        # fixture path being mixed-case, so we spy on readshd() to
        # check the exact string it receives.
        with mock.patch.object(pu, "readshd", wraps=pu.readshd) as spy:
            pu.plotshd(SHD_PATH, freq=30)
        called_filename = spy.call_args.kwargs.get("filename", spy.call_args.args[0] if spy.call_args.args else None)
        self.assertEqual(called_filename, SHD_PATH)

    def test_basic_call_does_not_crash(self):
        fig = pu.plotshd(SHD_PATH, freq=30, units="km")
        self.assertIsNone(fig)  # no (m, n, p) given -> draws into current figure, returns None

    def test_subplot_mode_returns_figure(self):
        fig = pu.plotshd(SHD_PATH, freq=30, m=1, n=1, p=1)
        self.assertIsNotNone(fig)

    def test_axis_argument_is_used(self):
        fig, ax = plt.subplots()
        result = pu.plotshd(SHD_PATH, freq=30, axis=ax)
        self.assertIsNone(result)
        self.assertTrue(len(ax.collections) > 0)  # pcolor drew into this axis


# ======================================================================
# plotshd_from_pressure_field
# ======================================================================
@unittest.skipUnless(os.path.exists(SHD_PATH), "real_field.shd fixture not present")
class TestPlotshdFromPressureField(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def _get_real_pressure(self, freq=30):
        from propa.kraken_toolbox.read_shd import readshd
        *_, pressure = readshd(SHD_PATH, freq=freq)
        return pressure

    def test_default_title_does_not_crash(self):
        # NOTE: regression test for the fixed bug -- the default-title
        # branch used to do `title = PlotTitle.replace(...)` then, on a
        # separate line, `+f'...'` -- never assigned back, and unary
        # '+' on a string raises `TypeError: bad operand type for
        # unary +: 'str'`. This means calling this function without an
        # explicit 'title' ALWAYS crashed.
        pressure = self._get_real_pressure()
        pu.plotshd_from_pressure_field(SHD_PATH, pressure_field=pressure, freq=30)
        # If we get here without raising, the bug is fixed.

    def test_explicit_title_still_works(self):
        pressure = self._get_real_pressure()
        pu.plotshd_from_pressure_field(
            SHD_PATH, pressure_field=pressure, freq=30, title="My custom title"
        )


# ======================================================================
# plot_tl_profile / plot_tl_profile_multi_freq (new functions)
# ======================================================================
@unittest.skipUnless(os.path.exists(SHD_PATH), "real_field.shd fixture not present")
class TestPlotTlProfile(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_single_frequency_profile(self):
        fig = pu.plot_tl_profile(SHD_PATH, freq=30, rcv_depth=0.0, units="km")
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 1)

    def test_multi_freq_profile_overlays_one_line_per_frequency(self):
        fig = pu.plot_tl_profile_multi_freq(
            SHD_PATH, freqs=[10, 20, 30], rcv_depth=0.0, units="km"
        )
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 3)

    def test_units_km_is_smaller_than_units_m(self):
        # Sanity check tying this module's TL-profile helpers to the
        # m/km fix already validated in read_shd.py / test_read_shd.py.
        fig_km = pu.plot_tl_profile(SHD_PATH, freq=30, rcv_depth=0.0, units="km")
        r_km = fig_km.axes[0].lines[0].get_xdata()
        fig_m = pu.plot_tl_profile(SHD_PATH, freq=30, rcv_depth=0.0, units="m")
        r_m = fig_m.axes[0].lines[0].get_xdata()
        np.testing.assert_allclose(np.asarray(r_km) * 1000.0, r_m)


# ======================================================================
# plot_ssp / plot_attenuation / plot_density
# ======================================================================
class TestPlotSsp(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_shear_wave_curve_is_plotted_when_present(self):
        # NOTE: regression test for the fixed bug -- an erroneous
        # `cs = 0` right before the "should I hide the S-wave curve?"
        # check discarded the real cs array, so the S-wave curve was
        # hidden EVERY TIME cp was not all-zero (i.e. for any normal
        # environment with real elastic shear-wave data).
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        cp = np.array([1500.0, 1550.0, 1600.0])
        cs = np.array([200.0, 250.0, 300.0])  # real, non-zero S-wave data
        pu.plot_ssp(cp_ssp=cp, cs_ssp=cs, z=z, ax=ax)
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertIn("S-wave", labels)
        self.assertIn("C-wave", labels)

    def test_shear_wave_curve_hidden_when_genuinely_zero(self):
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        cp = np.array([1500.0, 1550.0, 1600.0])
        cs = np.zeros(3)  # a fluid medium: genuinely no shear waves
        pu.plot_ssp(cp_ssp=cp, cs_ssp=cs, z=z, ax=ax)
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertNotIn("S-wave", labels)
        self.assertIn("C-wave", labels)

    def test_scalar_inputs_are_broadcast(self):
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        pu.plot_ssp(cp_ssp=1500.0, cs_ssp=0.0, z=z, ax=ax)  # must not raise

    def test_z_bottom_shades_domains(self):
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        pu.plot_ssp(cp_ssp=1500.0, cs_ssp=0.0, z=z, z_bottom=50.0, ax=ax)
        # color_domains draws 2 fill_between collections when z_bottom is given
        self.assertGreaterEqual(len(ax.collections), 2)


class TestPlotAttenuationAndDensity(unittest.TestCase):
    def tearDown(self):
        _close_all_figures()

    def test_plot_attenuation_basic(self):
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        pu.plot_attenuation(ap=0.5, ash=0.0, z=z, ax=ax)
        labels = [line.get_label() for line in ax.get_lines()]
        self.assertIn("C-wave", labels)
        self.assertNotIn("S-wave", labels)

    def test_plot_density_basic(self):
        fig, ax = plt.subplots()
        z = np.array([0.0, 50.0, 100.0])
        pu.plot_density(rho=1.5, z=z, ax=ax)
        self.assertEqual(len(ax.get_lines()), 1)


class TestPlotBathymetry(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp(prefix="plot_bathymetry_test_")

    def tearDown(self):
        _close_all_figures()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_basic_plot(self):
        from propa.kraken_toolbox.src.kraken_env import Bathymetry

        path = os.path.join(self.tmp_dir, "bathy.csv")
        with open(path, "w") as f:
            f.write("0,100\n5,80\n10,110\n")
        bathy = Bathymetry(path, units="km")

        fig = pu.plot_bathymetry(bathy)
        ax = fig.axes[0]
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [0, 5, 10])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [100, 80, 110])

    def test_axis_argument_is_used(self):
        from propa.kraken_toolbox.src.kraken_env import Bathymetry

        path = os.path.join(self.tmp_dir, "bathy.csv")
        with open(path, "w") as f:
            f.write("0,100\n5,80\n")
        bathy = Bathymetry(path, units="km")

        fig, ax = plt.subplots()
        result = pu.plot_bathymetry(bathy, ax=ax)
        self.assertIs(result, fig)
        self.assertEqual(len(ax.lines), 1)


if __name__ == "__main__":
    unittest.main()
