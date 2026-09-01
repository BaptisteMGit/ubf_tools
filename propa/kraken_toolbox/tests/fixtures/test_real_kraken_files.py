#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Regression tests using REAL KRAKEN/FIELD input/output files (not
synthetic ones), provided by the project owner:
    fixtures/real_kraken.env / .flp  (input files)
    fixtures/real_kraken.prt         (KRAKEN's human-readable output,
                                       used here as the ground truth to
                                       compare read_modes.py against)
    fixtures/real_kraken.mod         (KRAKEN's binary mode file)
    fixtures/real_field.shd          (FIELD's binary pressure field file)

The scenario ("Test CPU time for sensibility study") is a 5-frequency
(10/20/30/40/50 Hz) broadband, range-independent run with a
pressure-release (vacuum) surface and an acousto-elastic bottom -- a
good real-world case that a purely synthetic test cannot fully stand
in for (in particular, it validates read_modes.py's per-frequency
wavenumber extraction against genuinely different mode counts per
frequency: 1, 3, 4, 5, 7 modes respectively).

NOTE on fixture size: real_kraken.mod is ~800 KB. If that is too large
to keep versioned in your repository, this file (and only this file)
can be safely skipped/removed -- every other test file in this suite
uses small, in-test-generated synthetic data.

Run with either:
    python -m unittest discover -s propa/kraken_toolbox/tests
or (if pytest is installed in your environment):
    pytest propa/kraken_toolbox/tests
"""
import os
import unittest

import numpy as np

from propa.kraken_toolbox.read_modes import readmodes, readmodes_bin
from propa.kraken_toolbox.read_shd import readshd_bin

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
MOD_PATH = os.path.join(FIXTURES_DIR, "real_kraken.mod")
SHD_PATH = os.path.join(FIXTURES_DIR, "real_field.shd")

# Ground truth phase speeds (m/s) per frequency, transcribed from
# real_kraken.prt (KRAKEN's own human-readable output for this run) --
# phase speed = 2*pi*f / real(k), reported directly in the .prt file.
EXPECTED_PHASE_SPEEDS = {
    10: [1623.408570],
    20: [1550.923830, 1813.079580, 3829.510823],
    30: [1527.469258, 1620.129589, 1920.074603, 3044.551897],
    40: [1517.242807, 1573.928228, 1698.666862, 1986.537538, 2775.467237],
    50: [1511.859823, 1550.035974, 1619.335167, 1762.585583, 2031.662739, 2643.850913, 5919.242360],
}


@unittest.skipUnless(os.path.exists(MOD_PATH), "real_kraken.mod fixture not present")
class TestReadModesAgainstRealKrakenOutput(unittest.TestCase):
    def test_mode_count_per_frequency_matches_prt(self):
        for freq, expected_speeds in EXPECTED_PHASE_SPEEDS.items():
            Modes = readmodes_bin(MOD_PATH, freq=freq)
            self.assertEqual(
                Modes["M"], len(expected_speeds),
                f"frequency {freq} Hz: expected {len(expected_speeds)} modes, got {Modes['M']}",
            )

    def test_phase_speeds_match_prt_for_every_frequency(self):
        # Phase speed c = 2*pi*f / Re(k). This is the strongest possible
        # regression check available without a live KRAKEN executable:
        # it validates the full binary record-skipping logic (advancing
        # through a genuinely variable number of modes per frequency),
        # not just a single fixed-size record.
        for freq, expected_speeds in EXPECTED_PHASE_SPEEDS.items():
            Modes = readmodes_bin(MOD_PATH, freq=freq)
            phase_speed = 2 * np.pi * freq / Modes["k"].real
            np.testing.assert_allclose(
                phase_speed, expected_speeds, rtol=1e-5,
                err_msg=f"phase speed mismatch at {freq} Hz",
            )

    def test_full_readmodes_does_not_raise_and_tags_boundary_conditions(self):
        # This run has a vacuum top and an acousto-elastic bottom
        # (per real_kraken.env / .prt): exercises both branches of
        # readmodes()'s Top/Bot wavenumber calculation.
        Modes = readmodes(MOD_PATH, freq=30)
        self.assertEqual(Modes["Top"]["BC"], "V")
        self.assertEqual(Modes["Bot"]["BC"], "A")
        self.assertEqual(Modes["Top"]["rho"], 1.0)  # vacuum fallback
        self.assertIn("k2", Modes["Bot"])  # acousto-elastic branch populated


@unittest.skipUnless(os.path.exists(SHD_PATH), "real_field.shd fixture not present")
class TestReadShdAgainstRealFieldOutput(unittest.TestCase):
    def test_frequencies_and_grid_dimensions_match_flp(self):
        # real_kraken.flp requests 10 receiver ranges and 21 receiver
        # depths -- the grid FIELD actually wrote out.
        _, _, freqVec, _, _, _, Pos, pressure = readshd_bin(SHD_PATH, freq=list(EXPECTED_PHASE_SPEEDS))
        np.testing.assert_array_equal(freqVec, [10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(Pos["r"]["r"].size, 10)
        self.assertEqual(Pos["r"]["z"].size, 21)
        self.assertEqual(pressure.shape, (5, 1, 1, 21, 10))

    def test_pressure_is_near_zero_at_the_free_surface(self):
        # Physical sanity check: with a pressure-release (vacuum) top
        # boundary, the pressure field must vanish at the surface
        # (z=0, the first receiver depth) for every frequency and
        # range -- this cannot be true by construction of the parser,
        # it can only be true if the actual complex pressure values
        # were read from the right file offsets.
        _, _, _, _, _, _, Pos, pressure = readshd_bin(SHD_PATH, freq=list(EXPECTED_PHASE_SPEEDS))
        self.assertEqual(Pos["r"]["z"][0], 0.0)
        surface_amplitude = np.abs(pressure[:, 0, 0, 0, :])
        self.assertTrue(np.all(surface_amplitude < 1e-10))

    def test_readshd_bin_header_fields_known_to_be_unreliable_for_this_run(self):
        # Documents (rather than asserts a "correct" value for) the
        # caveat recorded in read_shd.py's docstring: for this
        # broadband/coherent FIELD run, 'PlotType' and 'freq0' are not
        # meaningfully populated by FIELD.exe, even though every other
        # header field parses correctly at the same byte offsets. This
        # test exists so that if a future FIELD version starts
        # populating them, it gets noticed (test will start failing,
        # which is a good thing to investigate).
        title, PlotType, _freqVec, freq0, *_ = readshd_bin(SHD_PATH)
        self.assertEqual(title, "Test CPU time for sensibility study")
        self.assertEqual(PlotType, "")  # blank in this real file -- see caveat


if __name__ == "__main__":
    unittest.main()
