#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Pekeris waveguide sensibility study: vary each of the 5 fundamental
Pekeris parameters INDEPENDENTLY around a common baseline (everything
else held fixed), and observe how a scalar quantity derived from
KRAKEN's own Green's function responds -- here, the Green's function
level (dB) at a fixed reference range and receiver depth (see
propa.kraken_toolbox.sensibility.compute_qoi_green_level_at_reference_range).

The 5 parameters swept, one at a time:
    - depth  (water depth, m)               -- 10 to 5000 m (as requested)
    - c1     (water sound speed, m/s)
    - c2     (bottom sound speed, m/s)
    - rho2   (bottom density, kg/m3)
    - attn2  (bottom compressional attenuation, dB/wavelength)

All the actual machinery (environment construction, KRAKEN/FIELD
execution, Green's function extraction, per-parameter working
directory, results CSV) lives in the generic, reusable
propa.kraken_toolbox.sensibility module -- this script only defines
WHICH parameters to sweep and over WHAT range, then calls
run_sensibility_study() once per parameter. Use that module directly
for your own sensibility studies (e.g. this project's own, more
elaborate RTF sensibility study can share the same 'all_arg_dict'
baseline and 'build_kraken_env'/'compute_green_function' building
blocks).

Filesystem layout (see propa.kraken_toolbox.sensibility's own module
docstring for the full rationale):
    io_files/<param_name>/       -- one REUSED working directory per
        swept parameter (not per value) -- keeps disk usage bounded.
    data/sensibility/<param_name>.csv  -- one results file per swept
        parameter: columns [value, qoi_dB].

Run this script directly to write the '.env'/'.flp' files for the LAST
value of each parameter (a quick sanity/dry-run check, no KRAKEN run).
Set the KRAKEN_EXAMPLES_RUN_KRAKEN=1 environment variable (and make
sure kraken.exe/field.exe are reachable) to actually run every value of
every parameter and produce the results CSVs + sensibility plots.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from propa.kraken_toolbox.sensibility import (
    baseline_arg_dict,
    run_sensibility_study,
    load_sensibility_result,
    plot_sensibility_result,
)

HERE = os.path.dirname(os.path.abspath(__file__))
COMPUTATION_ROOT = os.path.join(HERE, "io_files")
RESULT_ROOT = os.path.join(HERE, "data", "sensibility")
RUN_KRAKEN = os.environ.get("KRAKEN_EXAMPLES_RUN_KRAKEN", "0") == "1"

# ----------------------------------------------------------------------
# Baseline: same Pekeris waveguide for every parameter's sweep, except
# for the one parameter actually being varied at a time.
# ----------------------------------------------------------------------
BASELINE = baseline_arg_dict()
print("Baseline Pekeris waveguide:")
for k, v in BASELINE.items():
    if k != "r_rcv":  # skip the long range-grid array in this printout
        print(f"    {k} = {v}")

# ----------------------------------------------------------------------
# The 5 parameters to sweep, one at a time, and their ranges.
# ----------------------------------------------------------------------
SWEEP_PLAN = {
    # test_arg_name: (values, nice label, units)
    "depth": (np.linspace(10.0, 5000.0, 30), "Water depth", "m"),
    "c1": (np.linspace(1450.0, 1550.0, 20), "Water sound speed", "m/s"),
    "c2": (np.linspace(1550.0, 1900.0, 20), "Bottom sound speed", "m/s"),
    "rho2": (np.linspace(1.0, 2.5, 20) * 1e3, "Bottom density", "kg/m3"),
    "attn2": (np.linspace(0.0, 2.0, 20), "Bottom attenuation", "dB/wavelength"),
}


def run_all_sweeps():
    """Run (or dry-run) every parameter's sweep in SWEEP_PLAN, one after
    another, saving one results CSV per parameter."""
    all_results = {}
    for test_arg_name, (values, _label, _units) in SWEEP_PLAN.items():
        print(f"\n=== Sweeping '{test_arg_name}' ({values.size} values) ===")
        results = run_sensibility_study(
            test_arg_name=test_arg_name,
            test_arg_values=values,
            all_arg_dict=BASELINE,
            computation_root=COMPUTATION_ROOT,
            result_root=RESULT_ROOT,
            run_kraken=RUN_KRAKEN,
        )
        all_results[test_arg_name] = results
    return all_results


def plot_all_sweeps():
    """Reload every parameter's results CSV (must already exist -- see
    run_all_sweeps() with RUN_KRAKEN=1) and save one sensibility plot
    per parameter."""
    for test_arg_name, (_values, label, units) in SWEEP_PLAN.items():
        results = load_sensibility_result(RESULT_ROOT, test_arg_name)
        fig = plot_sensibility_result(
            results, test_arg_name, param_label=label, param_units=units
        )
        fig_path = os.path.join(RESULT_ROOT, f"{test_arg_name}.png")
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Wrote {fig_path}")


if __name__ == "__main__":
    run_all_sweeps()

    if RUN_KRAKEN:
        plot_all_sweeps()
    else:
        print(
            "\nKRAKEN_EXAMPLES_RUN_KRAKEN is not set: only wrote the "
            "'.env'/'.flp' files for the LAST value of each parameter, "
            "as a dry run (see io_files/<param_name>/). Set "
            "KRAKEN_EXAMPLES_RUN_KRAKEN=1 (with kraken.exe/field.exe "
            "reachable) to actually run every value of every parameter "
            "and produce the results CSVs and sensibility plots in "
            f"{RESULT_ROOT}."
        )
