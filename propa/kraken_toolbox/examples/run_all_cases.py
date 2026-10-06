#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Run every example case in this folder, one after another.

Usage:
    python run_all_cases.py                 # write .env/.flp + environment
                                             # overview figures for every
                                             # case (no KRAKEN/FIELD run)
    python run_all_cases.py --run-kraken    # also run KRAKEN/FIELD and
                                             # produce the mode-shape/TL
                                             # figures for every case
                                             # (requires kraken.exe/
                                             # field.exe to be reachable)
    python run_all_cases.py --only 1 3 9    # restrict to specific cases
    python run_all_cases.py --parallel      # pass parallel=True through
                                             # to the broadband +
                                             # range-dependent cases
                                             # (4, 7, 8, 9)

Each case is run in its OWN subprocess (not imported in-process): this
keeps matplotlib figures, global state, and any mid-script failure
fully contained to that one case, and lets each 'run_case.py' remain a
completely standalone, independently copyable example (see this
folder's README.md) -- exactly as if you had 'cd'ed into that case's
directory and run it yourself.

'RUN_KRAKEN' inside each case script is controlled here via the
KRAKEN_EXAMPLES_RUN_KRAKEN environment variable (set to "1" or "0"),
rather than by editing every 'run_case.py' by hand.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def discover_cases():
    """Return the sorted list of (case_number, folder_name, script_path)
    for every 'case_NN_*' folder in this directory that contains a
    'run_case.py'."""
    cases = []
    for name in sorted(os.listdir(HERE)):
        folder = os.path.join(HERE, name)
        script = os.path.join(folder, "run_case.py")
        if name.startswith("case_") and os.path.isfile(script):
            # 'case_01_...' -> case number 1
            try:
                number = int(name.split("_")[1])
            except (IndexError, ValueError):
                continue
            cases.append((number, name, script))
    return cases


def run_one_case(number, name, script, run_kraken, parallel):
    """Run a single case's 'run_case.py' as a subprocess.

    Returns:
        bool: True on success (exit code 0), False otherwise. Output is
        streamed live (not captured) so progress is visible while a
        (potentially long, if run_kraken=True) case is running.
    """
    env = os.environ.copy()
    env["KRAKEN_EXAMPLES_RUN_KRAKEN"] = "1" if run_kraken else "0"
    if parallel:
        env["KRAKEN_EXAMPLES_PARALLEL"] = "1"

    print("=" * 78)
    print(f"Case {number}: {name}")
    print("=" * 78)

    result = subprocess.run(
        [sys.executable, os.path.basename(script)],
        cwd=os.path.dirname(script),
        env=env,
    )

    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit code {result.returncode})"
    print(f"--> {status}\n")
    return ok


def main(run_kraken=False, only=None, parallel=False):
    """Run every discovered case (or only the ones listed in 'only').

    Args:
        run_kraken (bool): forwarded to every case as
            KRAKEN_EXAMPLES_RUN_KRAKEN.
        only (list[int]|None): case numbers to restrict to. None runs
            every case found.
        parallel (bool): forwarded to every case as
            KRAKEN_EXAMPLES_PARALLEL (only cases 4, 7, 8, 9 read it, for
            their broadband + range-dependent KrakenManager(parallel=...)
            call -- see each case's own PARALLEL variable).

    Returns:
        dict[int, bool]: case number -> success, in run order.
    """
    cases = discover_cases()
    if only is not None:
        cases = [c for c in cases if c[0] in only]
        missing = set(only) - {c[0] for c in cases}
        if missing:
            print(f"Warning: no case folder found for number(s): {sorted(missing)}")

    if not cases:
        print("No cases found to run.")
        return {}

    results = {}
    for number, name, script in cases:
        results[number] = run_one_case(number, name, script, run_kraken, parallel)

    print("=" * 78)
    print("Summary")
    print("=" * 78)
    for number, name, _ in cases:
        status = "OK" if results[number] else "FAILED"
        print(f"  Case {number:<2} ({name}): {status}")

    n_failed = sum(1 for ok in results.values() if not ok)
    if n_failed:
        print(f"\n{n_failed} case(s) failed.")
    else:
        print("\nAll cases completed successfully.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-kraken", action="store_true",
        help="Also run KRAKEN/FIELD and produce mode-shape/TL figures "
             "(requires kraken.exe/field.exe to be reachable). Without "
             "this flag, only '.env'/'.flp' and the environment overview "
             "figures are produced.",
    )
    parser.add_argument(
        "--only", type=int, nargs="+", default=None, metavar="N",
        help="Restrict to these case numbers, e.g. --only 1 3 9.",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Forward parallel=True to KrakenManager for the broadband "
             "+ range-dependent cases (4, 7, 8, 9).",
    )
    args = parser.parse_args()

    results = main(run_kraken=args.run_kraken, only=args.only, parallel=args.parallel)
    sys.exit(1 if any(not ok for ok in results.values()) else 0)
