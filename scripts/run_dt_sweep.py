"""
Sweep runner for step_response_test.py.

Runs one simulation per (Kp, physics-frequency) combination, each in its own
subprocess. This is deliberate: Isaac Sim's AppLauncher / SimulationApp is
meant to be created once per process. Trying to tear it down and relaunch it
repeatedly inside a single long-running script is fragile (the app, GPU
buffers, and PhysX state are heavyweight singletons), and any leftover state
from one run could quietly bleed into the next -- the last thing you want when
the experiment itself is about isolating dt effects. A fresh process per run
guarantees each test starts from a truly clean state.

Usage:
    python run_dt_sweep.py

Adjust KP_VALUES / FREQ_VALUES below to change the sweep.
"""

import itertools
import os
import subprocess
import sys
import time

# Script being swept (must be in the same folder, or give a full/relative path)
SCRIPT = "step_response_test.py"

# Kp values to test
KP_VALUES = [2000.0, 1000.0]

# Physics frequencies to test, in Hz (physics_dt = 1/freq).
# 120 Hz is your current baseline; 1200 Hz is the "trust this as ground truth" end.
# These are all multiples of 120 so the derived `decimation` (see main script)
# comes out to a clean integer for every run.
FREQ_VALUES = [120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200]
# Control loop rate to hold constant across the sweep (passed through to the
# main script so only physics fidelity changes, not how often the controller runs)
CONTROL_HZ = 60.0


def main():
    os.makedirs("data", exist_ok=True)

    combos = list(itertools.product(KP_VALUES, FREQ_VALUES))
    print(f"Running {len(combos)} simulations ({len(KP_VALUES)} Kp values x {len(FREQ_VALUES)} frequencies)...\n")

    failures = []
    start_all = time.time()

    for i, (kp, freq) in enumerate(combos, start=1):
        dt = 1.0 / freq
        print(f"[{i}/{len(combos)}] Kp={kp:g}  freq={freq} Hz  (dt={dt:.6f}s)")

        t0 = time.time()
        result = subprocess.run(
            [
                sys.executable, SCRIPT,
                "--kp", str(kp),
                "--freq", str(freq),
                "--control-hz", str(CONTROL_HZ),
            ],
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  -> FAILED (exit code {result.returncode}) after {elapsed:.1f}s -- continuing with next run\n")
            failures.append((kp, freq))
        else:
            print(f"  -> done in {elapsed:.1f}s\n")

    total_elapsed = time.time() - start_all
    print(f"All runs finished in {total_elapsed/60:.1f} min. CSVs are in ./data/")

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for kp, freq in failures:
            print(f"  Kp={kp:g}, freq={freq} Hz")


if __name__ == "__main__":
    main()
