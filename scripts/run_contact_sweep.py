"""
Sweep runner for fall_analysis_test.py.

Full grid over physics dt and PhysX solver iteration counts (every combination
of FREQ_VALUES x POS_ITERS_VALUES x VEL_ITERS_VALUES), to see interaction
effects between them and where the fall/contact response stops matching the
trusted "ground truth" config, plus how much each setting costs in
simulation wall-clock time.

contact_offset and max_depenetration_velocity are held fixed throughout.

For each (freq, pos_iters, vel_iters) setting:
  - 1 run saves the full movement trace (angles + pistoning) to data2/
  - 4 more runs are timing-only (no CSV) to average out wall-clock noise
All RUNS_PER_COMBO runs' wall-clock times are averaged and:
  - written as one row of data2/timing_summary.csv
  - appended as a repeated last column ("Mean_Wall_Time_Sec") onto every row
    of that setting's own movement CSV, so the timing travels with the data

Usage:
    python run_contact_sweep.py
"""

import csv
import itertools
import os
import signal
import statistics
import subprocess
import sys
import time

SCRIPT = "fall_analysis_test.py"

# "Ground truth" config: trusted as physically accurate (fine dt, high iterations).
# Only used here to label that point in the output; the grid below covers it
# automatically since its values are included in the lists.
GROUND_TRUTH_FREQ = 4800       # Hz  (dt ~= 0.000208s)
GROUND_TRUTH_POS_ITERS = 32
GROUND_TRUTH_VEL_ITERS = 16

# Your current production setting, just for labeling purposes (also already
# covered by the grid below).
CURRENT_PRODUCTION = (480, 16, 8)

# Full grid: every combination of these three lists is tested.
# 6 x 4 x 4 = 96 unique settings x RUNS_PER_COMBO runs each = 480 runs total.
FREQ_VALUES = [240, 360, 480, 960, 1200, 1920, 4800]
POS_ITERS_VALUES = [4, 8, 16, 32]
VEL_ITERS_VALUES = [1, 4, 8]

RUNS_PER_COMBO = 5         # 1 movement-saving run + 4 timing-only runs
RUN_TIMEOUT_SEC = 300      # kill + move on if a single run hangs
POLL_INTERVAL_SEC = 5      # how often to check on the child process
HEARTBEAT_EVERY_SEC = 30   # how often to print "still running" while waiting

DATA_DIR = "data2"
SUMMARY_CSV = os.path.join(DATA_DIR, "timing_summary.csv")


def _kill_process_group(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def build_combos():
    """Returns list of (tag, freq, pos_iters, vel_iters) -- the full grid."""
    combos = []
    for freq, pos_iters, vel_iters in itertools.product(FREQ_VALUES, POS_ITERS_VALUES, VEL_ITERS_VALUES):
        key = (freq, pos_iters, vel_iters)
        if key == (GROUND_TRUTH_FREQ, GROUND_TRUTH_POS_ITERS, GROUND_TRUTH_VEL_ITERS):
            tag = "ground_truth"
        elif key == CURRENT_PRODUCTION:
            tag = "current_production"
        else:
            tag = "grid"
        combos.append((tag, freq, pos_iters, vel_iters))
    return combos


def run_once(freq, pos_iters, vel_iters, save_movement):
    """Runs one subprocess, returns (ok, wall_time_sec_or_None).

    Polls instead of doing one big blocking communicate(timeout=...) so we
    can print heartbeats (a quiet run isn't necessarily a hung run -- Isaac
    Sim's first launch in a session can take a couple minutes on its own)
    and so Ctrl+C kills the child's process group instead of leaving an
    orphaned Isaac Sim process behind.
    """
    cmd = [
        sys.executable, SCRIPT,
        "--freq", str(freq),
        "--pos-iters", str(pos_iters),
        "--vel-iters", str(vel_iters),
    ]
    if save_movement:
        cmd.append("--save-movement")

    proc = subprocess.Popen(
        cmd,
        start_new_session=True,  # own process group so we can kill it (and any children)
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    start = time.time()
    timed_out = False
    last_heartbeat = 0

    try:
        while proc.poll() is None:
            elapsed = time.time() - start
            if elapsed > RUN_TIMEOUT_SEC:
                timed_out = True
                break
            if elapsed - last_heartbeat >= HEARTBEAT_EVERY_SEC:
                print(f"    ... still running ({elapsed:.0f}s elapsed, timeout at {RUN_TIMEOUT_SEC}s)")
                last_heartbeat = elapsed
            time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n  Ctrl+C received -- killing this run's process group, then exiting")
        _kill_process_group(proc)
        proc.wait()
        raise

    if timed_out:
        print(f"    TIMED OUT after {RUN_TIMEOUT_SEC}s, killing process group")
        _kill_process_group(proc)

    try:
        stdout, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        stdout, _ = proc.communicate()

    returncode = -1 if timed_out else proc.returncode

    wall_time = None
    for line in (stdout or "").splitlines():
        if line.startswith("SIM_WALL_TIME_SEC:"):
            wall_time = float(line.split(":", 1)[1].strip())
            break

    ok = returncode == 0 and wall_time is not None
    if not ok:
        print(f"    FAILED (exit={returncode}). Last output lines:")
        for line in (stdout or "").splitlines()[-15:]:
            print(f"      {line}")

    return ok, wall_time


def append_mean_time_column(csv_path, mean_time):
    """Appends a repeated 'Mean_Wall_Time_Sec' column onto every row of an
    already-written movement CSV, so the timing average travels with the
    data without needing to cross-reference the summary file."""
    if not os.path.exists(csv_path):
        print(f"    (movement CSV not found at {csv_path}, skipping mean-time column)")
        return

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return

    rows[0].append("Mean_Wall_Time_Sec")
    for row in rows[1:]:
        row.append(f"{mean_time:.6f}")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def write_summary_row(row, write_header):
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "tag", "freq_hz", "dt", "pos_iters", "vel_iters",
                "n_ok", "n_total", "mean_wall_time_sec", "std_wall_time_sec",
                "movement_csv",
            ])
        writer.writerow(row)


def load_completed_combos():
    """Reads timing_summary.csv (if it exists) and returns the set of
    (freq, pos_iters, vel_iters) combos already recorded, so a rerun after an
    interruption skips what's already done instead of redoing everything."""
    completed = set()
    if not os.path.exists(SUMMARY_CSV):
        return completed
    with open(SUMMARY_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                completed.add((float(row["freq_hz"]), int(row["pos_iters"]), int(row["vel_iters"])))
            except (KeyError, ValueError):
                continue
    return completed


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    write_header = not os.path.exists(SUMMARY_CSV)
    completed = load_completed_combos()

    combos = build_combos()
    print(f"Running {len(combos)} settings x {RUNS_PER_COMBO} runs each "
          f"= {len(combos) * RUNS_PER_COMBO} total simulations.\n")

    start_all = time.time()

    try:
        for i, (tag, freq, pos_iters, vel_iters) in enumerate(combos, start=1):
            if (float(freq), pos_iters, vel_iters) in completed:
                print(f"[{i}/{len(combos)}] ({tag}) freq={freq} Hz  pos_iters={pos_iters}  vel_iters={vel_iters} "
                      f"-- already in {SUMMARY_CSV}, skipping")
                continue

            run_tag = f"freq{int(round(freq))}hz_pos{pos_iters}_vel{vel_iters}"
            movement_csv = f"movement_{run_tag}.csv"
            movement_path = os.path.join(DATA_DIR, movement_csv)
            print(f"[{i}/{len(combos)}] ({tag}) freq={freq} Hz  pos_iters={pos_iters}  vel_iters={vel_iters}")

            times = []
            for run_idx in range(RUNS_PER_COMBO):
                save_movement = (run_idx == 0)
                t0 = time.time()
                ok, wall_time = run_once(freq, pos_iters, vel_iters, save_movement)
                elapsed = time.time() - t0
                if ok:
                    print(f"  run {run_idx + 1}/{RUNS_PER_COMBO}: sim={wall_time:.3f}s (process took {elapsed:.1f}s)")
                    times.append(wall_time)
                else:
                    print(f"  run {run_idx + 1}/{RUNS_PER_COMBO}: failed/timed out after {elapsed:.1f}s -- skipped")

            if times:
                mean_t = statistics.mean(times)
                std_t = statistics.stdev(times) if len(times) > 1 else 0.0
                append_mean_time_column(movement_path, mean_t)
            else:
                mean_t, std_t = float("nan"), float("nan")

            write_summary_row(
                [tag, freq, 1.0 / freq, pos_iters, vel_iters,
                 len(times), RUNS_PER_COMBO, f"{mean_t:.6f}", f"{std_t:.6f}", movement_csv],
                write_header,
            )
            write_header = False  # header only goes in once

            print(f"  -> mean sim time: {mean_t:.3f}s (n={len(times)}/{RUNS_PER_COMBO})\n")
    except KeyboardInterrupt:
        print(f"\nInterrupted by user after {i}/{len(combos)} settings. "
              f"Whatever finished is already saved in {SUMMARY_CSV} and data2/movement_*.csv -- "
              f"just rerun this script to continue (it appends, it won't overwrite what's there).")
        return

    total_elapsed = time.time() - start_all
    print(f"All settings finished in {total_elapsed / 60:.1f} min.")
    print(f"Movement CSVs + {SUMMARY_CSV} are in {DATA_DIR}/")


if __name__ == "__main__":
    main()
