"""Convert CBOAMD dipole.dat / polarizability.dat to compressed float32 .npz.

The reduction is purely structural. A column is dropped only if it is bitwise
zero at every step, or bitwise identical to another column at every step. The
uniform time column is replaced by (t0, dt) after checking uniformity. Whatever
survives is stored as float32. Nothing is resampled, smoothed, filtered,
truncated in time, or quantized.

Every assumption is asserted per file and the script aborts on the first
violation, so no trace can be silently degraded. In particular it does NOT
assume that columns 5-7 duplicate columns 2-4: they are the bare and dressed
dipole respectively and differ by up to 5e-2 a.u. at strong coupling.

Time resolution in particular is never reduced: every trace is written at its
original 0.5 fs sampling. Decimation was measured and rejected. Keeping every
4th sample moves the lambda = 0.008 peak by 35 cm^-1, which is disqualifying
for figures whose entire content is peak positions and Rabi splittings.

Usage:
    python3 convert_traces.py SRC_ROOT DST_ROOT

SRC_ROOT is the directory holding the lambda=* AIMD runs and the
nep_md/lambda=* MLIP runs. DST_ROOT receives the same run-directory names with
dipole.npz and polarizability.npz inside.
"""

from __future__ import print_function

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from co2io import COLUMN_LAYOUTS  # noqa: E402

# Runs whose position.dat is shipped. Only the field-free AIMD run is needed:
# SI/measure_nu2_aimd.py measures the bend frequency nu2 = 680.4 cm^-1 from it.
# The 100 ps MLIP position traces are ten times larger and no figure reads
# them, so they stay in the raw release tier.
POSITION_RUNS = {"lambda=0_qmode"}

# Relative tolerance for the uniform-time-grid check. The time column is
# written at 18 significant digits, so genuine uniformity shows up at the
# 1e-12 level; anything looser means the run was restarted or decimated.
DT_RTOL = 1e-9


def read_header(path):
    with open(path) as fh:
        first = fh.readline()
    return first.strip() if first.startswith("#") else ""


def convert_file(src, dst, kind):
    names = COLUMN_LAYOUTS[kind]
    header = read_header(src)
    raw = np.loadtxt(src, comments="#")

    if raw.ndim != 2 or raw.shape[1] != len(names):
        raise SystemExit(
            "ABORT %s: expected %d columns for kind '%s', found %s"
            % (src, len(names), kind, raw.shape))

    nstep = raw.shape[0]

    # --- step column: must be exactly 0..nstep-1 -------------------------
    step = raw[:, names.index("step")]
    if not np.array_equal(step, np.arange(nstep, dtype=step.dtype)):
        raise SystemExit("ABORT %s: step column is not 0..N-1" % src)

    # --- time column: must be a uniform grid ------------------------------
    t = raw[:, names.index("time")]
    diffs = np.diff(t)
    dt = float(np.median(diffs))
    if dt <= 0:
        raise SystemExit("ABORT %s: non-positive dt %r" % (src, dt))
    max_dev = float(np.abs(diffs - dt).max()) if nstep > 1 else 0.0
    if max_dev > DT_RTOL * abs(dt):
        raise SystemExit(
            "ABORT %s: time grid not uniform, max deviation %.3e vs dt %.6f"
            % (src, max_dev, dt))
    t0 = float(t[0])

    # --- classify the remaining columns -----------------------------------
    payload_names = [n for n in names if n not in ("step", "time")]

    zeros = []
    dups = {}
    keep = []
    for name in payload_names:
        col = raw[:, names.index(name)]
        if np.count_nonzero(col) == 0:
            zeros.append(name)
            continue
        source = None
        for kept in keep:
            if np.array_equal(col, raw[:, names.index(kept)]):
                source = kept
                break
        if source is not None:
            dups[name] = source
        else:
            keep.append(name)

    # A run with the polarizability switched off writes an all-zero
    # polarizability.dat, so `keep` is legitimately empty and the whole file
    # reduces to its shape plus the time grid.
    data = np.zeros((len(keep), nstep), dtype=np.float32)
    ref = np.zeros((len(keep), nstep), dtype=np.float64)
    for i, name in enumerate(keep):
        ref[i] = raw[:, names.index(name)]
        data[i] = ref[i]

    # --- round-trip check on the float32 encoding -------------------------
    # Assert that the stored float32 reproduces the float64 source to the
    # relative precision float32 can carry. This catches any column whose
    # dynamic range would be clipped (overflow to inf, underflow to zero).
    back = data.astype(np.float64)
    if not np.all(np.isfinite(back)):
        raise SystemExit("ABORT %s: float32 encoding produced non-finite values" % src)
    lost = (ref != 0) & (back == 0)
    if lost.any():
        raise SystemExit("ABORT %s: float32 underflow in columns %s" % (src, keep))
    if len(keep):
        scale = np.maximum(np.abs(ref).max(axis=1, keepdims=True), 1e-300)
        rel = float((np.abs(back - ref) / scale).max())
    else:
        rel = 0.0
    if rel > 1e-6:
        raise SystemExit(
            "ABORT %s: float32 round-trip error %.3e exceeds 1e-6" % (src, rel))

    np.savez_compressed(
        dst,
        kind=kind,
        columns=np.array(keep),
        data=data,
        dt=np.float64(dt),
        t0=np.float64(t0),
        nstep=np.int64(nstep),
        zeros=np.array(zeros) if zeros else np.array([], dtype="U1"),
        dups=json.dumps(dups),
        header=header,
    )
    return nstep, keep, zeros, dups, rel


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src_root, dst_root = sys.argv[1], sys.argv[2]

    run_dirs = []
    for entry in sorted(os.listdir(src_root)):
        if entry.startswith("lambda="):
            run_dirs.append((entry, os.path.join(src_root, entry)))
    nep_root = os.path.join(src_root, "nep_md")
    if os.path.isdir(nep_root):
        for entry in sorted(os.listdir(nep_root)):
            if entry.startswith("lambda="):
                run_dirs.append((os.path.join("nep_md", entry),
                                 os.path.join(nep_root, entry)))

    total_in = total_out = 0
    n_files = 0
    for rel_name, run_dir in run_dirs:
        out_dir = os.path.join(dst_root, rel_name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        kinds = ["dipole", "polarizability"]
        if rel_name in POSITION_RUNS:
            kinds.append("position")
        for kind in kinds:
            src = os.path.join(run_dir, kind + ".dat")
            if not os.path.exists(src):
                print("  skip (absent): %s/%s.dat" % (rel_name, kind))
                continue
            dst = os.path.join(out_dir, kind + ".npz")
            nstep, keep, zeros, dups, relerr = convert_file(src, dst, kind)
            size_in = os.path.getsize(src)
            size_out = os.path.getsize(dst)
            total_in += size_in
            total_out += size_out
            n_files += 1
            print("%-34s %-16s N=%-7d keep=%-2d zero=%-2d dup=%-2d "
                  "%6.1f MB -> %5.2f MB (%4.1fx)  relerr=%.1e"
                  % (rel_name, kind, nstep, len(keep), len(zeros), len(dups),
                     size_in / 1e6, size_out / 1e6,
                     float(size_in) / size_out, relerr))
            print("      keep: %s" % ", ".join(keep))
            if zeros:
                print("      zero: %s" % ", ".join(zeros))
            if dups:
                print("      dup : %s" % ", ".join(
                    "%s=%s" % (k, v) for k, v in sorted(dups.items())))

    print("\n%d files: %.1f MB -> %.1f MB (%.1fx)"
          % (n_files, total_in / 1e6, total_out / 1e6,
             float(total_in) / max(total_out, 1)))


if __name__ == "__main__":
    main()
