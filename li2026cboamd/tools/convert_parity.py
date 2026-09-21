"""Convert DeePMD `dp test` parity output to compressed float32 .npz.

The liquid force and atomic-dipole parity files are about 56 MB of ASCII each,
which is the whole reason SI Fig. S6 could not simply be shipped as-is. They
are dense tables with no redundant columns, so the only reduction available is
the float32 re-encoding, worth roughly 14x.

The single-molecule NEP parity files (energy_train.out and friends) are small
enough to ship as the original ASCII and are not handled here.

Usage:
    python3 convert_parity.py SRC_ROOT DST_ROOT

SRC_ROOT is the DeePMD training directory holding the three
*_aug_pbe0_*after210_clean model directories.
"""

from __future__ import print_function

import os
import sys

import numpy as np

# (model directory, file name, output stem)
PARITY_FILES = [
    ("ener_model_aug_pbe0_after210_clean", "test_out.e.out", "ener_energy"),
    ("ener_model_aug_pbe0_after210_clean", "test_out.f.out", "ener_force"),
    ("dipole_model_aug_pbe0_true_dipole_after210_clean", "test_out.out",
     "dipole_atomic"),
    ("polar_model_aug_pbe0_after210_clean", "test_out.out", "polar_global"),
]


def convert_file(src, dst):
    with open(src) as fh:
        header = fh.readline().strip()
    raw = np.loadtxt(src, comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] % 2 != 0:
        raise SystemExit(
            "ABORT %s: %d columns is odd, expected paired data/prediction "
            "halves" % (src, raw.shape[1]))

    data = np.array(raw, dtype=np.float32)
    scale = max(float(np.abs(raw).max()), 1e-300)
    rel = float((np.abs(data.astype(np.float64) - raw) / scale).max())
    if rel > 1e-6:
        raise SystemExit("ABORT %s: float32 round-trip error %.3e" % (src, rel))
    if not np.all(np.isfinite(data)):
        raise SystemExit("ABORT %s: non-finite values after encoding" % src)

    np.savez_compressed(dst, data=data, header=header)
    return raw.shape, rel


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src_root, dst_root = sys.argv[1], sys.argv[2]

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)

    total_in = total_out = 0
    for model_dir, fname, stem in PARITY_FILES:
        src = os.path.join(src_root, model_dir, fname)
        if not os.path.exists(src):
            raise SystemExit("ABORT: missing %s" % src)
        dst = os.path.join(dst_root, stem + ".npz")
        shape, rel = convert_file(src, dst)
        size_in = os.path.getsize(src)
        size_out = os.path.getsize(dst)
        total_in += size_in
        total_out += size_out
        print("%-16s %-14s %6.1f MB -> %5.2f MB (%4.1fx)  relerr=%.1e"
              % (stem, "x".join(str(s) for s in shape),
                 size_in / 1e6, size_out / 1e6,
                 float(size_in) / size_out, rel))

    print("\n%d parity files: %.1f MB -> %.1f MB (%.1fx)"
          % (len(PARITY_FILES), total_in / 1e6, total_out / 1e6,
             float(total_in) / max(total_out, 1)))


if __name__ == "__main__":
    main()
