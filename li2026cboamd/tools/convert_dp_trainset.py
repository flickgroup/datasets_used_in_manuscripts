"""Halve the DeePMD training set by storing the bulk arrays as float32.

DeePMD-kit reads float32 `.npy` directly, so the converted tree stays a
drop-in input for `dp train`. Only the three large per-atom arrays are
converted:

    coord.npy          (1972, 576)  Angstrom, box is about 16 A
    force.npy          (1972, 576)  eV/Angstrom
    atomic_dipole.npy  (1972, 576)  e*Angstrom

The small arrays keep float64. `energy.npy` in particular holds total energies
of order 1e4 eV, where float32 would leave about 1 meV of absolute rounding.
That is a tenth of the model's own energy RMSE rather than a negligible
fraction of it, and the array is 16 kB, so there is nothing to gain by
converting it. box.npy, dipole.npy and polarizability.npy are likewise too
small to matter.

The tree holds four subsets: C64O128 (energy, force and both tensors),
atomic_dipole, global_dipole and global_polar. They duplicate coord.npy and
box.npy rather than symlinking so that each is consumable by `dp train` as-is.
global_dipole is not read by any shipped input.json; it is kept because it is
the natural comparison set for the O-site sum quoted in Table S2.

Usage:
    python3 convert_dp_trainset.py TRAIN_DATA_ROOT
"""

from __future__ import print_function

import os
import sys

import numpy as np

CONVERT = {"coord.npy", "force.npy", "atomic_dipole.npy"}

# Largest acceptable relative round-trip error for a converted array.
REL_TOL = 1e-6


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = sys.argv[1]

    before = after = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if not fname.endswith(".npy"):
                continue
            path = os.path.join(dirpath, fname)
            size_in = os.path.getsize(path)
            before += size_in

            if fname not in CONVERT:
                after += size_in
                continue

            arr = np.load(path)
            if arr.dtype == np.float32:
                after += size_in
                continue
            if arr.dtype != np.float64:
                raise SystemExit("ABORT %s: unexpected dtype %s"
                                 % (path, arr.dtype))

            small = arr.astype(np.float32)
            back = small.astype(np.float64)
            if not np.all(np.isfinite(back)):
                raise SystemExit("ABORT %s: non-finite after conversion" % path)
            scale = max(float(np.abs(arr).max()), 1e-300)
            rel = float(np.abs(back - arr).max()) / scale
            if rel > REL_TOL:
                raise SystemExit("ABORT %s: round-trip error %.3e" % (path, rel))

            np.save(path, small)
            size_out = os.path.getsize(path)
            after += size_out
            print("%-58s %-14s %5.1f MB -> %5.1f MB  relerr=%.1e"
                  % (os.path.relpath(path, root), str(arr.shape),
                     size_in / 1e6, size_out / 1e6, rel))

    print("\ntrain set: %.1f MB -> %.1f MB" % (before / 1e6, after / 1e6))


if __name__ == "__main__":
    main()
