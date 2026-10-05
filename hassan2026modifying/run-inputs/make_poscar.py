#!/usr/bin/env python3
"""Write the POSCAR for one stacking at one interlayer distance.

The six POSCAR_* files in this directory are the bilayer cells at R = 3.0 Å,
the first point of the scan. Every other geometry is the same cell with the two
layers moved symmetrically about the mid-plane at z = 10 Å, so the upper layer
sits at z = 10 + R/2 and the lower one at z = 10 - R/2. Nothing else changes:
the in-plane lattice is held at its relaxed monolayer value and the 20 Å of
vacuum along z is fixed, which is what makes E(R) - E(10 Å) meaningful.

    python make_poscar.py POSCAR_hbn_aa 3.6425          # -> stdout
    python make_poscar.py POSCAR_hbn_aa 10.0 -o POSCAR  # the reference cell

To regenerate a whole scan the way the manuscript did:

    for R in $(seq 3.0 0.0025 5.1); do
        python make_poscar.py POSCAR_hbn_aa $R -o scan/R$R/POSCAR
    done
"""

import argparse
import sys
from pathlib import Path

MIDPLANE_ANG = 10.0  # half of the 20 Å cell height


def retarget(lines, distance, midplane=MIDPLANE_ANG):
    """Rewrite the z column of a Cartesian POSCAR to the requested distance."""
    counts = [int(n) for n in lines[6].split()]
    start = 8  # comment, scale, 3 lattice vectors, species, counts, "Cartesian"
    if not lines[7].lower().startswith(("c", "k")):
        raise ValueError("expected a Cartesian POSCAR; direct coordinates are "
                         "not handled because the z scaling would change too")

    upper, lower = midplane + distance / 2, midplane - distance / 2
    out = list(lines[:start])
    for line in lines[start:start + sum(counts)]:
        x, y, z = line.split()[:3]
        # Which layer an atom belongs to is decided by which side of the
        # mid-plane it started on; the input cells are symmetric about it.
        new_z = upper if float(z) > midplane else lower
        out.append(f"{float(x):22.16f}{float(y):22.16f}{new_z:22.16f}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("poscar", type=Path, help="a POSCAR_* template from this directory")
    p.add_argument("distance", type=float, help="interlayer distance R in Angstrom")
    p.add_argument("-o", "--out", type=Path, help="output file (default: stdout)")
    args = p.parse_args()

    lines = args.poscar.read_text().splitlines()
    out = retarget(lines, args.distance)
    text = "\n".join(out) + "\n"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}  (R = {args.distance} A)", file=sys.stderr)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
