"""Convert liquid RDF summaries to compressed float32 .npz.

These files are small (about 40 kB each), so the conversion buys little space.
It is done anyway so that every array under data/ is read through the same
co2io interface and carries its provenance header as structured metadata
rather than as a comment a reader has to parse by hand.

Usage:
    python3 convert_rdf.py SRC_ROOT DST_ROOT

SRC_ROOT is the directory holding offresonant/ and
resonant/; the RDF summaries live in each tuning's postprocess/ directory.
"""

from __future__ import print_function

import json
import os
import re
import sys

import numpy as np

RDF_COLUMNS = ["r_ang", "g_CC", "g_CO", "g_OO"]


def parse_meta(lines):
    """Extract the case name, frame count and box from the comment block."""
    meta = {}
    blob = " ".join(lines)
    m = re.search(r"^#\s*([^:]+):", lines[0]) if lines else None
    if m:
        meta["case"] = m.group(1).strip()
    m = re.search(r"(\d+)\s+frames from\s+(\d+)\s+trajectories", blob)
    if m:
        meta["n_frames"] = int(m.group(1))
        meta["n_trajectories"] = int(m.group(2))
    m = re.search(r"stride=(\d+)", blob)
    if m:
        meta["stride"] = int(m.group(1))
    m = re.search(r"box L=([0-9.]+)", blob)
    if m:
        meta["box_L_ang"] = float(m.group(1))
    for key in ("N_C", "N_O"):
        m = re.search(key + r"=(\d+)", blob)
        if m:
            meta[key] = int(m.group(1))
    return meta


def convert_file(src, dst):
    header_lines = []
    with open(src) as fh:
        for line in fh:
            if line.startswith("#"):
                header_lines.append(line.rstrip())
            else:
                break

    raw = np.loadtxt(src, comments="#")
    if raw.ndim != 2 or raw.shape[1] != len(RDF_COLUMNS):
        raise SystemExit("ABORT %s: expected %d columns, found %s"
                         % (src, len(RDF_COLUMNS), raw.shape))

    r = raw[:, 0]
    if not np.all(np.diff(r) > 0):
        raise SystemExit("ABORT %s: r axis is not strictly increasing" % src)

    data = np.array(raw[:, 1:].T, dtype=np.float32)
    ref = raw[:, 1:].T
    scale = np.maximum(np.abs(ref).max(axis=1, keepdims=True), 1e-300)
    rel = float((np.abs(data.astype(np.float64) - ref) / scale).max())
    if rel > 1e-6:
        raise SystemExit("ABORT %s: float32 round-trip error %.3e" % (src, rel))

    np.savez_compressed(
        dst,
        r=np.array(r, dtype=np.float32),
        columns=np.array(RDF_COLUMNS[1:]),
        data=data,
        meta=json.dumps(parse_meta(header_lines)),
        header="\n".join(header_lines),
    )
    return raw.shape[0], rel


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src_root, dst_root = sys.argv[1], sys.argv[2]

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)

    total_in = total_out = 0
    n_files = 0
    for tuning in ("offresonant", "resonant"):
        pp = os.path.join(src_root, tuning, "postprocess")
        if not os.path.isdir(pp):
            continue
        for fname in sorted(os.listdir(pp)):
            if not (fname.startswith("rdf_s1_") and fname.endswith(".dat")):
                continue
            src = os.path.join(pp, fname)
            out_dir = os.path.join(dst_root, tuning)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            out = os.path.join(tuning, fname[:-len(".dat")] + ".npz")
            dst = os.path.join(dst_root, out)
            nrow, rel = convert_file(src, dst)
            size_in = os.path.getsize(src)
            size_out = os.path.getsize(dst)
            total_in += size_in
            total_out += size_out
            n_files += 1
            print("%-70s %4d rows  %6.1f kB -> %5.1f kB  relerr=%.1e"
                  % (out, nrow, size_in / 1e3, size_out / 1e3, rel))

    print("\n%d RDF summaries: %.1f kB -> %.1f kB"
          % (n_files, total_in / 1e3, total_out / 1e3))


if __name__ == "__main__":
    main()
