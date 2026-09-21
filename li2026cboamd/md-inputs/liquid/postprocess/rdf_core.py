"""Shared liquid-CO2 g(r) machinery: dump parsing + site-site RDF computation.

Used by plot_rdf_s1.py (per-configuration figure) and plot_rdf_compare.py
(overlay of several configurations). Distances use the minimum-image convention
in the fixed cubic NVE box (valid to L/2 ~ 8.1 A). ALL pairs are counted
(intra- + intermolecular), so the intramolecular O-O distance (~2.3 A) appears
as the sharp low-r O-O peak, matching paper Fig S1.

Normalization (ideal-gas spherical-shell, rho = N/V):
  like  X-X : g = <hist_{i<j}> / [ 0.5 N_X (N_X-1) * shell(r)/V ]
  cross C-O : g = <hist_{c,o}>  / [ N_C N_O          * shell(r)/V ]
"""

import glob
import os

import numpy as np

TYPE_C = 1          # carbon atom type (mass 12)
TYPE_O = 2          # oxygen atom type (mass 16)


def iter_frames(path, stride):
    """Stream (box_lengths[3], types[N], xyz[N,3]) from a LAMMPS custom dump.

    Skipped frames (frame_index % stride != 0) are advanced without parsing
    coordinates. A truncated trailing frame (job still writing) ends iteration
    cleanly, so partial dumps are usable.
    """
    with open(path) as f:
        frame = 0
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            if not f.readline():                       # timestep value
                return
            f.readline()                               # ITEM: NUMBER OF ATOMS
            n_line = f.readline()
            if not n_line:
                return
            natoms = int(n_line)
            f.readline()                               # ITEM: BOX BOUNDS ...
            box = []
            for _ in range(3):
                bl = f.readline()
                if not bl:
                    return
                lo, hi = bl.split()[:2]
                box.append(float(hi) - float(lo))
            hdr = f.readline()                         # ITEM: ATOMS ...
            if not hdr:
                return
            cols = hdr.split()[2:]
            ti, xi = cols.index("type"), cols.index("x")

            if frame % stride:                         # skip: don't parse atoms
                ok = True
                for _ in range(natoms):
                    if not f.readline():
                        ok = False
                        break
                if not ok:
                    return
                frame += 1
                continue

            types = np.empty(natoms, dtype=np.int32)
            xyz = np.empty((natoms, 3), dtype=np.float64)
            ok = True
            for i in range(natoms):
                al = f.readline()
                if not al:
                    ok = False
                    break
                t = al.split()
                types[i] = int(float(t[ti]))
                xyz[i] = (t[xi], t[xi + 1], t[xi + 2])
            if not ok:                                 # truncated final frame
                return
            yield np.asarray(box), types, xyz
            frame += 1


def pair_hist(a, b, box, bins, same):
    """Minimum-image pair-distance histogram between point sets a and b.

    same=True: like species -> unordered pairs (i<j).
    same=False: cross species -> every ordered (a_i, b_j) pair once.
    """
    diff = a[:, None, :] - b[None, :, :]               # (nA, nB, 3)
    diff -= box * np.round(diff / box)                 # minimum image (cubic)
    d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    if same:
        d = d[np.triu_indices(a.shape[0], k=1)]
    else:
        d = d.ravel()
    return np.histogram(d, bins=bins)[0]


def find_dumps(case_dir, pattern="traj_*/traj.dump"):
    """Sorted list of dump files under a case directory."""
    return sorted(glob.glob(os.path.join(case_dir, pattern)))


def compute_rdf(files, rmax=8.0, dr=0.02, stride=1,
                type_c=TYPE_C, type_o=TYPE_O, progress=None):
    """Ensemble-averaged C-C, C-O, O-O g(r) over a list of dump files.

    progress: optional callable(k, n, relpath, nframes) for per-file logging.
    Returns dict: r, g_cc, g_co, g_oo, nframes, nc, no, L (box length).
    """
    if not files:
        raise ValueError("compute_rdf: no dump files given")
    bins = np.arange(0.0, rmax + dr, dr)
    h_cc = np.zeros(len(bins) - 1, dtype=np.float64)
    h_co = np.zeros_like(h_cc)
    h_oo = np.zeros_like(h_cc)
    nframes = 0
    vol_sum = 0.0
    nc = no = 0

    for k, fn in enumerate(files, 1):
        ff = 0
        for box, types, xyz in iter_frames(fn, stride):
            c = xyz[types == type_c]
            o = xyz[types == type_o]
            nc, no = c.shape[0], o.shape[0]
            h_cc += pair_hist(c, c, box, bins, same=True)
            h_oo += pair_hist(o, o, box, bins, same=True)
            h_co += pair_hist(c, o, box, bins, same=False)
            vol_sum += float(box[0] * box[1] * box[2])
            nframes += 1
            ff += 1
        if progress:
            progress(k, len(files), fn, ff)

    if nframes == 0:
        raise ValueError("compute_rdf: no complete frames found")

    r = 0.5 * (bins[:-1] + bins[1:])
    vol = vol_sum / nframes
    shell = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    def norm(hist, na, nb, like):
        ideal_pairs = (0.5 * na * (na - 1)) if like else (na * nb)
        ideal = ideal_pairs * shell / vol
        return np.divide(hist / nframes, ideal,
                         out=np.zeros_like(hist), where=ideal > 0)

    return dict(r=r,
                g_cc=norm(h_cc, nc, nc, True),
                g_co=norm(h_co, nc, no, False),
                g_oo=norm(h_oo, no, no, True),
                nframes=nframes, nc=nc, no=no, L=vol ** (1.0 / 3.0))
