#!/usr/bin/env python3
"""Liquid-CO2 radial distribution functions g(r): C-C, C-O, O-O (paper Fig S1).

Set (1): the "bare" S1-style three-panel RDF figure for ONE configuration,
reproducing Fig. S1 of Mathur et al., J. Phys. Chem. B 2023, 127, 4562
(10.1021/acs.jpcb.3c00610). Reads the LAMMPS custom dumps (traj_*/traj.dump)
written every 20 steps; type 1 = C, type 2 = O.

Outputs (into --outdir):
  <prefix>.dat   columns: r[A]  g_CC  g_CO  g_OO
  <prefix>.png / .pdf   three-panel figure

Usage (from johannes_runs/postprocess, after activating dp-gpu):
  python plot_rdf_s1.py                                      # case 10 (default)
  python plot_rdf_s1.py --case-dir ../11_cavity_..._lam010   # any case
  python plot_rdf_s1.py --stride 5                           # faster
"""

import argparse
import os
from pathlib import Path

import numpy as np

import rdf_core as R


def parse_args():
    here = Path(__file__).resolve().parent
    default_case = here.parent / "10_out_of_cavity_ir_40traj_newmodels"
    p = argparse.ArgumentParser(description="Liquid-CO2 g(r) (paper Fig S1).")
    p.add_argument("--case-dir", default=str(default_case),
                   help="run directory containing traj_*/traj.dump")
    p.add_argument("--glob", default="traj_*/traj.dump")
    p.add_argument("--rmax", type=float, default=8.0, help="max r [A] (<= L/2)")
    p.add_argument("--dr", type=float, default=0.02, help="bin width [A]")
    p.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    p.add_argument("--type-c", type=int, default=R.TYPE_C)
    p.add_argument("--type-o", type=int, default=R.TYPE_O)
    p.add_argument("--outdir", default=str(here), help="output directory")
    p.add_argument("--prefix", default=None,
                   help="output basename (default rdf_s1_<casename>)")
    p.add_argument("--xlim", type=float, nargs=2, default=(2.0, 7.0))
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    case = os.path.normpath(args.case_dir)
    name = os.path.basename(case)
    prefix = args.prefix or f"rdf_s1_{name}"

    files = R.find_dumps(case, args.glob)
    if not files:
        raise SystemExit(f"[plot_rdf_s1] no dumps under {case}/{args.glob}")

    def log(k, n, fn, ff):
        print(f"[plot_rdf_s1] {k}/{n}  {os.path.relpath(fn, case)}  frames={ff}")

    d = R.compute_rdf(files, rmax=args.rmax, dr=args.dr, stride=args.stride,
                      type_c=args.type_c, type_o=args.type_o, progress=log)
    r = d["r"]

    os.makedirs(args.outdir, exist_ok=True)
    dat = os.path.join(args.outdir, f"{prefix}.dat")
    np.savetxt(dat, np.column_stack([r, d["g_cc"], d["g_co"], d["g_oo"]]),
               header=(f"{name}: liquid-CO2 g(r), {d['nframes']} frames from "
                       f"{len(files)} trajectories (stride={args.stride})\n"
                       f"box L={d['L']:.6f} A, N_C={d['nc']}, N_O={d['no']}\n"
                       f"r[A]  g_CC  g_CO  g_OO"))
    print(f"[plot_rdf_s1] wrote {dat}  ({d['nframes']} frames, L={d['L']:.4f} A)")

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, g, title in zip(axes, (d["g_cc"], d["g_co"], d["g_oo"]),
                            ("C-C", "C-O", "O-O")):
        ax.plot(r, g, "b-", lw=1.4)
        ax.axhline(1.0, ls=":", c="0.6", lw=0.7)
        ax.set_xlim(*args.xlim)
        sel = (r >= args.xlim[0]) & (r <= args.xlim[1])
        ax.set_ylim(0.0, max(1.05, 1.1 * float(g[sel].max())))
        ax.set_title(title)
        ax.set_xlabel(r"$r\ (\mathrm{\AA})$")
    axes[0].set_ylabel(r"$g(r)$")
    fig.suptitle(f"Fig S1 RDFs: {name} ({d['nframes']} frames)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"{prefix}.{ext}"), dpi=300)
    print(f"[plot_rdf_s1] wrote {os.path.join(args.outdir, prefix)}.png (+pdf)")


if __name__ == "__main__":
    main()
