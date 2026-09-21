#!/usr/bin/env python3
"""Step 2: fill missing labels with zero-field PBE0/aug-cc-pVDZ single points.

For every config with any missing label, computes exactly what is missing:
  energy/forces : one RKS (conv 1e-8) + analytic gradient
  dipole        : from the same SCF
  pol           : six extra finite-field SCFs (+-2e-4 au along x, y, z)
Units match the run logs / NEP labels: energy eV (unshifted), forces eV/A,
dipole a.u., polarizability a.u.

Resumable: results append to recompute_out.jsonl keyed by (split, index).
A config is skipped only if an existing entry covers ALL labels the dataset
record is missing (so widening the missing set, e.g. after dropping the
x-only logged chi, re-queues configs whose earlier entries lack pol).
When both an old and a new entry exist for a key, the later line wins in
03_assemble_xyz.py. Parallel over configs (--nproc, default 8).
Run on a single core; no scheduler needed.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HA_EV = 27.211386245988
BOHR_A = 0.529177210903
DE = 2e-4
LABELS = ("energy_eV", "forces", "dipole", "pol")


def compute(task):
    split, idx, rec = task
    from pyscf import gto, dft
    need_pol = rec["pol"] is None
    mol = gto.M(atom=[[s, tuple(p)] for s, p in zip(["O", "C", "O"],
                                                    rec["positions"])],
                basis="aug-cc-pvdz", unit="Angstrom", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.conv_tol = 1e-8
    e = mf.kernel()
    assert mf.converged
    dip = mf.dip_moment(unit="AU", verbose=0)
    grad = mf.nuc_grad_method().kernel()          # Ha/Bohr
    forces = (-grad * HA_EV / BOHR_A).tolist()
    out = {"split": split, "idx": idx,
           "energy_eV": float(e * HA_EV),
           "forces": forces, "dipole": dip.tolist()}
    if need_pol:
        mol.set_common_orig([0, 0, 0])
        ao_dip = mol.intor_symmetric("int1e_r", comp=3)
        dm0 = mf.make_rdm1()
        chi = np.zeros((3, 3))
        for a in range(3):
            mus = []
            for sgn in (+1, -1):
                mff = dft.RKS(mol)
                mff.xc = "pbe0"
                mff.conv_tol = 1e-8
                h = mff.get_hcore(mol) + sgn * DE * ao_dip[a]
                mff.get_hcore = lambda *args, hh=h: hh
                mff.kernel(dm0=dm0)
                mus.append(mff.dip_moment(unit="AU", verbose=0))
            chi[a] = (np.array(mus[0]) - np.array(mus[1])) / (2 * DE)
        chi = 0.5 * (chi + chi.T)
        out["pol"] = chi.reshape(9).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=8)
    args = ap.parse_args()

    done = {}
    out_path = ROOT / "recompute_out.jsonl"
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done[(r["split"], r["idx"])] = r

    tasks = []
    for split in ("train", "test"):
        data = json.load(open(ROOT / f"dataset_{split}.json"))
        for i, rec in enumerate(data):
            missing = [k for k in LABELS if rec[k] is None]
            if not missing:
                continue
            have = done.get((split, i))
            if have is not None and all(k in have for k in missing):
                continue
            tasks.append((split, i, rec))
    print(f"{len(tasks)} configs to compute ({len(done)} entries on file)")

    with Pool(args.nproc) as pool, open(out_path, "a") as fh:
        for k, res in enumerate(pool.imap_unordered(compute, tasks)):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(tasks)}", flush=True)
    print("done. next: python 03_assemble_xyz.py")


if __name__ == "__main__":
    main()
