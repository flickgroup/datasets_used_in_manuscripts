#!/usr/bin/env python3
"""Step 3: merge labels and write train.xyz/test.xyz for the three NEP models.

Writes the extended-XYZ label format the NEP models are trained on, verified
against the earlier LDA training set: 18 A cubic Lattice, pbc "T T T",
Properties=species:S:1:pos:R:3:force:R:3, comment keys energy=<shifted eV>,
dipole="dx dy dz" (a.u., total), pol="9 components" (a.u., total).

The energy shift is chosen as -round(mean unshifted train energy, 2) and
written to energy_shift_eV.txt (quoted in the SI). Recomputed labels
(recompute_out.jsonl) take precedence over missing entries; where BOTH a
logged and a recomputed value exist the logged one is kept (log provenance
mirrors the original protocol) unless --prefer-recomputed is given.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIRS = ["ener_model_nep", "dipole_model_nep", "polarizability_model_nep"]
LAT = "1.800000e+01 0.000000e+00 0.000000e+00 0.000000e+00 1.800000e+01 " \
      "0.000000e+00 0.000000e+00 0.000000e+00 1.800000e+01"

ap = argparse.ArgumentParser()
ap.add_argument("--prefer-recomputed", action="store_true")
args = ap.parse_args()

recomp = {}
p = ROOT / "recompute_out.jsonl"
if p.exists():
    for line in open(p):
        r = json.loads(line)
        recomp[(r["split"], r["idx"])] = r

data = {}
for split in ("train", "test"):
    data[split] = json.load(open(ROOT / f"dataset_{split}.json"))
    for i, rec in enumerate(data[split]):
        r = recomp.get((split, i))
        if r is None:
            continue
        for k in ("energy_eV", "forces", "dipole", "pol"):
            if k in r and (rec[k] is None or args.prefer_recomputed):
                rec[k] = r[k]

incomplete = [(s, i) for s in data for i, rec in enumerate(data[s])
              if any(rec[k] is None for k in ("energy_eV", "forces",
                                              "dipole", "pol"))]
if incomplete:
    raise SystemExit(f"{len(incomplete)} configs still missing labels; "
                     "run 02_recompute_labels.py first")

energies = np.array([r["energy_eV"] for r in data["train"]])
shift = round(-energies.mean(), 2)
(ROOT / "energy_shift_eV.txt").write_text(f"{shift}\n")
print(f"energy shift: {shift:+.2f} eV (mean train energy {energies.mean():.2f})")


def write_xyz(path, records):
    with open(path, "w") as f:
        for rec in records:
            e = rec["energy_eV"] + shift
            dip = " ".join(f"{v:e}" for v in rec["dipole"])
            pol = " ".join(f"{v:e}" for v in rec["pol"])
            f.write("3\n")
            f.write(f'Lattice="{LAT}" '
                    f"Properties=species:S:1:pos:R:3:force:R:3 "
                    f"energy={e:e} dipole=\"{dip}\" pol=\"{pol}\" "
                    f'pbc="T T T"\n')
            for sym, pos, frc in zip(["O", "C", "O"], rec["positions"],
                                     rec["forces"]):
                f.write(f"{sym} " + " ".join(f"{v:e}" for v in pos) + " "
                        + " ".join(f"{v:e}" for v in frc) + "\n")


for md in MODEL_DIRS:
    for split in ("train", "test"):
        write_xyz(ROOT / md / f"{split}.xyz", data[split])
print(f"wrote train.xyz ({len(data['train'])}) / test.xyz "
      f"({len(data['test'])}) into {', '.join(MODEL_DIRS)}")
print("next: run nep in each model directory")
