#!/usr/bin/env python3
"""Step 1: sample configurations + collect available labels from the campaign.

Protocol (q-mode trajectories are
EXCLUDED from the training set since q-mode and e-mode-polar sample the same
distribution -> 17 e-mode trajectories, 1360 train / 680 test, mirroring
the earlier LDA counts):
  TRAIN: 80 frames/trajectory, RANDOM (fixed per-trajectory seed) from steps
         0..15999 (i.e. excluding the final 2 ps), minimum spacing 50 steps.
  TEST:  40 frames/trajectory, deterministic stride 100 over steps
         16000..19900 (the reserved final 2 ps).
Only trajectories with >= 20000 completed steps are used.

If dataset_{train,test}.v1.json exist, the frame selection of any trajectory
present there is REUSED verbatim (keeps previously computed labels valid);
fresh sampling only happens for new trajectories, with a deterministic
crc32-based seed (the v1 run used salted hash(), which is not reproducible
across processes).

Label sources (bare, zero-field only; anything else is left missing and
handled by 02_recompute_labels.py):
  energy  energy.dat col 2 (eV, unshifted total).
  forces  force_bare.dat cols 2..10 if present (post-PR#30 runs);
          else force.dat for lambda=0 / photons-off runs (bare there);
          else MISSING (pre-PR e-mode cavity runs log total forces only).
  dipole  dipole.dat cols 5-7 (bare mu0, a.u.).
  pol     ALWAYS MISSING -> recomputed. The logged polarizability.dat
          contains only the x-components (the MD needs just chi.lambda
          with lambda || x); chi_yy/chi_zz are logged as exact zeros in
          every run, so no logged chi is a valid full-tensor label.

Output: dataset_train.json, dataset_test.json (one record per config with
labels or null), plus a summary of missing labels.
"""
import json
import re
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
SYMBOLS = ["O", "C", "O"]          # atom order of co2-disp.xyz
N_TRAIN, N_TEST = 80, 40
TRAIN_MAX_STEP = 16000             # exclusive
MIN_GAP = 50                       # steps (25 fs)
TEST_STRIDE = 100


def load_indexed(path, ncols=None):
    d = np.loadtxt(path, comments="#")
    return {int(s): d[i] for i, s in enumerate(d[:, 0])}, d


def sample_steps(all_steps, rng):
    pool = [s for s in all_steps if s < TRAIN_MAX_STEP]
    order = rng.permutation(len(pool))
    train = []
    for i in order:
        s = pool[i]
        if all(abs(s - t) >= MIN_GAP for t in train):
            train.append(s)
        if len(train) == N_TRAIN:
            break
    test = [s for s in range(TRAIN_MAX_STEP, 20000, TEST_STRIDE)][:N_TEST]
    return sorted(train), test


prev = {}
for split in ("train", "test"):
    p = ROOT / f"dataset_{split}.v1.json"
    if p.exists():
        for r in json.load(open(p)):
            prev.setdefault(r["source"], {}).setdefault(split, []).append(
                r["step"])

records = {"train": [], "test": []}
summary = []
for d in sorted(CAMPAIGN.glob("lambda=*")):
    if not (d / "md.log").exists():
        continue
    nsteps = sum(1 for _ in open(d / "md.log")) - 1
    if nsteps < 20000:
        summary.append(f"SKIP {d.name}: only {nsteps} steps")
        continue
    m = re.match(r"lambda=([\d.]+)_(\w+)", d.name)
    lam, variant = float(m.group(1)), m.group(2)
    if variant == "qmode" and lam > 0:
        summary.append(f"SKIP {d.name}: q-mode excluded")
        continue
    bare_run = (lam == 0)

    pos_idx, _ = load_indexed(d / "position.dat")
    en_idx, _ = load_indexed(d / "energy.dat")
    dip_idx, _ = load_indexed(d / "dipole.dat")
    fb = d / "force_bare.dat"
    if fb.exists():
        f_idx, _ = load_indexed(fb)
        f_src = "force_bare.dat"
    elif bare_run:
        f_idx, _ = load_indexed(d / "force.dat")
        f_src = "force.dat(lambda=0)"
    else:
        f_idx, f_src = None, "MISSING"

    if d.name in prev:
        train_steps = sorted(prev[d.name]["train"])
        test_steps = sorted(prev[d.name]["test"])
        sel_src = "reused v1 selection"
    else:
        rng = np.random.default_rng(42 + zlib.crc32(d.name.encode()) % 10000)
        train_steps, test_steps = sample_steps(sorted(pos_idx), rng)
        sel_src = "fresh (crc32 seed)"

    for split, steps in (("train", train_steps), ("test", test_steps)):
        for s in steps:
            row = pos_idx[s]
            pos = row[2:11].reshape(3, 3).tolist()
            rec = {"source": d.name, "step": int(s), "lam": lam,
                   "variant": variant, "positions": pos,
                   "energy_eV": None, "forces": None,
                   "dipole": None, "pol": None}
            rec["energy_eV"] = float(en_idx[s][2])
            rec["dipole"] = dip_idx[s][5:8].tolist()
            if f_idx is not None:
                rec["forces"] = f_idx[s][2:11].reshape(3, 3).tolist()
            records[split].append(rec)
    summary.append(f"OK   {d.name}: {len(train_steps)}/{len(test_steps)} "
                   f"train/test, forces from {f_src}, {sel_src}")

for split in ("train", "test"):
    out = ROOT / f"dataset_{split}.json"
    json.dump(records[split], open(out, "w"))
    n = len(records[split])
    miss = {k: sum(1 for r in records[split] if r[k] is None)
            for k in ("energy_eV", "forces", "dipole", "pol")}
    print(f"{split}: {n} configs, missing: {miss}  -> {out.name}")
for line in summary:
    print(" ", line)
print("next: python remap_salvage.py (if v1 results exist), "
      "then python 02_recompute_labels.py")
