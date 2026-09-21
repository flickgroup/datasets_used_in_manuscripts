#!/usr/bin/env python3
"""Satellite origin hunt, part 2 (MLIP 100 ps runs, 0.33 cm-1 resolution).

 a. deep peak list (>0.3% of max) over 20-3800 for lambda=0.3 polar/nopolar,
    dressed AND bare dipole -> find modulation lines, hidden partners,
    bend / symmetric-stretch activity.
 b. satellite offsets vs lambda for all couplings, both variants.
"""
import sys
from pathlib import Path
import numpy as np

# --- repository layout ------------------------------------------------------
# This script runs from inside its own folder in the li2026cboamd repository and
# reads only files shipped in that repository. tools/co2io.py hides the .npz
# encoding: its loaders return exactly the arrays the original ASCII files
# held.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import co2io  # noqa: E402

DATA = REPO / "data"


BASE = DATA / "single_molecule"
AU_TO_CM = 219474.63
OMEGA_C = 2442.0


def spec_of(f, col):
    d = co2io.load_trace_columns(f)
    t, x = d[:, 1], d[:, col]
    dt = t[1] - t[0]
    sig = x - x.mean()
    sp = np.abs(np.fft.rfft(sig)) * dt
    fr = 2 * np.pi * np.fft.rfftfreq(len(sig), d=dt) * AU_TO_CM
    return fr, sp


def peaks(fr, sp, lo, hi, frac, ref=None):
    m = np.flatnonzero((fr >= lo) & (fr <= hi))
    smax = ref if ref else sp[m].max()
    out = []
    for i in m[1:-1]:
        if sp[i] >= sp[i - 1] and sp[i] >= sp[i + 1] and sp[i] >= frac * smax:
            out.append((fr[i], sp[i] / smax))
    merged = []
    for fq, h in sorted(out, key=lambda x: -x[1]):
        if all(abs(fq - m0) > 3 for m0, _ in merged):
            merged.append((fq, h))
    return sorted(merged)


print("=== a. deep peak lists, lambda=0.3, MLIP 100 ps ===")
for tag in ("polar", "nopolar"):
    f = BASE / f"nep_md/lambda=0.3_{tag}" / "dipole.npz"
    for col, name in [(2, "dressed"), (5, "bare")]:
        fr, sp = spec_of(f, col)
        gmax = sp[(fr >= 20) & (fr <= 3800)].max()
        pk = peaks(fr, sp, 20, 3800, 0.003, ref=gmax)
        print(f"\n{tag} {name} mu_x  (100*peak/max):")
        for fq, h in pk:
            print(f"    {fq:8.1f}  {100*h:6.2f}%")

print("\n=== b. LP/UP satellites vs lambda (dressed mu_x, MLIP 100 ps) ===")
print("tag      lam    mainLP  satLP(off)      mainUP  satUP(off)")
for tag in ("polar", "nopolar"):
    for lam in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        f = BASE / f"nep_md/lambda={lam:g}_{tag}" / "dipole.npz"
        fr, sp = spec_of(f, 2)
        lp = peaks(fr, sp, 800, OMEGA_C - 1, 0.02)
        up = peaks(fr, sp, OMEGA_C + 1, 3800, 0.02)
        def fmt(pk):
            if not pk:
                return "-", "-"
            main = max(pk, key=lambda x: x[1])
            sats = [p for p in pk if abs(p[0] - main[0]) > 3]
            s = ", ".join(f"{p[0]:.0f}({p[0]-main[0]:+.0f},{100*p[1]:.0f}%)"
                          for p in sats) or "-"
            return f"{main[0]:.1f}", s
        mlp, slp = fmt(lp)
        mup, sup = fmt(up)
        print(f"{tag:8s} {lam:4.2f}  {mlp:>7s}  {slp:<22s} {mup:>7s}  {sup}")
