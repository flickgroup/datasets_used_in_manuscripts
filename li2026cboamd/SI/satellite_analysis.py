#!/usr/bin/env python3
"""Analyze the satellite peaks near LP/UP at lambda=0.3 (PBE0 campaign).

Questions:
 1. chi-included LP (~1197): AIMD shows a satellite ~1270. Real or 10 ps artifact?
 2. chi-neglected UP: satellite structure. Same question.

Tests:
 a. list local maxima near LP and UP for AIMD (10 ps) and MLIP (100 ps),
 b. truncate the MLIP trajectory to 10 ps -> if satellites appear at the same
    places as AIMD, they are finite-time/nonstationarity artifacts,
 c. print satellite offsets vs candidate physical frequencies (bend ~660,
    2*bend Fermi region ~1330, LP/UP combinations).
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

RUNS = {
    "AIMD polar":    BASE / "lambda=0.3_emode_polar",
    "AIMD nopolar":  BASE / "lambda=0.3_emode_nopolar",
    "MLIP polar":    BASE / "nep_md/lambda=0.3_polar",
    "MLIP nopolar":  BASE / "nep_md/lambda=0.3_nopolar",
}


def spectrum(f, nmax=None):
    d = co2io.load_trace_columns(f)
    if nmax:
        d = d[:nmax]
    t, mux = d[:, 1], d[:, 2]
    dt = t[1] - t[0]
    sig = mux - mux.mean()
    spec = np.abs(np.fft.rfft(sig)) * dt
    freq = 2 * np.pi * np.fft.rfftfreq(len(sig), d=dt) * AU_TO_CM
    return freq, spec, len(sig)


def peaks_in(freq, spec, lo, hi, frac=0.05):
    m = np.flatnonzero((freq >= lo) & (freq <= hi))
    smax = spec[m].max()
    out = []
    for i in m[1:-1]:
        if spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1] and spec[i] >= frac * smax:
            out.append((freq[i], spec[i] / smax))
    # merge peaks closer than 4 cm-1, keep the taller
    merged = []
    for fq, h in sorted(out, key=lambda x: -x[1]):
        if all(abs(fq - m0) > 4 for m0, _ in merged):
            merged.append((fq, h))
    return sorted(merged)


def report(label, freq, spec, nsteps):
    ps = nsteps * 0.5 / 1000
    print(f"\n=== {label}  ({ps:.0f} ps, df={freq[1]-freq[0]:.2f} cm-1) ===")
    for name, lo, hi in [("LP region", 800, OMEGA_C - 1),
                         ("UP region", OMEGA_C + 1, 3800)]:
        pk = peaks_in(freq, spec, lo, hi)
        if not pk:
            print(f"  {name}: no peaks above threshold")
            continue
        main = max(pk, key=lambda x: x[1])[0]
        print(f"  {name}: " + ", ".join(
            f"{fq:.1f} ({100*h:.0f}%{'':s}{', off %+.1f' % (fq-main) if abs(fq-main)>2 else ''})"
            for fq, h in pk))


for label, d in RUNS.items():
    freq, spec, n = spectrum(d / "dipole.npz")
    report(label, freq, spec, n)

# windowing test: MLIP truncated to 10 ps (20000 steps)
for tag in ("polar", "nopolar"):
    f = BASE / f"nep_md/lambda=0.3_{tag}" / "dipole.npz"
    freq, spec, n = spectrum(f, nmax=20000)
    report(f"MLIP {tag} TRUNCATED to 10 ps", freq, spec, n)
    # also a later 10 ps window for nonstationarity check
    d = co2io.load_trace_columns(f)[100000:120000]
    t, mux = d[:, 1], d[:, 2]
    dt = t[1] - t[0]
    sig = mux - mux.mean()
    sp = np.abs(np.fft.rfft(sig)) * dt
    fr = 2 * np.pi * np.fft.rfftfreq(len(sig), d=dt) * AU_TO_CM
    report(f"MLIP {tag} window 50-60 ps", fr, sp, len(sig))

# full-range peak list for MLIP 100 ps (is there ANY reproducible sideband?)
print("\n=== full-range peaks, MLIP 100 ps, >2% of global max, 400-3600 ===")
for tag in ("polar", "nopolar"):
    f = BASE / f"nep_md/lambda=0.3_{tag}" / "dipole.npz"
    freq, spec, n = spectrum(f)
    pk = peaks_in(freq, spec, 400, 3600, frac=0.02)
    print(f"  {tag}: " + ", ".join(f"{fq:.1f} ({100*h:.1f}%)" for fq, h in pk))
