"""Measure the CO2 bend frequency nu2 from the lambda=0 NEP-MD run.

Primary route: |FFT| of the perpendicular dipole components (mu_y, mu_z)
from dipole.dat, peak in the 400-900 cm-1 window. Fallback diagnostics:
O-C-O bend-angle spectrum from position.dat.
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


AU_TO_CM = 219474.63
BASE = str(DATA / "single_molecule" / "lambda=0_qmode")

d = co2io.load_trace_columns(BASE + "/dipole.npz")
t = d[:, 1]
dt = t[1] - t[0]

for col, name in [(3, "mu_y"), (4, "mu_z")]:
    sig = d[:, col] - d[:, col].mean()
    spec = np.abs(np.fft.rfft(sig)) * dt
    freq = 2 * np.pi * np.fft.rfftfreq(len(sig), d=dt) * AU_TO_CM
    m = (freq >= 400) & (freq <= 900)
    fm, sm = freq[m], spec[m]
    i = int(np.argmax(sm))
    med = float(np.median(sm))
    print(f"{name}: peak {fm[i]:.1f} cm-1  amp {sm[i]:.4g}  "
          f"median {med:.4g}  contrast {sm[i]/med:.1f}")
    # top 5 local maxima for context
    loc = [j for j in range(1, len(sm) - 1)
           if sm[j] >= sm[j - 1] and sm[j] >= sm[j + 1]]
    loc = sorted(loc, key=lambda j: sm[j], reverse=True)[:5]
    print("   top local maxima:",
          ", ".join(f"{fm[j]:.1f} ({sm[j]:.3g})" for j in loc))

# fallback: bend-angle spectrum from position.dat
try:
    p = co2io.load_trace_columns(BASE + "/position.npz")
    tp = p[:, 1]
    dtp = tp[1] - tp[0]
    O1 = p[:, 2:5]
    C = p[:, 5:8]
    O2 = p[:, 8:11]
    v1 = O1 - C
    v2 = O2 - C
    cosang = np.sum(v1 * v2, axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    sig = ang - ang.mean()
    spec = np.abs(np.fft.rfft(sig)) * dtp
    freq = 2 * np.pi * np.fft.rfftfreq(len(sig), d=dtp) * AU_TO_CM
    m = (freq >= 400) & (freq <= 900)
    fm, sm = freq[m], spec[m]
    i = int(np.argmax(sm))
    med = float(np.median(sm))
    print(f"bend angle: peak {fm[i]:.1f} cm-1  contrast {sm[i]/med:.1f}  "
          f"mean angle {ang.mean():.2f} deg")
except Exception as exc:
    print("bend-angle fallback failed:", exc)
