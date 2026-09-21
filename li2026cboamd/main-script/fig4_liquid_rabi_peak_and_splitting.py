"""Manuscript Fig. 4 (liquid Rabi peak positions & splitting), dense-lambda version.

Auto-discovers every cavity case in data/liquid/spectra/offresonant
(lambda = 0.01 ... 0.10, chi included = polar / chi neglected = nopolar) and
extracts peak positions from the Hamming-smoothed averaged dressed spectra
(spectrum_direct_dressed_x_avg40_hamming11.npz, column 4).

Branches are TRACKED by continuity in lambda rather than assigned to fixed
frequency windows, because with chi included the lower stretch polariton
descends continuously through the 1200 cm-1 region into the bending region,
and the upper polariton persists near ~2496 cm-1 with strongly suppressed
intensity (1.5% -> 0.2% of the spectrum maximum for lambda = 0.04 -> 0.09):

  upper branch:  strongest local max in [out_of_cavity, 4400], floor 0.1%
  lower branch:  local max nearest below/at the previous lambda's position
                 (search window [prev-500, prev+100]), floor 0.1%
  bend peak:     strongest local max in [450, 760] at least 40 cm-1 below the
                 lower branch (the 674.6 cm-1 bare bend, red-shifting under
                 strong chi-included coupling)
"""
import sys
from pathlib import Path
import re

import matplotlib.pyplot as plt
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


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent
SPEC = DATA / "liquid" / "spectra" / "offresonant"
OUTDIR = ROOT

colors = {"inc": "#cc163a", "neg": "#1177b0", "res": "#1b7837", "ref": "0.45"}

out_of_cavity = 2407.4
low_reference = 674.6
FLOOR_REL = 0.001


def local_maxima(freq, inten, lo, hi, floor):
    idx = np.flatnonzero((freq >= lo) & (freq <= hi))
    return [i for i in idx[1:-1]
            if inten[i] >= inten[i - 1] and inten[i] >= inten[i + 1]
            and inten[i] >= floor]


def band_centroid(freq, inten, lo, hi, floor, frac=0.3):
    """Band centroid of the strongest band in [lo, hi].

    At strong chi-neglected coupling the upper polariton is a 300-400 cm-1 wide
    multiplet of near-degenerate sub-maxima; picking the single strongest bin
    hops between sub-clusters from one lambda to the next (e.g. 3633 -> 3940
    for lambda 0.09 -> 0.10 while the band centroids step smoothly 3706 ->
    3856). The centroid of the contiguous region above 30% of the band
    maximum is stable (50% stays inside one sub-cluster; 30% spans the band;
    for a sharp peak both reduce to the peak position), and reduces to the peak position for a sharp peak.
    """
    idx = np.flatnonzero((freq >= lo) & (freq <= hi))
    if idx.size < 3 or inten[idx].max() < floor:
        return np.nan
    imax = idx[np.argmax(inten[idx])]

    def region(f):
        thr = f * inten[imax]
        a = b = imax
        while a - 1 >= idx[0] and inten[a - 1] >= thr:
            a -= 1
        while b + 1 <= idx[-1] and inten[b + 1] >= thr:
            b += 1
        return a, b

    a, b = region(0.5)
    if freq[b] - freq[a] <= 150.0:
        # sharp peak: report its position directly. A centroid here is fragile:
        # e.g. the weak chi-included upper peak (pinned at 2497.5 cm-1 for
        # lambda >= 0.03) has a lambda-independent residual neighbor at
        # ~2441 cm-1 that crosses the 50% inclusion threshold as the main peak
        # fades, dragging a centroid left although the peak does not move.
        return float(freq[imax])
    # genuinely broad band (multiplet): centroid over the 30% region so all
    # sub-clusters contribute (the strongest bin alone hops between them).
    a, b = region(frac)
    w = inten[a:b + 1]
    return float(np.sum(freq[a:b + 1] * w) / np.sum(w))


def load(path):
    d = co2io.load_spectrum_usecols(path, (0, 4))
    return d[:, 0], d[:, 1]


def collect(variant, data=None):
    cases = []
    if data is None:
        data = SPEC
    for path in sorted(data.glob(f"*_cavity_ir_40traj_newmodels_{variant}_lam*/"
                                 "spectrum_direct_dressed_x_avg40_hamming11.npz")):
        m = re.search(rf"{variant}_lam(\d+)", path.parent.name)
        cases.append((int(m.group(1)) / 100.0, path))
    cases.sort()

    rows = []
    prev_lower = out_of_cavity
    for lam, path in cases:
        freq, inten = load(path)
        floor = FLOOR_REL * inten[(freq >= 300) & (freq <= 4400)].max()
        # upper branch: half-maximum band centroid (see band_centroid docstring)
        upper = band_centroid(freq, inten, out_of_cavity, 4400, floor)
        # lower branch: strongest local max in a window tracking the previous
        # position, clamped above the bend zone (the descended chi-included
        # branch bottoms out at ~785 cm-1, the bend at <760 cm-1) and below
        # the upper-branch zone
        lo_win, hi_win = max(prev_lower - 500.0, 765.0), min(prev_lower + 100.0, out_of_cavity)
        lowb = local_maxima(freq, inten, lo_win, hi_win, floor)
        if lowb:
            lower = float(freq[max(lowb, key=lambda i: inten[i])])
            prev_lower = lower
        else:
            lower = np.nan
        # bend peak (disjoint zone below the branch window)
        bd = local_maxima(freq, inten, 450, 760, floor)
        bend = float(freq[max(bd, key=lambda i: inten[i])]) if bd else np.nan
        rows.append((lam, bend, lower, upper))
    return rows


SPEC_RES = DATA / "liquid" / "spectra" / "resonant"

inc = collect("polar")
neg = collect("nopolar")
res = collect("polar", SPEC_RES)   # omega_c retuned per lambda (omega_c/n_eff)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.609), gridspec_kw={"wspace": 0.34})

ax = axes[0]
ax.axhline(out_of_cavity, color=colors["ref"], lw=0.9, ls="--", zorder=0)
ax.axhline(low_reference, color=colors["ref"], lw=0.9, ls="--", zorder=0)

for rows, key in [(inc, "inc"), (neg, "neg"), (res, "res")]:
    lam0 = [0.0] + [r[0] for r in rows]
    lower = [out_of_cavity] + [r[2] for r in rows]
    upper = [out_of_cavity] + [r[3] for r in rows]
    bend = [low_reference] + [r[1] for r in rows]
    ax.plot(lam0, upper, color=colors[key], marker="o", ms=4.2, lw=1.2, ls="--")
    ax.plot(lam0, lower, color=colors[key], marker="o", ms=4.2, lw=1.2, ls="--")
    ax.plot(lam0, bend, color=colors[key], marker="o", ms=4.0, lw=1.0, ls="--", alpha=0.8)

ax.plot([], [], color=colors["inc"], marker="o", ms=4.2, lw=1.2, ls="--", label=r"$\chi$ included")
ax.plot([], [], color=colors["res"], marker="o", ms=4.2, lw=1.2, ls="--", label=r"$\chi$ included, $\omega_c/n_{\mathrm{eff}}$")
ax.plot([], [], color=colors["neg"], marker="o", ms=4.2, lw=1.2, ls="--", label=r"$\chi$ neglected")
ax.set_xlim(-0.004, 0.106)
ax.set_ylim(420, 4100)
ax.set_xlabel(r"Coupling strength $\lambda$")
ax.set_ylabel(r"Frequency [cm$^{-1}$]")
ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.tick_params(direction="in", top=True, right=True)
ax.legend(frameon=False, loc="upper left", handlelength=1.3, fontsize=9,
          handletextpad=0.5, labelspacing=0.3)
# "a" to the left of the top y-tick label (4000)
ax.text(-0.26, 0.973, "a", transform=ax.transAxes, fontsize=14, fontfamily="cmb10", va="center")

ax = axes[1]
for rows, key, lab in [(inc, "inc", r"$\chi$ included"), (neg, "neg", r"$\chi$ neglected"),
                       (res, "res", r"$\chi$ included, $\omega_c/n_{\mathrm{eff}}$")]:
    lam0 = [0.0] + [r[0] for r in rows]
    split = [0.0] + [r[3] - r[2] for r in rows]
    ax.plot(lam0, split, "o--", color=colors[key], ms=4.2, lw=1.1, label=lab)
ax.set_xlim(-0.004, 0.106)
ax.set_ylim(-60, 2500)
ax.set_xlabel(r"Coupling strength $\lambda$")
ax.set_ylabel(r"Rabi Splitting [cm$^{-1}$]")
ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.tick_params(direction="in", top=True, right=True)
# "b" to the left of the top y-tick label (2500)
ax.text(-0.26, 1.0, "b", transform=ax.transAxes, fontsize=14, fontfamily="cmb10", va="center")

fig.savefig(OUTDIR / "fig4_liquid_rabi_peak_and_splitting.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "fig4_liquid_rabi_peak_and_splitting.pdf", bbox_inches="tight")

print("chi      lambda   bend    lower    upper   splitting")
for rows, tag in [(inc, "included"), (neg, "neglected"), (res, "resonant")]:
    for lam, bend, lower, upper in rows:
        print(f"{tag:9s} {lam:5.2f} {bend:7.1f} {lower:8.1f} {upper:8.1f} {upper-lower:9.1f}")
