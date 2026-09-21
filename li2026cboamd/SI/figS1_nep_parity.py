#!/usr/bin/env python3
"""Fig. S1 (single_molecule_nep_parity): parity plots of the retrained
PBE0/aug-cc-pVDZ NEP models, reproducing the layout of the LDA SI figure:

  a Energy [meV/atom]      b Forces [meV/A]
  c Dipole [a.u.]          d chi_xx [a.u.]
  e chi_yy and chi_zz      f Off-diagonal chi [10^n a.u.]

Data: the predicted-vs-reference files GPUMD's nep writes, shipped in
models/single_molecule_nep/{ener_model_nep2,dipole_model_nep,
polarizability_model_nep}/ as the original ASCII (they are small).
Each file holds N prediction columns followed by N reference columns
(N=1 energy, 3 forces/dipole, 6 polarizability: xx yy zz xy yz zx).
Dipole and polarizability outputs are per atom and are multiplied by 3
to molecular totals, as in the SI (Table S1 convention).

Output: figS1_nep_parity.pdf/.png next to this script + RMSE table
(molecular units for dipole/pol, meV/atom and meV/A for energy/forces).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN_NEP = REPO / "models" / "single_molecule_nep"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

C_TRAIN = "#7fa8cf"   # soft blue, as in the LDA SI figure
C_TEST = "#d16f6b"    # soft red


def load(model_dir, stem, split):
    """Return (pred, ref) arrays of shape (nconf, N)."""
    d = np.atleast_2d(np.loadtxt(TRAIN_NEP / model_dir / f"{stem}_{split}.out"))
    half = d.shape[1] // 2
    return d[:, :half], d[:, half:]


def panel(ax, series, title, xlabel, ylabel, legend=False):
    """series: {'train': (ref, pred), 'test': (ref, pred)} (flattened)."""
    for split, color in (("train", C_TRAIN), ("test", C_TEST)):
        ref, pred = series[split]
        ax.plot(ref, pred, "o", ms=3.5, mew=0, color=color, alpha=0.7,
                label=split, zorder=2 if split == "train" else 3)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--", color="k", lw=1.2, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="center right", frameon=False, handletextpad=0.1)


def rmse(series):
    return {s: float(np.sqrt(np.mean((series[s][1] - series[s][0]) ** 2)))
            for s in ("train", "test")}


def collect(model_dir, stem, cols, scale):
    out = {}
    for split in ("train", "test"):
        pred, ref = load(model_dir, stem, split)
        out[split] = (ref[:, cols].ravel() * scale,
                      pred[:, cols].ravel() * scale)
    return out


# energy: eV/atom -> meV/atom; forces: eV/A -> meV/A
S_ENER = collect("ener_model_nep2", "energy", [0], 1e3)
S_FORCE = collect("ener_model_nep2", "force", [0, 1, 2], 1e3)
# dipole/pol: per-atom -> molecular (x3)
S_DIP = collect("dipole_model_nep", "dipole", [0, 1, 2], 3.0)
S_XX = collect("polarizability_model_nep", "polarizability", [0], 3.0)
S_YZ = collect("polarizability_model_nep", "polarizability", [1, 2], 3.0)
S_OFF = collect("polarizability_model_nep", "polarizability", [3, 4, 5], 3.0)

# scale off-diagonal chi to O(1), power-of-ten label as in the SI
off_max = max(np.abs(v).max() for pair in S_OFF.values() for v in pair)
exp = int(np.floor(np.log10(off_max))) if off_max > 0 else 0
S_OFF = {s: (r / 10.0 ** exp, p / 10.0 ** exp) for s, (r, p) in S_OFF.items()}
off_unit = rf"[$10^{{{exp}}}$ a.u.]"

fig, axes = plt.subplots(3, 2, figsize=(8.0, 11.5))
specs = [
    (S_ENER, "Energy", "energy", "[meV/atom]", True),
    (S_FORCE, "Forces", "force", r"[meV $\mathrm{\AA}^{-1}$]", False),
    (S_DIP, "Dipole", "dipole", "[a.u.]", False),
    (S_XX, r"$\chi_{xx}$", r"$\chi_{xx}$", "[a.u.]", False),
    (S_YZ, r"$\chi_{yy}$ and $\chi_{zz}$", r"$\chi_{yy},\chi_{zz}$",
     "[a.u.]", False),
    (S_OFF, r"Off-diagonal $\chi$", r"off-diagonal $\chi$", off_unit, False),
]
for ax, letter, (series, title, qty, unit, leg) in zip(
        axes.ravel(), "abcdef", specs):
    panel(ax, series, title, f"DFT {qty} {unit}", f"NEP {qty} {unit}",
          legend=leg)
    ax.text(-0.22, 1.04, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"figS1_nep_parity.{ext}", dpi=300,
                bbox_inches="tight")

print(f"{'target':<28}{'unit':<12}{'RMSE train':>14}{'RMSE test':>14}")
rows = [("Energy", "meV/atom", S_ENER), ("Force", "meV/A", S_FORCE),
        ("Dipole", "a.u.", S_DIP), ("chi_xx", "a.u.", S_XX),
        ("chi_yy,chi_zz", "a.u.", S_YZ),
        (f"off-diag chi", f"1e{exp} a.u.", S_OFF)]
for name, unit, series in rows:
    r = rmse(series)
    print(f"{name:<28}{unit:<12}{r['train']:>14.3e}{r['test']:>14.3e}")

# Combined polarizability RMSE over all six tensor components (xx, yy, zz,
# xy, yz, zx), molecular units: this is the single "Polarizability" number
# quoted in Table S1 of the SI.
S_POL_ALL = collect("polarizability_model_nep", "polarizability",
                    [0, 1, 2, 3, 4, 5], 3.0)
r = rmse(S_POL_ALL)
print(f"{'Polarizability (Table S1)':<28}{'a.u.':<12}"
      f"{r['train']:>14.3e}{r['test']:>14.3e}")
print("wrote figS1_nep_parity.pdf/.png")
