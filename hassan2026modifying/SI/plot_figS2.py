#!/usr/bin/env python3
"""Fig. S2: equilibrium interlayer distance and layer breathing mode of graphene.

The graphene counterpart of main-text Fig. 2: (a) R0 and (b) the LBM frequency
of bilayer graphene in AA and AB stacking against the coupling strength λ, for
ω = 2 eV and a single out-of-plane mode.

Run from inside SI/:  python plot_figS2.py  ->  fig_S2/fig_S2.{png,pdf}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# The loader, the fit and the binding curves live with the main-text scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main-script"))
import pmbd_bilayer as pb  # noqa: E402

LAMBDAS = pb.LAMBDAS
LAMBDA_LABEL = r"Coupling Strength $\boldsymbol{\lambda_{\alpha}}$ [a.u.]"

fits = {}
for name in ("graphene_aa", "graphene_ab"):
    system = pb.load(name)
    r0, freq, _ = system.fit_series(LAMBDAS)
    fits[name] = (system.stacking, r0, freq)

pb.use_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=pb.FIGSIZE)

for ax, column, ylabel in (
    (ax1, 1, r"Interlayer Distance ($R_0$) [$\mathrm{\AA}$]"),
    (ax2, 2, "Vibrational Frequency [cm$^{-1}$]"),
):
    for fit in fits.values():
        stacking = fit[0]
        ax.plot(LAMBDAS, fit[column], "o--",
                color=pb.STACKING_COLOR[stacking], markersize=7, label=stacking)
    ax.set_xlabel(LAMBDA_LABEL, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

ax1.tick_params(labelsize=17)
ax2.tick_params(labelsize=17)

# Panel (a)'s curves rise towards both corners, so its legend is placed by hand.
ax1.legend(loc=[0.62, 0.15], ncol=1, frameon=False, fontsize=20)
ax2.legend(loc="best", ncol=1, frameon=False, fontsize=20)

ax1.text(0.17, 3.62, "a)", fontsize=20)
ax2.text(0.17, 79.35, "b)", fontsize=20)

pb.add_schematic(ax1, "graphene_interlayer_schematic", (0.11, 3.54), 0.205)
pb.add_schematic(ax2, "graphene_lbm_schematic", (0.09, 76.5), 0.205)

plt.subplots_adjust(hspace=0.25, wspace=0.35, top=0.9, bottom=0.1,
                    left=0.1, right=0.9)

pb.save(fig, "fig_S2", "fig_S2")
plt.close(fig)

print(f"{'lambda':>8}{'AA':>22}{'AB':>22}")
for i, lam in enumerate(LAMBDAS):
    cells = "".join(f"  {r0[i]:7.4f} A {freq[i]:7.2f} cm-1"
                    for _, r0, freq in fits.values())
    print(f"{lam:8.2f}{cells}")
for stacking, _, freq in fits.values():
    print(f"{stacking}: LBM changes by {100 * (freq[-1] / freq[0] - 1):+.1f}% "
          f"from {freq[0]:.1f} to {freq[-1]:.1f} cm-1 over lambda = 0 to 0.2")
