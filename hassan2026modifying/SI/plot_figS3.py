#!/usr/bin/env python3
"""Fig. S3: change in the equilibrium binding energy with coupling strength.

E(λ) - E(λ = 0) at the equilibrium interlayer distance, in meV/atom, for
(a) bilayer hBN in AA, AB, AA' and AB1 stacking and (b) bilayer graphene in AA
and AB. E is measured against the same cell with the layers 10 Å apart, so this
is the cavity-induced change in the binding energy alone. Both the energy and
the distance it is evaluated at come from the cubic fit of the binding-curve
minimum.

Run from inside SI/:  python plot_figS3.py  ->  fig_S3/fig_S3.{png,pdf}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main-script"))
import pmbd_bilayer as pb  # noqa: E402

LAMBDAS = pb.LAMBDAS
LAMBDA = r"\boldsymbol{\lambda_{\alpha}}"

PANELS = (("hbn_aa", "hbn_ab", "hbn_aap", "hbn_ab1"),
          ("graphene_aa", "graphene_ab"))
LEGEND_LABEL = {"AB1": "AB$_1$"}

pb.use_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=pb.FIGSIZE)

results = {}
for ax, names in zip((ax1, ax2), PANELS):
    for name in names:
        system = pb.load(name)
        _, _, e_min = system.fit_series(LAMBDAS)
        shift = (e_min - e_min[0]) * 1000.0 / system.n_atoms
        results[system.label] = shift
        ax.plot(LAMBDAS, shift, "o--", color=pb.STACKING_COLOR[system.stacking],
                markersize=7,
                label=LEGEND_LABEL.get(system.stacking, system.stacking))
    ax.set_xlabel(rf"Coupling Strength ${LAMBDA}$ [a.u.]", fontsize=20)

ax1.set_ylabel(rf"E(${LAMBDA}$) $-$ E(${LAMBDA} = 0$) [meV/atom]", fontsize=20)
ax1.tick_params(labelsize=17)
ax2.tick_params(labelsize=15)

ax1.legend(loc="best", ncol=1, frameon=False, fontsize=17)
ax2.legend(loc="best", ncol=1, frameon=False, fontsize=17)

pb.add_schematic(ax1, "hbn_aap", (0.045, 10.7), 0.17)
pb.add_schematic(ax2, "graphene_ab", (0.06, 1.5), 0.17)

plt.subplots_adjust(hspace=0.25, wspace=0.25, top=0.9, bottom=0.1,
                    left=0.1, right=0.9)

pb.save(fig, "fig_S3", "fig_S3")
plt.close(fig)

print(f"{'lambda':>8}" + "".join(f"{n:>14}" for n in results))
for i, lam in enumerate(LAMBDAS):
    print(f"{lam:8.2f}" + "".join(f"{s[i]:14.3f}" for s in results.values()))
print("(meV/atom, positive = less strongly bound inside the cavity)")
