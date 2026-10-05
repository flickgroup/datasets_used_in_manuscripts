#!/usr/bin/env python3
"""Fig. 2: equilibrium interlayer distance and layer breathing mode of hBN.

(a) R0 and (b) the LBM frequency of bilayer hBN in AA, AB, AB1 and AA' stacking,
as a function of the coupling strength λ, for ω = 2 eV and a single out-of-plane
mode. Both come from the cubic fit of the binding-curve minimum; see
`pmbd_bilayer.fit_minimum`.

Run from inside main-script/:  python plot_fig2.py  ->  fig_2/fig_2.{png,pdf}
"""

import matplotlib.pyplot as plt

import pmbd_bilayer as pb

LAMBDAS = pb.LAMBDAS

# Each panel lists the stackings in its own order, so that its legend runs top
# to bottom roughly as its curves do.
ORDER_R0 = ("hbn_aa", "hbn_ab1", "hbn_aap", "hbn_ab")
ORDER_FREQ = ("hbn_aap", "hbn_ab", "hbn_ab1", "hbn_aa")
LEGEND_LABEL = {"AB1": "AB$_1$"}

fits = {}
for name in ORDER_R0:
    system = pb.load(name)
    r0, freq, _ = system.fit_series(LAMBDAS)
    fits[name] = (system.stacking, r0, freq)

pb.use_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=pb.FIGSIZE)

for name in ORDER_R0:
    stacking, r0, _ = fits[name]
    ax1.plot(LAMBDAS, r0, "o--", color=pb.STACKING_COLOR[stacking], markersize=7,
             label=LEGEND_LABEL.get(stacking, stacking))
ax1.set_xlabel(r"Coupling Strength $\mathrm{\lambda_{\alpha}}$ [a.u.]", fontsize=20)
ax1.set_ylabel(r"Interlayer Distance ($R_0$) [$\mathrm{\AA}$]", fontsize=20)
ax1.tick_params(labelsize=17)

for name in ORDER_FREQ:
    stacking, _, freq = fits[name]
    ax2.plot(LAMBDAS, freq, "o--", color=pb.STACKING_COLOR[stacking], markersize=7,
             label=LEGEND_LABEL.get(stacking, stacking))
ax2.set_xlabel(r"Coupling Strength $\boldsymbol{\lambda_{\alpha}}$ [a.u.]", fontsize=20)
ax2.set_ylabel("Vibrational Frequency [cm$^{-1}$]", fontsize=20)
ax2.tick_params(labelsize=15)

ax1.legend(loc=[0.65, 0.035], ncol=1, frameon=False, fontsize=18)
ax2.legend(loc="best", ncol=1, frameon=False, fontsize=18)

ax1.text(0.15, 4.245, "a)", fontsize=20)
ax2.text(0.1, 74, "b)", fontsize=20)

pb.add_schematic(ax1, "hbn_interlayer_schematic", (0.07, 4.1), 0.175)
pb.add_schematic(ax2, "hbn_lbm_schematic", (0.06, 43), 0.2)

plt.subplots_adjust(hspace=0.25, wspace=0.25, top=0.9, bottom=0.1,
                    left=0.1, right=0.9)

pb.save(fig, "fig_2", "fig_2")
plt.close(fig)

print(f"{'lambda':>8}" + "".join(f"{fits[n][0]:>22}" for n in ORDER_R0))
for i, lam in enumerate(LAMBDAS):
    cells = "".join(f"  {fits[n][1][i]:7.4f} A {fits[n][2][i]:7.2f} cm-1"
                    for n in ORDER_R0)
    print(f"{lam:8.2f}{cells}")
