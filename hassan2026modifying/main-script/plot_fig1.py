#!/usr/bin/env python3
"""Fig. 1: interlayer binding curves inside and outside the cavity.

(a) bilayer hBN in AA, AB, AB1 and AA' stacking, (b) bilayer graphene in AA and
AB, each at λ = 0 (black) and λ = 0.14 a.u. (red), ω = 2 eV, mode polarised out
of plane. Energies are E(R) - E(10 Å) in meV/atom.

Run from inside main-script/:  python plot_fig1.py  ->  fig_1/fig_1.{png,pdf}
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pmbd_bilayer as pb

LAM_BARE, LAM_CAVITY = 0.0, 0.14
LAM_COLOR = {LAM_BARE: "black", LAM_CAVITY: "red"}

# Line style per stacking, shared by both panels. The curves are drawn in this
# order, both coupling strengths of one stacking before the next.
HBN = {"hbn_aa": "-.", "hbn_ab": "--", "hbn_aap": "-", "hbn_ab1": ":"}
GRAPHENE = {"graphene_aa": "-.", "graphene_ab": "--"}

# Curve labels placed by hand in data coordinates (meV/atom): the text position,
# its colour, then the arrow's tip and tail.
HBN_LABELS = [
    (3.80,  -6.25, "AA",     "red",   (3.75,  -8.25),  (3.88,  -6.375)),
    (3.55,  -9.00, "AB",     "red",   (3.70, -10.75),  (3.63,  -9.125)),
    (3.27, -12.00, "AA'",    "red",   (3.62, -11.00),  (3.46, -11.75)),
    (3.55,  -4.50, "AB$_1$", "red",   (3.35,  -6.25),  (3.63,  -4.625)),
    (3.55, -16.25, "AA",     "black", (3.70, -18.75),  (3.63, -16.375)),
    (3.15, -19.50, "AA'",    "black", (3.31, -24.00),  (3.25, -19.75)),
    (3.40, -21.25, "AB$_1$", "black", (3.43, -23.375), (3.50, -21.50)),
    (3.70, -24.50, "AB",     "black", (3.40, -24.50),  (3.70, -24.00)),
]
GRAPHENE_LABELS = [
    (3.30, -19.50,  "AA", "black", (3.63, -18.00),   (3.48, -19.00)),
    (3.73, -22.25,  "AB", "black", (3.48, -22.00),   (3.73, -21.875)),
    (3.57, -15.00,  "AA", "red",   (3.62, -16.625),  (3.65, -15.125)),
    (3.87, -21.125, "AB", "red",   (3.44, -20.6875), (3.80, -20.75)),
]


def annotate(ax, entries):
    """Curve labels with an arrow pointing at the curve they name."""
    for x, y, text, color, tip, tail in entries:
        ax.text(x, y, text, color=color, fontsize=20)
        ax.annotate("", xy=tip, xytext=tail,
                    arrowprops=dict(arrowstyle="->, head_width=0.2",
                                    color=color, lw=2))


systems = {name: pb.load(name) for name in (*HBN, *GRAPHENE)}

pb.use_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=pb.FIGSIZE, sharey=True)

for ax, panel in ((ax1, HBN), (ax2, GRAPHENE)):
    for name, style in panel.items():
        for lam in (LAM_BARE, LAM_CAVITY):
            ax.plot(systems[name].r, systems[name].curve_mev_per_atom(lam), style,
                    color=LAM_COLOR[lam], linewidth=2)
    r = systems[name].r  # one R grid for every system
    ax.plot(r, 0 * r, "--", color="gray")
    ax.set_xlim([3.0, 5.0])
    ax.tick_params(labelsize=17)

# Clipped just below the deepest well, the hBN AB minimum; panel (b) shares it.
ax1.set_ylim([-25.75, 0.25])

annotate(ax1, HBN_LABELS)
annotate(ax2, GRAPHENE_LABELS)

ax1.set_ylabel("Energy E(R) [meV/atom]", fontsize=20)
for ax in (ax1, ax2):
    ax.set_xlabel(r"Interlayer Distance R [$\mathrm{\AA}$]", fontsize=20)

handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=10,
           label=r"$\boldsymbol{\lambda_{\alpha}} = 0$"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10,
           label=rf"$\boldsymbol{{\lambda_\alpha}} = {LAM_CAVITY}$ a.u."),
]
ax1.legend(handles=handles, loc="best", ncol=1, frameon=False, fontsize=18)
ax2.legend(handles=handles, loc="best", ncol=1, frameon=False, fontsize=18)

# Structure renders, each with the stacking it shows named beside it, plus the
# element key over the hBN AA render.
for name, xy, label, label_xy in (
    ("hbn_aa",  (4.25, -22.625), "AA",     (4.155, -25.625)),
    ("hbn_ab",  (4.78, -12.50),  "AB",     (4.70,  -16.25)),
    ("hbn_ab1", (4.52, -18.75),  "AB$_1$", (4.73,  -19.25)),
    ("hbn_aap", (4.78, -22.625), "AA'",    (4.69,  -25.625)),
):
    pb.add_schematic(ax1, name, xy, 0.08)
    ax1.text(*label_xy, label, fontsize=20, weight="bold")

ax1.text(4.199, -23.625, "N", color=(176 / 255, 185 / 255, 230 / 255),
         fontsize=20, weight="bold")
ax1.text(4.2, -22.5, "B", color=(31 / 255, 162 / 255, 15 / 255),
         fontsize=21.5, weight="bold")

ax1.text(3.6, -2.5, "a)", fontsize=20)
ax2.text(3.6, -2.5, "b)", fontsize=20)

for name, xy, label, label_xy in (
    ("graphene_aa", (3.7, -10.0), "AA", (4.00, -10.50)),
    ("graphene_ab", (4.5, -17.5), "AB", (4.77, -17.75)),
):
    pb.add_schematic(ax2, name, xy, 0.1)
    ax2.text(*label_xy, label, fontsize=20, weight="bold")
ax2.text(4.45, -13.5, "C", color=(0, 0, 0), fontsize=20, weight="bold")

plt.subplots_adjust(hspace=0.25, wspace=0.15, top=0.9, bottom=0.1,
                    left=0.1, right=0.9)

pb.save(fig, "fig_1", "fig_1")
plt.close(fig)

for system in systems.values():
    r0_0, _, e_0 = system.fit(LAM_BARE)
    r0, _, e = system.fit(LAM_CAVITY)
    print(f"{system.label:>13}  R0 {r0_0:.3f} -> {r0:.3f} A   "
          f"E_min {e_0 * 1000 / system.n_atoms:7.2f} -> "
          f"{e * 1000 / system.n_atoms:7.2f} meV/atom")
