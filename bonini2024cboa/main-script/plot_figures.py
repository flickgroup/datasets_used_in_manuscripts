#!/usr/bin/env python3
"""
Render Figs. 2-6 and Table I from the .dat files produced by reproduce.py and
make_table1.py.

The layout is not approximated. Every geometric constant below was read out of
the content streams of the manuscript's own figure PDFs, so the axes, colorbar,
labels, ticks and annotations land on the published coordinates and the output
page is the same size as the published one. See LAYOUT_SOURCE.

Spectrum panels (Figs. 3-6) follow the paper's encoding:
  * x = lambda, y = hbar*omega (meV)
  * band COLOR = photon character, inferno sliced at [0.15, 0.85]
  * band WIDTH = IR intensity, strongest line in a panel MAXWIDTH points across
  * a gray dashed line along every branch, which is what makes the zero-IR
    branches visible, since those get no band width at all
Layout: 2x2 grid, columns "chi neglected" / "chi included",
rows "multiple cavity modes" / "single cavity mode", shared colorbar.

Every figure is written as BOTH .png and .pdf.
"""
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "data"))
FIGS = os.path.abspath(os.path.join(HERE, "figures"))
os.makedirs(FIGS, exist_ok=True)

LAYOUT_SOURCE = (
    "publications/20240924_bonini_resubmission/main/{octopusCO2_compare,"
    "CO2multi,FeCOmulti,BNmulti,HfS2multi}*.pdf, matplotlib 3.5.1"
)

# ---------------------------------------------------------------------------
# Everything in this block is in PostScript points, measured from the
# manuscript PDFs. Do not round them: they are what makes the output overlay
# the published figures.
# ---------------------------------------------------------------------------
FONT_SIZE = 12.0            # every label except the panel letters and the title

# --- the 2x2 spectrum figures (Figs. 3-6) ---
FIG_H = 356.45              # page height, identical for all four
PANEL_W = 162.0
PANEL_H = 123.5454545455
COL_DX = 194.4              # right column's left edge, from the left column's
ROW_Y = (189.7045454545, 41.45)          # top row, bottom row
CBAR_DX = 365.04            # colorbar left edge, from the left column's
CBAR_W, CBAR_Y, CBAR_H = 12.96, 50.45, 252.0
# Panel letter offset from the axes top-left corner, bold, baseline anchored.
# HfS2 sits tighter to its frame than the other three in the published figures,
# so the offset is per system rather than shared.
LETTER_SIZE = 18.0
TITLE_PAD = 5.0             # column title baseline above the axes top
ROWLAB_X = 16.325           # rotated row label, centred on its row
SUPTITLE_DX = 161.75        # title centre, from the axes left edge
SUPTITLE_Y = 338.5625
MAXWIDTH = 15.0             # band width at the panel's strongest IR line
DASH_GRAY = "0.5"           # branch overlay colour
DASH_LW = 1.2               # at this width matplotlib's "--" is [4.44 1.92],
                            # exactly the published dash pattern

# --- Fig. 2 ---
F2_FIGW, F2_FIGH = 395.121875, 320.45
F2_AX = (53.121875, 41.45, 334.8, 271.8)
F2_XLIM = (-0.015, 0.315)                # matplotlib's 5% margin on 0..0.30
F2_YLIM = (150.0, 324.968)
F2_XSTEP, F2_YSTEP = 0.05, 20.0
F2_LEGEND_ANCHOR = (0.988817, 0.842937)  # legend upper-right, in axes fraction
F2_LW = 1.5
F2_MS = 6.0                              # circle markers of radius 3 pt
# Reference points are red rather than the main text's tab:orange, following
# the supplement's version of this comparison (figs/compare_explicit.pdf).
F2_POINT_COLOR = "red"
F2_LINE_COLOR = "tab:blue"

# The "IR abs." pointer of Fig. 3 panel D, in data coordinates, converted from
# the published arrow at x = 404.546875 pt spanning y = 99.837578..114.087089.
IR_ANNOT = {"CO2": dict(panel="D", lam=0.130, y0=229.603, y1=273.417,
                        text_lam=0.105, text_y=190.02)}
# Enlarge that arrow about its midpoint, and its heads with it. A deliberate
# departure from the published size, which is small enough to be easy to miss.
IR_ARROW_SCALE = 1.6
IR_ARROW_HEAD = 15.0        # heads grow less than the shaft, so the shaft still
                            # reads as a shaft rather than two heads meeting

# The manuscript panels came from qedmft.plotting's
# plot_f_vs_lambdas_colorchar_widthir, which slices inferno at [0.15, 0.85].
base_cmap = mpl.colormaps["inferno"]
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "c_partial", base_cmap(np.linspace(0.15, 0.85, base_cmap.N)))

# system -> paper figure, title, and the measured axis setup
SYS = {
    "CO2":   dict(fig="Fig3", title=r"CO$_2$",     ylim=(50, 430), ystep=100,
                  figw=485.2125,   axleft=69.746875, letter=(-16.2, 0.0)),
    "FeCO5": dict(fig="Fig4", title=r"Fe(CO)$_5$", ylim=(125, 280), ystep=50,
                  figw=485.2125,   axleft=69.746875, letter=(-16.2, 0.0)),
    "hBN":   dict(fig="Fig5", title=r"h-BN",       ylim=(50, 230), ystep=50,
                  figw=485.2125,   axleft=69.746875, letter=(-16.2, 0.0)),
    "HfS2":  dict(fig="Fig6", title=r"HfS$_2$",    ylim=(0, 50),   ystep=10,
                  figw=477.571875, axleft=62.10625,  letter=(-3.24, 3.71)),
}
XLIM = (0.0, 0.15)
XSTEP = 0.05
COL_TITLE = {0: r"$\chi$ neglected", 1: r"$\chi$ included"}
ROW_LABEL = {0: "multiple cavity modes", 1: "single cavity mode"}
POS = {"A": (0, 0), "B": (0, 1), "C": (1, 0), "D": (1, 1)}   # panel -> row, col


def _panel_file(name, panel):
    return os.path.join(DATA, f"{name}_{SYS[name]['fig']}_panel{panel}.dat")


def _load_panel(name, panel):
    """Return (lambda, freqs, chars, irns) from the combined panel file
    (layout: lambda | freqs(N) | photon_char(N) | IR(N))."""
    d = np.loadtxt(_panel_file(name, panel))
    lam = d[:, 0]
    n = (d.shape[1] - 1) // 3
    return lam, d[:, 1:1 + n], d[:, 1 + n:1 + 2 * n], d[:, 1 + 2 * n:1 + 3 * n]


def draw_panel(ax, lam, freqs, chars, irns, maxwidth=MAXWIDTH):
    """One spectrum panel: coloured bands plus the dashed branch overlay."""
    norm = mcolors.Normalize(0.0, 1.0)
    ir = np.abs(irns)
    irscale = ir.max() if ir.max() > 0 else 1.0
    for j in range(freqs.shape[1]):
        ws = maxwidth * (ir[:, j] / irscale)
        pts = np.array([lam, freqs[:, j]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=CMAP, norm=norm, zorder=3)
        lc.set_array(chars[:-1, j])
        lc.set_linewidth(ws[:-1])
        ax.add_collection(lc)
    for j in range(freqs.shape[1]):
        ax.plot(lam, freqs[:, j], "--", color=DASH_GRAY, lw=DASH_LW, zorder=4)


def plot_system(name):
    cfg = SYS[name]
    W, H = cfg["figw"], FIG_H
    axleft = cfg["axleft"]

    def rect(x, y, w, h):
        return [x / W, y / H, w / W, h / H]

    with plt.rc_context({"font.size": FONT_SIZE}):
        fig = plt.figure(figsize=(W / 72.0, H / 72.0))
        axs = {}
        for panel, (r, c) in POS.items():
            ax = fig.add_axes(rect(axleft + (COL_DX if c else 0.0),
                                   ROW_Y[r], PANEL_W, PANEL_H))
            axs[panel] = ax
            lam, freqs, chars, irns = _load_panel(name, panel)
            draw_panel(ax, lam, freqs, chars, irns)
            ax.set_xlim(*XLIM)
            ax.set_ylim(*cfg["ylim"])
            ax.xaxis.set_major_locator(mticker.MultipleLocator(XSTEP))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(cfg["ystep"]))
            if r == 0:
                ax.set_xticklabels([])                 # shared with the row below
                ax.set_title(COL_TITLE[c], fontsize=FONT_SIZE, pad=TITLE_PAD)
            else:
                ax.set_xlabel(r"$\lambda$")
            if c == 0:
                ax.set_ylabel(r"$\hbar\omega$ (meV)")
            else:
                ax.set_yticklabels([])                 # shared with the column left
            ldx, ldy = cfg["letter"]
            fig.text((axleft + (COL_DX if c else 0.0) + ldx) / W,
                     (ROW_Y[r] + PANEL_H + ldy) / H, panel,
                     fontsize=LETTER_SIZE, fontweight="bold",
                     ha="left", va="baseline")

        for r in (0, 1):
            fig.text(ROWLAB_X / W, (ROW_Y[r] + PANEL_H / 2.0) / H, ROW_LABEL[r],
                     rotation=90, rotation_mode="anchor", ha="center",
                     va="baseline", fontsize=FONT_SIZE)
        fig.text((axleft + SUPTITLE_DX) / W, SUPTITLE_Y / H, cfg["title"],
                 fontsize=1.2 * FONT_SIZE, ha="center", va="baseline")

        cax = fig.add_axes(rect(axleft + CBAR_DX, CBAR_Y, CBAR_W, CBAR_H))
        cbar = fig.colorbar(
            ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(0, 1)),
            cax=cax, label="photon character")
        cbar.set_ticks([0, 1])

        if name in IR_ANNOT:
            a = IR_ANNOT[name]
            ax = axs[a["panel"]]
            mid = 0.5 * (a["y0"] + a["y1"])
            half = 0.5 * (a["y1"] - a["y0"]) * IR_ARROW_SCALE
            ax.annotate("", xy=(a["lam"], mid + half),
                        xytext=(a["lam"], mid - half),
                        arrowprops=dict(arrowstyle="<->", color="black",
                                        lw=1.0,
                                        mutation_scale=IR_ARROW_HEAD),
                        zorder=6)
            ax.text(a["text_lam"], a["text_y"], "IR abs.", ha="left",
                    va="baseline", fontsize=FONT_SIZE, zorder=6)

        for ext in ("png", "pdf"):
            out = os.path.join(FIGS, f"{name}_{cfg['fig']}.{ext}")
            fig.savefig(out, dpi=200)      # no bbox_inches: the page is exact
            print("wrote", out)
        plt.close(fig)


def plot_fig2():
    # combined efield file: lambda | freqs(N) | photon_char(N)
    d = np.loadtxt(os.path.join(DATA, "CO2_Fig2_splitting_efield_response.dat"))
    lam = d[:, 0]
    n = (d.shape[1] - 1) // 2
    freqs = d[:, 1:1 + n]
    octo = np.loadtxt(os.path.join(DATA, "CO2_Fig2_splitting_octopus.dat"))
    # cols 1..3 are the symmetric stretch and the lower and upper polariton.
    # The symmetric stretch is IR-inactive, so it does not couple to the cavity
    # and sits flat at 169 meV; the published figure plots it, so it is kept.
    oct_lam, oct_meV = octo[:, 0], octo[:, 1:]

    with plt.rc_context({"font.size": FONT_SIZE}):
        fig = plt.figure(figsize=(F2_FIGW / 72.0, F2_FIGH / 72.0))
        x, y, w, h = F2_AX
        ax = fig.add_axes([x / F2_FIGW, y / F2_FIGH,
                           w / F2_FIGW, h / F2_FIGH])
        ax.set_xlim(*F2_XLIM)
        ax.set_ylim(*F2_YLIM)
        # The published figure draws three blue branches: the two polaritons and
        # the flat symmetric stretch. Rather than selecting by photon character,
        # plot every branch and let the y-limits crop to the resonant window,
        # which leaves exactly those three.
        for j in range(freqs.shape[1]):
            if freqs[:, j].max() >= F2_YLIM[0] and freqs[:, j].min() <= F2_YLIM[1]:
                ax.plot(lam, freqs[:, j], color=F2_LINE_COLOR, lw=F2_LW)
        ax.plot(oct_lam, oct_meV, "o", color=F2_POINT_COLOR, ms=F2_MS, zorder=5)
        ax.plot([], [], color=F2_LINE_COLOR, lw=F2_LW,
                label="via electric field response")
        ax.plot([], [], "o", color=F2_POINT_COLOR, ms=F2_MS,
                label="via explicit CBOA QEDFT")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(F2_XSTEP))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(F2_YSTEP))
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\hbar\omega$ (meV)")
        ax.legend(loc="upper right", bbox_to_anchor=F2_LEGEND_ANCHOR,
                  borderaxespad=0.0, fontsize=FONT_SIZE)
        for ext in ("png", "pdf"):
            out = os.path.join(FIGS, f"CO2_Fig2_splitting.{ext}")
            fig.savefig(out, dpi=200)
            print("wrote", out)
        plt.close(fig)


def plot_table1():
    """Render the reproduced Table I (data/Table_I.dat) as png+pdf."""
    path = os.path.join(DATA, "Table_I.dat")
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.split())
    # columns: System omega chi_axis chi_00 |Z*| Degenerate
    paper = {"CO2": (312, 26, 0.75, "False"),
             "Fe(CO)5": (252, 115, 0.60, "False"),
             "h-BN": (165, 37, 1.10, "True"),
             "HfS2": (18, 316, 0.98, "True")}
    col_labels = ["System", r"$\omega_{res}$ (meV)", r"$\chi_{11}$ ($a_0^3$)",
                  r"$|Z^*_{res}|$ ($e/\sqrt{amu}$)", "Degenerate",
                  "(paper $\\omega$/$\\chi$/$|Z^*|$/deg)"]
    cell = []
    for r in rows:
        sysn = r[0]
        p = paper.get(sysn, ("", "", "", ""))
        cell.append([sysn, f"{float(r[1]):.0f}", f"{float(r[2]):.1f}",
                     f"{float(r[4]):.2f}", r[5],
                     f"{p[0]} / {p[1]} / {p[2]} / {p[3]}"])
    fig, ax = plt.subplots(figsize=(13.5, 1.8))
    ax.axis("off")
    tab = ax.table(cellText=cell, colLabels=col_labels, loc="center",
                   cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.6)
    for j in range(len(col_labels)):
        tab[0, j].set_facecolor("#dfe6ee")
        tab[0, j].set_text_props(fontweight="bold")
    ax.set_title("TABLE I.  Selected matter properties (reproduced vs paper)",
                 fontsize=11, loc="left")
    for ext in ("png", "pdf"):
        out = os.path.join(FIGS, f"Table_I.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    for name in SYS:
        plot_system(name)
    plot_fig2()
    plot_table1()
