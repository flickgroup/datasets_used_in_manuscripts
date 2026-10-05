#!/usr/bin/env python3
"""Shared loader and analysis for the periodic-pMBD bilayer binding curves.

Figs. 1, 2, S2, S3 and S4 read the same six binding curves, so the reading,
the assembly of the total energy and the harmonic fit live here rather than
being repeated in each plotting script. The scripts in main-script/ and SI/
import this module; the SI ones add main-script/ to sys.path first.

Physics, all from the HDF5 files written by tools/build_dataset.py:

    E(λ, R) = E_DFT(R) + [ E_MBD(R) + ptexc(λ, R) - ene(λ, R) ] * Hartree

E_DFT is the VASP PBE total energy of the four-atom bilayer cell in eV and is
independent of λ. E_MBD is the q-converged MBD energy in Hartree. The bracket's
last two terms are the coupled and uncoupled pMBD exchange-correlation energies,
whose difference is the cavity-induced shift. The two pieces come off different
q-grids on purpose: the shift converges quickly and is taken from a 5 x 3 x 1
run, while E_MBD does not and is taken from a separate 37 x 37 x 1 run at λ = 0. Binding curves are reported
against the same cell with the layers pulled 10 Å apart,

    ΔE(λ, R) = [ E(λ, R) - E(λ, 10 Å) ] * 1000 / n_atoms   in meV/atom.
"""

from pathlib import Path

import h5py
import numpy as np

# Hartree in eV. This is 2 * P_Ry with the octopus value P_Ry = 13.60569193
# that libphotonmbd used to produce the raw output, not the current CODATA
# number: keeping it identical reproduces the published figures bit for bit.
HARTREE_EV = 2.0 * 13.60569193

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Mass of a single layer inside the unit cell, in amu. The layer breathing mode
# of two identical layers has reduced mass m_layer / 2, hence ω = sqrt(2k/m).
# graphene: 2 * 12.011 (two C per layer). hBN: one B plus one N.
LAYER_MASS_AMU = {"graphene": 2 * 12.011, "hBN": 24.812}

# Half-width of the fit window, in grid points either side of the discrete
# minimum. The grid spacing is 0.0025 Å, so these are ±0.0875 Å for hBN and
# ±0.0625 Å for graphene. Values as used for the published figures.
FIT_HALF_WIDTH = {"hBN": 35, "graphene": 25}

# sqrt(eV / (Å² amu)) -> cm⁻¹, i.e. ν̃ = ω / (2πc) with
# 1 eV/Å² = 16.02176634 J/m² and 1 amu = 1.6605402e-27 kg.
CM_PER_SQRT_EV_ANG2_AMU = (
    np.sqrt(16.02176634) * 10 ** 3.5 / (2 * np.pi * 2.99792458 * np.sqrt(1.6605402))
)

# Colours used for the stackings throughout the manuscript.
STACKING_COLOR = {"AA": "r", "AB": "b", "AA'": "g", "AB1": "m"}


class Bilayer:
    """One stacking of one material: the full (λ, R) grid of binding energies."""

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.material = f.attrs["material"]
            self.stacking = f.attrs["stacking"]
            self.n_atoms = int(f.attrs["n_atoms"])
            self.omega_ev = float(f.attrs["omega_ev"])
            self.r = f["r"][:]
            # Unscaled coupling strengths, as plotted and as quoted in the
            # manuscript. The solver ran each one at lam * sqrt(N_q) with
            # N_q = 15, the cavity mode coupling to all periodic images at
            # once; see run-inputs/README.md.
            self.lam = f["lambda"][:]
            e_dft = f["E_dft"][:]
            e_mbd = f["E_mbd"][:]
            shift = f["ptexc"][:] - f["ene"][:]
            ref = f["reference"]
            e_dft_ref = float(ref["E_dft"][()])
            e_mbd_ref = float(ref["E_mbd"][()])
            shift_ref = ref["ptexc"][:] - ref["ene"][:]

        # Total energy of the cell in eV, and of the R = 10 Å reference.
        total = e_dft + (e_mbd + shift) * HARTREE_EV
        total_ref = e_dft_ref + (e_mbd_ref + shift_ref) * HARTREE_EV

        #: binding energy in eV per cell, shape (n_lambda, n_r)
        self.binding_ev = total - total_ref[:, None]

    @property
    def label(self):
        return f"{self.material} {self.stacking}"

    def index(self, lam):
        """Row index of a coupling strength."""
        i = int(np.argmin(np.abs(self.lam - lam)))
        if abs(self.lam[i] - lam) > 1e-9:
            raise KeyError(f"{self.label}: no data for lambda = {lam}")
        return i

    def curve_mev_per_atom(self, lam):
        """ΔE(R) at one coupling strength, in meV/atom."""
        return self.binding_ev[self.index(lam)] * 1000.0 / self.n_atoms

    def fit(self, lam):
        """Harmonic fit of the minimum. Returns (R0 [Å], ω [cm⁻¹], E_min [eV])."""
        curve = self.binding_ev[self.index(lam)]
        r0, k, e_min = fit_minimum(self.r, curve, FIT_HALF_WIDTH[self.material])
        return r0, lbm_frequency(k, self.material), e_min

    def fit_series(self, lambdas):
        """`fit` over several coupling strengths, as three arrays."""
        out = np.array([self.fit(lam) for lam in lambdas])
        return out[:, 0], out[:, 1], out[:, 2]


def fit_minimum(r, energy, half_width):
    """Cubic fit around the discrete minimum of `energy`.

    Returns (R0, k, E_min): the stationary point of the cubic inside the fit
    window, its curvature there, and the energy at it. Units follow the input,
    so with r in Å and energy in eV, k comes out in eV/Å².

    A cubic rather than a parabola because the well is visibly asymmetric over
    the ±0.06-0.09 Å window used; the extra degree of freedom absorbs the
    asymmetry instead of biasing R0.
    """
    r, energy = np.asarray(r, float), np.asarray(energy, float)
    i = int(np.argmin(energy))
    lo, hi = i - half_width, i + half_width + 1
    if lo < 0 or hi > len(r):
        raise ValueError(f"fit window [{lo}, {hi}) runs off the {len(r)}-point grid")
    x, y = r[lo:hi], energy[lo:hi]

    poly = np.poly1d(np.polyfit(x, y, deg=3))
    roots = np.polyder(poly, 1).roots
    inside = [z.real for z in roots if np.isreal(z) and x[0] <= z.real <= x[-1]]
    if not inside:
        raise ValueError("cubic fit has no stationary point inside the window")
    r0 = inside[0]
    return r0, float(np.polyder(poly, 2)(r0)), float(poly(r0))


def lbm_frequency(k, material):
    """Layer breathing mode frequency in cm⁻¹ from a curvature k in eV/Å²."""
    return CM_PER_SQRT_EV_ANG2_AMU * np.sqrt(2.0 * k / LAYER_MASS_AMU[material])


def load(name, data_dir=DATA_DIR):
    """Load one system by file stem, e.g. 'hbn_aa' or 'graphene_ab'."""
    return Bilayer(Path(data_dir) / f"{name}.h5")


IMAGES_DIR = Path(__file__).resolve().parents[1] / "images"


def add_schematic(ax, name, xy, zoom):
    """Overlay the structure render images/<name>.png at data coordinates `xy`.

    The insets are artwork rather than data, so a missing PNG prints a note and
    the figure is drawn without it.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import matplotlib.pyplot as plt

    path = IMAGES_DIR / f"{name}.png"
    if not path.exists():
        print(f"  (no {path.name} in {IMAGES_DIR.name}/, inset omitted)")
        return
    box = OffsetImage(plt.imread(path), zoom=zoom)
    ax.add_artist(AnnotationBbox(box, xy, frameon=False))


def save(fig, outdir, stem):
    """Write <outdir>/<stem>.png and .pdf.

    matplotlib's default 100 dpi and a tight bounding box, as for the published
    PNGs: the manuscript scales each one by a fixed factor in \\includegraphics,
    so a regenerated PNG keeps its printed size only at the same pixel size.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{stem}.{ext}", bbox_inches="tight")
    print(f"Saved -> {outdir}/{stem}.png, .pdf")


def use_style():
    """matplotlib defaults, apart from the heavier axes frame of the manuscript."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.linewidth": 1.8})


GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
FIGSIZE = (7 * GOLDEN_RATIO, 7)

# Coupling strengths shown in every λ scan, in a.u.
LAMBDAS = np.round(np.arange(0.0, 0.20 + 0.001, 0.02), 2)
