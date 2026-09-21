"""Manuscript Fig. 6: collective scaling of the Rabi splitting (PBE0 only).

Style/setup matches fig1-fig5. Chi-neglected channel throughout.

Panel a: Rabi splitting vs lambda for the single CO2 molecule (PBE0/aug-cc-pVDZ
MLIP, extracted live from data/single_molecule/nep_md/lambda=*_nopolar/,
the 100 ps NEP-MD runs shown in fig1/fig2) and for liquid CO2
(PBE0 DP models, extracted from the averaged dressed spectra as in fig4),
together with the parameter-free prediction
    Omega_liq(lambda) = Omega_1(lambda) x sqrt(Sum_i cos^2 theta_i) x d_liq/d_gas
built from the linear fit Omega_1 = c1*lambda and the two factors MEASURED in
panel b / from the same trajectories (no adjustable parameters).

Panel b: the local-field measurement: instantaneous liquid dipole mu_x vs the
collective stretch descriptor X = Sum_i s_i cos(theta_i) computed from the
out-of-cavity trajectory frames (s_i = asymmetric-stretch coordinate, theta_i =
angle between molecular axis and x). The regression slope is the effective
per-molecule dipole derivative in the liquid; the gas-phase PBE0/aug-cc-pVDZ
slope (4.589 a.u./A, PySCF finite difference, mu(dC=0.05 A)/0.05, conv 1e-10)
is shown for comparison.

Frames are parsed from traj.dump (every 20 steps) of ALL 40 out-of-cavity
trajectories; molecules assigned per frame by the two nearest O per C under
the minimum image. Parsed descriptors are cached in
fig6_scatter_cache_40traj.npz.

Only that cache is shipped. The traj.dump files it was built from are about
3 GB and live in the reduced raw release tier, not in this repository, so the
parse below runs only where that tier is mounted. Delete the cache and this
script rebuilds it there; everywhere else it reads the cache and panel b is
reproduced from it.
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
OUTDIR = ROOT
SINGLE = DATA / "single_molecule"
LIQ = DATA / "liquid" / "spectra" / "offresonant"
NOCAV = LIQ / "10_out_of_cavity_ir_40traj_newmodels"
NTRAJ_SCATTER = 40         # out-of-cavity trajectories used for panel b (all)
CACHE = DATA / "liquid" / f"fig6_scatter_cache_{NTRAJ_SCATTER}traj.npz"

AU_TO_CM = 219474.63
OMEGA_C_SINGLE = 2442.0
OMEGA_C_LIQ = 2407.4
GAS_DMU = 4.589            # a.u./A, PySCF PBE0/aug-cc-pVDZ (see docstring)

colors = {"single": "#cc163a", "liq": "#1177b0", "pred": "#1b7837"}


# ---------- panel a: single-molecule splittings (100 ps NEP-MD runs) --------
def spectrum(dipole_file):
    d = co2io.load_trace_columns(dipole_file)
    t, mux = d[:, 1], d[:, 2]
    dt = t[1] - t[0]
    sig = mux - mux.mean()
    spec = np.abs(np.fft.rfft(sig))
    freq = 2 * np.pi * np.fft.rfftfreq(len(sig), d=dt) * AU_TO_CM
    return freq, spec


def strongest_peak(freq, spec, lo, hi, floor):
    idx = np.flatnonzero((freq >= lo) & (freq <= hi))
    cand = [i for i in idx[1:-1]
            if spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1]
            and spec[i] >= floor]
    return float(freq[max(cand, key=lambda i: spec[i])]) if cand else np.nan


def single_series():
    rows = []
    for d in sorted((SINGLE / "nep_md").glob("lambda=*_nopolar")):
        lam = float(re.search(r"lambda=([\d.]+)_", d.name).group(1))
        f = d / "dipole.npz"
        if lam == 0 or not f.exists():
            continue
        freq, spec = spectrum(f)
        floor = 0.01 * spec[(freq >= 500) & (freq <= 3800)].max()
        lo = strongest_peak(freq, spec, 800, OMEGA_C_SINGLE - 1, floor)
        hi = strongest_peak(freq, spec, OMEGA_C_SINGLE + 1, 3800, floor)
        if np.isfinite(lo) and np.isfinite(hi):
            rows.append((lam, hi - lo))
    return np.array(sorted(rows))


# ---------- panel a: liquid splittings (as in fig4, chi neglected) ----------
def band_centroid(freq, inten, lo, hi, floor, frac=0.3):
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
        return float(freq[imax])
    a, b = region(frac)
    w = inten[a:b + 1]
    return float(np.sum(freq[a:b + 1] * w) / np.sum(w))


def liquid_series():
    rows = []
    for path in sorted(LIQ.glob("*_cavity_ir_40traj_newmodels_nopolar_lam*/"
                                "spectrum_direct_dressed_x_avg40_hamming11.npz")):
        lam = int(re.search(r"nopolar_lam(\d+)", path.parent.name).group(1)) / 100.0
        d = co2io.load_spectrum_usecols(path, (0, 4))
        freq, inten = d[:, 0], d[:, 1]
        floor = 0.001 * inten[(freq >= 300) & (freq <= 4400)].max()
        lo = strongest_peak(freq, inten, 1200, OMEGA_C_LIQ, floor)
        hi = band_centroid(freq, inten, OMEGA_C_LIQ, 4400, floor)
        rows.append((lam, hi - lo))
    return np.array(sorted(rows))


# ---------- panel b: orientation + local-field regression -------------------
def parse_dump(path):
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline())
            f.readline()
            n = int(f.readline())
            f.readline()
            L = float(f.readline().split()[1])
            f.readline(); f.readline(); f.readline()
            data = np.array([f.readline().split()[:5] for _ in range(n)],
                            dtype=float)
            data = data[np.argsort(data[:, 0])]
            t = data[:, 1].astype(int)
            frames.append((step, L, data[t == 1, 2:5], data[t == 2, 2:5]))
    return frames


def scatter_data():
    if CACHE.exists():
        d = np.load(CACHE)
        return d["X"], d["MU"], float(d["sum_cos2"])
    X, MU, cos2 = [], [], []
    for i in range(NTRAJ_SCATTER):
        traj = NOCAV / f"traj_{i:02d}"
        d = np.loadtxt(traj / "cboamd_output.dat", comments="#")
        lookup = {int(s): k for k, s in enumerate(d[:, 0])}
        for step, L, C, O in parse_dump(traj / "traj.dump"):
            if step not in lookup:
                continue
            dd = O[None, :, :] - C[:, None, :]
            dd -= L * np.round(dd / L)
            r = np.linalg.norm(dd, axis=2)
            idx = np.argsort(r, axis=1)[:, :2]
            b1 = dd[np.arange(64), idx[:, 0]]
            b2 = dd[np.arange(64), idx[:, 1]]
            e = b1 - b2
            e /= np.linalg.norm(e, axis=1)[:, None]
            s = 0.5 * (np.linalg.norm(b1, axis=1) - np.linalg.norm(b2, axis=1))
            X.append(np.sum(s * e[:, 0]))
            MU.append(d[lookup[step], 2])
            cos2.append(np.sum(e[:, 0] ** 2))
    X, MU, sum_cos2 = np.array(X), np.array(MU), float(np.mean(cos2))
    np.savez(CACHE, X=X, MU=MU, sum_cos2=sum_cos2)
    return X, MU, sum_cos2


single = single_series()
liq = liquid_series()
X, MU, SUM_COS2 = scatter_data()
d_eff = abs(np.cov(X, MU)[0, 1] / np.var(X))
ENH = d_eff / GAS_DMU
# mass-weighted resonant-mode effective charge Z*_res in e/sqrt(amu),
# following Bonini and Flick, J. Chem. Phys. 161, 154104 (2024):
# Z = d[a.u./A] * a0[A/bohr] * sqrt(1/(2 m_O) + 1/m_C)
BOHR_A = 0.529177
MASS_FAC = np.sqrt(1.0 / (2 * 15.999) + 1.0 / 12.011)
Z_LIQ = d_eff * BOHR_A * MASS_FAC
Z_GAS = GAS_DMU * BOHR_A * MASS_FAC
c1 = np.sum(single[:, 0] * single[:, 1]) / np.sum(single[:, 0] ** 2)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.609), gridspec_kw={"wspace": 0.42})

ax = axes[0]
lam_grid = np.linspace(0, 0.31, 50)
ax.plot(lam_grid, c1 * lam_grid, "-", color=colors["single"], lw=1.0, alpha=0.5)
ax.plot(single[:, 0], single[:, 1], "o", color=colors["single"], ms=4.5,
        label=r"single CO$_2$")
ax.plot(liq[:, 0], liq[:, 1], "o", color=colors["liq"], ms=4.5,
        label=r"liquid CO$_2$")
ax.plot(lam_grid, c1 * lam_grid * np.sqrt(SUM_COS2) * ENH, "--",
        color=colors["pred"], lw=1.4)
# the prediction formula as a direct green annotation in the empty band
# between the single-molecule line and the liquid data (a legend row this
# wide cannot avoid the steep green line)
ax.text(0.305, 1600,
        r"$\Omega_1\sqrt{\Sigma_i\cos^2\theta_i}\;"
        r"Z^{*,\mathrm{liq}}_{\mathrm{res}}/Z^{*,\mathrm{gas}}_{\mathrm{res}}$",
        ha="right", va="center", fontsize=9, color=colors["pred"])
ax.set_xlim(0, 0.31)
ax.set_ylim(0, 2900)   # headroom so the legend (10 pt) clears the topmost liquid points
ax.set_xlabel(r"Coupling strength $\lambda$")
ax.set_ylabel(r"Rabi Splitting [cm$^{-1}$]")
ax.set_xticks([0.0, 0.1, 0.2, 0.3])
ax.tick_params(direction="in", top=True, right=True)
# the wedge between the liquid prediction line and the single-molecule line
# (upper right) is free of data; short handles keep the legend clear of the
# steep green prediction line and the topmost liquid points
ax.legend(frameon=False, loc="upper right", handlelength=1.3,
          borderaxespad=0.02, fontsize=9, labelspacing=0.3)
ax.text(-0.26, 0.973, "a", transform=ax.transAxes, fontsize=14,
        fontfamily="cmb10", va="center")

ax = axes[1]
# binned profile instead of the raw scatter: mean dipole (with SEM bars) per
# descriptor bin; the raw per-frame cloud carries large bend/induced-dipole
# scatter that obscures the linear response
sl = np.cov(X, MU)[0, 1] / np.var(X)
off = MU.mean() - sl * X.mean()
edges = np.linspace(np.quantile(X, 0.005), np.quantile(X, 0.995), 12)
cent, mean, sem = [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    m = (X >= a) & (X < b)
    if m.sum() < 20:
        continue
    cent.append(0.5 * (a + b))
    mean.append(MU[m].mean())
    sem.append(MU[m].std() / np.sqrt(m.sum()))
xg = np.linspace(edges[0], edges[-1], 10)
ax.plot(xg, sl * xg + off, "-", color=colors["liq"], lw=1.6,
        label=rf"liquid:" "\n" rf"$Z^{{*}}_{{\mathrm{{res}}}}={Z_LIQ:.2f}$ "
              rf"$e\,\mathrm{{amu}}^{{-1/2}}$")
ax.plot(xg, np.sign(sl) * GAS_DMU * xg + off, "--", color=colors["single"],
        lw=1.4,
        label=rf"single molecule:" "\n" rf"$Z^{{*}}_{{\mathrm{{res}}}}={Z_GAS:.2f}$ "
              rf"$e\,\mathrm{{amu}}^{{-1/2}}$")
ax.errorbar(cent, mean, yerr=sem, fmt="o", color="0.25", ms=4.0,
            elinewidth=1.0, capsize=2.0, zorder=5)
# plain dot as the legend handle: the SEM bars are invisible at this scale
ax.plot([], [], "o", color="0.25", ms=4.0, label="liquid MD")
ax.set_xlabel(r"$\sum_i s_i\cos\theta_i$ [$\mathring{\mathrm{A}}$]")
ax.set_ylabel(r"$\mu_x$ [a.u.]")
ax.tick_params(direction="in", top=True, right=True)
# data descends from upper left to lower right; at 9 pt with tight padding
# the legend fits in the free lower-left quadrant below the lines
ax.legend(frameon=False, loc="lower left", handlelength=0.8, fontsize=9,
          handletextpad=0.3, labelspacing=0.25, borderaxespad=0.1,
          borderpad=0.1)
ax.text(-0.30, 0.973, "b", transform=ax.transAxes, fontsize=14,
        fontfamily="cmb10", va="center")

fig.savefig(OUTDIR / "fig6_rabi_scaling.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "fig6_rabi_scaling.pdf", bbox_inches="tight")

print(f"single-molecule points: {single[:,0].tolist()}  c1 = {c1:.0f}/lambda")
print(f"Sum cos^2 theta = {SUM_COS2:.2f}   d_eff = {d_eff:.3f} au/A   enh = {ENH:.3f}")
print(f"prediction factor = {np.sqrt(SUM_COS2)*ENH:.2f}")
c_liq = np.sum(liq[:, 0] * liq[:, 1]) / np.sum(liq[:, 0] ** 2)
print(f"liquid fit {c_liq:.0f}/lambda   predicted {c1*np.sqrt(SUM_COS2)*ENH:.0f}/lambda "
      f"({100*(c1*np.sqrt(SUM_COS2)*ENH/c_liq-1):+.1f}%)")
