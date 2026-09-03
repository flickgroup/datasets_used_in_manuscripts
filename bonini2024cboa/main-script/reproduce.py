#!/usr/bin/env python3
"""
Reproduce the vibro-/phonon-polariton spectra figures of
Bonini, Ahmadabadi & Flick, J. Chem. Phys. 161, 154104 (2024),
"Cavity Born-Oppenheimer approximation for molecules and materials via
electric field response".

Figures reproduced (the ones backed by the data shipped in this folder):
  Fig. 3  CO2          (4 panels A/B/C/D)
  Fig. 4  Fe(CO)5      (4 panels A/B/C/D)
  Fig. 5  h-BN         (4 panels A/B/C/D)
  Fig. 6  HfS2         (4 panels A/B/C/D)
  Fig. 2  CO2 splitting via electric-field response (blue curves only;
          the orange explicit-CBOA-QEDFT points were computed with Octopus
          and are NOT part of this data set).

Panel convention (same as the paper):
  A : chi neglected, multiple cavity modes (7 harmonics, 3rd resonant)
  B : chi included,  multiple cavity modes
  C : chi neglected, single resonant cavity mode
  D : chi included,  single resonant cavity mode

The physics/solver is the qedmft package of John Bonini
(https://github.com/jrbp/qedmft), which is NOT shipped here; see the folder
README for how to make it importable. Only the I/O layer (py4vasp, which cannot
be installed) is replaced, by vasp_reader.py, which reads the identical arrays
out of vaspout.h5 and applies the identical unit conversions as
qedmft.io.adiabatic_dat_from_p4vsp.
"""
import os
import sys
import io as _io
import contextlib
from dataclasses import replace
import numpy as np

from vasp_reader import adiabatic_dat_from_vaspout
from qedmft import adiabatic as qad
from qedmft.units import EV_PER_HARTREE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "../matter-data/vasp-dfpt"))
OUTDATA = os.path.abspath(os.path.join(HERE, "data"))
os.makedirs(OUTDATA, exist_ok=True)

HA2MEV = EV_PER_HARTREE * 1e3

# ----------------------------------------------------------------------------
# system definitions: data path + cavity polarization axis (the resonant
# IR-active mode is found automatically as the largest |Z*| along that axis).
# These reproduce the omega_res values of Table I.
# ----------------------------------------------------------------------------
SYSTEMS = {
    "CO2":     dict(path="CO2",   axis=[0, 0, 1], fig="Fig3"),
    "FeCO5":   dict(path="FeCO5", axis=[1, 0, 0], fig="Fig4"),
    "hBN":     dict(path="hBN",   axis=[1, 0, 0], fig="Fig5"),
    "HfS2":    dict(path="HfS2",  axis=[1, 0, 0], fig="Fig6"),
}

NLAMBDA = 1000
LAMBDA_MAX_SPECTRA = 0.15     # Figs 3-6 horizontal axis
LAMBDA_MAX_FIG2 = 0.30        # Fig 2 horizontal axis
NHARM_MULTI = 7
ARES = 3                      # resonant harmonic index in the multi-mode case


@contextlib.contextmanager
def _quiet():
    """Swallow stdout.

    Kept for older qedmft checkouts, which had debug prints in
    get_coupled_hmat / hmat_to_freqs / solve_all_qad. The current upstream
    (github.com/jrbp/qedmft) prints nothing, so this is a no-op there.
    """
    with contextlib.redirect_stdout(_io.StringIO()):
        yield


def find_resonant_mode(dat, axis):
    """Return (omega_res_Ha, mode_index) for the largest |Z*_axis| mode."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    with _quiet():
        freqs, eigdispls = qad.hmat_to_freqs(dat.cmat, dat.masses_flat,
                                             freq_only=False)
    Zmode = eigdispls @ dat.zmat            # (nmode, 3)
    proj = np.abs(Zmode @ axis)             # coupling strength along axis
    k = int(np.argmax(proj))
    return freqs[k], k


def build_photons(wres, axis, multi):
    """Single resonant mode, or 7 harmonics with the 3rd resonant.

    multi=False: one mode at wres, lambda direction = axis (unit).
    multi=True : modes a=1..7 with omega_a=(a/3) wres and lambda_a=(a/3) axis,
                 so that scaling by lambda_res gives lambda_a=(a/3) lambda_res
                 and omega_a=(a/3) omega_res (3rd harmonic resonant).
    """
    d = np.asarray(axis, float)
    d = d / np.linalg.norm(d)
    if not multi:
        fs = np.array([wres])
        ds = d[None, :].copy()
    else:
        a = np.arange(1, NHARM_MULTI + 1)
        fs = (a / ARES) * wres
        ds = (a[:, None] / ARES) * d[None, :]
    return qad.PhotonModes(freqs=fs, lambdas=ds)


def solve_panel(dat, axis, multi, chi, lambdas):
    """Return freqs(meV), photon-char, IR-intensity arrays of shape
    (n_lambda, n_total_modes) for one panel."""
    mat = dat if chi else replace(dat, chi0=np.zeros_like(dat.chi0))
    wres, _ = find_resonant_mode(dat, axis)
    photons = build_photons(wres, axis, multi)
    phts = photons.range_scale_lambda(lambdas)
    # numpy >= 2 raises spurious divide/overflow/invalid warnings out of the
    # threaded BLAS matmul in get_coupled_hmat (the Fe(CO)5 multi-mode panels
    # trigger them at every lambda, including lambda = 0). Every operand and
    # result there is finite, which the assertions below check, so the warnings
    # are silenced rather than worked around.
    with _quiet(), np.errstate(divide="ignore", over="ignore",
                               invalid="ignore"):
        freqs, displs, irvs, irns = qad.solve_all_qad(phts, mat)
        # upstream qedmft.pht_char reads phts[0].nmodes, so it wants the list
        chars = qad.pht_char(displs, phts)
    for name, arr in (("freqs", freqs), ("chars", chars), ("irns", irns)):
        assert np.isfinite(arr).all(), f"non-finite {name} in this panel"
    return freqs * HA2MEV, chars, irns, wres


def write_dat(fname, lambdas, mat, header):
    """Write lambda + per-mode columns to a .dat file."""
    out = np.column_stack([lambdas, mat])
    np.savetxt(os.path.join(OUTDATA, fname), out, header=header,
               fmt="%.8e")


def write_combined(fname, lambdas, freqs, chars, irns, header):
    """One file per panel: lambda | freqs(N) | photon_char(N) | IR(N)."""
    out = np.column_stack([lambdas, freqs, chars, irns])
    np.savetxt(os.path.join(OUTDATA, fname), out, header=header, fmt="%.8e")


PANELS = [("A", True, False), ("B", True, True),
          ("C", False, False), ("D", False, True)]
PANEL_DESC = {
    "A": "chi neglected, multiple cavity modes (7 harmonics, 3rd resonant)",
    "B": "chi included,  multiple cavity modes (7 harmonics, 3rd resonant)",
    "C": "chi neglected, single resonant cavity mode",
    "D": "chi included,  single resonant cavity mode",
}


def main():
    lambdas = np.linspace(0, LAMBDA_MAX_SPECTRA, NLAMBDA)
    results = {}
    for name, cfg in SYSTEMS.items():
        dat = adiabatic_dat_from_vaspout(os.path.join(DATA, cfg["path"]))
        results[name] = {}
        wres_meV = None
        fig = cfg["fig"]
        for panel, multi, chi in PANELS:
            freqs, chars, irns, wres = solve_panel(
                dat, cfg["axis"], multi, chi, lambdas)
            wres_meV = wres * HA2MEV
            results[name][panel] = dict(freqs=freqs, chars=chars, irns=irns)
            n = freqs.shape[1]
            hdr = (f"{name} {fig} panel {panel}: {PANEL_DESC[panel]}\n"
                   f"omega_res = {wres_meV:.3f} meV ; axis = {cfg['axis']} ; "
                   f"nmodes = {n}\n"
                   f"col0 = lambda ; "
                   f"cols 1..{n} = frequency (meV) ; "
                   f"cols {n+1}..{2*n} = photon character ; "
                   f"cols {2*n+1}..{3*n} = IR intensity (arb. u.)")
            write_combined(f"{name}_{fig}_panel{panel}.dat",
                           lambdas, freqs, chars, irns, hdr)
        print(f"{name:8s} ({fig}): omega_res = {wres_meV:8.2f} meV  "
              f"(axis {cfg['axis']}), wrote 4 combined .dat files")

    # ---- Fig 2: CO2 splitting via electric field response (blue curves) ----
    lam2 = np.linspace(0, LAMBDA_MAX_FIG2, NLAMBDA)
    dat = adiabatic_dat_from_vaspout(os.path.join(DATA, SYSTEMS["CO2"]["path"]))
    freqs2, chars2, irns2, wres2 = solve_panel(
        dat, SYSTEMS["CO2"]["axis"], multi=False, chi=True, lambdas=lam2)
    n2 = freqs2.shape[1]
    hdr2 = ("CO2 Fig2 splitting via electric-field response (Eq. 30), single "
            "resonant cavity mode, chi included (VASP matter data).\n"
            f"omega_res = {wres2*HA2MEV:.3f} meV ; nmodes = {n2}.  The paper's "
            "Fig 2 used Octopus for both curves; the Octopus explicit-CBOA "
            "points are in CO2_Fig2_splitting_octopus.dat.\n"
            f"col0 = lambda ; cols 1..{n2} = frequency (meV) ; "
            f"cols {n2+1}..{2*n2} = photon character")
    write_dat("CO2_Fig2_splitting_efield_response.dat", lam2,
              np.column_stack([freqs2, chars2]), hdr2)

    # Octopus explicit-CBOA finite-difference reference points (paper Fig 2).
    # ../bonini-data/octopus_CO2_CBOA_FD.txt : row0 = lambda, rows 1..3 =
    # polariton branch frequencies in cm^-1. Convert cm^-1 -> meV.
    HA_TO_CM1 = 219474.6313702
    octo = np.loadtxt(os.path.abspath(os.path.join(
        HERE, "../matter-data/octopus/octopus_CO2_CBOA_FD.txt")))
    oct_meV = octo[1:].T / HA_TO_CM1 * 1000 * EV_PER_HARTREE
    write_dat("CO2_Fig2_splitting_octopus.dat", octo[0], oct_meV,
              "CO2 Fig2 splitting via explicit CBOA-QEDFT finite differences "
              "(Octopus).\nSource: matter-data/octopus/octopus_CO2_CBOA_FD.txt, "
              "converted cm^-1 -> meV.\n"
              "col0 = lambda ; col1 = symmetric stretch, 1363.8 cm^-1, "
              "IR-inactive so uncoupled and flat in lambda ; "
              "col2 = lower polariton ; col3 = upper polariton (meV)")
    print("Fig2    : CO2 splitting (efield-response + Octopus points) written")
    results["_fig2"] = dict(lambdas=lam2, freqs=freqs2)

    np.savez(os.path.join(OUTDATA, "_all_results.npz"),
             lambdas=lambdas, lam2=lam2,
             **{f"{n}_{p}_{q}": results[n][p][q]
                for n in SYSTEMS for p in "ABCD"
                for q in ("freqs", "chars", "irns")})
    print("\nAll .dat files written to:", OUTDATA)


if __name__ == "__main__":
    main()
