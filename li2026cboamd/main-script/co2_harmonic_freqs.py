#!/usr/bin/env python3
"""Harmonic frequencies of CO2 at PBE0/aug-cc-pVDZ (conv 1e-10).

Verifies the gray marker lines in fig1 (nu2 bend, nu1 symmetric stretch)
and the nu3/omega_c value. Bond length optimized by parabola fit over a
symmetric-stretch scan, then analytic RKS Hessian + harmonic analysis.
"""
import numpy as np
from pyscf import gto, dft
from pyscf.hessian import thermo


def energy(r):
    mol = gto.M(atom=[["O", (-r, 0, 0)], ["C", (0, 0, 0)], ["O", (r, 0, 0)]],
                basis="aug-cc-pvdz", unit="Angstrom", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.conv_tol = 1e-10
    return mf.kernel(), mol, mf

rs = np.array([1.160, 1.165, 1.170, 1.175, 1.180])
es = np.array([energy(r)[0] for r in rs])
c = np.polyfit(rs, es, 2)
r0 = -c[1] / (2 * c[0])
print(f"optimized C-O bond length: {r0:.6f} A")

e0, mol, mf = energy(r0)
hess = mf.Hessian().kernel()
freq = thermo.harmonic_analysis(mol, hess)
print("harmonic frequencies [cm-1]:", np.round(freq["freq_wavenumber"], 1))
