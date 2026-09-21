#!/usr/bin/env python3
"""Gas-phase dipole derivative of CO2 at PBE0/aug-cc-pVDZ (conv 1e-10).

Verifies the gas-phase slope drawn in Fig. 6(b) and used in the Fig. 6(a)
prediction: the finite-difference dipole derivative obtained by displacing
the C atom by 0.05 A along the molecular axis,

    d = mu_x(dC = 0.05 A) / 0.05 = 4.589 a.u./A

(GAS_DMU in fig6_rabi_scaling.py). The reference geometry is the one the
whole MD campaign uses, C-O = 1.170708 A (md-inputs/single_molecule/
co2-disp.xyz); at the re-optimized PBE0 bond length of 1.163638 A
(co2_harmonic_freqs.py) the same recipe gives 4.627 a.u./A instead. Forward
and central differences agree to the printed digits.

Also converts both the gas-phase slope and the liquid regression slope
d_eff = 5.549 a.u./A (printed by fig6_rabi_scaling.py) to the resonant-mode
Born effective charge of the SI,

    Z*_res = d * a0 * sqrt(1/(2 m_O) + 1/m_C),

giving Z*_gas = 0.82 and Z*_liq = 0.99 e/amu^{1/2} as quoted there.
"""
import numpy as np
from pyscf import gto, dft

R_CO = 1.170708    # A, C-O distance of the MD reference geometry
DISP = 0.05        # A, C displacement along the molecular axis
A0 = 0.529177      # Bohr radius in A
M_O, M_C = 15.999, 12.011   # amu


def dipole_x(dc):
    """mu_x (a.u.) with the C atom displaced by dc A along x."""
    mol = gto.M(atom=[["O", (-R_CO, 0, 0)], ["C", (dc, 0, 0)],
                      ["O", (R_CO, 0, 0)]],
                basis="aug-cc-pvdz", unit="Angstrom", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf.dip_moment(unit="AU", verbose=0)[0]


mu = dipole_x(DISP)
d_gas = mu / DISP
print(f"reference geometry: C-O = {R_CO} A (co2-disp.xyz)")
print(f"mu_x(dC={DISP} A) = {mu:.6f} a.u.")
print(f"gas-phase dipole derivative d = {d_gas:.3f} a.u./A "
      f"(GAS_DMU in fig6_rabi_scaling.py)")

conv = A0 * np.sqrt(1.0 / (2.0 * M_O) + 1.0 / M_C)
d_liq = 5.549   # a.u./A, liquid regression slope printed by fig6
print(f"Z*_res gas    = {d_gas * conv:.2f} e/amu^0.5")
print(f"Z*_res liquid = {d_liq * conv:.2f} e/amu^0.5   (from d_eff = {d_liq})")
