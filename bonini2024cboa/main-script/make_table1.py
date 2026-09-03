#!/usr/bin/env python3
"""
Reproduce Table I of the paper (selected matter properties):
    System | omega_res (meV) | chi_11 (a0^3) | |Z*_res| (e/sqrt(amu)) | Degenerate

All quantities come from the same VASP DFPT data used for the spectra:
  omega_res : frequency of the resonant IR-active mode (largest |Z*| along axis)
  chi_11    : (1,1) element of the electronic polarizability tensor chi0
              [chi0 = V*(eps_clamped - I)/(4 pi), atomic units a0^3; for the 2D
               solids this is the value per (periodic) unit cell]
  |Z*_res|  : mode effective charge of the resonant mode, converted to e/sqrt(amu)
  Degenerate: whether the resonant mode has a (near-)degenerate partner
"""
import os
import numpy as np
from vasp_reader import adiabatic_dat_from_vaspout
from qedmft import adiabatic as qad
from qedmft.units import EV_PER_HARTREE, MASS_AMU_FACT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "../matter-data/vasp-dfpt"))
OUTDATA = os.path.abspath(os.path.join(HERE, "data"))
HA2MEV = EV_PER_HARTREE * 1e3
SQRT_AMU = np.sqrt(MASS_AMU_FACT)        # e/sqrt(m_e) -> e/sqrt(amu)

SYSTEMS = {
    "CO2":     dict(path="CO2",   axis=[0, 0, 1], label="CO2"),
    "FeCO5":   dict(path="FeCO5", axis=[1, 0, 0], label="Fe(CO)5"),
    "hBN":     dict(path="hBN",   axis=[1, 0, 0], label="h-BN"),
    "HfS2":    dict(path="HfS2",  axis=[1, 0, 0], label="HfS2"),
}
# paper Table I values for reference
PAPER = {"CO2": (312, 26, 0.75, "False"),
         "FeCO5": (252, 115, 0.60, "False"),
         "hBN": (165, 37, 1.10, "True"),
         "HfS2": (18, 316, 0.98, "True")}

rows = []
for name, cfg in SYSTEMS.items():
    dat = adiabatic_dat_from_vaspout(os.path.join(DATA, cfg["path"]))
    axis = np.array(cfg["axis"], float)
    axis = axis / np.linalg.norm(axis)
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        freqs, eigdispls = qad.hmat_to_freqs(dat.cmat, dat.masses_flat,
                                             freq_only=False)
    Zmode = eigdispls @ dat.zmat
    proj = np.abs(Zmode @ axis)
    k = int(np.argmax(proj))
    wres_meV = freqs[k] * HA2MEV
    Zres = np.sqrt((Zmode[k] ** 2).sum()) * SQRT_AMU      # e/sqrt(amu)
    # chi_11 = (1,1) element along the coupling axis (here diagonal x or z)
    iax = int(np.argmax(np.abs(axis)))
    chi11 = dat.chi0[iax, iax]                            # a0^3
    # also report the conventional [0,0] element
    chi00 = dat.chi0[0, 0]
    # degeneracy: another mode within 1 meV with comparable |Z*|
    near = [j for j in range(len(freqs)) if j != k
            and abs(freqs[j] * HA2MEV - wres_meV) < 1.0
            and np.sqrt((Zmode[j] ** 2).sum()) * SQRT_AMU > 0.5 * Zres]
    degenerate = "True" if near else "False"
    rows.append((cfg["label"], wres_meV, chi11, chi00, Zres, degenerate))
    p = PAPER[name]
    print(f"{cfg['label']:9s} omega_res={wres_meV:7.1f} (paper {p[0]:>4})  "
          f"chi_axis={chi11:7.2f} chi_00={chi00:7.2f} (paper {p[1]:>4})  "
          f"|Z*|={Zres:5.3f} (paper {p[2]:.2f})  "
          f"deg={degenerate} (paper {p[3]})")

# write Table_I.dat
hdr = ("Table I  -- Selected matter properties (reproduced).\n"
       "Polarizability chi_11 in Bohr^3 (a0^3); for 2D crystals it is per u.c.\n"
       "|Z*_res| in e/sqrt(amu).  omega_res in meV.\n"
       "Columns: System  omega_res(meV)  chi_axis(a0^3)  chi_00(a0^3)  "
       "|Z*_res|(e/sqrt(amu))  Degenerate")
with open(os.path.join(OUTDATA, "Table_I.dat"), "w") as f:
    f.write("# " + hdr.replace("\n", "\n# ") + "\n")
    for r in rows:
        f.write(f"{r[0]:10s} {r[1]:10.3f} {r[2]:10.3f} {r[3]:10.3f} "
                f"{r[4]:10.4f} {r[5]:>8s}\n")
print("\nwrote", os.path.join(OUTDATA, "Table_I.dat"))
