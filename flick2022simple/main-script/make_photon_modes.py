"""Regenerate the octopus photon-mode files for Figure 2 of
Flick, Phys. Rev. Lett. 129, 143201 (2022).

The Fig. 2 runs in ../matter-data/octopus-fullerenes/ each reference a mode file by name
through `PhotonmodesFilename` in their `inp`. Those files are 37.6 MiB each, 564 MiB for
the full set, so they are not shipped: this script rewrites them byte-for-byte.

Extracted from the mode-file writer at the end of cell 2 of dissipation-wang-lambdas2.ipynb,
wrapped in the loop over the five dissipation constants the paper uses. The notebook emitted
only whichever kappa was set in cell 1, so producing the set meant re-running that cell by
hand five times.

The file format octopus expects is

    <number of modes> 5
    frequency | lambda | pol_x | pol_y | pol_z
    ...

with the polarisation vector along x throughout. Filenames are
`photon_modes_{nom}_{omegac}_{kappac}.dat` with omegac and kappac in Hartree, which is
what makes them look like `..._0.11024797619388696_0.0003674932539796232.dat`.

Verified: the kappa = 0.01 output is byte-identical to the file used for the published runs
(md5 017a19e3b7e2f06d2ad3e0df2aa18e56).

Run from this directory:  python3 make_photon_modes.py [outdir]
"""

import os
import sys

import numpy as np

# Hartree in eV, as octopus defines it: 2 * P_Ry with P_Ry = 13.60569193
# (src/basic/global.F90, src/basic/unit_system.F90). The original scripts imported this
# from a personal constants module; inlined so this folder depends on nothing outside
# numpy and matplotlib.
P_Har = 2.0 * 13.60569193

# Cavity parameters, from cell 1 of the notebook.
omegac = 3 / P_Har        # 3 eV cavity mode, in Hartree
lambdac = 0.1             # total coupling strength, a.u.
nom = 400000              # number of photon modes; see the note in plot_fig2.py

# The five dissipation constants of Fig. 2(b), in eV.
KAPPAS_EV = [0.01, 0.05, 0.1, 0.5, 1]


def lorentzian(omega, kappa, omegac, dom):
    return (dom*1/2/np.pi*kappa/((omega-omegac)**2+(kappa/2)**2)) #multiply with dom?


def write_dissipative(kappa_ev, outdir):
    """Write the mode file for one non-zero dissipation constant."""
    kappac = kappa_ev / P_Har

    omega_array = np.linspace(0, 0.3, nom)
    dom = omega_array[1] - omega_array[0]
    lambda_array = 0*omega_array

    for ii in range(0, nom):
        lambda_array[ii] = np.sqrt(lambdac**2*lorentzian(omega_array[ii], kappac, omegac, dom))

    # octopus requires the following structure:
    # frequency | lambda | pol_x | pol_y | pol_z
    # where (pol_x, pol_y, pol_z) is the polarization
    # vector.
    outputname = f'photon_modes_{nom}_{omegac}_{kappac}.dat'
    path = os.path.join(outdir, outputname)
    f = open(path, 'w')

    string = f'{nom} 5\n'
    f.write(string)

    for ii in range(0, nom):
        string = '%2.15f  %2.15f  %2.15f  %2.15f  %2.15f\n' % (omega_array[ii], lambda_array[ii], 1, 0, 0)
        f.write(string)

    string = '# nom: %2d dom: %2.4f omegac: %2.4f lambdac: %2.4f kappac: %2.4f \n' % (nom, dom, omegac, lambdac, kappac)
    f.write(string)

    f.close()
    return outputname


def write_single_mode(outdir):
    """Write the kappa = 0 reference file.

    This one is not a Lorentzian: it is the no-dissipation limit, a single mode at the
    cavity frequency carrying the entire coupling lambdac. The original was written by
    hand rather than by cell 2, and its formatting differs accordingly: plain float
    repr instead of %2.15f, and no trailing comment line. Reproduced here as-is so the
    `inp` files that name it stay runnable.
    """
    outputname = f'photon_modes_1_{omegac}_0.dat'
    path = os.path.join(outdir, outputname)
    with open(path, 'w') as f:
        f.write('1 5\n')
        f.write(f'{omegac} {lambdac} 1 0 0\n')
    return outputname


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(outdir, exist_ok=True)

    for kappa_ev in KAPPAS_EV:
        name = write_dissipative(kappa_ev, outdir)
        print(f'kappa = {kappa_ev:>4} eV  ->  {name}')

    name = write_single_mode(outdir)
    print(f'kappa =    0 eV  ->  {name}')


if __name__ == '__main__':
    main()
