"""Rebuild the Figure 2(b) data files from the raw octopus runs.

Reads the `Photon ex.` energy out of

    ../matter-data/octopus-fullerenes/<system>/OEP-gga-<kappa>/static/info

and writes ../main-script/data/<system>_photon-ex.dat, two columns: the dissipation
constant kappa in eV, and the GA electron-photon exchange energy in Hartree. plot_fig2.py
scales those by P_Har/N_at to get the eV-per-atom quantity the figure plots.

Why this script exists: the published .dat files were assembled by hand. The
`grep_etot.sh` sitting next to the fullerene runs is a stale copy of the one used for
Fig. 1, walking `1..20` and `f1..f20` subdirectories that do not exist there and grepping
`Total       =` rather than `Photon ex.`, so it cannot have produced them. This script is
the missing step, so that main-script/data/ is derivable from matter-data/ rather than
taken on trust.

The rows follow the runs present under matter-data/, which are the runs the paper used.
C180 therefore yields five rows where C20 and C60 yield six: the paper has no C180 point
at kappa = 0.5, so that run is not shipped.

Run from anywhere:  python3 tools/collect_photon_ex.py
Add --check to compare against the existing files instead of overwriting them.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, os.pardir, 'matter-data', 'octopus-fullerenes')
OUT = os.path.join(HERE, os.pardir, 'main-script', 'data')

SYSTEMS = ['C20', 'C60', 'C180']

# Row order of the published files: the dissipative runs by ascending kappa, then the
# single-mode (no dissipation) run last, recorded as kappa = 0.
RUN_PREFIX = 'OEP-gga-'
SINGLE_MODE_DIR = 'OEP-gga-single-mode'


def read_photon_ex(info_path):
    """Return the `Photon ex.` energy in Hartree from an octopus static/info file."""
    with open(info_path) as f:
        for line in f:
            if line.strip().startswith('Photon ex.'):
                return float(line.split('=')[1])
    raise ValueError(f'no "Photon ex." line in {info_path}')


def kappas_present(system):
    """The dissipation constants this system has runs for, ascending.

    Read off the directory names rather than hardcoded, so the shipped runs decide the
    rows and a system with an incomplete scan needs no special case.
    """
    kappas = []
    for name in os.listdir(os.path.join(RUNS, system)):
        if name.startswith(RUN_PREFIX) and name != SINGLE_MODE_DIR:
            kappas.append(float(name[len(RUN_PREFIX):]))
    return sorted(kappas)


def collect(system):
    """Return the list of (kappa_ev, energy_hartree) rows for one fullerene."""
    rows = []
    for kappa in kappas_present(system):
        # directory names carry the kappa value as it was typed: 0.01, 0.05, 0.1, 0.5, 1
        name = f'{RUN_PREFIX}{kappa:g}'
        rows.append((kappa, read_photon_ex(os.path.join(RUNS, system, name, 'static', 'info'))))
    rows.append((0, read_photon_ex(os.path.join(RUNS, system, SINGLE_MODE_DIR, 'static', 'info'))))
    return rows


def format_rows(rows):
    return ''.join(f'{kappa:<6g} {energy:.8f}\n' for kappa, energy in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='compare against the existing files instead of writing')
    args = parser.parse_args()

    failed = False
    for system in SYSTEMS:
        rows = collect(system)
        text = format_rows(rows)
        path = os.path.join(OUT, f'{system}_photon-ex.dat')

        if args.check:
            existing = open(path).read() if os.path.exists(path) else None
            status = 'OK' if existing == text else 'MISMATCH'
            if existing != text:
                failed = True
            print(f'  {system}: {status} ({len(rows)} rows)')
        else:
            with open(path, 'w') as f:
                f.write(text)
            print(f'  {system}: wrote {len(rows)} rows to {os.path.relpath(path, os.getcwd())}')

    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
