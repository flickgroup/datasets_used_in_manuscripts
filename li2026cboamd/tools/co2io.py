"""Readers for the compressed .npz files shipped in data/.

Every array in data/ is a lossless float32 re-encoding of an ASCII file
produced by the MD and postprocessing codes. The encoding drops columns that
are bitwise zero or bitwise identical to another column, and replaces the
uniform time column by a (t0, dt) pair. The readers below undo all of that, so
a script sees the same named arrays it would have got from the original file.

Nothing is resampled, smoothed, filtered, truncated in time, or quantized. See
tools/convert_traces.py and tools/convert_spectra.py for the writers, which
carry the per-file assertions and the measured precision argument.
"""

import json

import numpy as np

# Column layout of the ASCII files written by CBOAMD, in order.
#
# dipole.dat carries two physically distinct dipoles: columns 2-4 are the
# dressed dipole (the one that enters the cavity equations of motion and the
# IR spectra) and columns 5-7 are the bare, field-free dipole of the same
# geometry. They coincide exactly at lambda = 0 and differ by up to 5e-2 a.u.
# at lambda = 0.3, so neither can be reconstructed from the other.
DIPOLE_COLUMNS = [
    "step", "time",
    "mu_x", "mu_y", "mu_z",
    "mu_bare_x", "mu_bare_y", "mu_bare_z",
]

# polarizability.dat repeats "polxy" in its header where it means "polyx".
# The names below are the corrected ones, in the file's own column order.
POLARIZABILITY_COLUMNS = [
    "step", "time",
    "chi_xx", "chi_xy", "chi_xz",
    "chi_yx", "chi_yy", "chi_yz",
    "chi_zx", "chi_zy", "chi_zz",
]

# position.dat holds the three CO2 atoms in the order C, O, O, coordinates in
# Angstrom followed by velocities. Only the field-free AIMD run lambda=0_qmode
# is shipped: it is the run the SI bend frequency nu2 = 680.4 cm^-1 is measured
# from. The other runs' position traces are in the raw release tier.
POSITION_COLUMNS = (
    ["step", "time"]
    + ["r_%s_%s" % (atom, axis)
       for atom in ("C", "O1", "O2") for axis in ("x", "y", "z")]
    + ["v_%s_%s" % (atom, axis)
       for atom in ("C", "O1", "O2") for axis in ("x", "y", "z")]
)

COLUMN_LAYOUTS = {
    "dipole": DIPOLE_COLUMNS,
    "polarizability": POLARIZABILITY_COLUMNS,
    "position": POSITION_COLUMNS,
}


def _rehydrate(npz):
    """Rebuild the full name -> array mapping from a stored trace."""
    stored = {name: npz["data"][i].astype(np.float64)
              for i, name in enumerate(npz["columns"])}

    nstep = int(npz["nstep"])
    for name in npz["zeros"]:
        stored[str(name)] = np.zeros(nstep, dtype=np.float64)
    for dropped, source in json.loads(str(npz["dups"])).items():
        stored[dropped] = stored[source].copy()

    t0, dt = float(npz["t0"]), float(npz["dt"])
    stored["step"] = np.arange(nstep, dtype=np.float64)
    stored["time"] = t0 + dt * np.arange(nstep, dtype=np.float64)
    return stored


def load_trace(path):
    """Load a dipole.npz or polarizability.npz trace.

    Returns a dict of float64 arrays keyed by the names in DIPOLE_COLUMNS or
    POLARIZABILITY_COLUMNS, plus "dt" (atomic units) and "header" (the
    original ASCII header line).
    """
    with np.load(str(path), allow_pickle=False) as npz:
        out = _rehydrate(npz)
        out["dt"] = float(npz["dt"])
        out["header"] = str(npz["header"])
        out["kind"] = str(npz["kind"])
    return out


def load_trace_columns(path):
    """Load a trace as the dense 2D array the original ASCII file held.

    Column order is exactly COLUMN_LAYOUTS[kind], so this reproduces
    np.loadtxt(original_dat) to float32 precision. Used by the round-trip
    check in convert_traces.py and by scripts that index columns positionally.
    """
    trace = load_trace(path)
    names = COLUMN_LAYOUTS[trace["kind"]]
    return np.column_stack([trace[n] for n in names])


def load_spectrum(path):
    """Load an averaged liquid spectrum.

    Returns (freq, columns) where columns is a dict keyed by the spectrum
    column names (mean_intensity, mean_hamming11, ...) and freq is in cm^-1.
    The stored window is 0 to 6000 cm^-1; the negative half of the original
    file is the mirror image of the positive half and is not stored.
    """
    with np.load(str(path), allow_pickle=False) as npz:
        freq = npz["freq"].astype(np.float64)
        columns = {str(name): npz["data"][i].astype(np.float64)
                   for i, name in enumerate(npz["columns"])}
        meta = json.loads(str(npz["meta"]))
    return freq, columns, meta


def load_spectrum_usecols(path, usecols=(0, 4)):
    """Positional-column access matching np.loadtxt(dat, usecols=...).

    The figure scripts read usecols=(0, 4), i.e. the frequency axis and the
    Hamming-smoothed mean intensity. Column indices refer to the original
    ASCII layout.
    """
    freq, columns, _ = load_spectrum(path)
    names = ["freq_cm-1", "mean_intensity", "std_intensity", "sem_intensity",
             "mean_hamming11", "std_hamming11", "sem_hamming11"]
    full = {"freq_cm-1": freq}
    full.update(columns)
    return np.column_stack([full[names[i]] for i in usecols])


def load_rdf(path):
    """Load an RDF summary. Returns (r, {g_CC, g_CO, g_OO}, meta)."""
    with np.load(str(path), allow_pickle=False) as npz:
        r = npz["r"].astype(np.float64)
        columns = {str(name): npz["data"][i].astype(np.float64)
                   for i, name in enumerate(npz["columns"])}
        meta = json.loads(str(npz["meta"]))
    return r, columns, meta


def load_rdf_columns(path):
    """RDF as the dense (r, g_CC, g_CO, g_OO) array the ASCII file held."""
    r, columns, _ = load_rdf(path)
    return np.column_stack([r, columns["g_CC"], columns["g_CO"],
                            columns["g_OO"]])


def load_parity(path):
    """Load a DeePMD dp test parity file. Returns the dense 2D array."""
    with np.load(str(path), allow_pickle=False) as npz:
        return npz["data"].astype(np.float64)
