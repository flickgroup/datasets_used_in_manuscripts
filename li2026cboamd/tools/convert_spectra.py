"""Convert averaged liquid spectra to compressed float32 .npz.

The ASCII spectra span the full FFT range, about -33356 to +33356 cm^-1, and
the lower half is redundant: the spectrum is |FFT| of a real signal, so it is
exactly symmetric about its DC bin. Only the DC bin and everything above it up
to 6000 cm^-1 is stored. The symmetry is verified per column, per file, before
anything is dropped.

Two quirks of the producing code (ir_from_cboamd.py and
reconstruct_dressed_dipole_spectra.py) are measured here rather than silently
corrected, because the published figures were drawn against these exact
arrays:

1. The frequency axis is built as np.arange(-pi, pi, 2*pi/n) independently of
   the fftshift bin ordering. For the odd n used here it lands half a bin
   (about 0.834 cm^-1) below the true bin centres, so the DC bin is labelled
   -0.834 rather than 0.

2. hamming_smooth returns smoothed[window_len//2 - 1 : -window_len//2], i.e.
   [4:-6] for an 11-point window where the centred slice is [5:-5]. The three
   *_hamming11 columns are therefore displaced by one bin (about 1.668 cm^-1)
   relative to the three raw columns they smooth.

Both are recorded in the per-file metadata. Neither is large next to the
hundreds of cm^-1 of Rabi splitting the figures report, and neither affects a
splitting at all, since a splitting is a difference of two peak positions on
the same axis.

Nothing is resampled, smoothed, filtered, truncated in time, or quantized.

Usage:
    python3 convert_spectra.py SRC_ROOT DST_ROOT

SRC_ROOT is the directory holding offresonant/ and
resonant/. Output names encode the tuning and the case directory.
"""

from __future__ import print_function

import json
import os
import sys

import numpy as np

SPECTRUM_COLUMNS = [
    "freq_cm-1", "mean_intensity", "std_intensity", "sem_intensity",
    "mean_hamming11", "std_hamming11", "sem_hamming11",
]

# Stored window. The figures plot up to 4400 cm^-1; 6000 leaves headroom for
# reanalysis without carrying the whole Nyquist range.
FREQ_HI = 6000.0

# Highest frequency any shipped script reads. Asserted to be inside the window.
MAX_FREQ_USED = 4400.0

# A column counts as exactly mirror-symmetric below this relative deviation.
# The true value is float64 rounding, around 1e-16.
SYMMETRY_TOL = 1e-12

SPECTRUM_FILES = [
    "spectrum_direct_dressed_x_avg40_hamming11.dat",
    "spectrum_direct_x_avg40_hamming11.dat",
]


def parse_meta(header):
    """Pull the key=value tail off the spectrum header line."""
    meta = {}
    for token in header.lstrip("#").split():
        if "=" in token:
            key, value = token.split("=", 1)
            meta[key] = value
    return meta


def find_symmetry_centre(col, guess, src, name):
    """Row index about which `col` mirrors exactly, searched near `guess`."""
    scale = max(float(np.abs(col).max()), 1e-300)
    found = []
    for centre in range(max(guess - 4, 1), min(guess + 5, len(col) - 1)):
        npair = min(centre, len(col) - 1 - centre)
        if npair < 100:
            continue
        off = np.arange(1, npair + 1)
        dev = float(np.abs(col[centre - off] - col[centre + off]).max()) / scale
        if dev < SYMMETRY_TOL:
            found.append(centre)
    if not found:
        raise SystemExit(
            "ABORT %s: column %s has no exact mirror-symmetry centre near the "
            "zero crossing; the lower half is not redundant" % (src, name))
    if len(found) > 1:
        raise SystemExit(
            "ABORT %s: column %s has more than one symmetry centre %s"
            % (src, name, found))
    return found[0]


def convert_file(src, dst):
    with open(src) as fh:
        header = fh.readline().strip()
    raw = np.loadtxt(src, comments="#")

    if raw.ndim != 2 or raw.shape[1] != len(SPECTRUM_COLUMNS):
        raise SystemExit(
            "ABORT %s: expected %d columns, found %s"
            % (src, len(SPECTRUM_COLUMNS), raw.shape))

    freq = raw[:, 0]
    df = float(np.median(np.diff(freq)))
    guess = int(np.argmin(np.abs(freq)))

    # --- per-column symmetry centres --------------------------------------
    # The raw and the Hamming-smoothed columns do not share a centre, because
    # of the off-by-one slice in hamming_smooth. Each is located and checked
    # on its own terms.
    centres = {}
    for col in range(1, raw.shape[1]):
        name = SPECTRUM_COLUMNS[col]
        centres[name] = find_symmetry_centre(raw[:, col], guess, src, name)

    start = min(centres.values())

    # Keeping every row from `start` upward guarantees that each dropped row
    # i < start has its mirror partner 2*c - i inside the kept range for every
    # column c, since c >= start > i.
    for name, centre in centres.items():
        if 2 * centre - (start - 1) >= len(freq):
            raise SystemExit(
                "ABORT %s: mirror partner of the last dropped row falls "
                "outside the array for column %s" % (src, name))

    # --- record the two axis quirks ---------------------------------------
    dc_label = float(freq[centres["mean_intensity"]])
    if abs(abs(dc_label) - 0.5 * abs(df)) > 1e-6 * abs(df):
        raise SystemExit(
            "ABORT %s: DC bin is labelled %.6f cm^-1, which is not the "
            "expected half-bin offset of %.6f" % (src, dc_label, 0.5 * df))

    smooth_shift = (centres["mean_hamming11"] - centres["mean_intensity"])

    # --- window and encode -------------------------------------------------
    keep = np.zeros(len(freq), dtype=bool)
    keep[start:] = True
    keep &= freq <= FREQ_HI
    if not keep.any():
        raise SystemExit("ABORT %s: no samples in the window" % src)
    if freq[keep].max() < MAX_FREQ_USED:
        raise SystemExit(
            "ABORT %s: window tops out at %.1f cm^-1, below the %.1f cm^-1 "
            "the figures read" % (src, freq[keep].max(), MAX_FREQ_USED))

    window = raw[keep]
    data = np.array(window[:, 1:].T, dtype=np.float32)

    ref = window[:, 1:].T
    scale = np.maximum(np.abs(ref).max(axis=1, keepdims=True), 1e-300)
    rel = float((np.abs(data.astype(np.float64) - ref) / scale).max())
    if rel > 1e-6:
        raise SystemExit("ABORT %s: float32 round-trip error %.3e" % (src, rel))

    meta = parse_meta(header)
    meta["bin_width_cm-1"] = df
    meta["dc_bin_index_in_source"] = int(centres["mean_intensity"])
    meta["dc_bin_label_cm-1"] = dc_label
    meta["freq_axis_note"] = (
        "Frequency labels are verbatim from the ASCII source. They sit half a "
        "bin (%.4f cm^-1) below the true FFT bin centres because the "
        "producing code builds the axis independently of the fftshift "
        "ordering. Add %.4f cm^-1 to recover the true frequency."
        % (abs(dc_label), abs(dc_label)))
    meta["hamming_shift_bins"] = int(smooth_shift)
    meta["hamming_note"] = (
        "The *_hamming11 columns are displaced by %d bin(s) (%.4f cm^-1) "
        "relative to the raw columns, because hamming_smooth slices [4:-6] "
        "where the centred slice is [5:-5]."
        % (smooth_shift, smooth_shift * df))

    np.savez_compressed(
        dst,
        freq=np.array(window[:, 0], dtype=np.float32),
        columns=np.array(SPECTRUM_COLUMNS[1:]),
        data=data,
        meta=json.dumps(meta),
        header=header,
    )
    return int(keep.sum()), len(freq), rel, dc_label, smooth_shift


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src_root, dst_root = sys.argv[1], sys.argv[2]

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)

    total_in = total_out = 0
    n_files = 0
    offsets = set()
    shifts = set()
    for tuning in ("offresonant", "resonant"):
        tune_root = os.path.join(src_root, tuning)
        if not os.path.isdir(tune_root):
            continue
        for case in sorted(os.listdir(tune_root)):
            case_dir = os.path.join(tune_root, case)
            if not os.path.isdir(case_dir):
                continue
            for fname in SPECTRUM_FILES:
                src = os.path.join(case_dir, fname)
                if not os.path.exists(src):
                    continue
                out_dir = os.path.join(dst_root, tuning, case)
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                out = os.path.join(tuning, case, fname[:-len(".dat")] + ".npz")
                dst = os.path.join(dst_root, out)
                nkeep, ntot, rel, dc, shift = convert_file(src, dst)
                offsets.add(round(dc, 6))
                shifts.add(shift)
                size_in = os.path.getsize(src)
                size_out = os.path.getsize(dst)
                total_in += size_in
                total_out += size_out
                n_files += 1
                print("%-78s %5d/%-6d %6.2f MB -> %5.3f MB (%4.1fx) relerr=%.0e"
                      % (out, nkeep, ntot, size_in / 1e6, size_out / 1e6,
                         float(size_in) / size_out, rel))

    print("\n%d spectra: %.1f MB -> %.1f MB (%.1fx)"
          % (n_files, total_in / 1e6, total_out / 1e6,
             float(total_in) / max(total_out, 1)))
    print("DC-bin labels seen:      %s cm^-1" % sorted(offsets))
    print("Hamming shifts seen:     %s bin(s)" % sorted(shifts))


if __name__ == "__main__":
    main()
