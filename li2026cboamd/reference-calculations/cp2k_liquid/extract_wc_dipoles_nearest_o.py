#!/usr/bin/env python3
"""Extract CO2 WC-derived dipoles by assigning each Wannier center to nearest O.

This script expects a parsed DeepMD-style dataset directory containing
``source_paths.txt`` and ``C64O128/set.000/{box,coord}.npy``.  The source paths
must point to CP2K frame directories with ``co2.xyz``, ``CELL``, and
``co2-HOMO_centers_s1-1_0.data``.

Output atom order is canonicalized to 64 C atoms followed by 128 O atoms.
Each Wannier center is assigned to the nearest oxygen atom only, under PBC.
The resulting per-O minimum-image WC displacement is written as
``atomic_dipole``; C entries are kept zero, matching the previous label
convention.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


N_C = 64
N_O = 128
N_ATOMS = N_C + N_O
N_WC_EXPECTED = 4 * N_O


def read_xyz_raw(filename: Path) -> tuple[list[str], np.ndarray]:
    lines = filename.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=float)


def canonical_perm(symbols: list[str], frame_id: int) -> np.ndarray:
    c_idx = [i for i, symbol in enumerate(symbols) if symbol == "C"]
    o_idx = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    if len(c_idx) != N_C or len(o_idx) != N_O:
        raise ValueError(
            f"frame {frame_id}: expected {N_C} C and {N_O} O, "
            f"got {len(c_idx)} C and {len(o_idx)} O"
        )
    return np.asarray(c_idx + o_idx, dtype=int)


def read_wc(filename: Path) -> np.ndarray:
    centers: list[list[float]] = []
    for line in filename.read_text().splitlines()[2:]:
        if not line.strip():
            continue
        parts = line.split()
        if parts[0] == "X":
            centers.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(centers, dtype=float)


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return (delta / cell - np.rint(delta / cell)) * cell


def atomic_dipole_nearest_o(
    coord: np.ndarray, wc: np.ndarray, cell: np.ndarray, frame_id: int
) -> tuple[np.ndarray, np.ndarray]:
    if coord.shape != (N_ATOMS, 3):
        raise ValueError(f"frame {frame_id}: unexpected coord shape {coord.shape}")
    if wc.shape != (N_WC_EXPECTED, 3):
        raise ValueError(f"frame {frame_id}: unexpected WC shape {wc.shape}")

    oxygen_coord = coord[N_C:]
    dist_o = minimum_image(wc[:, None, :] - oxygen_coord[None, :, :], cell)
    nearest_o = np.argmin(np.linalg.norm(dist_o, axis=2), axis=1)

    wc_displacements = np.zeros((N_O, 3))
    counts = np.zeros(N_O, dtype=int)
    for oxygen_i in range(N_O):
        wc_idx = np.where(nearest_o == oxygen_i)[0]
        counts[oxygen_i] = wc_idx.size
        if wc_idx.size == 0:
            raise ValueError(
                f"frame {frame_id}: oxygen local index {oxygen_i} has no WCs"
            )
        wc_displacements[oxygen_i] = dist_o[wc_idx, oxygen_i].mean(axis=0)

    atomic_dipole = np.zeros((N_ATOMS, 3))
    atomic_dipole[N_C:] = wc_displacements
    return atomic_dipole, counts


def load_sources(dataset_dir: Path) -> list[tuple[int, Path]]:
    source_file = dataset_dir / "source_paths.txt"
    frames: list[tuple[int, Path]] = []
    for line in source_file.read_text().splitlines():
        frame_id, source_path = line.split(maxsplit=1)
        frames.append((int(frame_id), Path(source_path)))
    return frames


def write_view(dataset_dir: Path, name: str, label: str, label_array: np.ndarray) -> None:
    raw_root = dataset_dir / "C64O128"
    setdir = raw_root / "set.000"
    view = dataset_dir / name
    if view.exists():
        shutil.rmtree(view)
    view_set = view / "set.000"
    view_set.mkdir(parents=True)
    shutil.copy2(raw_root / "type.raw", view / "type.raw")
    shutil.copy2(raw_root / "type_map.raw", view / "type_map.raw")
    shutil.copy2(setdir / "box.npy", view_set / "box.npy")
    shutil.copy2(setdir / "coord.npy", view_set / "coord.npy")
    shutil.copy2(dataset_dir / "frame_ids.txt", view / "frame_ids.txt")
    np.save(view_set / f"{label}.npy", label_array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    raw_root = dataset_dir / "C64O128"
    setdir = raw_root / "set.000"
    if args.no_overwrite and (setdir / "atomic_dipole.npy").exists():
        raise SystemExit(f"Refusing to overwrite {setdir / 'atomic_dipole.npy'}")

    frames = load_sources(dataset_dir)
    atomic_dipole = np.zeros((len(frames), N_ATOMS, 3))
    count_hist: dict[int, int] = {}
    anomalies: list[dict[str, object]] = []

    for idx, (frame_id, source_dir) in enumerate(frames):
        symbols, coord_raw = read_xyz_raw(source_dir / "co2.xyz")
        coord = coord_raw[canonical_perm(symbols, frame_id)]
        cell = np.loadtxt(source_dir / "CELL")
        wc = read_wc(source_dir / "co2-HOMO_centers_s1-1_0.data")
        atomic_dipole[idx], counts = atomic_dipole_nearest_o(
            coord, wc, cell, frame_id
        )

        values, nums = np.unique(counts, return_counts=True)
        for value, num in zip(values.tolist(), nums.tolist()):
            count_hist[value] = count_hist.get(value, 0) + num
        if not np.all(counts == 4):
            anomalies.append(
                {
                    "id": frame_id,
                    "min_count": int(counts.min()),
                    "max_count": int(counts.max()),
                    "counts": {
                        str(int(value)): int(num)
                        for value, num in zip(values.tolist(), nums.tolist())
                    },
                }
            )

        if (idx + 1) % 500 == 0 or idx + 1 == len(frames):
            print(f"processed {idx + 1}/{len(frames)}", flush=True)

    dipole = atomic_dipole.sum(axis=1)
    atomic_flat = atomic_dipole.reshape(len(frames), N_ATOMS * 3)
    dipole_flat = dipole.reshape(len(frames), 3)

    np.savetxt(raw_root / "atomic_dipole.raw", atomic_flat, fmt="%12.8e")
    np.savetxt(raw_root / "dipole.raw", dipole_flat, fmt="%12.8e")
    np.save(setdir / "atomic_dipole.npy", atomic_flat)
    np.save(setdir / "dipole.npy", dipole_flat)
    write_view(dataset_dir, "atomic_dipole", "atomic_dipole", atomic_flat)
    write_view(dataset_dir, "global_dipole", "dipole", dipole_flat)

    (dataset_dir / "nearest_o_wc_count_anomalies.json").write_text(
        json.dumps(anomalies, indent=2) + "\n"
    )
    summary_path = dataset_dir / "extraction_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["wc_assignment"] = {
        "standard_method": (
            "nearest O atom only, using minimum-image WC-minus-O "
            "displacements for atomic_dipole"
        ),
        "n_frames": len(frames),
        "oxygen_wc_count_histogram_over_all_oxygen_atoms": {
            str(key): int(value) for key, value in sorted(count_hist.items())
        },
        "n_frames_with_non_four_wc_per_oxygen": len(anomalies),
        "non_four_wc_per_oxygen_frames_file": "nearest_o_wc_count_anomalies.json",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote nearest-O atomic_dipole and dipole labels for {len(frames)} frames")
    print("oxygen WC count histogram:", dict(sorted(count_hist.items())))
    print("frames with non-four WCs per O:", len(anomalies))


if __name__ == "__main__":
    main()
