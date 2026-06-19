#!/usr/bin/env python3
"""
Prepare binary input for the standalone C++ Auger pair generator.

This bridge keeps the existing Python parsing, carrier-concentration, NSCF
reading, and symmetry-expansion logic in one place, then exports the initialized
PairGenerator state to a compact binary file consumed by ``pair_generation_calc``.

Examples
--------
Generate exact-kpoint table input::

    python -m auger.cpp.prepare_pair_generation_input \\
        --task exact_kpoints \\
        --results_folder /path/to/results \\
        --auger_type eeh \\
        --CB_window 0.5 \\
        --VB_window 0.5 \\
        --num_to_keep 100000 \\
        --poscar_path /path/to/POSCAR \\
        --is_expanded_from_irreducible \\
        --output cpp_pair_input.bin

Generate nearest-kpoint pair-table input::

    python -m auger.cpp.prepare_pair_generation_input \\
        --task pairs \\
        --approach nearest_kpoint \\
        --results_folder /path/to/results \\
        --auger_type eeh \\
        --CB_window 0.5 \\
        --VB_window 0.5 \\
        --num_to_keep all \\
        --output cpp_pair_input.bin

Then run::

    ./pair_generation_calc cpp_pair_input.bin output_pairs.csv
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from typing import Iterable, List, Sequence, Union

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from auger import utilities as ut
from auger.calculator import AugerCalculator
from auger.pairs import PairGenerator


MAGIC = b"AUGPAIR1"
VERSION = 1

TASK_CODES = {"exact_kpoints": 0, "pairs": 1}
AUGER_TYPE_CODES = {"eeh": 0, "ehh": 1}
APPROACH_CODES = {"nearest_kpoint": 0, "exact_kpoint": 1}
SEARCH_MODE_CODES = {"Brute_Force": 0, "Max_Heap": 1}


def _as_path_list(value: Union[str, Sequence[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _parse_count(value: str) -> int:
    if str(value).lower() == "all":
        return -1
    return int(value)


def _parse_folder_index(value: str):
    if str(value).lower() == "all":
        return "all"
    return int(value)


def _vec3(value) -> List[float]:
    arr = np.asarray(value, dtype=float).reshape(3)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _write_string(fh, text: str) -> None:
    raw = str(text).encode("utf-8")
    fh.write(struct.pack("<i", len(raw)))
    fh.write(raw)


def _write_vec3(fh, value) -> None:
    fh.write(struct.pack("<3d", *_vec3(value)))


def _write_state(fh, state: dict) -> None:
    fh.write(struct.pack("<ii", int(state["band_index"]), int(state["k_index"])))
    fh.write(struct.pack("<d", float(state["energy"])))
    _write_vec3(fh, state["k"])
    fh.write(struct.pack("<dd", float(state["kw"]), float(state["P"])))


def _write_state_list(fh, states: Iterable[dict]) -> None:
    states = list(states)
    fh.write(struct.pack("<q", len(states)))
    for state in states:
        _write_state(fh, state)


def _write_string_set(fh, values: Iterable[str]) -> None:
    values = sorted({str(v) for v in values})
    fh.write(struct.pack("<q", len(values)))
    for value in values:
        _write_string(fh, value)


def _write_exact_row(fh, row: dict, auger_type: str) -> None:
    _write_string(fh, row["partial_pair_id"])

    e_indices = [
        int(row["E1_index"]),
        int(row["E2_index"]),
        int(row["E3_index"]),
        int(row["E4_index"]),
    ]
    wc_indices = [
        int(row["k1_wc_index"]),
        int(row["k2_wc_index"]),
        int(row["k3_wc_index"]),
        int(row["k4_wc_index"]),
    ]
    nscf_indices = [
        int(row["k1_nscf_index"]),
        int(row["k2_nscf_index"]),
        int(row["k3_nscf_index"]),
        int(row["k4_nscf_index"]),
    ]
    # Exact-kpoint pair tables store k*_index as NSCF-local indices, matching
    # PairGenerator._build_exact_kpoint_pairs and the WAVECAR mapping used later.
    k_indices = list(nscf_indices)
    fh.write(struct.pack("<4i", *e_indices))
    fh.write(struct.pack("<4i", *k_indices))
    fh.write(struct.pack("<4i", *wc_indices))
    fh.write(struct.pack("<4i", *nscf_indices))

    energies = [float(row[f"E{i}"]) for i in range(1, 5)]
    fh.write(struct.pack("<4d", *energies))

    if auger_type == "eeh":
        vectors = [
            row["k1"],
            row["k2_target_cart"],
            row["k3"],
            row["k4"],
        ]
        kws = [
            float(row["kw1"]),
            float(row["k2_weight"]),
            float(row["kw3"]),
            float(row["kw4"]),
        ]
        mapped = row["k2_target_cart_mapped"]
    else:
        vectors = [
            row["k1"],
            row["k2"],
            row["k3"],
            row["k4_target_cart"],
        ]
        kws = [
            float(row["kw1"]),
            float(row["kw2"]),
            float(row["kw3"]),
            float(row["k4_weight"]),
        ]
        mapped = row["k4_target_cart_mapped"]

    for vec in vectors:
        _write_vec3(fh, vec)
    fh.write(struct.pack("<4d", *kws))
    _write_vec3(fh, mapped)


def _collect_skip_ids(files: Sequence[str], *, kind: str, auger_type: str) -> set[str]:
    ids: set[str] = set()
    for path in _as_path_list(files):
        if not os.path.exists(path):
            print(f"  Warning: continuation file not found, skipping: {path}")
            continue
        rows = ut.read_csv(path)
        if kind == "exact_kpoints":
            ids.update(str(row["partial_pair_id"]) for row in rows if row.get("partial_pair_id") is not None)
        else:
            ids.update(
                str(row["pair_id"])
                for row in rows
                if row.get("pair_id") is not None and row.get("pair_type", auger_type) == auger_type
            )
    return ids


def _ensure_carriers(calc: AugerCalculator, delta_n: float, force_recompute: bool = False) -> None:
    missing = not hasattr(calc, "Efn") or not hasattr(calc, "Efp")
    if missing or force_recompute:
        print("  Carrier data not available in band_info.txt; calculating carrier concentrations.")
        calc.calculate_carrier_concentrations(delta_n=delta_n)


def _build_initialized_generator(
    calc: AugerCalculator,
    *,
    auger_type: str,
    CB_window: float,
    VB_window: float,
    approach: str,
    search_mode: str,
    num_to_keep: int,
    poscar_path: str | None,
    is_expanded_from_irreducible: bool,
    vasp_folder_to_expand,
) -> PairGenerator:
    return PairGenerator(
        auger_type,
        (
            calc,
            CB_window,
            VB_window,
            approach,
            search_mode,
            num_to_keep,
            "",
            poscar_path,
            is_expanded_from_irreducible,
            True,
            vasp_folder_to_expand,
        ),
    )


def _build_exact_row_generator(
    calc: AugerCalculator,
    *,
    auger_type: str,
    CB_window: float,
    VB_window: float,
    search_mode: str,
    num_to_keep: int,
    poscar_path: str | None,
    nscf_folders: Sequence[str],
    exact_kpoints_csv: Sequence[str],
) -> PairGenerator:
    gen = PairGenerator(
        auger_type,
        (
            calc,
            CB_window,
            VB_window,
            "exact_kpoint",
            search_mode,
            num_to_keep,
            "",
            poscar_path,
            False,
            False,
            "all",
        ),
    )
    gen._prepare_exact_kpoint_data(list(nscf_folders), list(exact_kpoints_csv))
    return gen


def prepare(
    *,
    results_folder: str,
    output: str,
    task: str,
    auger_type: str,
    approach: str,
    CB_window: float,
    VB_window: float,
    num_to_keep: str,
    search_mode: str = "Max_Heap",
    multiplier: int = 1,
    T: float = 300.0,
    nd: float = 0.0,
    delta_n: float = 0.0,
    force_recompute_carriers: bool = False,
    poscar_path: str | None = None,
    is_expanded_from_irreducible: bool = False,
    vasp_folder_to_expand: Union[int, str] = "all",
    nscf_folders: Sequence[str] | None = None,
    exact_kpoints_csv: Sequence[str] | None = None,
    continue_from_files: Sequence[str] | None = None,
) -> None:
    t0 = time.time()
    if task not in TASK_CODES:
        raise ValueError(f"Unsupported task: {task}")
    if auger_type not in AUGER_TYPE_CODES:
        raise ValueError(f"Unsupported auger_type: {auger_type}")
    if approach not in APPROACH_CODES:
        raise ValueError(f"Unsupported approach: {approach}")
    if search_mode not in SEARCH_MODE_CODES:
        raise ValueError(f"Unsupported search_mode: {search_mode}")

    desired_total = _parse_count(num_to_keep)

    calc = AugerCalculator(T=T, nd=nd)
    calc.import_parsed_BS_data(from_folder=results_folder)
    _ensure_carriers(calc, delta_n=delta_n, force_recompute=force_recompute_carriers)

    skip_partial_ids: set[str] = set()
    skip_pair_ids: set[str] = set()
    exact_rows: list[dict] = []

    if task == "exact_kpoints":
        approach = "exact_kpoint"
        skip_partial_ids = _collect_skip_ids(
            _as_path_list(continue_from_files),
            kind="exact_kpoints",
            auger_type=auger_type,
        )
        if skip_partial_ids:
            print(f"  Existing exact-kpoint partial_pair_id values: {len(skip_partial_ids):,}")
        gen = _build_initialized_generator(
            calc,
            auger_type=auger_type,
            CB_window=CB_window,
            VB_window=VB_window,
            approach=approach,
            search_mode=search_mode,
            num_to_keep=desired_total,
            poscar_path=poscar_path,
            is_expanded_from_irreducible=is_expanded_from_irreducible,
            vasp_folder_to_expand=vasp_folder_to_expand,
        )
    elif approach == "exact_kpoint":
        nscf_folders = _as_path_list(nscf_folders)
        exact_kpoints_csv = _as_path_list(exact_kpoints_csv)
        if not nscf_folders:
            raise ValueError("--nscf_folders is required for exact_kpoint pair generation")
        if not exact_kpoints_csv:
            raise ValueError("--exact_kpoints_csv is required for exact_kpoint pair generation")
        skip_pair_ids = _collect_skip_ids(
            _as_path_list(continue_from_files),
            kind="pairs",
            auger_type=auger_type,
        )
        if skip_pair_ids:
            print(f"  Existing pair_id values: {len(skip_pair_ids):,}")
        gen = _build_exact_row_generator(
            calc,
            auger_type=auger_type,
            CB_window=CB_window,
            VB_window=VB_window,
            search_mode=search_mode,
            num_to_keep=desired_total,
            poscar_path=poscar_path,
            nscf_folders=nscf_folders,
            exact_kpoints_csv=exact_kpoints_csv,
        )
        exact_rows = list(gen.exact_kpoints_dict.values())
    else:
        skip_pair_ids = _collect_skip_ids(
            _as_path_list(continue_from_files),
            kind="pairs",
            auger_type=auger_type,
        )
        if skip_pair_ids:
            print(f"  Existing pair_id values: {len(skip_pair_ids):,}")
        gen = _build_initialized_generator(
            calc,
            auger_type=auger_type,
            CB_window=CB_window,
            VB_window=VB_window,
            approach=approach,
            search_mode=search_mode,
            num_to_keep=desired_total,
            poscar_path=poscar_path,
            is_expanded_from_irreducible=False,
            vasp_folder_to_expand="all",
        )

    previous_count = len(skip_partial_ids) if task == "exact_kpoints" else len(skip_pair_ids)
    if desired_total >= 0:
        print(f"  Requested total rows: {desired_total:,}")
        print(f"  Previous continuation rows: {previous_count:,}")
        print(f"  New rows requested from C++: {max(desired_total - previous_count, 0):,}")

    kpoints = np.asarray(calc.kpoints, dtype=np.float64)
    kpoint_weights = np.asarray(calc.kpoints_weights, dtype=np.float64)
    energies = np.ascontiguousarray(calc.data_energies, dtype=np.float64)
    reciprocal_lattice = np.ascontiguousarray(calc.reciprocal_lattice, dtype=np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with open(output, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<i", VERSION))
        fh.write(
            struct.pack(
                "<iiiiqqiiiqddd",
                TASK_CODES[task],
                AUGER_TYPE_CODES[auger_type],
                APPROACH_CODES[approach],
                SEARCH_MODE_CODES[search_mode],
                int(desired_total),
                int(previous_count),
                int(multiplier),
                int(calc.firstCB_index),
                int(calc.num_of_bands),
                int(len(kpoints)),
                float(calc.T),
                float(calc.Efn),
                float(calc.Efp),
            )
        )
        fh.write(struct.pack("<9d", *reciprocal_lattice.reshape(-1)))

        fh.write(struct.pack("<q", len(kpoints)))
        for kpt, kw in zip(kpoints, kpoint_weights):
            _write_vec3(fh, kpt)
            fh.write(struct.pack("<d", float(kw)))

        fh.write(struct.pack("<iq", int(energies.shape[0]), int(energies.shape[1])))
        fh.write(energies.tobytes(order="C"))

        _write_state_list(fh, getattr(gen, "E1_energies", []))
        _write_state_list(fh, getattr(gen, "E2_energies", []))
        _write_state_list(fh, getattr(gen, "E3_energies", []))
        _write_state_list(fh, getattr(gen, "E4_energies", []))
        _write_string_set(fh, skip_partial_ids)
        _write_string_set(fh, skip_pair_ids)

        fh.write(struct.pack("<q", len(exact_rows)))
        for row in exact_rows:
            _write_exact_row(fh, row, auger_type)

    elapsed = time.time() - t0
    print(f"  Wrote C++ pair-generation input: {output}")
    print(f"  Preparation time: {elapsed:.2f} s")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=sorted(TASK_CODES), required=True)
    parser.add_argument("--auger_type", choices=sorted(AUGER_TYPE_CODES), required=True)
    parser.add_argument("--approach", choices=sorted(APPROACH_CODES), default="nearest_kpoint")
    parser.add_argument("--search_mode", choices=sorted(SEARCH_MODE_CODES), default="Max_Heap")
    parser.add_argument("--CB_window", type=float, required=True)
    parser.add_argument("--VB_window", type=float, required=True)
    parser.add_argument("--num_to_keep", default="all")
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--T", type=float, default=300.0)
    parser.add_argument("--nd", type=float, default=0.0)
    parser.add_argument("--delta_n", type=float, default=0.0)
    parser.add_argument("--force_recompute_carriers", action="store_true")
    parser.add_argument("--poscar_path")
    parser.add_argument("--is_expanded_from_irreducible", action="store_true")
    parser.add_argument("--vasp_folder_to_expand", default="all")
    parser.add_argument("--nscf_folders", nargs="*")
    parser.add_argument("--exact_kpoints_csv", nargs="*")
    parser.add_argument("--continue_from_files", nargs="*")
    args = parser.parse_args(argv)

    prepare(
        results_folder=args.results_folder,
        output=args.output,
        task=args.task,
        auger_type=args.auger_type,
        approach=args.approach,
        search_mode=args.search_mode,
        CB_window=args.CB_window,
        VB_window=args.VB_window,
        num_to_keep=args.num_to_keep,
        multiplier=args.multiplier,
        T=args.T,
        nd=args.nd,
        delta_n=args.delta_n,
        force_recompute_carriers=args.force_recompute_carriers,
        poscar_path=args.poscar_path,
        is_expanded_from_irreducible=args.is_expanded_from_irreducible,
        vasp_folder_to_expand=_parse_folder_index(args.vasp_folder_to_expand),
        nscf_folders=args.nscf_folders,
        exact_kpoints_csv=args.exact_kpoints_csv,
        continue_from_files=args.continue_from_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
