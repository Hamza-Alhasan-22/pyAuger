#!/usr/bin/env python3
"""Compare C++ matrix elements against the Python implementation for a subset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AugerCalculator = None
prepare_cpp_input = None
utilities = None


def import_auger_modules() -> None:
    global AugerCalculator, prepare_cpp_input, utilities
    if AugerCalculator is not None:
        return
    from auger import AugerCalculator as _AugerCalculator
    from auger import utilities as _utilities
    from auger.cpp import prepare_cpp_input as _prepare_cpp_input

    AugerCalculator = _AugerCalculator
    utilities = _utilities
    prepare_cpp_input = _prepare_cpp_input


FIELDS = [
    "|M|^2",
    "|M(G=0)|^2",
    "|Md|^2",
    "|Mx|^2",
    "|Md(G=0)|^2",
    "|Mx(G=0)|^2",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_dielectric(value: str):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return utilities.normalize_dielectric_input(parsed)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_folder", required=True)
    parser.add_argument("--wavecar_files", nargs="+", required=True)
    parser.add_argument("--pairs_csv", nargs="+", required=True)
    parser.add_argument("--auger_type", choices=["eeh", "ehh"], required=True)
    parser.add_argument(
        "--dielectric",
        required=True,
        help="Scalar dielectric or JSON 3x3 tensor.",
    )
    parser.add_argument("--firstCB_index", type=int, required=True)
    parser.add_argument("--lastVB_index", type=int, required=True)
    parser.add_argument("--cpp_executable", default=str(REPO_ROOT / "auger" / "cpp" / "matrix_element_calc"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_matrix_elements", default="5")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--T", type=float, default=300.0)
    parser.add_argument("--nd", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    import_auger_modules()
    dielectric = parse_dielectric(args.dielectric)

    n_me: str | int = args.num_matrix_elements
    if n_me != "all":
        n_me = int(n_me)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.results_folder) / "cpp_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cpp_bin = out_dir / f"{args.auger_type}_cpp_input.bin"
    cpp_jsonl = out_dir / f"{args.auger_type}_matrix_elements_cpp_1.jsonl"
    cpp_config = out_dir / f"{args.auger_type}_cpp_config.json"

    calc = AugerCalculator(T=args.T, nd=args.nd)
    calc.assign_firstCB_and_lastVB(args.firstCB_index, args.lastVB_index)
    calc.import_parsed_BS_data(args.results_folder)
    calc.read_auger_pairs(args.pairs_csv)

    print("\nRunning Python matrix elements...")
    calc.calculate_matrix_elements(
        auger_type=args.auger_type,
        wavecar_files=args.wavecar_files,
        dielectric_constant=dielectric,
        num_matrix_elements=n_me,
        add_suffix_name="python_validation",
    )
    python_rows = calc.matrix_elements_dicts[args.auger_type]

    print("\nPreparing C++ binary input...")
    prepare_cpp_input.prepare(
        results_folder=args.results_folder,
        wavecar_files=args.wavecar_files,
        pairs_csv=args.pairs_csv,
        auger_type=args.auger_type,
        dielectric=dielectric,
        firstCB_index=args.firstCB_index,
        lastVB_index=args.lastVB_index,
        mapping_csv=None,
        output_path=str(cpp_bin),
        num_matrix_elements=n_me,
        continue_from_files=None,
        T=args.T,
        nd=args.nd,
    )

    cpp_config.write_text(json.dumps({
        "input_binary": str(cpp_bin),
        "output_jsonl": str(cpp_jsonl),
        "num_threads": args.num_threads,
        "overwrite": True,
        "append": False,
        "resume": False,
        "log_file": None,
        "progress_interval": 1,
    }, indent=2) + "\n")

    print("\nRunning C++ matrix elements...")
    subprocess.run([args.cpp_executable, "--config", str(cpp_config)], check=True)
    cpp_rows = read_jsonl(cpp_jsonl)

    py_map = {row["pair_id"]: row for row in python_rows if "error" not in row}
    cpp_map = {row["pair_id"]: row for row in cpp_rows if "error" not in row}
    errors = [row for row in cpp_rows if "error" in row]
    if errors:
        raise SystemExit(f"C++ returned {len(errors)} error rows; first error: {errors[0]}")

    missing = sorted(set(py_map) - set(cpp_map))
    extra = sorted(set(cpp_map) - set(py_map))
    if missing or extra:
        raise SystemExit(f"Pair-id mismatch. Missing in C++: {missing[:5]}, extra in C++: {extra[:5]}")

    failures = []
    for pid, py_row in py_map.items():
        cpp_row = cpp_map[pid]
        for field in FIELDS:
            if field not in py_row or field not in cpp_row:
                failures.append((pid, field, "missing field"))
                continue
            if not np.isclose(float(py_row[field]), float(cpp_row[field]), rtol=args.rtol, atol=args.atol):
                failures.append((pid, field, py_row[field], cpp_row[field]))

    if failures:
        print("\nValidation failures:")
        for item in failures[:20]:
            print(f"  {item}")
        raise SystemExit(f"{len(failures)} field comparisons failed.")

    print(f"\nValidation passed for {len(py_map)} pairs with rtol={args.rtol:g}, atol={args.atol:g}.")


if __name__ == "__main__":
    main()
