#!/usr/bin/env python3
"""
Prepare a binary input file for the standalone C++ matrix-element calculator.

This script extracts all needed wavefunction data from WAVECAR files and pair
information from the AugerCalculator, writing a single ``.bin`` file that the
C++ executable reads.

Usage (from your job script)::

    python -m auger.cpp.prepare_cpp_input \\
        --results_folder  /path/to/results/exact_kpoint \\
        --wavecar_files   /path/to/VASP1/WAVECAR /path/to/NSCF_1/WAVECAR ... \\
        --pairs_csv       /path/to/auger_eeh_pairs_47.csv \\
        --auger_type      eeh \\
        --dielectric      15.7 \\
        --firstCB_index   10 \\
        --lastVB_index    9 \\
        --mapping_csv     /path/to/kpoints_mapping.csv \\
        --output          cpp_input.bin

Then run the C++ calculator::

    ./matrix_element_calc cpp_input.bin output_1.jsonl [num_threads]

The output JSONL chunks have the same format as the Python version and can be
loaded back with ``AugerCalculator.read_matrix_elements()``.

``--dielectric`` may be a scalar or a JSON 3x3 tensor such as
``'[[16,0,0],[0,15,0],[0,0,14]]'``. Tensor input is passed to the C++
calculator for directional q-dependent model dielectric screening.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Allow running as ``python -m auger.cpp.prepare_cpp_input`` from project root
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from auger import utilities as ut
from auger.calculator import AugerCalculator
from auger.constants import ALPHA_PENN, EPSILON_0, HBAR, M_E, MATRIX_FACTOR, eV

import vaspwfc as vwfc


def _parse_dielectric_cli(value: str):
    """Parse scalar or JSON 3x3 tensor from --dielectric."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return ut.normalize_dielectric_input(parsed)[0]


# ──────────────────────────────────────────────────────────────────────────────
# Binary-file writer
# ──────────────────────────────────────────────────────────────────────────────
# Format (all little-endian):
#
#   Header
#   ------
#   char[8]      magic = "AUGERCPP"
#   int32        version = 2
#   int32        auger_type  (0 = eeh, 1 = ehh)
#   float64[9]   true_Bcell  (row-major 3x3)
#   float64      a_fit, b_fit, c_fit
#   float64      inv_debye
#   float64      matrix_factor
#   float64      V_m3
#   float64      eV_const
#   int32        dielectric_mode  (0 = scalar/Penn, 1 = tensor/directional Penn)
#   float64      dielectric_scalar_for_debye
#   float64[9]   dielectric_tensor  (row-major; zeros for scalar mode)
#
#   Wavefunction data  (shared across pairs)
#   -----------------------------------------
#   int32        num_wfc_entries
#   For each wfc entry:
#       int32         nG   (number of G-vectors)
#       int32[nG*3]   G    (row-major, flattened)
#       float64[nG*2] C    (re, im interleaved)
#
#   Pair data
#   ---------
#   int32        num_pairs
#   For each pair:
#       int32       id_len
#       char[id_len] pair_id
#       float64[12]  k1[3], k2[3], k3[3], k4[3]
#       int32[4]     wfc_idx_1..4  (indices into wfc table)
#       int32[3]     G_prime
# ──────────────────────────────────────────────────────────────────────────────


def _write_header(f, auger_type_int, true_Bcell, a, b, c, inv_debye,
                  matrix_factor, V_m3, eV_const, dielectric_mode,
                  dielectric_scalar, dielectric_tensor):
    f.write(b"AUGERCPP")
    f.write(struct.pack("<i", 2))           # version
    f.write(struct.pack("<i", auger_type_int))
    f.write(struct.pack("<9d", *true_Bcell.ravel()))
    f.write(struct.pack("<7d", a, b, c, inv_debye, matrix_factor, V_m3, eV_const))
    f.write(struct.pack("<i", int(dielectric_mode)))
    f.write(struct.pack("<d", float(dielectric_scalar)))
    f.write(struct.pack("<9d", *np.asarray(dielectric_tensor, dtype=float).reshape(-1)))


def _write_wfc_entries(f, wfc_list):
    """wfc_list: list of (G_array_Nx3_int32, C_array_N_complex128)."""
    f.write(struct.pack("<i", len(wfc_list)))
    for G_arr, C_arr in wfc_list:
        nG = len(G_arr)
        f.write(struct.pack("<i", nG))
        f.write(np.ascontiguousarray(G_arr, dtype=np.int32).tobytes())
        # Interleave re/im
        interleaved = np.empty(nG * 2, dtype=np.float64)
        interleaved[0::2] = C_arr.real
        interleaved[1::2] = C_arr.imag
        f.write(interleaved.tobytes())


def _write_pairs(f, pair_records):
    """pair_records: list of dict with keys pair_id, k1..k4, wfc_idx_1..4, G_prime."""
    f.write(struct.pack("<i", len(pair_records)))
    for rec in pair_records:
        pid = rec["pair_id"].encode("utf-8")
        f.write(struct.pack("<i", len(pid)))
        f.write(pid)
        for ki in ("k1", "k2", "k3", "k4"):
            f.write(struct.pack("<3d", *rec[ki]))
        f.write(struct.pack("<4i",
                            rec["wfc_idx_1"], rec["wfc_idx_2"],
                            rec["wfc_idx_3"], rec["wfc_idx_4"]))
        f.write(struct.pack("<3i", *rec["G_prime"]))


# ──────────────────────────────────────────────────────────────────────────────
# Main preparation logic
# ──────────────────────────────────────────────────────────────────────────────

def prepare(
    results_folder: str,
    wavecar_files: List[str],
    pairs_csv: List[str],
    auger_type: str,
    dielectric,
    firstCB_index: int,
    lastVB_index: int,
    output_path: str,
    mapping_csv: str | None = None,
    num_matrix_elements: str | int = "all",
    continue_from_files: str | List[str] | None = None,
    T: float = 300.0,
    nd: float = 0.0,
):
    t0 = time.time()

    # ---- Set up AugerCalculator to get derived quantities ----
    ac = AugerCalculator(T=T, nd=nd)
    ac.assign_firstCB_and_lastVB(firstCB_index, lastVB_index)
    ac.import_parsed_BS_data(from_folder=results_folder)
    dielectric_value, dielectric_is_tensor, dielectric_scalar, dielectric_tensor = (
        ut.normalize_dielectric_input(dielectric)
    )
    if dielectric_tensor is None:
        dielectric_tensor_to_write = np.zeros((3, 3), dtype=float)
    else:
        dielectric_tensor_to_write = dielectric_tensor
    ac.dielectric_constant = dielectric_value
    with open(os.path.join(results_folder, "band_info.txt"), "a") as f:
        f.write(
            "dielectric_constant_used_in_matrix_elements "
            f"{json.dumps(ac.dielectric_constant)}\n"
        )

    # Read pairs
    ac.read_auger_pairs(pairs_csv)
    if mapping_csv:
        ac.set_kpoints_mapping(mapping_csv=mapping_csv, auger_type=auger_type)

    # ---- Derived physics parameters (mirror MatrixElements.__init__) ----
    V_m3 = ac.volume * 1e-30
    kB_T = 8.617333262145e-5 * ac.T
    q = eV

    dE_n = ac.Efn - ac.CBM
    if dE_n < 0 or dE_n < 1.5 * kB_T:
        lam_e = np.sqrt(dielectric_scalar * EPSILON_0 * kB_T * q
                        / ((ac.n * 1e6) * q ** 2)) * 1e10
    else:
        lam_e = np.sqrt(dielectric_scalar * EPSILON_0 * dE_n * eV
                        / (1.5 * (ac.n * 1e6) * q ** 2)) * 1e10

    dE_p = ac.VBM - ac.Efp
    if dE_p < 0 or dE_p < 1.5 * kB_T:
        lam_h = np.sqrt(dielectric_scalar * EPSILON_0 * kB_T * q
                        / ((ac.p * 1e6) * q ** 2)) * 1e10
    else:
        lam_h = np.sqrt(dielectric_scalar * EPSILON_0 * dE_p * eV
                        / (1.5 * (ac.p * 1e6) * q ** 2)) * 1e10

    inv_debye = np.sqrt(1.0 / lam_e ** 2 + 1.0 / lam_h ** 2)
    if dielectric_is_tensor:
        print(
            "  Dielectric tensor mode: Debye screening uses "
            f"trace(epsilon)/3 = {dielectric_scalar:.6g}"
        )
    print(f"  Inverse Debye length: {inv_debye:.6f} A^-1")

    if dielectric_is_tensor and abs(dielectric_scalar - 1.0) <= 1e-14:
        a_fit = np.inf
    else:
        a_fit = (dielectric_scalar - 1) ** -1
    b_fit = ALPHA_PENN / ac.q_TF ** 2
    c_fit = HBAR ** 2 / (4 * M_E ** 2 * ac.omega_p ** 2)

    # ---- Open WAVECARs ----
    wfcs = [vwfc.vaspwfc(wf) for wf in wavecar_files]
    true_Bcell = wfcs[0]._Bcell * (2 * np.pi)
    auger_type_int = 0 if auger_type == "eeh" else 1

    # ---- Select pairs ----
    sorted_pairs = sorted(
        ac.auger_pairs_dicts[auger_type],
        key=lambda x: x["probability"], reverse=True,
    )

    if isinstance(continue_from_files, str):
        continue_from_files = [continue_from_files]

    if continue_from_files:
        from auger.matrix_elements import MatrixElements
        done_ids: set = set()
        for cf in continue_from_files:
            if not os.path.exists(cf):
                print(f"  Warning: {cf} not found, skipping.")
                continue
            data = MatrixElements.read_matrix_elements_from_file(cf)
            for m in data:
                if m.get("error") is None:
                    done_ids.add(m["pair_id"])
        sorted_pairs = [p for p in sorted_pairs if p["pair_id"] not in done_ids]
        print(f"  {len(sorted_pairs):,} pairs remaining after skipping {len(done_ids):,}")

    if num_matrix_elements != "all":
        sorted_pairs = sorted_pairs[:max(0, int(num_matrix_elements))]

    print(f"  Preparing {len(sorted_pairs):,} pairs for C++ computation ...")

    # ---- Collect unique wavefunction states ----
    # key = (k_index, band_index, wc_index)  →  sequential ID
    wfc_key_to_idx: Dict[Tuple[int, int, int], int] = {}
    wfc_data_list: List[Tuple[np.ndarray, np.ndarray]] = []  # (G, C) pairs

    def _get_or_load_wfc(k_idx: int, band_idx: int, wc_idx: int) -> int:
        key = (k_idx, band_idx, wc_idx)
        if key in wfc_key_to_idx:
            return wfc_key_to_idx[key]
        idx = len(wfc_data_list)
        wfc_key_to_idx[key] = idx
        G = wfcs[wc_idx].gvectors(ikpt=k_idx + 1)
        C = wfcs[wc_idx].readBandCoeff(ispin=1, ikpt=k_idx + 1,
                                        iband=band_idx + 1, norm=True)
        wfc_data_list.append((G, C))
        return idx

    # Also need G-vectors per (k_index, wc_index) for common_G union.
    # The C++ code will build common_G from the 4 G-vector sets itself.

    # ---- Build pair records ----
    pair_records = []
    for i, pd in enumerate(sorted_pairs):
        # WAVECAR indices (0-based)
        if pd.get("k1_wc_index") is not None:
            wc_k1 = int(pd["k1_wc_index"]) - 1
            wc_k2 = int(pd["k2_wc_index"]) - 1
            wc_k3 = int(pd["k3_wc_index"]) - 1
            wc_k4 = int(pd["k4_wc_index"]) - 1
        else:
            wc_k1 = wc_k2 = wc_k3 = wc_k4 = 0

        k1_i = pd["k1_index"]; E1_i = pd["E1_index"]
        k2_i = pd["k2_index"]; E2_i = pd["E2_index"]
        k3_i = pd["k3_index"]; E3_i = pd["E3_index"]
        k4_i = pd["k4_index"]; E4_i = pd["E4_index"]

        wfc1 = _get_or_load_wfc(k1_i, E1_i, wc_k1)
        wfc2 = _get_or_load_wfc(k2_i, E2_i, wc_k2)
        wfc3 = _get_or_load_wfc(k3_i, E3_i, wc_k3)
        wfc4 = _get_or_load_wfc(k4_i, E4_i, wc_k4)

        # G_prime (Umklapp vector) — same logic as _calc_matrix_element
        k1 = np.asarray(pd["k1"])
        k2 = np.asarray(pd["k2"])
        k3 = np.asarray(pd["k3"])
        k4 = np.asarray(pd["k4"])

        kx_mapped = pd.get("k2_mapped" if auger_type == "eeh" else "k4_mapped")
        kx_mapped_frac = ut.to_fractional_coordinate(kx_mapped, true_Bcell)
        kx_raw = k2 if auger_type == "eeh" else k4
        kx_frac = ut.to_fractional_coordinate(kx_raw, true_Bcell)
        G_prime = np.array([int(round(x)) for x in kx_frac - kx_mapped_frac])

        pair_records.append({
            "pair_id": pd["pair_id"],
            "k1": k1, "k2": k2, "k3": k3, "k4": k4,
            "wfc_idx_1": wfc1, "wfc_idx_2": wfc2,
            "wfc_idx_3": wfc3, "wfc_idx_4": wfc4,
            "G_prime": G_prime,
        })

        if (i + 1) % 500 == 0 or (i + 1) == len(sorted_pairs):
            print(f"    Loaded {i+1:,}/{len(sorted_pairs):,} pairs  "
                  f"({len(wfc_data_list)} unique wfc states)", end="\r")

    print()
    print(f"  Total unique wavefunction states: {len(wfc_data_list):,}")

    # ---- Write binary file ----
    print(f"  Writing {output_path} ...")
    with open(output_path, "wb") as f:
        _write_header(f, auger_type_int, true_Bcell, a_fit, b_fit, c_fit,
                      inv_debye, MATRIX_FACTOR, V_m3, eV,
                      1 if dielectric_is_tensor else 0,
                      dielectric_scalar, dielectric_tensor_to_write)
        _write_wfc_entries(f, wfc_data_list)
        _write_pairs(f, pair_records)

    fsize_mb = os.path.getsize(output_path) / 1e6
    elapsed = time.time() - t0
    print(f"  Done. {fsize_mb:.1f} MB written in {elapsed:.1f}s")
    print(f"  Now run:  ./matrix_element_calc {output_path} output.jsonl [num_threads]")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Prepare binary input for the C++ matrix-element calculator.")
    p.add_argument("--results_folder", required=True,
                   help="Folder with band_info.txt and parsed data")
    p.add_argument("--wavecar_files", nargs="+", required=True,
                   help="Paths to WAVECAR files (order must match kpoints_mapping)")
    p.add_argument("--pairs_csv", nargs="+", required=True,
                   help="Auger pairs CSV file(s)")
    p.add_argument("--auger_type", required=True, choices=["eeh", "ehh"])
    p.add_argument(
        "--dielectric",
        type=_parse_dielectric_cli,
        required=True,
        help="Scalar dielectric or JSON 3x3 tensor, e.g. 16.8 or '[[16,0,0],[0,15,0],[0,0,14]]'",
    )
    p.add_argument("--firstCB_index", type=int, required=True)
    p.add_argument("--lastVB_index", type=int, required=True)
    p.add_argument("--mapping_csv", default=None,
                   help="kpoints_mapping.csv (Deprecated; not needed anymore)")
    p.add_argument("--output", default="cpp_input.bin")
    p.add_argument("--num_matrix_elements", default="all")
    p.add_argument("--continue_from", nargs="*", default=None,
                   help="JSONL file(s) with already-computed elements to skip")
    p.add_argument("--T", type=float, default=300.0, help="Temperature (K)")
    p.add_argument("--nd", type=float, default=0.0, help="Doping concentration")
    args = p.parse_args()

    n_me = args.num_matrix_elements
    if n_me != "all":
        n_me = int(n_me)

    prepare(
        results_folder=args.results_folder,
        wavecar_files=args.wavecar_files,
        pairs_csv=args.pairs_csv,
        auger_type=args.auger_type,
        dielectric=args.dielectric,
        firstCB_index=args.firstCB_index,
        lastVB_index=args.lastVB_index,
        mapping_csv=args.mapping_csv,
        output_path=args.output,
        num_matrix_elements=n_me,
        continue_from_files=args.continue_from,
        T=args.T,
        nd=args.nd,
    )


if __name__ == "__main__":
    main()
