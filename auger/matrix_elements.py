"""
MatrixElements — Coulomb matrix element |M|² calculation.

Computes the screened Coulomb interaction matrix elements for Auger
transitions using plane-wave coefficients read from VASP WAVECAR files
(via *pyvaspwfc*).

The heavy lifting runs in a multiprocessing ``Pool`` (spawn context) so that
each worker loads its own WAVECAR handle.  Three caches (G-vectors, plane-wave
coefficients, and coefficient dictionaries) are kept per-worker to avoid
redundant I/O.

Direct, exchange, and interference terms are all included::

    |M|² = |M_d|² + |M_x|² + |M_d - M_x|²  (× prefactor)
"""

from __future__ import annotations

import json
import os
from multiprocessing import cpu_count, get_context
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import vaspwfc as vwfc

from . import utilities as ut
from .constants import (
    ALPHA_PENN,
    ANGSTROM,
    EPSILON_0,
    HBAR,
    M_E,
    MATRIX_FACTOR,
    eV,
)


# ======================================================================
# Worker-level globals (initialised once per spawned process)
# ======================================================================
_wfcs: list = []
_Gvec_cache: Dict = {}
_Cg_cache: Dict = {}
_Dict_cache: Dict = {}


def _init_worker(wavecar_files: List[str]):
    """Initialiser for each Pool worker — opens WAVECAR handles."""
    global _wfcs
    _wfcs = [vwfc.vaspwfc(f) for f in wavecar_files]


def _get_Gvec(ikpt: int, wc_idx: int = 0) -> np.ndarray:
    key = (ikpt, wc_idx)
    if key not in _Gvec_cache:
        _Gvec_cache[key] = _wfcs[wc_idx].gvectors(ikpt=ikpt + 1)
    return _Gvec_cache[key]


def _get_Cg(ikpt: int, iband: int, wc_idx: int = 0) -> np.ndarray:
    key = (ikpt, iband, wc_idx)
    if key not in _Cg_cache:
        _Cg_cache[key] = _wfcs[wc_idx].readBandCoeff(
            ispin=1, ikpt=ikpt + 1, iband=iband + 1, norm=True
        )
    return _Cg_cache[key]


def _get_coeff_dict(ikpt: int, iband: int, wc_idx: int = 0) -> dict:
    key = (ikpt, iband, wc_idx)
    if key not in _Dict_cache:
        G = _get_Gvec(ikpt, wc_idx)
        C = _get_Cg(ikpt, iband, wc_idx)
        _Dict_cache[key] = {tuple(G[n]): C[n] for n in range(len(G))}
    return _Dict_cache[key]


# ======================================================================
# Worker function  (top-level so it can be pickled)
# ======================================================================
def _calc_matrix_element(args) -> dict:
    """
    Compute |M|² for one pair.  Runs inside a Pool worker.

    Returns ``{"pair_id": ..., "|M|^2": ...}`` on success,
    or ``{"pair_id": ..., "error": ...}`` on failure.
    """
    (pair_dict, auger_type, dielectric, inv_debye, matrix_factor,
     V_m3, a_fit, b_fit, c_fit, true_Bcell) = args

    try:
        pid = pair_dict["pair_id"]
        # Determine per-k WAVECAR indices (0-based for _wfcs list)
        if pair_dict.get("k1_wc_index") is not None:
            # Exact-kpoint path: k#_wc_index is 1-based → convert to 0-based
            wc_k1 = int(pair_dict["k1_wc_index"]) - 1
            wc_k2 = int(pair_dict["k2_wc_index"]) - 1
            wc_k3 = int(pair_dict["k3_wc_index"]) - 1
            wc_k4 = int(pair_dict["k4_wc_index"]) - 1
        else:
            # Non-exact pairs (nearest-kpoint): all k-points come from first WAVECAR
            wc_k1 = wc_k2 = wc_k3 = wc_k4 = 0
        for _lbl, _wi in (("k1", wc_k1), ("k2", wc_k2), ("k3", wc_k3), ("k4", wc_k4)):
            if _wi >= len(_wfcs):
                return {"pair_id": pid,
                        "error": f"{_lbl}_wc_index={_wi+1} >= {len(_wfcs)} WAVECAR files"}

        k1 = np.asarray(pair_dict["k1"])
        k2 = np.asarray(pair_dict["k2"])
        k3 = np.asarray(pair_dict["k3"])
        k4 = np.asarray(pair_dict["k4"])

        # Umklapp vector
        kx_mapped = pair_dict.get("k2_mapped" if auger_type == "eeh" else "k4_mapped")
        kx_mapped_frac = ut.to_fractional_coordinate(kx_mapped, true_Bcell)
        kx_raw = k2 if auger_type == "eeh" else k4
        kx_frac = ut.to_fractional_coordinate(kx_raw, true_Bcell)
        G_prime = np.array([int(round(x)) for x in kx_frac - kx_mapped_frac])

        k1_i = pair_dict["k1_index"]
        k2_i = pair_dict["k2_index"]
        k3_i = pair_dict["k3_index"]
        k4_i = pair_dict["k4_index"]
        E1_i = pair_dict["E1_index"]
        E2_i = pair_dict["E2_index"]
        E3_i = pair_dict["E3_index"]
        E4_i = pair_dict["E4_index"]

        # Each k-point reads from its own WAVECAR folder (wc_k1..wc_k4 already set above)
        G1 = _get_Gvec(k1_i, wc_k1)
        G2 = _get_Gvec(k2_i, wc_k2)
        G3 = _get_Gvec(k3_i, wc_k3)
        G4 = _get_Gvec(k4_i, wc_k4)

        common_G = np.unique(np.vstack((G1, G2, G3, G4)), axis=0)

        d1 = _get_coeff_dict(k1_i, E1_i, wc_k1)
        d2 = _get_coeff_dict(k2_i, E2_i, wc_k2)
        d3 = _get_coeff_dict(k3_i, E3_i, wc_k3)
        d4 = _get_coeff_dict(k4_i, E4_i, wc_k4)

        Md_sum = 0.0 + 0.0j
        Mx_sum = 0.0 + 0.0j

        for G in common_G:
            G_dot_B = np.dot(G, true_Bcell)
            Gp_minus_G = G_prime - G

            if auger_type == "eeh":
                I_34 = ut.I_ab(G, G3, d3, d4)
                I_12 = ut.I_ab(Gp_minus_G, G1, d1, d2)
                I_32 = ut.I_ab(G, G3, d3, d2)
                I_14 = ut.I_ab(Gp_minus_G, G1, d1, d4)
                arg_d = np.linalg.norm(k3 - k4 + G_dot_B)
                arg_x = np.linalg.norm(k3 - k2 + G_dot_B)
            else:
                I_34 = ut.I_ab(G, G2, d2, d1)   # I_21
                I_12 = ut.I_ab(Gp_minus_G, G3, d3, d4)  # I_34
                I_32 = ut.I_ab(G, G2, d2, d4)   # I_24
                I_14 = ut.I_ab(Gp_minus_G, G3, d3, d1)  # I_31
                arg_d = np.linalg.norm(k2 - k1 + G_dot_B)
                arg_x = np.linalg.norm(k2 - k4 + G_dot_B)

            eps_d = ut.calculate_epsilon(arg_d, a_fit, b_fit, c_fit)
            eps_x = ut.calculate_epsilon(arg_x, a_fit, b_fit, c_fit)
            Wd = ut.W(arg_d, eps_d, inv_debye)
            Wx = ut.W(arg_x, eps_x, inv_debye)

            Md_sum += I_34 * I_12 * Wd
            Mx_sum += I_32 * I_14 * Wx

        Md2 = np.abs(Md_sum) ** 2
        Mx2 = np.abs(Mx_sum) ** 2
        Mdx2 = np.abs(Md_sum - Mx_sum) ** 2
        M2 = (Md2 + Mx2 + Mdx2) * matrix_factor ** 2 / (V_m3 ** 2 * eV ** 2)

        return {"pair_id": pid, "|M|^2": float(M2)}

    except Exception as exc:  # noqa: BLE001
        return {"pair_id": pair_dict.get("pair_id", "unknown"), "error": str(exc)}


# ======================================================================
# MatrixElements class
# ======================================================================
class MatrixElements:
    """
    Manages parallel computation of Coulomb matrix elements for Auger pairs.

    Parameters
    ----------
    auger_instance : AugerCalculator
        The parent calculator (provides band data, volume, etc.).
    auger_type : {'eeh', 'ehh'}
    dielectric : float
        Macroscopic dielectric constant.
    wavecar_files : str or list[str]
        Path(s) to WAVECAR files.
    """

    def __init__(
        self,
        auger_instance,
        auger_type: str,
        dielectric: float,
        wavecar_files: Union[str, List[str]] = "WAVECAR",
    ):
        self.auger_type = auger_type
        self.auger = auger_instance
        self.dielectric = dielectric

        if isinstance(wavecar_files, str):
            wavecar_files = [wavecar_files]
        for wf in wavecar_files:
            if not os.path.exists(wf):
                raise FileNotFoundError(f"WAVECAR not found: {wf}")
        self.wavecar_files = wavecar_files

        self.V_m3 = auger_instance.volume * 1e-30
        self.inverse_debye = self._compute_debye_screening()

    # ---- Debye screening ----
    def _compute_debye_screening(self) -> float:
        """Compute combined electron + hole inverse Debye length (Å⁻¹)."""
        ai = self.auger
        kB_T = 8.617333262145e-5 * ai.T  # eV
        q = eV

        # Electron contribution
        dE_n = ai.Efn - ai.CBM
        if dE_n < 0 or dE_n < 1.5 * kB_T:
            lam_e = np.sqrt(self.dielectric * EPSILON_0 * kB_T * q
                            / ((ai.n * 1e6) * q ** 2)) * 1e10
        else:
            lam_e = np.sqrt(self.dielectric * EPSILON_0 * dE_n * eV
                            / (1.5 * (ai.n * 1e6) * q ** 2)) * 1e10

        # Hole contribution
        dE_p = ai.VBM - ai.Efp
        if dE_p < 0 or dE_p < 1.5 * kB_T:
            lam_h = np.sqrt(self.dielectric * EPSILON_0 * kB_T * q
                            / ((ai.p * 1e6) * q ** 2)) * 1e10
        else:
            lam_h = np.sqrt(self.dielectric * EPSILON_0 * dE_p * eV
                            / (1.5 * (ai.p * 1e6) * q ** 2)) * 1e10

        inv = np.sqrt(1.0 / lam_e ** 2 + 1.0 / lam_h ** 2)
        print(f"  Inverse Debye length: {inv:.6f} Å⁻¹")
        return inv

    # ---- I/O ----
    @staticmethod
    def read_matrix_elements_from_file(file_path: str) -> List[dict]:
        """Read a JSONL file of matrix elements."""
        with open(file_path, "r") as f:
            return [json.loads(line.strip()) for line in f]

    # ---- Main parallel calculation ----
    def calculate_matrix_elements_parallel(
        self,
        wavecar_files: Union[str, List[str]],
        num_matrix_elements: Union[int, str] = "all",
        add_suffix_name: str = "",
        continue_from_files: Union[str, List[str]] = [],
    ) -> List[dict]:
        """
        Compute |M|² for all (or top-N) pairs in parallel.

        Parameters
        ----------
        wavecar_files : str or list[str]
        num_matrix_elements : int or 'all'
        add_suffix_name : str
        continue_from_files : str or list[str]
            JSONL files with previously computed elements (to skip).

        Returns
        -------
        list[dict]
            Each dict has ``"pair_id"`` and ``"|M|^2"`` (in eV²).
        """
        if isinstance(wavecar_files, str):
            wavecar_files = [wavecar_files]
        if isinstance(continue_from_files, str):
            continue_from_files = [continue_from_files]

        suffix = f"_{add_suffix_name}" if add_suffix_name else ""
        out_dir = self.auger.results_folder.rstrip("/")
        output_file = os.path.join(
            out_dir, f"{self.auger_type}_matrix_elements_{self.auger.XX}{suffix}.jsonl"
        )

        sorted_pairs = sorted(
            self.auger.auger_pairs_dicts[self.auger_type],
            key=lambda x: x["probability"], reverse=True,
        )

        # ---- Skip already-computed pairs ----
        calculated: List[dict] = []
        if continue_from_files:
            done_ids: set = set()
            for cf in continue_from_files:
                if not os.path.exists(cf):
                    print(f"  Warning: {cf} not found, skipping.")
                    continue
                data = self.read_matrix_elements_from_file(cf)
                for m in data:
                    if m.get("error") is None:
                        done_ids.add(m["pair_id"])
                        calculated.append({"pair_id": m["pair_id"], "|M|^2": m["|M|^2"]})
            sorted_pairs = [p for p in sorted_pairs if p["pair_id"] not in done_ids]
            print(f"  {len(sorted_pairs):,} pairs remaining after skipping {len(done_ids):,}")

        if num_matrix_elements != "all":
            sorted_pairs = sorted_pairs[:max(0, num_matrix_elements)]

        # ---- Validate WAVECAR configuration ----
        if sorted_pairs and sorted_pairs[0].get("k1_wc_index") is not None:
            # Exact-kpoint path: k#_wc_index is 1-based; find highest folder index used
            max_wc_1based = max(
                (int(p.get(f"k{i}_wc_index", 1) or 1)
                 for p in sorted_pairs for i in range(1, 5)),
                default=1,
            )
            if max_wc_1based > len(wavecar_files):
                raise ValueError(
                    f"Pairs require k#_wc_index up to {max_wc_1based}, but only "
                    f"{len(wavecar_files)} WAVECAR file(s) provided."
                )

        n_workers = max(1, cpu_count() - 1)
        print(f"\n  Computing {len(sorted_pairs):,} matrix elements on {n_workers} cores …")

        # ---- Penn-model dielectric fit parameters ----
        a = (self.dielectric - 1) ** -1
        b = ALPHA_PENN / self.auger.q_TF ** 2
        c = HBAR ** 2 / (4 * M_E ** 2 * self.auger.omega_p ** 2)

        wfc0 = vwfc.vaspwfc(self.wavecar_files[0])
        true_Bcell = wfc0._Bcell * (2 * np.pi)

        args_list = [
            (d, self.auger_type, self.dielectric, self.inverse_debye,
             MATRIX_FACTOR, self.V_m3, a, b, c, true_Bcell)
            for d in sorted_pairs
        ]

        # ---- Write previously-known elements first ----
        if calculated:
            with open(output_file, "a") as f:
                for r in calculated:
                    f.write(json.dumps(r) + "\n")

        # ---- Pool execution ----
        results: List[dict] = []
        total = len(args_list)
        with open(output_file, "a") as f:
            ctx = get_context("spawn")
            with ctx.Pool(processes=n_workers,
                          initializer=_init_worker,
                          initargs=(wavecar_files,)) as pool:
                for result in pool.imap_unordered(_calc_matrix_element, args_list):
                    f.write(json.dumps(result) + "\n")
                    results.append(result)
                    f.flush()
                    done = len(results)
                    if done % 100 == 0 or done == total:
                        pct = done / total * 100
                        avg = np.mean([r.get("|M|^2", 0) for r in results[-100:]])
                        print(f"    [{done:6d}/{total}] {pct:5.1f}%  avg |M|²={avg:.4e} eV²",
                              end="\r")

        print(f"\n  Saved → {output_file}")
        return results + calculated
