"""
PairGenerator — Auger scattering channel (pair) identification and ranking.

Identifies all four-particle transition channels (E1, E2, E3, E4) for both
eeh (electron-electron-hole) and ehh (electron-hole-hole) Auger processes,
using one of two momentum-conservation approaches:

  * ``nearest_kpoint``  — map to closest SCF k-point
  * ``exact_kpoint``    — use NSCF-computed off-grid k-point

Two search algorithms are available:

  * ``Brute_Force`` — enumerate all combinations (exact but slow)
  * ``Max_Heap``    — priority-queue walk over the top-N most probable
"""

from __future__ import annotations

import ast
import heapq
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count, get_context
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from . import utilities as ut


# ======================================================================
# Pair data class
# ======================================================================
class Pair:
    """
    Container for a single Auger scattering channel.

    Can be constructed from a positional tuple (internal use) **or** from
    a ``dict`` (when reading back from CSV).
    """

    __slots__ = (
        "pair_id", "E1", "E2", "E3", "E4",
        "k1", "k2", "k3", "k4",
        "kw1", "kw2", "kw3", "kw4",
        "E1_index", "E2_index", "E3_index", "E4_index",
        "k1_index", "k2_index", "k3_index", "k4_index",
        "k1_nscf_index", "k2_nscf_index", "k3_nscf_index", "k4_nscf_index",
        "probability", "pair_type",
        "matrix_element", "mapped_kpoints",
        "k1_wc_index", "k2_wc_index", "k3_wc_index", "k4_wc_index",
    )

    def __init__(self, arg):
        if isinstance(arg, dict):
            self._init_from_dict(arg)
        else:
            self._init_from_tuple(arg)

    # ---- construction helpers ----
    def _init_from_tuple(self, t):
        (self.pair_id, self.E1, self.E2, self.E3, self.E4,
         self.k1, self.k2, self.k3, self.k4,
         self.kw1, self.kw2, self.kw3, self.kw4,
         self.E1_index, self.E2_index, self.E3_index, self.E4_index,
         self.k1_index, self.k2_index, self.k3_index, self.k4_index,
         self.probability, self.pair_type) = t
        self.matrix_element = None
        self.mapped_kpoints = [None, None, None, None]
        self.k1_nscf_index = None
        self.k2_nscf_index = None
        self.k3_nscf_index = None
        self.k4_nscf_index = None
        self.k1_wc_index = None
        self.k2_wc_index = None
        self.k3_wc_index = None
        self.k4_wc_index = None

    def _init_from_dict(self, d: dict):
        self.pair_id = d.get("pair_id")
        self.pair_type = d.get("pair_type")
        self.probability = d.get("probability")
        for attr in ("E1", "E2", "E3", "E4"):
            setattr(self, attr, d.get(attr))
        for attr in ("k1", "k2", "k3", "k4"):
            val = d.get(attr, [])
            setattr(self, attr, np.array(val) if val is not None else np.array([]))
        for attr in ("kw1", "kw2", "kw3", "kw4",
                     "E1_index", "E2_index", "E3_index", "E4_index",
                     "k1_index", "k2_index", "k3_index", "k4_index"):
            setattr(self, attr, d.get(attr))
        self.matrix_element = d.get("|M|^2")
        self.mapped_kpoints = [
            np.array(d[f"k{i}_mapped"]) if d.get(f"k{i}_mapped") is not None else None
            for i in range(1, 5)
        ]
        self.k1_nscf_index = d.get("k1_nscf_index")
        self.k2_nscf_index = d.get("k2_nscf_index")
        self.k3_nscf_index = d.get("k3_nscf_index")
        self.k4_nscf_index = d.get("k4_nscf_index")
        self.k1_wc_index = d.get("k1_wc_index")
        self.k2_wc_index = d.get("k2_wc_index")
        self.k3_wc_index = d.get("k3_wc_index")
        self.k4_wc_index = d.get("k4_wc_index")

    # ---- setters ----
    def set_matrix_element(self, val: float):
        self.matrix_element = val

    def set_mapped_kpoints(self, k1m, k2m, k3m, k4m):
        self.mapped_kpoints = [k1m, k2m, k3m, k4m]

    # ---- serialisation ----
    def get_pair_as_dict(self) -> dict:
        """Return a flat dict suitable for CSV / JSON output."""
        d: Dict[str, Any] = {
            "pair_id": self.pair_id,
            "pair_type": self.pair_type,
            "E1_index": self.E1_index, "E2_index": self.E2_index,
            "E3_index": self.E3_index, "E4_index": self.E4_index,
            "k1_index": self.k1_index, "k2_index": self.k2_index,
            "k3_index": self.k3_index, "k4_index": self.k4_index,
            "E1": self.E1, "E2": self.E2, "E3": self.E3, "E4": self.E4,
            "k1": _to_list(self.k1), "k2": _to_list(self.k2),
            "k3": _to_list(self.k3), "k4": _to_list(self.k4),
            "kw1": self.kw1, "kw2": self.kw2,
            "kw3": self.kw3, "kw4": self.kw4,
        }
        # Store the one non-None mapped k-point
        for i, km in enumerate(self.mapped_kpoints):
            if km is not None:
                d[f"k{i+1}_mapped"] = _to_list(km)
                break
        d["probability"] = self.probability
        if self.matrix_element is not None:
            d["|M|^2"] = self.matrix_element
        if self.k1_nscf_index is not None:
            d["k1_nscf_index"] = self.k1_nscf_index
        if self.k2_nscf_index is not None:
            d["k2_nscf_index"] = self.k2_nscf_index
        if self.k3_nscf_index is not None:
            d["k3_nscf_index"] = self.k3_nscf_index
        if self.k4_nscf_index is not None:
            d["k4_nscf_index"] = self.k4_nscf_index
        if self.k1_wc_index is not None:
            d["k1_wc_index"] = self.k1_wc_index
        if self.k2_wc_index is not None:
            d["k2_wc_index"] = self.k2_wc_index
        if self.k3_wc_index is not None:
            d["k3_wc_index"] = self.k3_wc_index
        if self.k4_wc_index is not None:
            d["k4_wc_index"] = self.k4_wc_index
        return d


def _to_list(arr):
    """Convert ndarray to list safely."""
    try:
        return arr.tolist()
    except AttributeError:
        return list(arr) if arr is not None else []


# ======================================================================
# PairGenerator
# ======================================================================
class PairGenerator:
    """
    Generate and rank Auger scattering pairs.

    Parameters
    ----------
    auger_type : {'eeh', 'ehh'}
    arg : tuple or None
        ``(auger_instance, CB_window, VB_window, approach, is_parallel,
        search_mode, num_top_pairs, table_name_suffix, poscar_path)``
        or *None* when constructing from stored CSV data.
    """

    def __init__(self, auger_type: str, arg=None):
        self.auger_type = auger_type
        self.pairs: List[Pair] = []
        self.marg_E1: Dict[str, dict] = {auger_type: {}}
        self.marg_E2: Dict[str, dict] = {auger_type: {}}
        self.marg_E3: Dict[str, dict] = {auger_type: {}}
        self.marg_E4: Dict[str, dict] = {auger_type: {}}

        if arg is not None:
            (self.auger_instance, self.CB_window, self.VB_window,
             self.approach, self.is_parallel, self.search_mode,
             self.num_top_pairs, table_name_suffix, poscar_path, is_Expand) = arg
            self.table_name_suffix = f"_{table_name_suffix}" if table_name_suffix else ""
            if is_Expand:
                if poscar_path is None:
                    raise ValueError("poscar_path is required for k-point expansion.")
                self._expand_irr_kpoints(poscar_path)

            # exact_kpoint + no expansion = post-NSCF pair building from CSV,
            # which reads all data from the CSV directly — no energy-state
            # initialisation needed.  All other cases (nearest_kpoint, or
            # exact_kpoint with expansion for the k-point list step) require it.
            if not (self.approach == "exact_kpoint" and not is_Expand):
                self._initialise_energy_states()
        else:
            self.table_name_suffix = ""

    # ------------------------------------------------------------------
    # Energy-state initialisation
    # ------------------------------------------------------------------
    def _expand_irr_kpoints(self, poscar_path: str):
        """
        Expand irreducible-wedge k-points to the full Brillouin zone.

        Uses spglib (via pymatgen) to find all symmetry-equivalent k-points
        for each irreducible k-point.  Eigenvalues are duplicated — they are
        identical for symmetry-related k-points — and the k-point weights are
        all set to 1 (uniform weighting over the full mesh).

        After this call the following attributes of ``self.auger_instance``
        are overwritten in-place:

        * ``data_energies``  — shape ``(nbands, N_full)``
        * ``kpoints``        — shape ``(N_full, 3)`` (Cartesian, Å⁻¹)
        * ``kpoints_weights`` — all ones, length ``N_full``
        * ``num_of_kpoints`` — updated to ``N_full``

        Parameters
        ----------
        poscar_path : str
            Path to the VASP POSCAR (or any pymatgen-readable structure file)
            that was used for the SCF calculation.
        """
        ai = self.auger_instance
        rl = ai.reciprocal_lattice                      # (3, 3) reciprocal lattice

        # ── 1. Parse kgrid from calculator ───────────────────────────
        kgrid = ai.kgrid                                 # may be tuple, list, or string
        if isinstance(kgrid, str):
            kgrid = ast.literal_eval(kgrid)
        mesh_dims = tuple(int(x) for x in kgrid)

        # ── 2. Full BZ mesh + irreducible mapping via spglib ─────────
        structure = Structure.from_file(poscar_path)
        analyzer = SpacegroupAnalyzer(structure)
        full_kpts_frac, ir_mapping = analyzer.get_ir_reciprocal_mesh_map(mesh_dims)
        # full_kpts_frac : (N_full, 3) fractional
        # ir_mapping     : (N_full,)   each entry = index of its irr. rep.

        n_full = len(full_kpts_frac)

        # ── 3. Build irr_index → [full_indices] lookup ───────────────
        #    irr_to_full[j] = list of full-mesh indices whose
        #    irreducible representative is j.
        irr_to_full: Dict[int, List[int]] = defaultdict(list)
        for fi, ii in enumerate(ir_mapping):
            irr_to_full[int(ii)].append(fi)

        # Unique irreducible indices (these are positions in full_kpts_frac)
        irr_indices = np.array(sorted(irr_to_full.keys()))  # (N_irr,)

        # ── 4. Match stored k-points → irreducible mesh indices ──────
        #    stored k-points come from VASP (irreducible wedge, Cartesian).
        #    Convert both sets to folded fractional coords and match.
        stored_kpts_frac = np.array([
            ut.to_fractional_coordinate(k, rl) for k in ai.kpoints
        ])  # (N_stored, 3)

        irr_kpts_frac = full_kpts_frac[irr_indices]  # (N_irr_mesh, 3)

        # Fold both sets to the same BZ convention for reliable matching
        stored_folded = np.array([
            ut.fold_kpoint_to_first_bz(k, convention="vasp_centered")
            for k in stored_kpts_frac
        ])
        irr_folded = np.array([
            ut.fold_kpoint_to_first_bz(k, convention="vasp_centered")
            for k in irr_kpts_frac
        ])
        
        # Vectorised nearest-neighbour in fractional coords (PBC-aware)
        # diff shape: (N_stored, N_irr_mesh, 3)
        diff = stored_folded[:, None, :] - irr_folded[None, :, :]
        diff -= np.round(diff)                              # periodic images
        dist2 = np.sum(diff ** 2, axis=2)                   # (N_stored, N_irr_mesh)
        best_match = np.argmin(dist2, axis=1)               # (N_stored,)

        # Map stored index → irreducible mesh index
        stored_to_irr_idx = irr_indices[best_match]         # (N_stored,)

        # Sanity check: every match should be very close
        min_dists = np.sqrt(np.min(dist2, axis=1))
        bad = np.where(min_dists > 0.05)[0]
        if len(bad) > 0:
            print(f"  ⚠  {len(bad)} stored k-points could not be matched "
                  f"to the spglib mesh (max err = {min_dists[bad].max():.4f}).")

        # ── 5. Build expanded arrays ─────────────────────────────────
        nbands = ai.data_energies.shape[0]

        new_energies = np.empty((nbands, n_full), dtype=ai.data_energies.dtype)
        new_kpoints_cart = np.empty((n_full, 3), dtype=np.float64)

        # Pre-compute Cartesian coords for the full mesh (vectorised)
        full_kpts_cart = full_kpts_frac @ rl              # (N_full, 3)

        for si in range(len(ai.kpoints)):
            irr_idx = int(stored_to_irr_idx[si])
            full_idxs = irr_to_full[irr_idx]              # all sym-equiv.
            new_energies[:, full_idxs] = ai.data_energies[:, si, np.newaxis]
            new_kpoints_cart[full_idxs] = full_kpts_cart[full_idxs]

        new_weights = np.full(n_full, 1.0 / (mesh_dims[0] * mesh_dims[1] * mesh_dims[2]))

        # ── 6. Overwrite auger_instance arrays ───────────────────────
        n_stored = len(ai.kpoints)
        ai.data_energies = new_energies
        ai.kpoints = new_kpoints_cart
        ai.kpoints_weights = new_weights
        ai.num_of_kpoints = n_full

        print(f"  Expanded k-points: {n_stored} irr → {n_full} full BZ  "
              f"(mesh {mesh_dims[0]}×{mesh_dims[1]}×{mesh_dims[2]})")
    
    def _initialise_energy_states(self):
        """
        Collect CB / VB states inside the energy windows and sort by
        Fermi–Dirac occupation probability.
        """
        print(f"\n  Initializing energy states …")
        print(f"    Type: {self.auger_type.upper()}  |  Approach: {self.approach}  |  Search: {self.search_mode}")

        ai = self.auger_instance
        self.E1_energies: list = []
        self.E2_energies: list = []
        self.E3_energies: list = []
        self.E4_energies: list = []

        cb_lo, cb_hi = ai.CBM, ai.CBM + self.CB_window
        vb_lo, vb_hi = ai.VBM - self.VB_window, ai.VBM

        for band_idx, band in enumerate(ai.data_energies):
            for k_idx, energy in enumerate(band):
                kpt = ai.kpoints[k_idx]
                kw = ai.kpoints_weights[k_idx]

                in_cb = cb_lo <= energy <= cb_hi
                in_vb = vb_lo <= energy <= vb_hi

                if in_cb:
                    d_e = _state_dict(band_idx, k_idx, energy, kpt, kw,
                                      ut.fermi_dirac(energy, ai.Efn, ai.T))
                    if self.auger_type == "eeh":
                        self.E1_energies.append(d_e)
                        self.E3_energies.append(d_e)
                        self.marg_E1[self.auger_type][(band_idx, k_idx)] = d_e
                        self.marg_E3[self.auger_type][(band_idx, k_idx)] = d_e
                    else:  # ehh
                        self.E1_energies.append(d_e)
                        self.marg_E1[self.auger_type][(band_idx, k_idx)] = d_e

                if in_vb:
                    d_h = _state_dict(band_idx, k_idx, energy, kpt, kw,
                                      1.0 - ut.fermi_dirac(energy, ai.Efp, ai.T))
                    if self.auger_type == "eeh":
                        self.E4_energies.append(d_h)
                        self.marg_E4[self.auger_type][(band_idx, k_idx)] = d_h
                    else:  # ehh
                        self.E2_energies.append(d_h)
                        self.E3_energies.append(d_h)
                        self.marg_E2[self.auger_type][(band_idx, k_idx)] = d_h
                        self.marg_E3[self.auger_type][(band_idx, k_idx)] = d_h

        # Sort each list by probability (descending) — required for Max_Heap
        for lst in (self.E1_energies, self.E2_energies,
                    self.E3_energies, self.E4_energies):
            lst.sort(key=lambda x: x["P"], reverse=True)

        # Summary
        if self.auger_type == "eeh":
            n1, n3, n4 = len(self.E1_energies), len(self.E3_energies), len(self.E4_energies)
            print(f"    E1(e): {n1:,}  |  E3(e): {n3:,}  |  E4(h): {n4:,}  |  combos: {n1*n3*n4:,}")
        else:
            n1, n2, n3 = len(self.E1_energies), len(self.E2_energies), len(self.E3_energies)
            print(f"    E1(e): {n1:,}  |  E2(h): {n2:,}  |  E3(h): {n3:,}  |  combos: {n1*n2*n3:,}")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def add_pair(self, pair: Pair):
        self.pairs.append(pair)

    def get_pairs(self, is_sorted: bool = True) -> List[dict]:
        if not self.pairs:
            raise ValueError("No pairs generated yet.")
        dicts = [p.get_pair_as_dict() for p in self.pairs]
        if is_sorted:
            dicts.sort(key=lambda x: x["probability"], reverse=True)
        return dicts

    def exclude_calculated_pairs(self, calculated_pairs: List[dict]):
        """Remove pairs whose ``pair_id`` appears in *calculated_pairs*."""
        ids = {p["pair_id"] for p in calculated_pairs}
        before = len(self.pairs)
        self.pairs = [p for p in self.pairs if p.pair_id not in ids]
        print(f"  Excluded {before - len(self.pairs):,} pairs  |  remaining: {len(self.pairs):,}")

    def read_pairs_from_csv(self, file_paths: Union[str, List[str]]) -> List[dict]:
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        raw = ut.read_csv(file_paths)
        self.pairs = [Pair(d) for d in raw]
        print(f"  Read {len(self.pairs):,} pairs from CSV")
        return raw

    def write_pairs_to_csv(self, to_path: str = "", is_sorted: bool = True) -> List[dict]:
        if not self.pairs:
            raise ValueError("No pairs to write.")
        data = self.get_pairs(is_sorted=is_sorted)
        name = f"auger_{self.auger_type}_pairs_{self.auger_instance.XX}{self.table_name_suffix}"
        self._write_pairs_chunked(data, name, to_path=to_path, checkpoint_only_current_chunk=False)
        print(f"  Saved {len(data):,} pairs → {name}_*.csv")
        return data

    def _write_pairs_chunked(
        self,
        data: List[dict],
        base_name: str,
        *,
        to_path: str = "",
        checkpoint_only_current_chunk: bool = False,
        chunk_size: int = 1_000_000,
    ) -> None:
        """Write pair tables as indexed CSV chunks ``..._1.csv``, ``..._2.csv``, ...

        When ``checkpoint_only_current_chunk`` is True, only the currently-active
        chunk file is rewritten (used for periodic 10k checkpoints to avoid
        touching earlier completed chunk files).
        """
        if to_path and not to_path.endswith("/"):
            to_path += "/"

        n = len(data)
        if n == 0:
            return

        # Always index from _1, even when total rows < chunk_size.
        active_idx = (n - 1) // chunk_size + 1

        if checkpoint_only_current_chunk:
            start = (active_idx - 1) * chunk_size
            end = n
            ut.write_to_csv(data[start:end], f"{base_name}_{active_idx}", folder_to_save=to_path)
            return

        # Full write: refresh all chunk files from _1.._N and remove stale extras.
        legacy_single_file = os.path.join(to_path.rstrip("/") if to_path else ".", f"{base_name}.csv")
        if os.path.exists(legacy_single_file):
            os.remove(legacy_single_file)

        parts = (n + chunk_size - 1) // chunk_size
        for i in range(1, parts + 1):
            start = (i - 1) * chunk_size
            end = min(i * chunk_size, n)
            ut.write_to_csv(data[start:end], f"{base_name}_{i}", folder_to_save=to_path)

        # Remove leftover chunk files from older larger writes.
        if to_path:
            folder = to_path.rstrip("/")
        else:
            folder = "."
        prefix = f"{base_name}_"
        suffix = ".csv"
        for fn in os.listdir(folder):
            if not (fn.startswith(prefix) and fn.endswith(suffix)):
                continue
            idx_part = fn[len(prefix):-len(suffix)]
            if idx_part.isdigit() and int(idx_part) > parts:
                os.remove(os.path.join(folder, fn))

    def _write_checkpoint_current_chunk(self, to_path: str = "", chunk_size: int = 1_000_000) -> None:
        """Write only the currently active chunk file during progressive generation."""
        n = len(self.pairs)
        if n == 0:
            return
        if to_path and not to_path.endswith("/"):
            to_path += "/"

        idx = (n - 1) // chunk_size + 1
        start = (idx - 1) * chunk_size
        chunk_dicts = [p.get_pair_as_dict() for p in self.pairs[start:n]]
        name = f"auger_{self.auger_type}_pairs_{self.auger_instance.XX}{self.table_name_suffix}_{idx}"
        ut.write_to_csv(chunk_dicts, name, folder_to_save=to_path)

    # ------------------------------------------------------------------
    # Momentum-conservation helpers
    # ------------------------------------------------------------------
    def find_closest_energy(self, energy: float, energies) -> Tuple[int, int, float]:
        """Return ``(band_index, k_index, energy)`` of the closest state."""
        best_band, best_k, best_E = 0, 0, 0.0
        min_diff = float("inf")
        for bi, band in enumerate(energies):
            for ki, e in enumerate(band):
                d = abs(e - energy)
                if d < min_diff:
                    min_diff = d
                    best_band, best_k, best_E = bi, ki, e
        return best_band, best_k, best_E

    def nearest_kpoint(self, k_diff: np.ndarray, E_diff: float,
                       actual_Bcell: np.ndarray) -> dict:
        """
        Find the nearest SCF k-point to the momentum-conserving vector
        and then the closest band energy at that k-point.
        """
        ai = self.auger_instance
        kx_frac = ut.to_fractional_coordinate(k_diff, actual_Bcell)
        kx_frac_mapped = ut.fold_kpoint_to_first_bz(kx_frac, convention="vasp_centered")
        kx_cart_mapped = ut.to_cartesian_coordinate(kx_frac_mapped, actual_Bcell)

        # Find nearest k-point
        dists = np.linalg.norm(ai.kpoints - kx_cart_mapped, axis=1)
        kx_index = int(np.argmin(dists))
        kx = ai.kpoints[kx_index]
        kwx = ai.kpoints_weights[kx_index]

        # Find closest band at that k-point
        if self.auger_type == "eeh":
            lo, hi = ai.firstCB_index, ai.num_of_bands
        else:
            lo, hi = 0, ai.firstCB_index
        best_bi, best_E, best_diff = lo, 0.0, float("inf")
        for bi in range(lo, hi):
            e = ai.data_energies[bi][kx_index]
            d = abs(e - E_diff)
            if d < best_diff:
                best_diff = d
                best_bi = bi
                best_E = e

        if self.auger_type == "eeh":
            Px = 1.0 - ut.fermi_dirac(best_E, ai.Efn, ai.T)
        else:
            Px = ut.fermi_dirac(best_E, ai.Efp, ai.T)

        return {
            "kx_target_cart": k_diff,
            "kx_target_frac": kx_frac,
            "kx_target_frac_mapped": kx_frac_mapped,
            "kx_target_cart_mapped": kx_cart_mapped,
            "nearest_kx_index": kx_index,
            "nearest_kx": kx,
            "nearest_kwx": kwx,
            "Ex_index": best_bi,
            "Ex": best_E,
            "Px": Px,
        }

    def exact_kpoint(self, k_diff: np.ndarray, actual_Bcell: np.ndarray) -> dict:
        """Fold the momentum-conserving k-point to the first BZ."""
        kx_frac = ut.to_fractional_coordinate(k_diff, actual_Bcell)
        kx_frac_mapped = ut.fold_kpoint_to_first_bz(kx_frac, convention="vasp_centered")
        kx_cart_mapped = ut.to_cartesian_coordinate(kx_frac_mapped, actual_Bcell)
        return {
            "kx_target_cart": k_diff,
            "kx_target_frac": kx_frac,
            "kx_target_frac_mapped": kx_frac_mapped,
            "kx_target_cart_mapped": kx_cart_mapped,
        }

    # ------------------------------------------------------------------
    # Exact k-point list generation (for NSCF workflow)
    # ------------------------------------------------------------------
    def generate_exact_kpoint_list(
        self,
        search_mode: str = "Brute_Force",
        num_kpoints: Union[int, str] = "all",
    ) -> List[dict]:
        """
        Build the list of off-grid k-points needed for the ``exact_kpoint``
        approach.  Writes partial pair info so the NSCF results can later be
        matched back.
        """
        print(f"  Generating exact k-point list … (search: {search_mode})")
        rl = self.auger_instance.reciprocal_lattice
        kpoints_list: List[dict] = []

        if self.auger_type == "eeh":
            for e1 in self.E1_energies:
                for e3 in self.E3_energies:
                    for e4 in self.E4_energies:
                        k2_diff = e3["k"] - e4["k"] + e1["k"]
                        res = self.exact_kpoint(k2_diff, rl)
                        pid = f"{e1['band_index']}-X-{e3['band_index']}-{e4['band_index']}-{e1['k_index']}-X-{e3['k_index']}-{e4['k_index']}"
                        kpoints_list.append({
                            "partial_pair_id": pid,
                            "P_134": e1["P"] * e3["P"] * e4["P"],
                            "E1_index": e1["band_index"], "E3_index": e3["band_index"], "E4_index": e4["band_index"],
                            "k1_index": e1["k_index"],    "k3_index": e3["k_index"],    "k4_index": e4["k_index"],
                            "E1": e1["energy"], "E3": e3["energy"], "E4": e4["energy"],
                            "k1": [float(x) for x in e1["k"]],
                            "k3": [float(x) for x in e3["k"]],
                            "k4": [float(x) for x in e4["k"]],
                            "kw1": e1["kw"], "kw3": e3["kw"], "kw4": e4["kw"],
                            "k1_frac": [float(x) for x in ut.to_fractional_coordinate(e1["k"], rl)],
                            "k3_frac": [float(x) for x in ut.to_fractional_coordinate(e3["k"], rl)],
                            "k4_frac": [float(x) for x in ut.to_fractional_coordinate(e4["k"], rl)],
                            "k2_target_cart": [float(x) for x in res["kx_target_cart"]],
                            "k2_target_frac": [float(x) for x in res["kx_target_frac"]],
                            "k2_target_frac_mapped": [float(x) for x in res["kx_target_frac_mapped"]],
                            "k2_target_cart_mapped": [float(x) for x in res["kx_target_cart_mapped"]],
                        })
        else:  # ehh
            for e1 in self.E1_energies:
                for e2 in self.E2_energies:
                    for e3 in self.E3_energies:
                        k4_diff = e3["k"] + e2["k"] - e1["k"]
                        res = self.exact_kpoint(k4_diff, rl)
                        pid = f"{e1['band_index']}-{e2['band_index']}-{e3['band_index']}-X-{e1['k_index']}-{e2['k_index']}-{e3['k_index']}-X"
                        kpoints_list.append({
                            "partial_pair_id": pid,
                            "P_123": e1["P"] * e2["P"] * e3["P"],
                            "E1_index": e1["band_index"], "E2_index": e2["band_index"], "E3_index": e3["band_index"],
                            "k1_index": e1["k_index"],    "k2_index": e2["k_index"],    "k3_index": e3["k_index"],
                            "E1": e1["energy"], "E2": e2["energy"], "E3": e3["energy"],
                            "k1": [float(x) for x in e1["k"]],
                            "k2": [float(x) for x in e2["k"]],
                            "k3": [float(x) for x in e3["k"]],
                            "kw1": e1["kw"], "kw2": e2["kw"], "kw3": e3["kw"],
                            "k1_frac": [float(x) for x in ut.to_fractional_coordinate(e1["k"], rl)],
                            "k2_frac": [float(x) for x in ut.to_fractional_coordinate(e2["k"], rl)],
                            "k3_frac": [float(x) for x in ut.to_fractional_coordinate(e3["k"], rl)],
                            "k4_target_cart": [float(x) for x in res["kx_target_cart"]],
                            "k4_target_frac": [float(x) for x in res["kx_target_frac"]],
                            "k4_target_frac_mapped": [float(x) for x in res["kx_target_frac_mapped"]],
                            "k4_target_cart_mapped": [float(x) for x in res["kx_target_cart_mapped"]],
                        })

        print(f"  Generated {len(kpoints_list):,} k-points")
        if num_kpoints != "all":
            if self.auger_type == "eeh":
                kpoints_list.sort(key=lambda x: x["P_134"], reverse=True)
            else:
                kpoints_list.sort(key=lambda x: x["P_123"], reverse=True)
            kpoints_list = kpoints_list[:num_kpoints]
            print(f"  Filtered to {len(kpoints_list):,}")
        return kpoints_list

    # ------------------------------------------------------------------
    # Internal: build a single pair from three known states + fourth
    # ------------------------------------------------------------------
    def _make_pair(
        self,
        e1: dict, e2_or_result: Any, e3: dict, e4_or_result: Any,
        *,
        is_eeh: bool,
        exact_entry: Optional[dict] = None,
    ) -> Pair:
        """
        Assemble a :class:`Pair` from three known states and the resolved
        fourth state.  Shared by brute-force and max-heap code paths.
        """
        ai = self.auger_instance

        if is_eeh:
            # e1, e3 are state dicts; e4_or_result is the E4 state dict;
            # e2_or_result is the resolution dict (nearest_kpoint)
            E1, E1_idx, k1, k1_idx, kw1, P1 = _unpack_state(e1, ai)
            E3, E3_idx, k3, k3_idx, kw3, P3 = _unpack_state(e3, ai)
            E4, E4_idx, k4, k4_idx, kw4, P4 = _unpack_state(e4_or_result, ai)

            if self.approach == "nearest_kpoint":
                r = e2_or_result
                E2, E2_idx = r["Ex"], r["Ex_index"]
                k2, k2_idx, kw2 = r["nearest_kx"], r["nearest_kx_index"], r["nearest_kwx"]
                P2 = r["Px"]
                k2_mapped = r["kx_target_cart_mapped"]
            elif self.approach == "exact_kpoint":
                ee = exact_entry
                E2, E2_idx = ee["E2"], ee["E2_index"]
                k2 = np.array(ee["k2_target_cart"])
                k2_idx = ee["k2_index"]
                kw2 = ee["k2_weight"]
                P2 = 1.0 - ut.fermi_dirac(E2, ai.Efn, ai.T)
                k2_mapped = np.array(ee["k2_target_cart_mapped"])
            else:
                raise ValueError(f"Unknown approach: {self.approach}")

            prob = P1 * P2 * P3 * P4
            pid = f"{E1_idx}-{E2_idx}-{E3_idx}-{E4_idx}-{k1_idx}-{k2_idx}-{k3_idx}-{k4_idx}"
            pair = Pair((pid, E1, E2, E3, E4,
                         k1, k2, k3, k4,
                         kw1, kw2, kw3, kw4,
                         E1_idx, E2_idx, E3_idx, E4_idx,
                         k1_idx, k2_idx, k3_idx, k4_idx,
                         prob, self.auger_type))
            if self.approach in ("nearest_kpoint", "exact_kpoint"):
                pair.set_mapped_kpoints(None, k2_mapped, None, None)
            if self.approach == "exact_kpoint" and exact_entry:
                pair.k1_wc_index = int(exact_entry["k1_wc_index"])
                pair.k2_wc_index = int(exact_entry["k2_wc_index"])
                pair.k3_wc_index = int(exact_entry["k3_wc_index"])
                pair.k4_wc_index = int(exact_entry["k4_wc_index"])
                pair.k1_nscf_index = int(exact_entry["k1_nscf_index"])
                pair.k2_nscf_index = int(exact_entry["k2_nscf_index"])
                pair.k3_nscf_index = int(exact_entry["k3_nscf_index"])
                pair.k4_nscf_index = int(exact_entry["k4_nscf_index"])

        else:  # ehh
            E1, E1_idx, k1, k1_idx, kw1, P1 = _unpack_state(e1, ai)
            E2, E2_idx, k2, k2_idx, kw2, P2 = _unpack_state(e2_or_result, ai)
            E3, E3_idx, k3, k3_idx, kw3, P3 = _unpack_state(e3, ai)

            if self.approach == "nearest_kpoint":
                r = e4_or_result
                E4, E4_idx = r["Ex"], r["Ex_index"]
                k4, k4_idx, kw4 = r["nearest_kx"], r["nearest_kx_index"], r["nearest_kwx"]
                P4 = r["Px"]
                k4_mapped = r["kx_target_cart_mapped"]
            elif self.approach == "exact_kpoint":
                ee = exact_entry
                E4, E4_idx = ee["E4"], ee["E4_index"]
                k4 = np.array(ee["k4_target_cart"])
                k4_idx = ee["k4_index"]
                kw4 = ee["k4_weight"]
                P4 = ut.fermi_dirac(E4, ai.Efn, ai.T)
                k4_mapped = np.array(ee["k4_target_cart_mapped"])
            else:
                raise ValueError(f"Unknown approach: {self.approach}")

            prob = P1 * P2 * P3 * P4
            pid = f"{E1_idx}-{E2_idx}-{E3_idx}-{E4_idx}-{k1_idx}-{k2_idx}-{k3_idx}-{k4_idx}"
            pair = Pair((pid, E1, E2, E3, E4,
                         k1, k2, k3, k4,
                         kw1, kw2, kw3, kw4,
                         E1_idx, E2_idx, E3_idx, E4_idx,
                         k1_idx, k2_idx, k3_idx, k4_idx,
                         prob, self.auger_type))
            if self.approach in ("nearest_kpoint", "exact_kpoint"):
                pair.set_mapped_kpoints(None, None, None, k4_mapped)
            if self.approach == "exact_kpoint" and exact_entry:
                pair.k1_wc_index = int(exact_entry["k1_wc_index"])
                pair.k2_wc_index = int(exact_entry["k2_wc_index"])
                pair.k3_wc_index = int(exact_entry["k3_wc_index"])
                pair.k4_wc_index = int(exact_entry["k4_wc_index"])
                pair.k1_nscf_index = int(exact_entry["k1_nscf_index"])
                pair.k2_nscf_index = int(exact_entry["k2_nscf_index"])
                pair.k3_nscf_index = int(exact_entry["k3_nscf_index"])
                pair.k4_nscf_index = int(exact_entry["k4_nscf_index"])

        return pair

    # ------------------------------------------------------------------
    # Brute Force
    # ------------------------------------------------------------------
    def brute_force_pairs(self, partial_ids: dict = {}):
        """
        Enumerate **all** triple combinations and resolve the fourth state
        using the ``nearest_kpoint`` approach.
        """
        is_eeh = self.auger_type == "eeh"

        if self.is_parallel and self.approach != "exact_kpoint":
            self._brute_force_parallel(is_eeh)
            return

        # ---- Serial path (also the only option for exact_kpoint) ----
        counter = 0
        rl = self.auger_instance.reciprocal_lattice

        if is_eeh:
            triples = _iter_eeh(self.E1_energies, self.E3_energies, self.E4_energies)
        else:
            triples = _iter_ehh(self.E1_energies, self.E2_energies, self.E3_energies)

        for state_a, state_b, state_c in triples:
            if is_eeh:
                e1, e3, e4 = state_a, state_b, state_c
                # Skip if already computed
                if partial_ids:
                    key = (e1["band_index"], e3["band_index"], e4["band_index"],
                           e1["k_index"], e3["k_index"], e4["k_index"])
                    if key in partial_ids:
                        continue
                E_diff = e3["energy"] - e4["energy"] + e1["energy"]
                k_diff = e3["k"] - e4["k"] + e1["k"]
            else:
                e1, e2, e3 = state_a, state_b, state_c
                if partial_ids:
                    key = (e1["band_index"], e2["band_index"], e3["band_index"],
                           e1["k_index"], e2["k_index"], e3["k_index"])
                    if key in partial_ids:
                        continue
                E_diff = e3["energy"] - e1["energy"] + e2["energy"]
                k_diff = e3["k"] - e1["k"] + e2["k"]

            exact_entry = None
            if self.approach == "nearest_kpoint":
                resolved = self.nearest_kpoint(k_diff, E_diff, rl)
            elif self.approach == "exact_kpoint":
                if is_eeh:
                    ppid = f"{e1['band_index']}-X-{e3['band_index']}-{e4['band_index']}-{e1['k_index']}-X-{e3['k_index']}-{e4['k_index']}"
                else:
                    ppid = f"{e1['band_index']}-{e2['band_index']}-{e3['band_index']}-X-{e1['k_index']}-{e2['k_index']}-{e3['k_index']}-X"
                exact_entry = self.exact_kpoints_dict.get(ppid)
                if exact_entry is None:
                    continue
                resolved = None
            else:
                raise ValueError(f"Unknown approach: {self.approach}")

            if is_eeh:
                pair = self._make_pair(e1, resolved, e3, e4,
                                       is_eeh=True, exact_entry=exact_entry)
            else:
                pair = self._make_pair(e1, e2, e3, resolved,
                                       is_eeh=False, exact_entry=exact_entry)

            self.add_pair(pair)
            counter += 1
            if counter % 10_000 == 0:
                self._write_checkpoint_current_chunk(to_path=self.auger_instance.results_folder)
                print(f"    pairs: {counter:,} …", end="\r")

    def _brute_force_parallel(self, is_eeh: bool):
        """Parallel brute-force using multiprocessing (fork, not spawn)."""
        n_cores = cpu_count()
        indices = list(range(len(self.E1_energies)))
        chunk_sz = max(1, len(indices) // n_cores)
        chunks = [indices[i:i + chunk_sz] for i in range(0, len(indices), chunk_sz)]
        print(f"    Parallel: {len(chunks)} chunks on {n_cores} cores")

        func = self._process_chunk_eeh if is_eeh else self._process_chunk_ehh
        with get_context("spawn").Pool(processes=n_cores) as pool:
            for chunk_pairs in pool.imap_unordered(func, [(c,) for c in chunks]):
                self.pairs.extend(chunk_pairs)

    def _process_chunk_eeh(self, args):
        idx_chunk, = args
        pairs_out = []
        rl = self.auger_instance.reciprocal_lattice
        for i in idx_chunk:
            e1 = self.E1_energies[i]
            for e3 in self.E3_energies:
                for e4 in self.E4_energies:
                    E_diff = e3["energy"] - e4["energy"] + e1["energy"]
                    k_diff = e3["k"] - e4["k"] + e1["k"]
                    if self.approach == "nearest_kpoint":
                        res = self.nearest_kpoint(k_diff, E_diff, rl)
                    else:
                        continue
                    pair = self._make_pair(e1, res, e3, None, is_eeh=True)
                    pairs_out.append(pair)
        return pairs_out

    def _process_chunk_ehh(self, args):
        idx_chunk, = args
        pairs_out = []
        rl = self.auger_instance.reciprocal_lattice
        for i in idx_chunk:
            e1 = self.E1_energies[i]
            for e2 in self.E2_energies:
                for e3 in self.E3_energies:
                    E_diff = e3["energy"] - e1["energy"] + e2["energy"]
                    k_diff = e3["k"] - e1["k"] + e2["k"]
                    if self.approach == "nearest_kpoint":
                        res = self.nearest_kpoint(k_diff, E_diff, rl)
                    else:
                        continue
                    pair = self._make_pair(e1, e2, e3, res, is_eeh=False)
                    pairs_out.append(pair)
        return pairs_out

    # ------------------------------------------------------------------
    # Max Heap
    # ------------------------------------------------------------------
    def max_heap_pairs(self, multiplier_top_k: int = 1):
        """
        Priority-queue walk: retrieve the *top-N* most probable pairs
        without enumerating all combinations.
        """
        is_eeh = self.auger_type == "eeh"
        rl = self.auger_instance.reciprocal_lattice

        if is_eeh:
            list_A, list_B, list_C = self.E1_energies, self.E3_energies, self.E4_energies
        else:
            list_A, list_B, list_C = self.E1_energies, self.E2_energies, self.E3_energies

        total = len(list_A) * len(list_B) * len(list_C)
        if self.num_top_pairs == -1:
            target = total
        else:
            target = min(self.num_top_pairs * multiplier_top_k, total)
        actual_top = min(self.num_top_pairs if self.num_top_pairs > 0 else total, total)

        heap: list = []
        visited: set = set()
        results: list = []

        def _push(i, j, k):
            if (i, j, k) in visited:
                return
            if i >= len(list_A) or j >= len(list_B) or k >= len(list_C):
                return
            visited.add((i, j, k))
            a, b, c = list_A[i], list_B[j], list_C[k]
            if is_eeh:
                E_diff = b["energy"] - c["energy"] + a["energy"]
                k_diff = b["k"] - c["k"] + a["k"]
            else:
                E_diff = c["energy"] - a["energy"] + b["energy"]
                k_diff = c["k"] - a["k"] + b["k"]
            if self.approach == "nearest_kpoint":
                res = self.nearest_kpoint(k_diff, E_diff, rl)
                Px = res["Px"]
            else:
                raise ValueError("Max_Heap only supports nearest_kpoint")
            prob = a["P"] * b["P"] * c["P"] * Px
            heapq.heappush(heap, (-prob, i, j, k, res))

        _push(0, 0, 0)

        while heap and len(results) < target:
            neg_p, i, j, k, res = heapq.heappop(heap)
            a, b, c = list_A[i], list_B[j], list_C[k]

            if is_eeh:
                pair = self._make_pair_from_heap(a, res, b, c, is_eeh=True)
            else:
                pair = self._make_pair_from_heap(a, b, c, res, is_eeh=False)

            self.add_pair(pair)
            results.append(pair)
            if len(results) % 10_000 == 0:
                self._write_checkpoint_current_chunk(to_path=self.auger_instance.results_folder)
                print(f"    pairs: {len(results):,} …", end="\r")

            for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                _push(i + di, j + dj, k + dk)

        # Trim to requested top N
        if self.num_top_pairs > 0 and self.num_top_pairs != -1:
            self.pairs = sorted(self.pairs, key=lambda p: p.probability, reverse=True)[:actual_top]

    def _make_pair_from_heap(self, e_a, res_or_e2, e_b, e_c_or_res, *, is_eeh: bool) -> Pair:
        """Build a Pair from max-heap pop results."""
        ai = self.auger_instance
        if is_eeh:
            e1, e3, e4 = e_a, e_b, e_c_or_res  # e_c_or_res = E4 state dict
            E1, E1_idx, k1, k1_idx, kw1, P1 = _unpack_state(e1, ai)
            E3, E3_idx, k3, k3_idx, kw3, P3 = _unpack_state(e3, ai)
            E4, E4_idx, k4, k4_idx, kw4, P4 = _unpack_state(e4, ai)
            res = res_or_e2  # resolution for E2
            E2, E2_idx = res["Ex"], res["Ex_index"]
            k2 = res["nearest_kx"]; k2_idx = res["nearest_kx_index"]; kw2 = res["nearest_kwx"]
            P2 = res["Px"]; k2_mapped = res["kx_target_cart_mapped"]
            prob = P1 * P2 * P3 * P4
            pid = f"{E1_idx}-{E2_idx}-{E3_idx}-{E4_idx}-{k1_idx}-{k2_idx}-{k3_idx}-{k4_idx}"
            pair = Pair((pid, E1, E2, E3, E4, k1, k2, k3, k4,
                         kw1, kw2, kw3, kw4,
                         E1_idx, E2_idx, E3_idx, E4_idx,
                         k1_idx, k2_idx, k3_idx, k4_idx,
                         prob, self.auger_type))
            if k2_mapped is not None:
                pair.set_mapped_kpoints(None, k2_mapped, None, None)
        else:
            e1, e2, e3 = e_a, res_or_e2, e_b  # res_or_e2 = E2 state dict, e_b = E3 state dict
            res = e_c_or_res  # resolution for E4
            E1, E1_idx, k1, k1_idx, kw1, P1 = _unpack_state(e1, ai)
            E2, E2_idx, k2, k2_idx, kw2, P2 = _unpack_state(e2, ai)
            E3, E3_idx, k3, k3_idx, kw3, P3 = _unpack_state(e3, ai)
            E4, E4_idx = res["Ex"], res["Ex_index"]
            k4 = res["nearest_kx"]; k4_idx = res["nearest_kx_index"]; kw4 = res["nearest_kwx"]
            P4 = res["Px"]; k4_mapped = res["kx_target_cart_mapped"]
            prob = P1 * P2 * P3 * P4
            pid = f"{E1_idx}-{E2_idx}-{E3_idx}-{E4_idx}-{k1_idx}-{k2_idx}-{k3_idx}-{k4_idx}"
            pair = Pair((pid, E1, E2, E3, E4, k1, k2, k3, k4,
                         kw1, kw2, kw3, kw4,
                         E1_idx, E2_idx, E3_idx, E4_idx,
                         k1_idx, k2_idx, k3_idx, k4_idx,
                         prob, self.auger_type))
            if k4_mapped is not None:
                pair.set_mapped_kpoints(None, None, None, k4_mapped)
        return pair

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------
    def create_pairs(
        self,
        multiplier: int = 1,
        nscf_folders: Optional[Union[str, List[str]]] = None,
        continue_from_files: Union[str, List[str]] = [],
        exact_kpoints_csv: Optional[Union[str, List[str]]] = None,
    ):
        """
        Route pair generation to the appropriate algorithm.

        Parameters
        ----------
        multiplier : int
            Expansion factor for Max_Heap (higher → more candidates explored).
        nscf_folders : str or list[str]
            Required for ``exact_kpoint`` approach.
        continue_from_files : str or list[str]
            CSV file(s) of previously-generated pairs (Brute_Force only).
        exact_kpoints_csv : str or None
            Explicit path to the ``exact_kpoints_<type>_<XX>.csv`` file.
            If *None*, the path is auto-discovered from ``results_folder``.
            Only used when *approach* = ``exact_kpoint``.
        """
        if isinstance(continue_from_files, str):
            continue_from_files = [continue_from_files]

        # ---- Prepare nscf folders ----
        if nscf_folders is not None:
            if isinstance(nscf_folders, str):
                nscf_folders = [nscf_folders]
            nscf_folders_ = nscf_folders
            nscf_folders = [f if f.endswith("/") else f + "/" for f in nscf_folders_]
            

        # ---- exact_kpoint: read CSV + NSCF, build pairs directly ----
        if self.approach == "exact_kpoint":
            self._prepare_exact_kpoint_data(nscf_folders, exact_kpoints_csv)
            self._build_exact_kpoint_pairs()
            print(f"\n  Pair creation complete: {len(self.pairs):,} pairs")
            return

        # ---- Load previously computed pairs (for continuation) ----
        partial_ids: dict = {}
        if continue_from_files:
            if self.search_mode != "Brute_Force":
                print("  Warning: continuation only works with Brute_Force. Switching.")
                self.search_mode = "Brute_Force"
            for fp in continue_from_files:
                if not os.path.isfile(fp):
                    print(f"  Warning: {fp} not found, skipping.")
                    continue
                prev = ut.read_csv([fp])
                for d in prev:
                    at = d.get("pair_type", "")
                    if at == "eeh":
                        key = (d["E1_index"], d["E3_index"], d["E4_index"],
                               d["k1_index"], d["k3_index"], d["k4_index"])
                    elif at == "ehh":
                        key = (d["E1_index"], d["E2_index"], d["E3_index"],
                               d["k1_index"], d["k2_index"], d["k3_index"])
                    else:
                        continue
                    partial_ids[key] = True
                    self.add_pair(Pair(d))

        # ---- Dispatch ----
        if self.search_mode == "Brute_Force":
            self.brute_force_pairs(partial_ids)
        elif self.search_mode == "Max_Heap":
            self.max_heap_pairs(multiplier_top_k=multiplier)
        else:
            raise ValueError(f"Unknown search_mode: '{self.search_mode}'")

        print(f"\n  Pair creation complete: {len(self.pairs):,} pairs")

    # ------------------------------------------------------------------
    # exact-kpoint data-loading helper
    # ------------------------------------------------------------------
    @staticmethod
    def _find_closest_band_at_kpoint(energy, data, kpt_idx):
        """Find closest band to *energy* at k-point column *kpt_idx*.

        Returns ``(band_index, energy)``.
        """
        best_band = 0
        best_E = 0.0
        min_diff = float("inf")
        for bi in range(len(data)):
            e = float(data[bi, kpt_idx])
            d = abs(e - energy)
            if d < min_diff:
                min_diff = d
                best_band = bi
                best_E = e
        return best_band, best_E

    def _prepare_exact_kpoint_data(self, nscf_folders, exact_kpoints_csv=None):
        """Read exact-kpoint CSV and NSCF EIGENVAL data, resolve E2 / E4.

                Requires the new CSV format produced by ``create_nscf_inputs``:
                each row must carry ``k#_wc_index`` (1-based folder) and
                ``k#_nscf_index`` (0-based within that folder) for every k-point.
        """
        ai = self.auger_instance
        is_eeh = self.auger_type == "eeh"

        if exact_kpoints_csv is not None:
            fname = exact_kpoints_csv
        else:
            res_folder = ai.results_folder.rstrip("/") + "/"
            fname = f"{res_folder}exact_kpoints_{self.auger_type}_{ai.XX}.csv"
        print(f"  Reading exact k-point CSV: {fname}")
        exact_kpoints = ut.read_csv(fname)

        target_wc_col  = "k2_wc_index"   if is_eeh else "k4_wc_index"
        target_nscf_col = "k2_nscf_index" if is_eeh else "k4_nscf_index"
        required_cols = {
            "k1_wc_index", "k2_wc_index", "k3_wc_index", "k4_wc_index",
            "k1_nscf_index", "k2_nscf_index", "k3_nscf_index", "k4_nscf_index",
            target_wc_col, target_nscf_col,
        }
        missing_cols = sorted(c for c in required_cols if c not in exact_kpoints[0])
        if missing_cols:
            raise ValueError(
                "exact k-point CSV is missing required new-format columns: "
                + ", ".join(missing_cols)
            )

        # Load each NSCF folder; key is 1-based wc_index.
        folder_data: dict    = {}
        folder_weights: dict = {}
        for fi, folder in enumerate(nscf_folders):
            data, _, _, weights = ut.read_nscf_results([folder])
            vbm_energy = float(np.max(data[ai.firstCB_index - 4:ai.firstCB_index]))
            if ai.scissor_shift != 0.0:
                data = data.copy()
                for bi in range(ai.firstCB_index, len(data)):
                    data[bi] = [e + ai.scissor_shift for e in data[bi]]
            key = fi + 1
            folder_data[key]    = data - vbm_energy
            folder_weights[key] = weights

        self.exact_kpoints_dict: dict = {}

        for item in exact_kpoints:
            target_wc   = int(item[target_wc_col])
            target_nscf = int(item[target_nscf_col])
            data_wc    = folder_data[target_wc]
            weights_wc = folder_weights[target_wc]

            if is_eeh:
                E_diff = item["E3"] - item["E4"] + item["E1"]
                E2_idx, E2 = self._find_closest_band_at_kpoint(
                    E_diff, data_wc, target_nscf)
                item["E2"]       = E2
                item["E2_index"] = E2_idx
                item["k2_index"] = target_nscf   # local within k2_wc_index folder
                item["k2_weight"] = float(weights_wc[target_nscf])
            else:
                E_diff = item["E3"] - item["E1"] + item["E2"]
                E4_idx, E4 = self._find_closest_band_at_kpoint(
                    E_diff, data_wc, target_nscf)
                item["E4"]       = E4
                item["E4_index"] = E4_idx
                item["k4_index"] = target_nscf   # local within k4_wc_index folder
                item["k4_weight"] = float(weights_wc[target_nscf])

            self.exact_kpoints_dict[item["partial_pair_id"]] = item
    # ------------------------------------------------------------------
    # Build pairs directly from exact-kpoint CSV rows
    # ------------------------------------------------------------------
    def _build_exact_kpoint_pairs(self):
        """
        Build :class:`Pair` objects directly from ``exact_kpoints_dict``.

        Called after :meth:`_prepare_exact_kpoint_data` has read the CSV,
        loaded NSCF eigenvalues, and resolved the unknown E2 (eeh) or E4
        (ehh) for every row.

        **All** k-point indices are NSCF-local — the NSCF KPOINTS file
        contains both the target k-points and the appended SCF k-points,
        so ``wavecar_files`` only needs NSCF WAVECARs.

        Cartesian coordinates come from the CSV.
        """
        ai = self.auger_instance
        is_eeh = self.auger_type == "eeh"

        for item in self.exact_kpoints_dict.values():

            if is_eeh:
                E1, E1_idx = item["E1"], item["E1_index"]
                E3, E3_idx = item["E3"], item["E3_index"]
                E4, E4_idx = item["E4"], item["E4_index"]

                k1 = np.array(item["k1"])
                k3 = np.array(item["k3"])
                k4 = np.array(item["k4"])

                kw1 = item["kw1"]
                kw3 = item["kw3"]
                kw4 = item["kw4"]

                # NSCF-local indices per k-point
                k1_idx = int(item["k1_nscf_index"])
                k3_idx = int(item["k3_nscf_index"])
                k4_idx = int(item["k4_nscf_index"])

                P1 = ut.fermi_dirac(E1, ai.Efn, ai.T)
                P3 = ut.fermi_dirac(E3, ai.Efn, ai.T)
                P4 = 1.0 - ut.fermi_dirac(E4, ai.Efp, ai.T)

                # Resolved E2 (excited CB electron from NSCF)
                E2, E2_idx = item["E2"], item["E2_index"]
                k2 = np.array(item["k2_target_cart"])
                k2_idx = int(item["k2_index"])   # local index within k2_wc_index folder
                kw2 = item["k2_weight"]
                P2 = 1.0 - ut.fermi_dirac(E2, ai.Efn, ai.T)
                k2_mapped = np.array(item["k2_target_cart_mapped"])

                prob = P1 * P2 * P3 * P4
                k1_wc = int(item["k1_wc_index"])
                k2_wc = int(item["k2_wc_index"])
                k3_wc = int(item["k3_wc_index"])
                k4_wc = int(item["k4_wc_index"])
                pid = (f"{E1_idx}-{E2_idx}-{E3_idx}-{E4_idx}"
                       f"-w{k1_wc}:{k1_idx}-w{k2_wc}:{k2_idx}"
                       f"-w{k3_wc}:{k3_idx}-w{k4_wc}:{k4_idx}")

                pair = Pair((pid, E1, E2, E3, E4,
                             k1, k2, k3, k4,
                             kw1, kw2, kw3, kw4,
                             E1_idx, E2_idx, E3_idx, E4_idx,
                             k1_idx, k2_idx, k3_idx, k4_idx,
                             prob, self.auger_type))
                pair.set_mapped_kpoints(None, k2_mapped, None, None)
                pair.k1_nscf_index = k1_idx
                pair.k2_nscf_index = k2_idx
                pair.k3_nscf_index = k3_idx
                pair.k4_nscf_index = k4_idx
                pair.k1_wc_index = k1_wc
                pair.k2_wc_index = k2_wc
                pair.k3_wc_index = k3_wc
                pair.k4_wc_index = k4_wc

            else:  # ehh
                E1, E1_idx = item["E1"], item["E1_index"]
                E2, E2_idx = item["E2"], item["E2_index"]
                E3, E3_idx = item["E3"], item["E3_index"]

                k1 = np.array(item["k1"])
                k2 = np.array(item["k2"])
                k3 = np.array(item["k3"])

                kw1 = item["kw1"]
                kw2 = item["kw2"]
                kw3 = item["kw3"]

                # NSCF-local indices per k-point
                k1_idx = int(item["k1_nscf_index"])
                k2_idx = int(item["k2_nscf_index"])
                k3_idx = int(item["k3_nscf_index"])

                P1 = ut.fermi_dirac(E1, ai.Efn, ai.T)
                P2 = 1.0 - ut.fermi_dirac(E2, ai.Efp, ai.T)
                P3 = 1.0 - ut.fermi_dirac(E3, ai.Efp, ai.T)

                # Resolved E4 (deep VB hole from NSCF)
                E4, E4_idx = item["E4"], item["E4_index"]
                k4 = np.array(item["k4_target_cart"])
                k4_idx = int(item["k4_index"])   # local index within k4_wc_index folder
                kw4 = item["k4_weight"]
                P4 = ut.fermi_dirac(E4, ai.Efp, ai.T)
                k4_mapped = np.array(item["k4_target_cart_mapped"])

                prob = P1 * P2 * P3 * P4
                k1_wc = int(item["k1_wc_index"])
                k2_wc = int(item["k2_wc_index"])
                k3_wc = int(item["k3_wc_index"])
                k4_wc = int(item["k4_wc_index"])
                pid = (f"{E1_idx}-{E2_idx}-{E3_idx}-{E4_idx}"
                       f"-w{k1_wc}:{k1_idx}-w{k2_wc}:{k2_idx}"
                       f"-w{k3_wc}:{k3_idx}-w{k4_wc}:{k4_idx}")

                pair = Pair((pid, E1, E2, E3, E4,
                             k1, k2, k3, k4,
                             kw1, kw2, kw3, kw4,
                             E1_idx, E2_idx, E3_idx, E4_idx,
                             k1_idx, k2_idx, k3_idx, k4_idx,
                             prob, self.auger_type))
                pair.set_mapped_kpoints(None, None, None, k4_mapped)
                pair.k1_nscf_index = k1_idx
                pair.k2_nscf_index = k2_idx
                pair.k3_nscf_index = k3_idx
                pair.k4_nscf_index = k4_idx
                pair.k1_wc_index = k1_wc
                pair.k2_wc_index = k2_wc
                pair.k3_wc_index = k3_wc
                pair.k4_wc_index = k4_wc
            self.add_pair(pair)

        print(f"    Built {len(self.pairs):,} pairs from exact k-point CSV")


# ======================================================================
# Module-level helpers
# ======================================================================
def _state_dict(band_index, k_index, energy, kpt, kw, P) -> dict:
    return {
        "band_index": band_index,
        "k_index": k_index,
        "energy": energy,
        "k": kpt,
        "kw": kw,
        "P": P,
    }


def _unpack_state(state: dict, ai):
    """Return ``(E, band_index, k, k_index, kw, P)`` from a state dict."""
    return (
        state["energy"],
        state["band_index"],
        ai.kpoints[state["k_index"]],
        state["k_index"],
        ai.kpoints_weights[state["k_index"]],
        state["P"],
    )


def _iter_eeh(E1_list, E3_list, E4_list):
    """Yield ``(e1, e3, e4)`` triple dicts."""
    for e1 in E1_list:
        for e3 in E3_list:
            for e4 in E4_list:
                yield e1, e3, e4


def _iter_ehh(E1_list, E2_list, E3_list):
    """Yield ``(e1, e2, e3)`` triple dicts."""
    for e1 in E1_list:
        for e2 in E2_list:
            for e3 in E3_list:
                yield e1, e2, e3
