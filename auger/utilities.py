"""
Utility functions for the Auger recombination package.

Includes helpers for:
- Band-structure I/O (read/write parsed data, CSV, JSONL)
- Fermi–Dirac statistics
- Brillouin-zone folding and coordinate transforms
- Delta-function approximations (Gaussian, Lorentzian, Rectangular)
- Coulomb / dielectric helper functions
- NSCF input-file generation
"""

from __future__ import annotations

import ast
import json
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import BSVasprun, Eigenval, Vasprun
from scipy.special import expit

from .constants import (
    ALPHA_PENN,
    ANGSTROM,
    CM_PER_ANGSTROM,
    EPSILON_0,
    HBAR,
    K_B_eV,
    M_E,
    eV,
)

# ═══════════════════════════════════════════════════════════════════════════
# Band-structure helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_firstCB_and_lastVB(
    data_energies: np.ndarray,
    Ef: float,
) -> Tuple[int, int]:
    """
    Identify the first conduction-band and last valence-band indices.

    Parameters
    ----------
    data_energies : ndarray, shape (nbands, nkpoints)
        Band energies at every k-point.
    Ef : float
        Reference Fermi level (eV), usually the VASP ``EFERMI``.

    Returns
    -------
    first_CB_index, last_VB_index : int
        Zero-based band indices.

    Raises
    ------
    ValueError
        If the gap is zero and the heuristic slope-change detection fails.
    """
    num_of_bands, num_of_kpoints = data_energies.shape

    first_CB_index = None
    last_VB_index = None

    for i in range(num_of_bands):
        if np.min(data_energies[i, :]) > Ef:
            first_CB_index = i
            break

    for i in range(num_of_bands - 1, -1, -1):
        if np.max(data_energies[i, :]) < Ef:
            last_VB_index = i
            break

    if first_CB_index is None or last_VB_index is None:
        raise ValueError(
            "Could not determine CB/VB indices automatically.  "
            "Assign them manually via AugerCalculator.assign_firstCB_and_lastVB()."
        )

    if first_CB_index - last_VB_index == 1:
        return first_CB_index, last_VB_index

    # --- Gapless / near-gapless fallback: slope-change heuristic ---
    bands_indices = range(last_VB_index, first_CB_index + 2)
    slopes = []
    for i in bands_indices:
        slope = np.polyfit(range(num_of_kpoints), data_energies[i, :], 1)[0]
        slopes.append(slope)

    sign_changes = [
        i for i in range(len(slopes) - 1) if slopes[i] * slopes[i + 1] < 0
    ]
    if len(sign_changes) != 1:
        raise ValueError(
            "Could not find CB/VB indices (band gap ≈ 0).  "
            "Assign them manually with AugerCalculator.assign_firstCB_and_lastVB()."
        )

    first_CB_band = bands_indices[sign_changes[0]]
    last_VB_band = bands_indices[sign_changes[0] + 1]
    first_CB_avg = np.mean(data_energies[first_CB_band, :])
    last_VB_avg = np.mean(data_energies[last_VB_band, :])

    if first_CB_avg > last_VB_avg:
        return first_CB_band, last_VB_band
    return last_VB_band, first_CB_band


# ═══════════════════════════════════════════════════════════════════════════
# File I/O
# ═══════════════════════════════════════════════════════════════════════════

def read_band_info(file_path: str) -> Dict:
    """Read the ``band_info.txt`` key-value file into a dictionary."""
    result_dict: Dict = {}

    def _try_int_list(s):
        return [int(i) for i in s.strip("[]").split(", ")]

    def _try_float_list(s):
        return [float(i) for i in s.strip("[]").split(", ")]

    with open(file_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            key = parts[0]
            raw = parts[1].strip() if len(parts) > 1 else ""

            # Attempt progressively broader conversions
            for converter in (int, float, _try_int_list, _try_float_list):
                try:
                    raw = converter(raw)
                    break
                except Exception:
                    continue

            result_dict[key] = raw

    return result_dict


def write_to_csv(
    dict_array: List[Dict],
    file_name: str,
    folder_to_save: str = "",
) -> None:
    """
    Write a list of dictionaries to one or more CSV files.

    Automatically splits into multiple files when the list exceeds 1 M rows
    (with suffixes ``_1.csv``, ``_2.csv``, …).
    """
    if folder_to_save and not folder_to_save.endswith("/"):
        folder_to_save += "/"

    n = len(dict_array)
    if n <= 1_000_000:
        pd.DataFrame(dict_array).to_csv(
            f"{folder_to_save}{file_name}.csv", index=False
        )
        return

    parts = int(np.ceil(n / 1_000_000))
    for i in range(parts):
        start = i * 1_000_000
        end = min((i + 1) * 1_000_000, n)
        pd.DataFrame(dict_array[start:end]).to_csv(
            f"{folder_to_save}{file_name}_{i + 1}.csv", index=False
        )


def read_csv(file_paths: Union[str, List[str]]) -> List[Dict]:
    """
    Read one or more CSV pair-table files and return a combined list of dicts.

    Automatically deserialises columns that store Python lists as strings
    (``k1``, ``k2``, …, ``k2_mapped``, etc.).
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    _list_cols = [
        "k1", "k2", "k3", "k4", "k2_mapped", "k4_mapped",
        "k1_frac", "k2_frac", "k3_frac", "k4_frac",
        "k2_target_cart", "k2_target_frac", "k2_target_frac_mapped",
        "k2_target_cart_mapped", "k4_target_cart", "k4_target_frac",
        "k4_target_frac_mapped", "k4_target_cart_mapped",
    ]
    _to_float_list = lambda x: [float(i) for i in x.strip("[]").split(", ")]

    result: List[Dict] = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        for col in _list_cols:
            if col in df.columns:
                df[col] = df[col].apply(_to_float_list)
        result.extend(df.to_dict("records"))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Fermi–Dirac statistics
# ═══════════════════════════════════════════════════════════════════════════

def fermi_dirac(E: float, E_Fermi: float, T: float = 300.0) -> float:
    """Fermi–Dirac distribution *f(E)* at temperature *T* (K)."""
    return float(expit((E_Fermi - E) / (K_B_eV * T)))


# ═══════════════════════════════════════════════════════════════════════════
# Brillouin-zone utilities
# ═══════════════════════════════════════════════════════════════════════════

def fold_kpoint_to_first_bz(
    kpoint: np.ndarray,
    convention: str = "vasp_centered",
) -> np.ndarray:
    """
    Fold a fractional k-point back into the first Brillouin zone.

    Parameters
    ----------
    kpoint : array-like, shape (3,)
        Fractional coordinates.
    convention : {'zero_to_one', 'centered', 'vasp_centered'}
        - ``zero_to_one`` →  [0, 1)
        - ``centered``    → [-0.5, 0.5)
        - ``vasp_centered`` → (-0.5, 0.5]   *(default)*

    Returns
    -------
    ndarray, shape (3,)
    """
    k = np.asarray(kpoint, dtype=float)

    if convention == "zero_to_one":
        return k - np.floor(k)

    if convention == "centered":
        k_folded = k - np.floor(k)
        return np.where(k_folded >= 0.5, k_folded - 1.0, k_folded)

    if convention == "vasp_centered":
        k_folded = k - np.floor(k)
        k_folded = np.where(k_folded > 0.5, k_folded - 1.0, k_folded)
        k_folded = np.where(np.abs(k_folded + 0.5) < 1e-10, 0.5, k_folded)
        return k_folded

    raise ValueError(
        f"Unsupported convention '{convention}'.  "
        "Choose 'zero_to_one', 'centered', or 'vasp_centered'."
    )


def to_fractional_coordinate(
    kpoint: np.ndarray,
    reciprocal_lattice: np.ndarray,
) -> np.ndarray:
    """Convert Cartesian k-point to fractional coordinates."""
    return np.dot(np.asarray(kpoint), np.linalg.inv(reciprocal_lattice))


def to_cartesian_coordinate(
    kpoint_frac: np.ndarray,
    reciprocal_lattice: np.ndarray,
) -> np.ndarray:
    """Convert fractional k-point to Cartesian coordinates."""
    return np.dot(np.asarray(kpoint_frac), reciprocal_lattice)


# ═══════════════════════════════════════════════════════════════════════════
# Delta-function approximations
# ═══════════════════════════════════════════════════════════════════════════

def delta_Gaussian(x: float, FWHM: float = 0.05) -> float:
    """Gaussian approximation to the Dirac delta function."""
    sigma = FWHM / 2.354_820_045_030_949_3  # FWHM / (2√(2 ln 2))
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x**2 / (2 * sigma**2))


def delta_Lorentzian(x: float, FWHM: float = 0.03) -> float:
    """Lorentzian approximation to the Dirac delta function."""
    half_width = 0.5 * FWHM
    return (half_width / np.pi) / (x**2 + half_width**2)


def delta_Rectangular(x: float, FWHM: float = 0.2) -> float:
    """Rectangular (box) approximation to the Dirac delta function."""
    half_width = FWHM / 2.0
    return np.where(np.abs(x) <= half_width, 1.0 / FWHM, 0.0)


DELTA_FUNCTIONS = {
    "Gaussian": delta_Gaussian,
    "Lorentzian": delta_Lorentzian,
    "Rectangular": delta_Rectangular,
}


# ═══════════════════════════════════════════════════════════════════════════
# Coulomb / dielectric helpers
# ═══════════════════════════════════════════════════════════════════════════

def I_ab(G, Ga, dicta, dictb):
    """
    Plane-wave overlap integral ⟨u_{a,k}|u_{b,k'}⟩ via a G-vector sum.

    Parameters
    ----------
    G : array-like
        Reciprocal-lattice vector (fractional).
    Ga : ndarray
        G-vectors of wavefunction *a*.
    dicta, dictb : dict
        Plane-wave coefficient dictionaries  {tuple(G): C_G}.

    Returns
    -------
    complex
        The overlap integral.
    """
    total = 0.0 + 0.0j
    for G1 in Ga:
        key_a = tuple(G1)
        key_b = tuple(G1 - G)
        ca = dicta.get(key_a)
        cb = dictb.get(key_b)
        if ca is not None and cb is not None:
            total += np.conj(ca) * cb
    return total


def calculate_epsilon(
    q: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> float:
    r"""
    k-dependent dielectric function:

    .. math::
        \varepsilon(q) = 1 + \frac{1}{a + b\,q^2 + c\,q^4}

    Parameters
    ----------
    q : array-like
        Wave-vector (Å⁻¹), 3-D vector or scalar magnitude.
    a, b, c : float
        Fitting parameters.
    """
    q_mag = float(np.linalg.norm(np.asarray(q)))
    return 1.0 + 1.0 / (a + b * q_mag**2 + c * (q_mag * 1e10) ** 4)


def W(q_mag: float, epsilon: float, lam: float) -> float:
    r"""
    Screened Coulomb interaction (without the :math:`4\pi e^2` prefactor).

    .. math::
        W(q) = \frac{1}{\varepsilon(q)\,(q^2 + \lambda^2)}

    Parameters
    ----------
    q_mag : float
        |q| in Å⁻¹.
    epsilon : float
        Dielectric function at *q*.
    lam : float
        Inverse Debye screening length (Å⁻¹).
    """
    return (1.0 / epsilon) * (1.0 / (q_mag**2 + lam**2))


# ═══════════════════════════════════════════════════════════════════════════
# Time formatting
# ═══════════════════════════════════════════════════════════════════════════

def convert_seconds(seconds: float) -> Tuple[int, int, int, int]:
    """Convert *seconds* to ``(days, hours, minutes, seconds)``."""
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return days, hours, minutes, secs


# ═══════════════════════════════════════════════════════════════════════════
# NSCF helpers
# ═══════════════════════════════════════════════════════════════════════════

def read_nscf_results(
    nscf_folders: Union[str, List[str]],
) -> Tuple[np.ndarray, list, list, list]:
    """
    Read band-structure data from one or more completed NSCF folders.

    Returns
    -------
    data_energies, kpoints_cart, kpoints_frac, kpoints_weights
    """
    if isinstance(nscf_folders, str):
        nscf_folders = [nscf_folders]

    all_data, all_cart, all_frac, all_weights = [], [], [], []

    for folder in nscf_folders:
        folder = folder.rstrip("/") + "/"
        bs = BSVasprun(f"{folder}vasprun.xml")
        bandstructure = bs.get_band_structure(kpoints_filename=f"{folder}KPOINTS")
        data = bandstructure.bands[Spin(1)]
        eigenvalues = Eigenval(f"{folder}EIGENVAL")
        XX = eigenvalues.nkpt
        cart = [bandstructure.kpoints[i].cart_coords for i in range(XX)]
        frac = [bandstructure.kpoints[i].frac_coords for i in range(XX)]
        weights = eigenvalues.kpoints_weights

        all_data.append(data)
        all_cart.append(cart)
        all_frac.append(frac)
        all_weights.append(weights)

    if len(nscf_folders) == 1:
        return all_data[0], all_cart[0], all_frac[0], all_weights[0]

    combined_data = np.concatenate(all_data, axis=1)
    combined_cart = [k for sublist in all_cart for k in sublist]
    combined_frac = [k for sublist in all_frac for k in sublist]
    combined_weights = [w for sublist in all_weights for w in sublist]
    return combined_data, combined_cart, combined_frac, combined_weights



def create_nscf_inputs(
    scf_folder: str,
    nscf_folder: str,
    exact_kpoints_table: Union[str, List[str]],
    auger_type: str = "eeh",
    num_kpoints_per_file: Union[int, str] = "all",
    nscf_settings: Optional[Dict[str, Union[int, str]]] = None,
) -> None:
    """
    Generate VASP NSCF input files from an SCF folder and exact-kpoint table.

    Collects ALL unique k-points needed across all Auger pairs — both the
    target k-points (k2 in eeh, k4 in ehh, deduplicated by their mapped
    fractional coordinates) and the SCF-side k-points (k1/k3/k4 in eeh,
    k1/k2/k3 in ehh, deduplicated by their original k-index) — and
    distributes them globally across NSCF folders, exactly
    ``num_kpoints_per_file`` k-points per folder.  No k-point is ever listed
    twice across the NSCF KPOINTS files.

    Each CSV row is updated with per-k-point location fields:

    * ``k#_wc_index``   — which NSCF folder (1-based) the k-point lives in
    * ``k#_nscf_index`` — its 0-based position within that folder's KPOINTS

    Parameters
    ----------
    scf_folder : str
        Path to the completed SCF calculation.
    nscf_folder : str
        Base output path for NSCF folder(s).
    exact_kpoints_table : str or list[str]
        CSV file(s) generated by ``create_exact_kpoint_list()``.
    auger_type : {'eeh', 'ehh'}
        Determines which k-points are "target" vs "SCF-side".
    num_kpoints_per_file : int or 'all'
        Split into multiple folders if int; single folder if ``'all'``.
    nscf_settings : dict, optional
        Additional INCAR settings to override defaults.
    """
    if isinstance(exact_kpoints_table, str):
        exact_kpoints_table = [exact_kpoints_table]

    exact_kpoints_dicts = read_csv(exact_kpoints_table)

    # Column names for target and SCF k-points.
    # eeh: target = k2,  SCF = k1, k3, k4
    # ehh: target = k4,  SCF = k1, k2, k3
    if auger_type == "eeh":
        scf_frac_cols   = ["k1_frac", "k3_frac", "k4_frac"]
        scf_idx_cols    = ["k1_index", "k3_index", "k4_index"]
        scf_wc_cols     = ["k1_wc_index", "k3_wc_index", "k4_wc_index"]
        scf_nscf_cols   = ["k1_nscf_index", "k3_nscf_index", "k4_nscf_index"]
        target_wc_col   = "k2_wc_index"
        target_nscf_col = "k2_nscf_index"
        choose_key = "k2_target_frac_mapped"
    else:  # ehh
        scf_frac_cols   = ["k1_frac", "k2_frac", "k3_frac"]
        scf_idx_cols    = ["k1_index", "k2_index", "k3_index"]
        scf_wc_cols     = ["k1_wc_index", "k2_wc_index", "k3_wc_index"]
        scf_nscf_cols   = ["k1_nscf_index", "k2_nscf_index", "k3_nscf_index"]
        target_wc_col   = "k4_wc_index"
        target_nscf_col = "k4_nscf_index"
        choose_key = "k4_target_frac_mapped"

    scf_folder  = scf_folder.rstrip("/") + "/"
    nscf_folder = nscf_folder.rstrip("/") + "/"

    # ── Step 1: Collect ALL unique k-points in encounter order ───────────
    # Target k-points: deduplicated by rounded mapped fractional coordinates.
    # SCF k-points:    deduplicated by their original k-index.
    unique_kpoints: Dict[tuple, list] = {}   # key → frac_coords
    for item in exact_kpoints_dicts:
        # Target k-point (k2 in eeh, k4 in ehh)
        target_frac = list(item[choose_key])
        t_key = ("target", tuple(round(x, 8) for x in target_frac))
        if t_key not in unique_kpoints:
            unique_kpoints[t_key] = target_frac
        # SCF k-points (unique by original SCF k-index)
        for fc, ic in zip(scf_frac_cols, scf_idx_cols):
            s_key = ("scf", int(item[ic]))
            if s_key not in unique_kpoints:
                unique_kpoints[s_key] = list(item[fc])

    all_kpt_keys  = list(unique_kpoints.keys())
    all_kpt_fracs = list(unique_kpoints.values())
    total_unique  = len(all_kpt_keys)
    n_target_unique = sum(1 for k in all_kpt_keys if k[0] == "target")
    n_scf_unique    = total_unique - n_target_unique

    # ── Step 2: Determine folder distribution ────────────────────────────
    if num_kpoints_per_file != "all" and num_kpoints_per_file > total_unique:
        print(
            f"Requested num_kpoints_per_file={num_kpoints_per_file} exceeds "
            f"total unique k-points ({total_unique}).  Using 'all' instead."
        )
        num_kpoints_per_file = "all"
    if num_kpoints_per_file == "all":
        num_folders = 1
        kpts_per_folder = [total_unique]
    else:
        num_folders = int(np.ceil(total_unique / num_kpoints_per_file))
        kpts_per_folder = [num_kpoints_per_file] * (num_folders - 1)
        kpts_per_folder.append(total_unique - (num_folders - 1) * num_kpoints_per_file)

    # Cumulative folder boundaries
    boundaries: List[int] = [0]
    for cnt in kpts_per_folder:
        boundaries.append(boundaries[-1] + cnt)

    print(f"\n{'─'*80}")
    print(f"Creating NSCF input files:")
    print(f"  Total unique k-points:  {total_unique:,}")
    print(f"    Unique target k-pts:  {n_target_unique:,}")
    print(f"    Unique SCF k-pts:     {n_scf_unique:,}")
    print(f"  Number of folders:      {num_folders}")
    if num_folders > 1:
        print(f"  K-points per folder:    {kpts_per_folder}")
    print(f"{'─'*80}\n")

    # ── Step 3: Assign (wc_index, nscf_index) to every unique k-point ────
    # wc_index is 1-based (folder 1, 2, …); nscf_index is 0-based within folder.
    key_to_location: Dict[tuple, Tuple[int, int]] = {}
    for gi, key in enumerate(all_kpt_keys):
        for fi in range(len(kpts_per_folder)):
            if boundaries[fi] <= gi < boundaries[fi + 1]:
                key_to_location[key] = (fi + 1, gi - boundaries[fi])
                break

    # ── Step 4: Build NSCF folders and write KPOINTS ─────────────────────
    _default_nscf: Dict[str, Union[int, str]] = {
        "ALGO": "Normal",
        "PREC": "Accurate",
        "ICHARG": 11,
        "LCHARG": "False",
        "LWAVE": "True",
        "ISYM": -1,
    }
    active_settings = nscf_settings if nscf_settings is not None else _default_nscf

    for fi in range(num_folders):
        cur_folder = f"{nscf_folder}NSCF_{auger_type}_{fi + 1}/"
        os.makedirs(cur_folder, exist_ok=True)

        # Copy SCF files
        for fname in ("POTCAR", "INCAR", "POSCAR", "CHGCAR"):
            try:
                shutil.copyfile(scf_folder + fname, cur_folder + fname)
            except FileNotFoundError:
                print(f"  ⚠  Could not copy '{fname}' to folder {fi + 1}")

        # Remove stale WAVECAR
        try:
            os.remove(cur_folder + "WAVECAR")
        except OSError:
            pass

        # Patch INCAR for NSCF mode
        incar_path = cur_folder + "INCAR"
        try:
            with open(incar_path, "r") as f:
                lines = f.readlines()
            with open(incar_path, "w") as f:
                written_keys: set = set()
                # Create a new line at the beginning:
                f.write("\n # Auto-generated INCAR for NSCF calculation\n")
                for line in lines:
                    key = line.split("=")[0].strip()
                    if key in active_settings:
                        f.write(f"{key} = {active_settings[key]}\n")
                        written_keys.add(key)
                    else:
                        f.write(line)
                for key, val in active_settings.items():
                    if key not in written_keys:
                        f.write(f"{key} = {val}\n")
        except FileNotFoundError:
            print(f"  ⚠  Could not modify INCAR in folder {fi + 1}")

        # Write KPOINTS — exactly kpts_per_folder[fi] k-points, no extras
        folder_kpts = all_kpt_fracs[boundaries[fi]:boundaries[fi + 1]]
        n_kpts = len(folder_kpts)
        with open(cur_folder + "KPOINTS", "w") as kf:
            kf.write("K-points for NSCF calculation\n")
            kf.write(f"{n_kpts}\n")
            kf.write("Reciprocal\n")
            for frac in folder_kpts:
                kf.write(f"  {frac[0]:.8f}  {frac[1]:.8f}  {frac[2]:.8f}  1\n")

        print(f"  ✓ Created folder {fi + 1}/{num_folders}: {cur_folder}")
        print(f"    K-points in this folder: {n_kpts:,}")

    # ── Step 5: Assign k#_wc_index / k#_nscf_index to every CSV row ──────
    for item in exact_kpoints_dicts:
        target_frac = list(item[choose_key])
        t_key = ("target", tuple(round(x, 8) for x in target_frac))
        t_wc, t_nscf = key_to_location[t_key]
        item[target_wc_col]   = t_wc
        item[target_nscf_col] = t_nscf

        for idx_col, wc_col, nscf_col in zip(scf_idx_cols, scf_wc_cols, scf_nscf_cols):
            s_key = ("scf", int(item[idx_col]))
            s_wc, s_nscf = key_to_location[s_key]
            item[wc_col]   = s_wc
            item[nscf_col] = s_nscf

        # Remove legacy columns no longer needed
        item.pop("wavecar", None)
        item.pop("wc_index", None)

    # ── Step 6: Persist updated CSV(s) ───────────────────────────────────
    if len(exact_kpoints_table) == 1:
        write_to_csv(
            exact_kpoints_dicts,
            exact_kpoints_table[0].replace(".csv", ""),
        )
    else:
        cur = 0
        for table_file in exact_kpoints_table:
            orig_len = len(read_csv([table_file]))
            write_to_csv(
                exact_kpoints_dicts[cur: cur + orig_len],
                table_file.replace(".csv", ""),
            )
            cur += orig_len

    print(f"\n✓ NSCF input creation complete!")
    print(f"{'─'*80}\n")
