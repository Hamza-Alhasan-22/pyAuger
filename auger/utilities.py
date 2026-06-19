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
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import BSVasprun, Eigenval, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
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

    # Raise an error if first_CB_index - last_VB_index != 1:
    if first_CB_index - last_VB_index != 1:
        raise ValueError(
            f"Identified first_CB_index={first_CB_index} and "
            f"last_VB_index={last_VB_index}, but expected them to be "
            f"consecutive.  Please assign them manually via "
            f"AugerCalculator.assign_firstCB_and_lastVB()."
        )
    return first_CB_index, last_VB_index
    
    
# ═══════════════════════════════════════════════════════════════════════════
# File I/O
# ═══════════════════════════════════════════════════════════════════════════

def read_band_info(file_path: str) -> Dict:
    """Read the ``band_info.txt`` key-value file into a dictionary."""
    result_dict: Dict = {}

    # Keys whose values are stored as JSON arrays (e.g. list of paths)
    _json_keys = {"vasp_folders", "dielectric_constant_used_in_matrix_elements"}

    def _try_int_list(s):
        return [int(i) for i in s.strip("[]()").split(", ")]

    def _try_float_list(s):
        return [float(i) for i in s.strip("[]()").split(", ")]

    with open(file_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            key = parts[0]
            raw = parts[1].strip() if len(parts) > 1 else ""

            if key in _json_keys:
                import json as _json
                result_dict[key] = _json.loads(raw)
                continue

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


def normalize_dielectric_input(dielectric) -> Tuple[Union[float, List[List[float]]], bool, float, Optional[np.ndarray]]:
    """
    Validate and normalize a scalar dielectric constant or 3x3 tensor.

    Returns
    -------
    normalized, is_tensor, scalar_for_screening, tensor
        ``normalized`` is JSON-serializable. ``scalar_for_screening`` is the
        scalar dielectric used by existing scalar-only pieces such as Debye
        screening. For tensor input this is ``trace(tensor) / 3``.
    """
    arr = np.asarray(dielectric, dtype=float)

    if arr.ndim == 0:
        value = float(arr)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Dielectric constant must be a positive finite scalar.")
        return value, False, value, None

    if arr.shape != (3, 3):
        raise ValueError(
            "physics.dielectric must be either a positive scalar or a 3x3 "
            f"dielectric tensor; got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Dielectric tensor must contain only finite values.")

    sym = 0.5 * (arr + arr.T)
    eigvals = np.linalg.eigvalsh(sym)
    if np.any(eigvals <= 0.0):
        raise ValueError(
            "Dielectric tensor must be positive definite in its symmetric part; "
            f"eigenvalues are {eigvals.tolist()}."
        )

    scalar = float(np.trace(arr) / 3.0)
    if scalar <= 0.0:
        raise ValueError("Dielectric tensor trace must imply a positive scalar average.")
    return arr.tolist(), True, scalar, arr


def directional_epsilon(q_vector: np.ndarray, epsilon_tensor: np.ndarray, *, assume_valid: bool = False) -> float:
    """
    Directional high-frequency longitudinal dielectric value.

    ``q_vector`` must be in Cartesian reciprocal coordinates. If ``|q|`` is
    numerically zero, the direction is undefined; this helper uses
    ``trace(eps) / 3`` as the isotropic directional average for that limit.
    """
    tensor = np.asarray(epsilon_tensor, dtype=float)
    if not assume_valid:
        _, _, _, tensor = normalize_dielectric_input(tensor)
    q = np.asarray(q_vector, dtype=float)
    if q.shape != (3,):
        raise ValueError(f"q_vector must have shape (3,), got {q.shape}.")

    q_mag = float(np.linalg.norm(q))
    if q_mag <= 1e-14:
        eps_l = float(np.trace(tensor) / 3.0)
    else:
        qhat = q / q_mag
        eps_l = float(qhat @ tensor @ qhat)

    if not np.isfinite(eps_l) or eps_l <= 0.0:
        raise ValueError(f"Directional dielectric epsilon_L must be positive; got {eps_l}.")
    return eps_l


def directional_dependent_epsilon(
    q_vector: np.ndarray,
    epsilon_tensor: np.ndarray,
    b: float,
    c: float,
    *,
    assume_valid: bool = False,
) -> float:
    r"""
    Directional q-dependent model dielectric function for tensor input.

    The scalar model is retained, but its scalar ``epsilon_inf`` is replaced by
    the longitudinal value

    .. math::
        \epsilon_{\infty,L}(\hat Q) = \hat Q^T \epsilon_\infty \hat Q

    before evaluating

    .. math::
        \epsilon(Q) = 1 + \left[
            (\epsilon_{\infty,L} - 1)^{-1} + b |Q|^2 + c |Q|^4
        \right]^{-1}

    with the same unit convention as :func:`calculate_epsilon`.
    """
    q = np.asarray(q_vector, dtype=float)
    eps_l = directional_epsilon(q, epsilon_tensor, assume_valid=assume_valid)
    if abs(eps_l - 1.0) <= 1e-14:
        a_dir = np.inf
    else:
        a_dir = (eps_l - 1.0) ** -1
    return calculate_epsilon(q, a_dir, b, c)


def directional_dependent_W(
    q_vector: np.ndarray,
    epsilon_tensor: np.ndarray,
    lam: float,
    b: float = 0.0,
    c: float = 0.0,
    *,
    assume_valid: bool = False,
) -> float:
    r"""
    Directionally screened Coulomb interaction for a dielectric tensor.

    This follows the same reduced convention as :func:`W`; the global Coulomb
    prefactor used by matrix-element calculations is applied elsewhere.
    Matrix-element callers supply ``b`` and ``c`` so tensor input uses the same
    q-dependent model as the scalar dielectric path, with only
    ``epsilon_inf`` made directional. The ``b=c=0`` defaults are kept only for
    backward-compatible direct helper calls.

    .. math::
        W(Q) = \frac{1}{\epsilon_{\mathrm{model}}(Q)(|Q|^2 + \lambda^2)}

    where ``epsilon_model`` is built from
    ``epsilon_inf_L = Qhat.T @ epsilon_tensor @ Qhat``.
    """
    q = np.asarray(q_vector, dtype=float)
    q_mag = float(np.linalg.norm(q))
    eps_model = directional_dependent_epsilon(
        q, epsilon_tensor, b, c, assume_valid=assume_valid
    )
    return W(q_mag, eps_model, lam)

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


def calculate_kpoints(poscar_path, kspacing):
    """
    Calculate the number of k-points in each reciprocal lattice direction based on a given KSPACING.
    
    Parameters:
        poscar_path (str): Path to the POSCAR file.
        kspacing (float): Value of KSPACING in Å^-1.
    
    Returns:
        tuple: A tuple containing the number of k-points (Ni1, Ni2, Ni3).
    """
    # Read the POSCAR file
    with open(poscar_path, 'r') as file:
        lines = file.readlines()
    
    # Extract lattice vectors
    a1 = np.array([float(x) for x in lines[2].split()])
    a2 = np.array([float(x) for x in lines[3].split()])
    a3 = np.array([float(x) for x in lines[4].split()])
    
    # Calculate the volume of the unit cell
    volume = np.dot(a1, np.cross(a2, a3))
    
    # Calculate reciprocal lattice vectors
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume
    
    # Calculate Ni for each direction
    Ni1 = max(1, np.ceil(np.linalg.norm(b1) / kspacing))
    Ni2 = max(1, np.ceil(np.linalg.norm(b2) / kspacing))
    Ni3 = max(1, np.ceil(np.linalg.norm(b3) / kspacing))
    
    return (Ni1, Ni2, Ni3)

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


def _as_path_list(value: Union[str, List[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _nscf_folder_index(folder: str, auger_type: str) -> Optional[int]:
    name = os.path.basename(os.path.normpath(folder))
    prefix = f"NSCF_{auger_type}_"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def _nscf_frac_key(frac: Sequence[float], digits: int = 8) -> Tuple[float, float, float]:
    folded = fold_kpoint_to_first_bz(np.asarray(frac, dtype=float), convention="vasp_centered")
    return tuple(round(float(x), digits) for x in folded)


def _read_nscf_kpoints_file(folder: str) -> List[List[float]]:
    path = os.path.join(folder, "KPOINTS")
    with open(path, "r") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"KPOINTS in {folder} is too short to parse.")
    try:
        count = int(lines[1].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not read KPOINTS count in {folder}.") from exc

    mode = lines[2].lower()
    if not (mode.startswith("r") or mode.startswith("k")):
        print(
            f"  Warning: KPOINTS in {folder} does not look like reciprocal "
            f"coordinates ('{lines[2]}'). Interpreting first 3 columns as fractional."
        )

    rows = lines[3:3 + count]
    if len(rows) != count:
        raise ValueError(f"KPOINTS in {folder} declares {count} points but contains {len(rows)}.")
    kpoints: List[List[float]] = []
    for row in rows:
        parts = row.split()
        if len(parts) < 3:
            raise ValueError(f"Malformed KPOINTS row in {folder}: {row}")
        kpoints.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return kpoints


def _read_incar_float_setting(folder: str, key_name: str) -> Optional[float]:
    path = os.path.join(folder, "INCAR")
    if not os.path.isfile(path):
        return None
    key_name = key_name.upper()
    with open(path, "r") as fh:
        for line in fh:
            raw = line.split("#", 1)[0].split("!", 1)[0]
            if "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            if key.strip().upper() != key_name:
                continue
            try:
                return float(value.split()[0])
            except (IndexError, ValueError):
                return None
    return None


def _row_has_location_columns(row: Dict, required_cols: Sequence[str]) -> bool:
    for col in required_cols:
        if col not in row:
            return False
        value = row[col]
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
    return True


def _active_nscf_settings(
    nscf_settings: Optional[Dict[str, Union[int, str]]],
    efermi: Optional[float],
) -> Dict[str, Union[int, float, str]]:
    settings: Dict[str, Union[int, float, str]] = {
        "ALGO": "Normal",
        "PREC": "Accurate",
        "ICHARG": 11,
        "LCHARG": "False",
        "LWAVE": "True",
        "ISYM": -1,
    }
    if efermi is not None:
        settings["EFERMI"] = efermi
    else:
        print("  Warning: no efermi provided; set EFERMI to keep NSCF energy references consistent.")
    if nscf_settings:
        settings.update(nscf_settings)
    return settings


def _patch_nscf_incar(incar_path: str, active_settings: Dict[str, Union[int, float, str]]) -> None:
    with open(incar_path, "r") as f:
        lines = f.readlines()
    with open(incar_path, "w") as f:
        written_keys: set = set()
        f.write("\n # Auto-generated INCAR for NSCF calculation\n")
        for line in lines:
            key = line.split("=")[0].strip()
            if key in active_settings:
                f.write(f"{key} = {active_settings[key]}\n")
                written_keys.add(key)
            else:
                f.write(line)
        f.write("\n")
        for key, val in active_settings.items():
            if key not in written_keys:
                f.write(f"{key} = {val}\n")


def _write_nscf_kpoints(path: str, folder_kpts: Sequence[Sequence[float]]) -> None:
    with open(path, "w") as kf:
        kf.write("K-points for NSCF calculation\n")
        kf.write(f"{len(folder_kpts)}\n")
        kf.write("Reciprocal\n")
        for frac in folder_kpts:
            kf.write(f"  {frac[0]:.8f}  {frac[1]:.8f}  {frac[2]:.8f}  1\n")


def _create_nscf_inputs_continued(
    scf_folder: str,
    nscf_folder: str,
    exact_kpoints_table: Union[str, List[str]],
    auger_type: str,
    num_kpoints_per_file: Union[int, str],
    nscf_settings: Optional[Dict[str, Union[int, str]]],
    efermi: Optional[float],
    continue_from_folders: Union[str, List[str]],
) -> None:
    exact_kpoints_table = _as_path_list(exact_kpoints_table)
    continue_folders = _as_path_list(continue_from_folders)

    print(f"\n{'-'*80}")
    print("Continuing NSCF input generation from previous folders.")
    print(
        "  Reminder: exact_kpoints_table should include both previous exact-kpoint "
        "CSV files and newly generated CSV files."
    )

    if auger_type == "eeh":
        scf_frac_cols   = ["k1_frac", "k3_frac", "k4_frac"]
        scf_idx_cols    = ["k1_index", "k3_index", "k4_index"]
        scf_wc_cols     = ["k1_wc_index", "k3_wc_index", "k4_wc_index"]
        scf_nscf_cols   = ["k1_nscf_index", "k3_nscf_index", "k4_nscf_index"]
        target_wc_col   = "k2_wc_index"
        target_nscf_col = "k2_nscf_index"
        choose_key = "k2_target_frac_mapped"
    else:
        scf_frac_cols   = ["k1_frac", "k2_frac", "k3_frac"]
        scf_idx_cols    = ["k1_index", "k2_index", "k3_index"]
        scf_wc_cols     = ["k1_wc_index", "k2_wc_index", "k3_wc_index"]
        scf_nscf_cols   = ["k1_nscf_index", "k2_nscf_index", "k3_nscf_index"]
        target_wc_col   = "k4_wc_index"
        target_nscf_col = "k4_nscf_index"
        choose_key = "k4_target_frac_mapped"

    required_cols = [target_wc_col, target_nscf_col] + scf_wc_cols + scf_nscf_cols

    previous_locations: Dict[Tuple[float, float, float], Tuple[int, int]] = {}
    previous_folder_indices: List[int] = []
    previous_kpoint_count = 0
    efermi_mismatches = 0

    for folder in continue_folders:
        if not os.path.isdir(folder):
            print(f"  Warning: previous NSCF folder not found, skipping: {folder}")
            continue
        folder_num = _nscf_folder_index(folder, auger_type)
        if folder_num is None:
            print(f"  Warning: could not infer NSCF folder number from {folder}; skipping.")
            continue

        try:
            kpoints = _read_nscf_kpoints_file(folder)
        except Exception as exc:
            print(f"  Warning: could not read previous KPOINTS from {folder}: {exc}")
            continue

        previous_folder_indices.append(folder_num)
        previous_kpoint_count += len(kpoints)
        for local_idx, frac in enumerate(kpoints):
            key = _nscf_frac_key(frac)
            existing = previous_locations.get(key)
            if existing is not None and existing != (folder_num, local_idx):
                print(
                    "  Warning: duplicate previous k-point detected; keeping "
                    f"first location {existing} and ignoring {(folder_num, local_idx)}."
                )
                continue
            previous_locations[key] = (folder_num, local_idx)

        old_efermi = _read_incar_float_setting(folder, "EFERMI")
        if old_efermi is not None and efermi is not None and abs(old_efermi - float(efermi)) > 1e-8:
            efermi_mismatches += 1
            print(
                f"  Warning: EFERMI mismatch in {folder}: previous {old_efermi}, "
                f"current {float(efermi)}."
            )

    if not previous_folder_indices:
        raise ValueError("continue_from_folders was provided, but no valid previous NSCF folders were loaded.")

    next_folder_index = max(previous_folder_indices) + 1
    print(f"  Previous NSCF folders loaded: {len(previous_folder_indices)}")
    print(f"  Previous KPOINTS rows read:   {previous_kpoint_count:,}")
    print(f"  Unique previous k-points:     {len(previous_locations):,}")
    print(f"  Next NSCF folder suffix:      _{next_folder_index}")
    if efermi_mismatches == 0:
        print("  EFERMI check: no mismatch detected.")

    table_rows: List[Tuple[str, List[Dict], bool]] = []
    exact_kpoints_dicts: List[Dict] = []
    print(f"  Exact-kpoint CSV files loaded: {len(exact_kpoints_table)}")
    for table_file in exact_kpoints_table:
        rows = read_csv([table_file])
        had_locations = all(_row_has_location_columns(row, required_cols) for row in rows)
        table_rows.append((table_file, rows, had_locations))
        exact_kpoints_dicts.extend(rows)
        status = "already annotated" if had_locations else "will be annotated"
        print(f"    {table_file} ({len(rows):,} rows, {status})")

    key_to_location: Dict[tuple, Tuple[int, int]] = {}
    key_to_new_index: Dict[tuple, int] = {}
    new_frac_to_index: Dict[Tuple[float, float, float], int] = {}
    new_kpt_fracs: List[List[float]] = []
    reused_previous_keys: set = set()

    def register_kpoint(key: tuple, frac: Sequence[float]) -> None:
        if key in key_to_location or key in key_to_new_index:
            return
        frac_key = _nscf_frac_key(frac)
        if frac_key in previous_locations:
            key_to_location[key] = previous_locations[frac_key]
            reused_previous_keys.add(frac_key)
            return
        if frac_key in new_frac_to_index:
            key_to_new_index[key] = new_frac_to_index[frac_key]
            return
        new_frac_to_index[frac_key] = len(new_kpt_fracs)
        key_to_new_index[key] = len(new_kpt_fracs)
        new_kpt_fracs.append(list(fold_kpoint_to_first_bz(np.asarray(frac, dtype=float), convention="vasp_centered")))

    for item in exact_kpoints_dicts:
        target_frac = list(item[choose_key])
        register_kpoint(("target", _nscf_frac_key(target_frac)), target_frac)
        for fc, ic in zip(scf_frac_cols, scf_idx_cols):
            register_kpoint(("scf", int(item[ic])), list(item[fc]))

    total_new = len(new_kpt_fracs)
    if num_kpoints_per_file != "all":
        num_kpoints_per_file = int(num_kpoints_per_file)
        if num_kpoints_per_file > total_new and total_new > 0:
            print(
                f"Requested num_kpoints_per_file={num_kpoints_per_file} exceeds "
                f"new unique k-points ({total_new}). Using 'all' instead."
            )
            num_kpoints_per_file = "all"

    if total_new == 0:
        kpts_per_folder: List[int] = []
    elif num_kpoints_per_file == "all":
        kpts_per_folder = [total_new]
    else:
        num_new_folders = int(np.ceil(total_new / int(num_kpoints_per_file)))
        kpts_per_folder = [int(num_kpoints_per_file)] * (num_new_folders - 1)
        kpts_per_folder.append(total_new - (num_new_folders - 1) * int(num_kpoints_per_file))

    boundaries: List[int] = [0]
    for cnt in kpts_per_folder:
        boundaries.append(boundaries[-1] + cnt)

    print(f"  Old k-points reused:         {len(reused_previous_keys):,}")
    print(f"  New unique k-points to write:{total_new:,}")
    print(f"  New NSCF folders to create:  {len(kpts_per_folder)}")
    if kpts_per_folder:
        print(f"  New k-points per folder:     {kpts_per_folder}")

    new_index_to_location: Dict[int, Tuple[int, int]] = {}
    for gi in range(total_new):
        for fi in range(len(kpts_per_folder)):
            if boundaries[fi] <= gi < boundaries[fi + 1]:
                folder_num = next_folder_index + fi
                new_index_to_location[gi] = (folder_num, gi - boundaries[fi])
                break
    for key, new_idx in key_to_new_index.items():
        key_to_location[key] = new_index_to_location[new_idx]

    scf_folder = scf_folder.rstrip("/\\")
    nscf_folder = nscf_folder.rstrip("/\\")
    active_settings = _active_nscf_settings(nscf_settings, efermi)

    for fi, count in enumerate(kpts_per_folder):
        folder_num = next_folder_index + fi
        cur_folder = os.path.join(nscf_folder, f"NSCF_{auger_type}_{folder_num}")
        if os.path.exists(cur_folder):
            raise FileExistsError(
                f"Refusing to modify existing NSCF folder during continuation: {cur_folder}"
            )
        os.makedirs(cur_folder, exist_ok=False)

        for fname in ("POTCAR", "INCAR", "POSCAR", "CHGCAR"):
            src = os.path.join(scf_folder, fname)
            dst = os.path.join(cur_folder, fname)
            try:
                shutil.copyfile(src, dst)
            except FileNotFoundError:
                print(f"  Warning: could not copy '{fname}' to folder {folder_num}")

        wav = os.path.join(cur_folder, "WAVECAR")
        try:
            os.remove(wav)
        except OSError:
            pass

        incar_path = os.path.join(cur_folder, "INCAR")
        try:
            _patch_nscf_incar(incar_path, active_settings)
        except FileNotFoundError:
            print(f"  Warning: could not modify INCAR in folder {folder_num}")

        folder_kpts = new_kpt_fracs[boundaries[fi]:boundaries[fi + 1]]
        _write_nscf_kpoints(os.path.join(cur_folder, "KPOINTS"), folder_kpts)
        print(f"  Created new folder NSCF_{auger_type}_{folder_num}: {cur_folder}")
        print(f"    K-points in this folder: {count:,}")

    for item in exact_kpoints_dicts:
        target_frac = list(item[choose_key])
        t_key = ("target", _nscf_frac_key(target_frac))
        t_wc, t_nscf = key_to_location[t_key]
        item[target_wc_col] = t_wc
        item[target_nscf_col] = t_nscf

        for idx_col, wc_col, nscf_col in zip(scf_idx_cols, scf_wc_cols, scf_nscf_cols):
            s_key = ("scf", int(item[idx_col]))
            s_wc, s_nscf = key_to_location[s_key]
            item[wc_col] = s_wc
            item[nscf_col] = s_nscf

        item.pop("wavecar", None)
        item.pop("wc_index", None)

    for table_file, rows, had_locations in table_rows:
        if had_locations:
            print(f"  Leaving previously annotated exact-kpoint CSV unchanged: {table_file}")
            continue
        print(f"  Writing NSCF indices to exact-kpoint CSV: {table_file}")
        write_to_csv(rows, table_file.replace(".csv", ""))

    print("\nNSCF continuation input creation complete!")
    print(f"{'-'*80}\n")


def create_nscf_inputs(
    scf_folder: str,
    nscf_folder: str,
    exact_kpoints_table: Union[str, List[str]],
    auger_type: str = "eeh",
    num_kpoints_per_file: Union[int, str] = "all",
    nscf_settings: Optional[Dict[str, Union[int, str]]] = None,
    efermi: Optional[float] = None,
    continue_from_folders: Union[str, List[str]] = [],
) -> None:
    """
    Generate VASP NSCF input files from an SCF folder and exact-kpoint table.

    Collects all physically unique k-points needed across all Auger pairs.
    Target k-points and SCF-side k-points are deduplicated together by folded
    fractional coordinate, so the same coordinate is never written twice just
    because it appears in different pair roles. The unique coordinates are
    distributed globally across NSCF folders, exactly
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
    efermi : float, optional
        Fermi level to use for the NSCF calculation as a reference level.
    continue_from_folders : str or list[str], optional
        Previous ``NSCF_<type>_<N>`` folders to reuse read-only. New folders
        start after the largest previous suffix, and existing KPOINTS rows are
        reused when exact-kpoint table rows require an already generated
        k-point.
    """
    if continue_from_folders:
        return _create_nscf_inputs_continued(
            scf_folder=scf_folder,
            nscf_folder=nscf_folder,
            exact_kpoints_table=exact_kpoints_table,
            auger_type=auger_type,
            num_kpoints_per_file=num_kpoints_per_file,
            nscf_settings=nscf_settings,
            efermi=efermi,
            continue_from_folders=continue_from_folders,
        )

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

    # ── Step 1: Collect all physically unique k-points in encounter order ──
    # Logical roles are still tracked so each CSV column can be assigned to the
    # correct NSCF row, but the KPOINTS file itself is keyed by folded
    # fractional coordinate across both target and SCF-side k-points.
    unique_kpoints: Dict[Tuple[float, float, float], list] = {}
    role_to_frac_key: Dict[tuple, Tuple[float, float, float]] = {}
    target_frac_keys: set = set()
    scf_frac_keys: set = set()

    def register_role(role_key: tuple, frac: Sequence[float], role_type: str) -> None:
        frac_key = _nscf_frac_key(frac)
        existing = role_to_frac_key.get(role_key)
        if existing is not None:
            if existing != frac_key:
                raise ValueError(
                    f"Inconsistent k-point coordinate for role {role_key}: "
                    f"first {existing}, later {frac_key}."
                )
            return

        role_to_frac_key[role_key] = frac_key
        if role_type == "target":
            target_frac_keys.add(frac_key)
        else:
            scf_frac_keys.add(frac_key)
        if frac_key not in unique_kpoints:
            folded = fold_kpoint_to_first_bz(
                np.asarray(frac, dtype=float),
                convention="vasp_centered",
            )
            unique_kpoints[frac_key] = [float(x) for x in folded]

    for item in exact_kpoints_dicts:
        # Target k-point (k2 in eeh, k4 in ehh)
        target_frac = list(item[choose_key])
        register_role(("target", _nscf_frac_key(target_frac)), target_frac, "target")
        # SCF k-points (unique by original SCF k-index)
        for fc, ic in zip(scf_frac_cols, scf_idx_cols):
            register_role(("scf", int(item[ic])), list(item[fc]), "scf")

    all_kpt_keys  = list(unique_kpoints.keys())
    all_kpt_fracs = list(unique_kpoints.values())
    total_unique  = len(all_kpt_keys)
    n_target_unique = len(target_frac_keys)
    n_scf_unique = len(scf_frac_keys)
    n_role_overlap = len(target_frac_keys & scf_frac_keys)

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
    if n_role_overlap:
        print(f"    Shared target/SCF coordinates reused: {n_role_overlap:,}")
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
    active_settings = _active_nscf_settings(nscf_settings, efermi)

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
                f.write("\n")
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
        t_key = role_to_frac_key[("target", _nscf_frac_key(target_frac))]
        t_wc, t_nscf = key_to_location[t_key]
        item[target_wc_col]   = t_wc
        item[target_nscf_col] = t_nscf

        for idx_col, wc_col, nscf_col in zip(scf_idx_cols, scf_wc_cols, scf_nscf_cols):
            s_key = role_to_frac_key[("scf", int(item[idx_col]))]
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


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive k-mesh refinement for NSCF around band extrema
# ═══════════════════════════════════════════════════════════════════════════

def _create_kpoints_mapping(
    folder_kpoint_counts: List[Tuple[str, int]],
    output_path: str,
) -> str:
    """
    Create a CSV mapping table that maps global k-point index ranges to
    WAVECAR files.

    Called inside :func:`generate_adaptive_nscf_inputs` after the folders
    are created, so that :meth:`AugerCalculator.set_kpoints_mapping` can
    later translate the global ``k*_index`` values (produced by
    ``parse_BS_data`` in multi-folder mode) into per-WAVECAR local indices
    and 1-based ``k*_wc_index`` values.

    Parameters
    ----------
    folder_kpoint_counts : list of (folder_path, nkpoints)
        Ordered list of (VASP folder path, number of k-points in that
        folder).  The order must match the order that will be passed to
        ``parse_BS_data(folder_path=[...])``.
    output_path : str
        Full path for the output CSV file.

    Returns
    -------
    str
        The *output_path* that was written.
    """
    rows: List[Dict] = []
    cumulative = 0
    for folder_path, nkpts in folder_kpoint_counts:
        wavecar = folder_path.rstrip("/") + "/WAVECAR"
        rows.append({
            "kpoint_first_index(included)": cumulative,
            "kpoint_last_index(excluded)": cumulative + nkpts,
            "wavecar_path": wavecar,
        })
        cumulative += nkpts

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  ✓ Wrote k-point mapping: {output_path}  ({len(rows)} entries, "
          f"{cumulative} total k-points)")
    return output_path


def _normalise_kgrid(kgrid: Union[int, str, Sequence[int]]) -> Tuple[int, int, int]:
    """Convert a user or VASP k-grid specification to a 3-integer tuple."""
    if isinstance(kgrid, str):
        kgrid = ast.literal_eval(kgrid)

    if isinstance(kgrid, int):
        return (kgrid, kgrid, kgrid)

    mesh_dims = tuple(int(x) for x in kgrid)
    if len(mesh_dims) != 3:
        raise ValueError(f"kgrid must contain exactly 3 values; got {mesh_dims}")
    return mesh_dims


def _expand_irr_kpoints_for_adaptive(
    data_energies: np.ndarray,
    stored_kpoints_frac: np.ndarray,
    mesh_dims: Tuple[int, int, int],
    poscar_path: str,
    match_tolerance: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand irreducible-wedge band energies to the full BZ mesh.

    This mirrors ``PairGenerator._expand_irr_kpoints`` but works directly on
    arrays used by ``generate_adaptive_nscf_inputs``.  The expanded arrays are
    used only to locate band extrema and avoid resampling symmetry-equivalent
    coarse-grid points.
    """
    data_energies = np.asarray(data_energies)
    stored_kpoints_frac = np.asarray(stored_kpoints_frac, dtype=float)

    if data_energies.ndim != 2:
        raise ValueError("data_energies must have shape (nbands, nkpoints).")
    if stored_kpoints_frac.ndim != 2 or stored_kpoints_frac.shape[1] != 3:
        raise ValueError("stored_kpoints_frac must have shape (nkpoints, 3).")
    if data_energies.shape[1] != stored_kpoints_frac.shape[0]:
        raise ValueError(
            "Number of stored k-points does not match the energy array: "
            f"{stored_kpoints_frac.shape[0]} k-points vs "
            f"{data_energies.shape[1]} energy columns."
        )

    structure = Structure.from_file(poscar_path)
    analyzer = SpacegroupAnalyzer(structure)
    full_kpts_frac, ir_mapping = analyzer.get_ir_reciprocal_mesh_map(mesh_dims)
    full_kpts_frac = np.asarray(full_kpts_frac, dtype=float)
    ir_mapping = np.asarray(ir_mapping, dtype=int)

    irr_to_full: Dict[int, List[int]] = defaultdict(list)
    for full_idx, irr_idx in enumerate(ir_mapping):
        irr_to_full[int(irr_idx)].append(full_idx)

    irr_indices = np.array(sorted(irr_to_full.keys()), dtype=int)
    irr_kpts_frac = full_kpts_frac[irr_indices]

    stored_folded = np.array([
        fold_kpoint_to_first_bz(k, convention="vasp_centered")
        for k in stored_kpoints_frac
    ])
    irr_folded = np.array([
        fold_kpoint_to_first_bz(k, convention="vasp_centered")
        for k in irr_kpts_frac
    ])

    diff = stored_folded[:, None, :] - irr_folded[None, :, :]
    diff -= np.round(diff)
    dist2 = np.sum(diff ** 2, axis=2)
    best_match = np.argmin(dist2, axis=1)
    min_dists = np.sqrt(np.min(dist2, axis=1))

    bad = np.where(min_dists > match_tolerance)[0]
    if len(bad) > 0:
        raise ValueError(
            f"{len(bad)} stored k-points could not be matched to the "
            f"symmetry-reduced {mesh_dims} mesh within tolerance "
            f"{match_tolerance}. Max error: {min_dists[bad].max():.4f}. "
            "Check poscar_path, kgrid, and the source KPOINTS mesh."
        )

    nbands = data_energies.shape[0]
    n_full = len(full_kpts_frac)
    new_energies = np.empty((nbands, n_full), dtype=data_energies.dtype)
    assigned = np.zeros(n_full, dtype=bool)

    stored_to_irr_idx = irr_indices[best_match]
    for stored_idx, irr_idx in enumerate(stored_to_irr_idx):
        full_indices = irr_to_full[int(irr_idx)]
        new_energies[:, full_indices] = data_energies[:, stored_idx, np.newaxis]
        assigned[full_indices] = True

    if not np.all(assigned):
        missing = int(np.count_nonzero(~assigned))
        raise ValueError(
            f"Could not assign energies for {missing} full-BZ k-points. "
            "The stored irreducible k-points do not cover the inferred mesh. "
            "Check poscar_path and kgrid."
        )

    full_kpts_folded = np.array([
        fold_kpoint_to_first_bz(k, convention="vasp_centered")
        for k in full_kpts_frac
    ])
    weights = np.full(n_full, 1.0 / np.prod(mesh_dims), dtype=float)

    print(
        f"  Expanded k-points for adaptive search: "
        f"{len(stored_kpoints_frac)} irr -> {n_full} full BZ "
        f"(mesh {mesh_dims[0]}x{mesh_dims[1]}x{mesh_dims[2]})"
    )
    return new_energies, full_kpts_folded, weights


def generate_adaptive_nscf_inputs(
    vasp_folder: str,
    output_base: str,
    firstCB_index: int,
    dense_grid: Union[int, Tuple[int, int, int]] = 5,
    radius_frac: float = 0.15,
    n_extrema: int = 1,
    num_kpoints_per_file: Union[int, str] = "all",
    chgcar_path: Optional[str] = None,
    nscf_settings: Optional[Dict[str, Union[int, str]]] = None,
    exact_kpoint_mode: bool = False,
    poscar_path: Optional[str] = None,
    kgrid: Optional[Union[int, str, Sequence[int]]] = None,
    match_tolerance: float = 0.05,
) -> List[str]:
    """
    Read a coarse VASP calculation, sample k-points densely around CBM/VBM,
    and create NSCF input folders ready for VASP.

    The resulting NSCF folders, once computed by VASP, can be fed directly
    into ``AugerCalculator.parse_BS_data(folder_path=[...])`` together
    with the original coarse folder. For nearest-kpoint workflows, the source
    calculation should already contain the full BZ grid (typically
    ``ISYM=-1``). For exact-kpoint workflows, set ``exact_kpoint_mode=True``
    to expand a symmetry-reduced source mesh to the full BZ before locating
    extrema.

    Parameters
    ----------
    vasp_folder : str
        Path to a completed (coarse) VASP calculation with at least
        ``vasprun.xml``, ``EIGENVAL``, ``KPOINTS``, ``POSCAR``, and
        ``POTCAR``.
    output_base : str
        Base directory where NSCF folders will be created
        (``NSCF_adaptive_1/``, ``NSCF_adaptive_2/``, …).
    firstCB_index : int
        Index of the first conduction band in the VASP output.  This is
        needed to identify which bands correspond to the CBM and VBM.
    dense_grid : int or (int, int, int)
        Number of sampling points per direction around each extremum
        in fractional coordinates.  A scalar is broadcast to all three
        directions.
    radius_frac : float
        Half-width of the sampling cube around each extremum in
        fractional reciprocal-space units (e.g. 0.15 means ±0.15
        around the extremum in each direction).
    n_extrema : int
        How many CBM / VBM extrema to refine around (sorted by energy).
        Use >1 if the material has degenerate pockets (e.g. L-point
        valleys in PbSe).
    num_kpoints_per_file : int or 'all'
        Maximum k-points per NSCF folder.  ``'all'`` puts everything in
        one folder.
    chgcar_path : str or None
        Path to a converged ``CHGCAR`` to copy into each NSCF folder.
        If ``None``, defaults to ``vasp_folder/CHGCAR``.
    nscf_settings : dict or None
        INCAR overrides for NSCF mode.  Defaults to standard NSCF
        settings (ICHARG=11, ISYM=-1, LWAVE=True, etc.).
    exact_kpoint_mode : bool
        If ``True``, treat ``vasp_folder`` as a symmetry-reduced calculation
        and expand its irreducible k-points to the full BZ for the extrema
        search. This mode requires a regular VASP k-mesh and a POSCAR.
    poscar_path : str or None
        Structure file used for symmetry expansion when
        ``exact_kpoint_mode=True``. Defaults to ``vasp_folder/POSCAR``.
    kgrid : int, sequence of 3 ints, str, or None
        K-point mesh dimensions used for symmetry expansion. If ``None``,
        the mesh is inferred from the source KPOINTS read by ``vasprun.xml``.
    match_tolerance : float
        Fractional-coordinate tolerance for matching stored irreducible
        k-points to the symmetry-reduced mesh.

    Returns
    -------
    list[str]
        Paths to the created NSCF folders, ready for VASP execution.
    """
    vasp_folder = vasp_folder.rstrip("/")
    output_base = output_base.rstrip("/")
    if chgcar_path is None:
        chgcar_path = f"{vasp_folder}/CHGCAR"

    if isinstance(dense_grid, int):
        dense_grid = (dense_grid, dense_grid, dense_grid)

    print(f"\n{'─'*80}")
    print(f"🔬 Adaptive k-mesh refinement")
    print(f"   Source folder:  {vasp_folder}")
    print(f"   Dense grid:     {dense_grid[0]}×{dense_grid[1]}×{dense_grid[2]}")
    print(f"   Radius (frac):  ±{radius_frac}")
    print(f"   # extrema:     {n_extrema}")
    print(f"   Exact mode:    {exact_kpoint_mode}")
    print(f"{'─'*80}")

    # ── Step 1: Read coarse band structure ───────────────────────────────
    bs = BSVasprun(f"{vasp_folder}/vasprun.xml")
    bandstructure = bs.get_band_structure(
        kpoints_filename=f"{vasp_folder}/KPOINTS"
    )
    data = bandstructure.bands[Spin(1)]
    Efermi = bandstructure.efermi
    data_shifted = data - Efermi

    firstCB, lastVB = firstCB_index, firstCB_index - 1

    eigenvalues = Eigenval(f"{vasp_folder}/EIGENVAL")
    source_nkpt = eigenvalues.nkpt
    if source_nkpt != len(bandstructure.kpoints):
        raise ValueError(
            "EIGENVAL and vasprun.xml disagree on the number of k-points: "
            f"{source_nkpt} vs {len(bandstructure.kpoints)}."
        )

    stored_frac_coords = np.array([
        fold_kpoint_to_first_bz(
            bandstructure.kpoints[i].frac_coords,
            convention="vasp_centered",
        )
        for i in range(source_nkpt)
    ])

    if stored_frac_coords.shape[0] != data_shifted.shape[1]:
        raise ValueError(
            "Band-structure k-point count does not match the energy array: "
            f"{stored_frac_coords.shape[0]} k-points vs "
            f"{data_shifted.shape[1]} energy columns."
        )

    data_for_extrema = data_shifted
    frac_coords = stored_frac_coords

    if exact_kpoint_mode:
        poscar_for_expansion = poscar_path or f"{vasp_folder}/POSCAR"
        if kgrid is None:
            try:
                resolved_kgrid = _normalise_kgrid(bs.kpoints.kpts[0])
            except Exception as exc:
                raise ValueError(
                    "exact_kpoint_mode=True requires a regular source k-grid. "
                    "Could not infer it from vasprun.xml/KPOINTS; pass kgrid "
                    "explicitly, for example kgrid=(6, 6, 6)."
                ) from exc
        else:
            resolved_kgrid = _normalise_kgrid(kgrid)

        data_for_extrema, frac_coords, _weights = _expand_irr_kpoints_for_adaptive(
            data_energies=data_shifted,
            stored_kpoints_frac=stored_frac_coords,
            mesh_dims=resolved_kgrid,
            poscar_path=poscar_for_expansion,
            match_tolerance=match_tolerance,
        )

    cb_energies = data_for_extrema[firstCB]  # shape (nkpt,)
    vb_energies = data_for_extrema[lastVB]   # shape (nkpt,)

    # ── Step 2: Identify CBM and VBM k-point locations ──────────────────
    # CBM: lowest energy k-points in the first CB
    cbm_indices = np.argsort(cb_energies)[:n_extrema]
    # VBM: highest energy k-points in the last VB
    vbm_indices = np.argsort(-vb_energies)[:n_extrema]

    extrema_kpoints = []
    for idx in cbm_indices:
        kf = frac_coords[idx]
        extrema_kpoints.append(("CBM", kf, float(cb_energies[idx])))
    for idx in vbm_indices:
        kf = frac_coords[idx]
        extrema_kpoints.append(("VBM", kf, float(vb_energies[idx])))

    print(f"\n  Band extrema identified:")
    for label, kf, en in extrema_kpoints:
        print(f"    {label}: k=({kf[0]:.4f}, {kf[1]:.4f}, {kf[2]:.4f})  "
              f"E-Ef={en:+.4f} eV")

    # ── Step 3: Generate dense k-point grid around each extremum ────────
    # Use a single rounding precision everywhere to avoid mismatched dedup
    _ROUND = 8
    dense_kpoints_set: Dict[tuple, list] = {}  # rounded frac → frac

    for label, center, _ in extrema_kpoints:
        offsets = [
            np.linspace(-radius_frac, radius_frac, dense_grid[d])
            for d in range(3)
        ]
        for di in offsets[0]:
            for dj in offsets[1]:
                for dk in offsets[2]:
                    kpt = np.array(center) + np.array([di, dj, dk])
                    kpt_folded = fold_kpoint_to_first_bz(kpt, "vasp_centered")
                    key = tuple(np.round(kpt_folded, _ROUND))
                    if key not in dense_kpoints_set:
                        dense_kpoints_set[key] = kpt_folded.tolist()

    # Remove k-points that already exist in the coarse grid
    existing_keys = set()
    for kf in frac_coords:
        existing_keys.add(tuple(np.round(kf, _ROUND)))

    new_kpoints = []
    for key, kf in dense_kpoints_set.items():
        if key not in existing_keys:
            new_kpoints.append(kf)

    total_new = len(new_kpoints)
    print(f"\n  New k-points to sample: {total_new}")
    print(f"  (Excluded {len(dense_kpoints_set) - total_new} already in coarse grid)")

    if total_new == 0:
        print("  ⚠  No new k-points needed — coarse grid already covers extrema.")
        print(f"{'─'*80}\n")
        return []

    # ── Step 4: Distribute k-points across NSCF folders ─────────────────
    if num_kpoints_per_file == "all":
        num_folders = 1
        kpts_per_folder = [total_new]
    else:
        num_folders = int(np.ceil(total_new / num_kpoints_per_file))
        kpts_per_folder = [num_kpoints_per_file] * (num_folders - 1)
        kpts_per_folder.append(total_new - (num_folders - 1) * num_kpoints_per_file)

    active_settings = _active_nscf_settings(nscf_settings, Efermi)

    os.makedirs(output_base, exist_ok=True)
    created_folders: List[str] = []
    offset = 0

    for fi in range(num_folders):
        cur_folder = f"{output_base}/NSCF_adaptive_{fi + 1}/"
        os.makedirs(cur_folder, exist_ok=True)

        # Copy input files
        for fname in ("POTCAR", "POSCAR"):
            try:
                shutil.copyfile(f"{vasp_folder}/{fname}", f"{cur_folder}{fname}")
            except FileNotFoundError:
                print(f"  ⚠  Could not copy '{fname}' to folder {fi + 1}")

        # Copy CHGCAR from specified path
        try:
            shutil.copyfile(chgcar_path, f"{cur_folder}CHGCAR")
        except FileNotFoundError:
            print(f"  ⚠  Could not copy CHGCAR from '{chgcar_path}' to folder {fi + 1}")

        # Remove stale WAVECAR
        try:
            os.remove(f"{cur_folder}WAVECAR")
        except OSError:
            pass

        # Write / patch INCAR
        incar_src = f"{vasp_folder}/INCAR"
        incar_dst = f"{cur_folder}INCAR"
        try:
            with open(incar_src, "r") as f:
                lines = f.readlines()
            with open(incar_dst, "w") as f:
                written_keys: set = set()
                f.write("\n # Auto-generated INCAR for adaptive NSCF calculation\n")
                for line in lines:
                    key = line.split("=")[0].strip()
                    if key in active_settings:
                        f.write(f"{key} = {active_settings[key]}\n")
                        written_keys.add(key)
                    else:
                        f.write(line)
                f.write("\n")
                for key, val in active_settings.items():
                    if key not in written_keys:
                        f.write(f"{key} = {val}\n")
        except FileNotFoundError:
            # No source INCAR — write from scratch
            with open(incar_dst, "w") as f:
                f.write("# Auto-generated INCAR for adaptive NSCF calculation\n")
                for key, val in active_settings.items():
                    f.write(f"{key} = {val}\n")

        # Write KPOINTS
        n_kpts = kpts_per_folder[fi]
        folder_kpts = new_kpoints[offset:offset + n_kpts]
        offset += n_kpts

        with open(f"{cur_folder}KPOINTS", "w") as kf:
            kf.write("Adaptive k-points around band extrema (NSCF)\n")
            kf.write(f"{n_kpts}\n")
            kf.write("Reciprocal\n")
            for frac in folder_kpts:
                kf.write(f"  {frac[0]:.8f}  {frac[1]:.8f}  {frac[2]:.8f}  1\n")

        created_folders.append(cur_folder)
        print(f"  ✓ Created folder {fi + 1}/{num_folders}: {cur_folder}  "
              f"({n_kpts} k-points)")

    # ── Step 5: Create k-point mapping CSV ─────────────────────────────
    # The mapping covers the coarse folder + all NSCF folders, in the
    # same order that will be passed to parse_BS_data(folder_path=[...]).
    folder_kpoint_counts: List[Tuple[str, int]] = [(vasp_folder, source_nkpt)]
    for fi in range(num_folders):
        folder_path = f"{output_base}/NSCF_adaptive_{fi + 1}"
        folder_kpoint_counts.append((folder_path, kpts_per_folder[fi]))

    mapping_csv = f"{output_base}/kpoints_mapping.csv"
    _create_kpoints_mapping(folder_kpoint_counts, mapping_csv)

    print(f"\n✓ Adaptive NSCF input creation complete!")
    print(f"  Total folders: {num_folders}")
    print(f"  Total new k-points: {total_new}")
    print(f"  Mapping CSV: {mapping_csv}")
    print(f"\n  Next steps:")
    print(f"    1. Run VASP in each NSCF folder")
    print(f"    2. Use parse_BS_data(folder_path=['{vasp_folder}'] + created_folders)")
    print(f"    3. After pairs are generated, call set_kpoints_mapping('{mapping_csv}')")
    print(f"{'─'*80}\n")

    return created_folders
