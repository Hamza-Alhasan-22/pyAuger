"""
AugerCalculator — main driver for Auger-recombination coefficient calculations.

Orchestrates the full workflow:
  1.  Parse / import band-structure data from VASP.
  2.  Compute carrier concentrations & quasi-Fermi levels.
  3.  Generate Auger pairs (eeh / ehh).
  4.  Calculate Coulomb matrix elements |M|².
  5.  Evaluate Auger coefficients C_n, C_p.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import BSVasprun, Eigenval, Vasprun
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from . import utilities as ut
from .constants import (
    ANGSTROM,
    BOHR_TO_ANGSTROM,
    CM_PER_ANGSTROM,
    EPSILON_0,
    HBAR,
    K_B_eV,
    M_E,
    eV,
)
from .matrix_elements import MatrixElements
from .pairs import Pair, PairGenerator


class AugerCalculator:
    """
    Main driver for calculating Auger recombination coefficients.

    Parameters
    ----------
    T : float
        Temperature in Kelvin (e.g. 300).
    nd : float
        Doping concentration in cm⁻³.
        Positive → n-type, negative → p-type, zero → intrinsic.

    Examples
    --------
    >>> calc = AugerCalculator(T=300, nd=1e17)
    >>> calc.parse_BS_data(folder_path="./vasp_scf", write_path="./parsed")
    >>> calc.import_parsed_BS_data(from_folder="./parsed")
    >>> calc.calculate_carrier_concentrations(delta_n=1e17)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, T: float, nd: float):
        self.T = T
        self.nd = nd
        self.parms_imported = False
        self.is_assigned_manually = False

        # Per-type containers
        self.is_auger_pairs_created: Dict[str, bool] = {}
        self.auger_pairs_dicts: Dict[str, list] = {}
        self.matrix_elements_dicts: Dict[str, list] = {}
        self.auger_coefficients: Dict[str, list] = {}
        self.is_matrix_elements_calculated: Dict[str, bool] = {}
        self.auger_pairs_object: Dict[str, Optional[PairGenerator]] = {}

        for typ in ("eeh", "ehh"):
            self.is_auger_pairs_created[typ] = False
            self.auger_pairs_dicts[typ] = []
            self.matrix_elements_dicts[typ] = []
            self.auger_coefficients[typ] = []
            self.is_matrix_elements_calculated[typ] = False
            self.auger_pairs_object[typ] = None

        self._print_banner()

    # ------------------------------------------------------------------
    # Pretty-printing helpers
    # ------------------------------------------------------------------
    def _print_banner(self):
        print(f"\n{'='*90}")
        print(f"{'AUGER RECOMBINATION CALCULATOR':^90}")
        print(f"{'='*90}")
        doping_type = "n-type" if self.nd > 0 else ("p-type" if self.nd < 0 else "intrinsic")
        print(f"│ {'Temperature:':<30} {f'{self.T} K':<55} │")
        print(f"│ {'Doping Concentration:':<30} {f'{self.nd:.2e} cm⁻³':<55} │")
        print(f"│ {'Doping Type:':<30} {doping_type:<55} │")
        print(f"{'='*90}\n")

    # ------------------------------------------------------------------
    # Band assignment (for zero-gap systems)
    # ------------------------------------------------------------------
    def assign_firstCB_and_lastVB(
        self,
        firstCB_index: int,
        lastVB_index: int,
    ) -> None:
        """
        Manually specify the first conduction-band and last valence-band indices.

        Call this **before** :meth:`parse_BS_data` or :meth:`import_parsed_BS_data`
        when the automatic detection would fail (e.g. zero-gap materials).
        """
        assert firstCB_index-lastVB_index == 1, "firstCB_index must be exactly 1 greater than lastVB_index."
        self.firstCB_index = firstCB_index
        self.lastVB_index = lastVB_index
        self.is_assigned_manually = True
        print(f"  First CB index: {firstCB_index}  |  Last VB index: {lastVB_index}")

    # ------------------------------------------------------------------
    # Band-structure parsing (VASP → .npy + band_info.txt)
    # ------------------------------------------------------------------
    def parse_BS_data(
        self,
        folder_path: str = ".",
        write_path: str = "",
        scissor_shift: float = 0.0,
        force_gap: Optional[float] = None,
    ) -> None:
        """
        Parse VASP output files and write band-structure arrays to disk.

        Reads ``KPOINTS``, ``EIGENVAL``, and ``vasprun.xml`` from *folder_path*
        and produces ``kgrid_X_XX.npy``, ``Egrid_X_XX.npy``, ``kw_X_XX.npy``,
        and ``band_info.txt`` in *write_path*.

        Parameters
        ----------
        folder_path : str
            Directory containing VASP output files.
        write_path : str
            Where to save parsed data (created if needed).
        scissor_shift : float
            Rigid shift (eV) applied to conduction bands.
        force_gap : float or None
            If given, overrides *scissor_shift* to enforce this exact gap.
        """
        print(f"\n{'─'*90}")
        print(f"📂 Parsing VASP data from: {folder_path}")
        print(f"{'─'*90}")

        bs = BSVasprun(f"{folder_path}/vasprun.xml")
        vrun = Vasprun(f"{folder_path}/vasprun.xml")

        dielectric_tensor = np.matrix(vrun.epsilon_static)
        dielectric = float(np.trace(dielectric_tensor) / 3)
        volume = bs.final_structure.volume  # Å³

        # Thomas–Fermi wave-vector
        nelec = vrun.parameters["NELECT"]
        n_val = nelec / volume  # Å⁻³
        n_val_au = n_val * BOHR_TO_ANGSTROM**3
        q_TF = 2 * (3 * n_val_au / np.pi) ** (1 / 6) / BOHR_TO_ANGSTROM  # Å⁻¹

        # Plasma frequency
        omega_p = np.sqrt(
            (n_val * 1e30) * eV**2 / (M_E * EPSILON_0)
        )  # rad/s
        
        # Structure analysis
        structure_ = Structure.from_file(f"{folder_path}/POSCAR")
        analyzer_ = SpacegroupAnalyzer(structure_)
        CrystalSystem = analyzer_.get_crystal_system()
        Space_group = analyzer_.get_space_group_symbol()

        reciprocal_lattice = bs.final_structure.lattice.reciprocal_lattice.matrix
        b1, b2, b3 = reciprocal_lattice.tolist()

        bandstructure = bs.get_band_structure(
            kpoints_filename=f"{folder_path}/KPOINTS"
        )
        material_name = bandstructure.structure.reduced_formula
        band_gap = round(bandstructure.get_band_gap()["energy"], 4)
        print(f"  Material: {material_name}  |  Band gap: {band_gap} eV")

        data = bandstructure.bands[Spin(1)]
        Efermi = round(bandstructure.efermi, 4)

        if self.is_assigned_manually:
            firstCB, lastVB = self.firstCB_index, self.lastVB_index
        else:
            firstCB, lastVB = ut.get_firstCB_and_lastVB(data, Efermi)

        data -= bandstructure.efermi

        if force_gap is not None:
            if scissor_shift != 0.0:
                print(f"  ⚠  force_gap overrides scissor_shift")
            CBM = np.min(data[firstCB])
            VBM = np.max(data[lastVB])
            scissor_shift = force_gap - (CBM - VBM)

        if scissor_shift != 0.0:
            print(f"  Applying scissor shift: {scissor_shift:+.4f} eV")
            data[firstCB:] += scissor_shift

        data -= np.max(data[lastVB])  # VBM → 0 eV
        CBM = float(np.min(data[firstCB]))
        VBM = float(np.max(data[lastVB]))
        band_gap_after_shift = CBM - VBM

        eigenvalues = Eigenval(f"{folder_path}/EIGENVAL")
        kpoints_weights = eigenvalues.kpoints_weights
        XX = eigenvalues.nkpt
        cart_coords = [bandstructure.kpoints[i].cart_coords for i in range(XX)]
        kgrid = bs.kpoints.kpts[0]
        X = int(kgrid[0])

        if write_path:
            os.makedirs(write_path, exist_ok=True)

        np.save(f"{write_path}/kgrid_{X}_{XX}.npy", cart_coords)
        np.save(f"{write_path}/Egrid_{X}_{XX}.npy", data)
        np.save(f"{write_path}/kw_{X}_{XX}.npy", kpoints_weights)

        keys = [
            "material_name", "Crystal_System", "Space_Group", "X", "XX", "E_Fermi", "nbands", "nkpoints",
            "kgrid", "scissor_shift", "band_gap", "band_gap_after_shift",
            "firstCB_index", "lastVB_index", "CBM", "VBM",
            "volume", "dielectric_constant",
            "b1", "b2", "b3", "NELECT", "q_TF", "omega_p",
        ]
        vals = [
            material_name, CrystalSystem, Space_group, X, XX, Efermi, eigenvalues.nbands, data.shape[1],
            kgrid, scissor_shift,
            band_gap,
            band_gap_after_shift,
            firstCB, lastVB, CBM, VBM,
            round(volume, 4), dielectric,
            b1, b2, b3, nelec, q_TF, omega_p,
        ]
        with open(f"{write_path}/band_info.txt", "w") as f:
            for k, v in zip(keys, vals):
                f.write(f"{k} {v}\n")

        print(f"  Saved to: {write_path}/")
        print(f"{'─'*90}\n")

    # ------------------------------------------------------------------
    # Import previously parsed data
    # ------------------------------------------------------------------
    def import_parsed_BS_data(
        self,
        from_folder: str,
    ) -> None:
        """
        Load band-structure arrays and metadata written by :meth:`parse_BS_data`.

        Parameters
        ----------
        from_folder : str
            Directory containing ``band_info.txt`` and the ``.npy`` files.
        """
        print(f"\n{'─'*90}")
        print(f"📥 Importing from: {from_folder}")
        print(f"{'─'*90}")

        from_folder = from_folder.rstrip("/")
        self.results_folder = from_folder

        info = ut.read_band_info(f"{from_folder}/band_info.txt")
        self.material_name = info["material_name"]
        self.scissor_shift = info["scissor_shift"]
        self.kgrid = info["kgrid"]
        self.band_gap = info["band_gap"]
        self.band_gap_after_shift = info["band_gap_after_shift"]
        self.Crystal_System = info["Crystal_System"]
        self.Space_Group = info["Space_Group"]
        self.X = info["X"]
        self.XX = info["XX"]
        self.E_Fermi = info["E_Fermi"]

        self.data_energies = np.load(f"{from_folder}/Egrid_{self.X}_{self.XX}.npy")
        self.kpoints = np.load(f"{from_folder}/kgrid_{self.X}_{self.XX}.npy")
        self.kpoints_weights = np.load(f"{from_folder}/kw_{self.X}_{self.XX}.npy")

        self.num_of_bands = info["nbands"]
        self.num_of_kpoints = info["nkpoints"]
        self.firstCB_index = info["firstCB_index"]
        self.lastVB_index = info["lastVB_index"]

        self.CBM = info.get("CBM", float(np.min(self.data_energies[self.firstCB_index])))
        self.VBM = info.get("VBM", float(np.max(self.data_energies[self.lastVB_index])))

        self.dielectric_constant = info.get("dielectric_constant") or None
        self.volume = info["volume"]
        self.reciprocal_lattice = np.array([info["b1"], info["b2"], info["b3"]])
        self.nelec = info["NELECT"]
        self.q_TF = info["q_TF"]
        self.omega_p = info["omega_p"]

        # Optionally restore carrier-concentration data
        for attr in ("Ef_eq", "ni", "n", "p", "delta_n", "Efn", "Efp"):
            if attr in info:
                setattr(self, attr, info[attr])

        self.parms_imported = True
        print(f"  Material: {self.material_name}")
        print(f"  K-grid:   {self.kgrid}  ({self.XX} irreducible k-points)")
        print(f"  Bands:    {self.num_of_bands}  |  Gap: {self.CBM - self.VBM:.4f} eV")
        print(f"{'─'*90}\n")

    # ------------------------------------------------------------------
    # Carrier concentrations
    # ------------------------------------------------------------------
    def calculate_carrier_concentrations(
        self,
        delta_n: float = 0.0,
        start_Ef: Optional[float] = None,
        end_Ef: Optional[float] = None,
        Nsteps_Ef: int = 500,
    ) -> Tuple[interp1d, interp1d]:
        """
        Compute carrier concentrations and (quasi-) Fermi levels.

        Parameters
        ----------
        delta_n : float
            Injected excess carrier density (cm⁻³). Zero = equilibrium.
        start_Ef, end_Ef : float or None
            Search bounds for Fermi level.  Defaults to VBM / CBM.
        Nsteps_Ef : int
            Number of interpolation points.

        Returns
        -------
        fn, fp : interp1d
            Interpolators for n(Ef) and p(Ef).
        """
        if not self.parms_imported:
            raise RuntimeError("Import band-structure data first.")

        t0 = time.time()
        print(f"\n{'─'*90}")
        print(f"⚙  Calculating carrier concentrations …")
        print(f"{'─'*90}")

        A0_cm = CM_PER_ANGSTROM
        start_Ef = self.VBM if start_Ef is None else start_Ef
        end_Ef = self.CBM if end_Ef is None else end_Ef
        Ef_arr = np.linspace(start_Ef, end_Ef, Nsteps_Ef)

        # n(Ef)
        n_arr = np.zeros(Nsteps_Ef)
        for idx, Ef in enumerate(Ef_arr):
            s = sum(
                ut.fermi_dirac(self.data_energies[m][k], Ef, self.T)
                * self.kpoints_weights[k]
                for m in range(self.firstCB_index, self.num_of_bands)
                for k in range(self.num_of_kpoints)
            )
            n_arr[idx] = 2.0 / (self.volume * A0_cm**3) * s
        fn = interp1d(Ef_arr, n_arr, bounds_error=False,
                      fill_value=(n_arr[0], n_arr[-1]))

        # p(Ef)
        p_arr = np.zeros(Nsteps_Ef)
        for idx, Ef in enumerate(Ef_arr):
            s = sum(
                (1.0 - ut.fermi_dirac(self.data_energies[m][k], Ef, self.T))
                * self.kpoints_weights[k]
                for m in range(self.lastVB_index + 1)
                for k in range(self.num_of_kpoints)
            )
            p_arr[idx] = 2.0 / (self.volume * A0_cm**3) * s
        fp = interp1d(Ef_arr, p_arr, bounds_error=False,
                      fill_value=(p_arr[0], p_arr[-1]))

        # --- Equilibrium ---
        def _charge_neutrality(Ef):
            return float(fn(Ef) - fp(Ef) - self.nd)

        if _charge_neutrality(start_Ef) * _charge_neutrality(end_Ef) >= 0:
            raise ValueError("Cannot bracket Ef.  Widen start_Ef / end_Ef.")

        Ef_eq = brentq(_charge_neutrality, start_Ef, end_Ef)
        n0, p0 = float(fn(Ef_eq)), float(fp(Ef_eq))
        self.ni = np.sqrt(n0 * p0)
        self.Ef_eq = Ef_eq

        if delta_n == 0.0:
            self.Efn = self.Efp = Ef_eq
            self.n, self.p = n0, p0
            self.delta_n = 0.0
            label = "EQUILIBRIUM"
        else:
            target_n = n0 + delta_n
            Efn = brentq(lambda E: float(fn(E)) - target_n, start_Ef, end_Ef)
            target_p = float(fn(Efn)) - self.nd
            Efp = brentq(lambda E: float(fp(E)) - target_p, start_Ef, end_Ef)
            self.Efn, self.Efp = Efn, Efp
            self.n, self.p = float(fn(Efn)), float(fp(Efp))
            self.delta_n = delta_n
            label = "NON-EQUILIBRIUM"

        # Persist to band_info
        extra = {"nd": self.nd, "Ef_eq": self.Ef_eq, "ni": self.ni,
                 "n": self.n, "p": self.p}
        if delta_n > 0:
            extra.update(delta_n=delta_n, Efn=self.Efn, Efp=self.Efp)
        with open(f"{self.results_folder}/band_info.txt", "a") as f:
            for k, v in extra.items():
                f.write(f"{k} {v}\n")

        elapsed = ut.convert_seconds(time.time() - t0)
        print(f"\n  {label} CARRIER CONCENTRATIONS")
        print(f"    n = {self.n:.4e} cm⁻³   |   p = {self.p:.4e} cm⁻³")
        print(f"    Efn = {self.Efn:.4f} eV    |   Efp = {getattr(self, 'Efp', self.Efn):.4f} eV")
        print(f"    ni  = {self.ni:.4e} cm⁻³")
        print(f"  ⏱  {elapsed[1]:02d}h {elapsed[2]:02d}m {elapsed[3]:02d}s")
        print(f"{'─'*90}\n")
        return fn, fp

    # ------------------------------------------------------------------
    # Energy-window auto-calculation
    # ------------------------------------------------------------------
    def calculate_energy_cutoffs(
        self,
        charge_threshold: float = 0.99,
        max_M: int = 30,
    ) -> Tuple[float, float]:
        """
        Determine CB_window and VB_window that capture *charge_threshold* of carriers.

        Returns ``(CB_window, VB_window)`` in eV, measured from CBM / VBM.
        """
        kT = K_B_eV * self.T
        A0_cm = CM_PER_ANGSTROM
        prefactor = 2.0 / (self.volume * A0_cm**3)

        # Electron cutoff
        target_e = self.n
        E_cut_e = self.Efn + max_M * kT
        for M in range(max_M + 1):
            E_lim = self.Efn + M * kT
            density = prefactor * sum(
                ut.fermi_dirac(self.data_energies[m][k], self.Efn, self.T)
                * self.kpoints_weights[k]
                for m in range(self.firstCB_index, self.num_of_bands)
                for k in range(self.num_of_kpoints)
                if self.data_energies[m][k] <= E_lim
            )
            if density >= charge_threshold * target_e:
                E_cut_e = E_lim
                break

        # Hole cutoff
        target_h = self.p
        E_cut_h = self.Efp - max_M * kT
        for M in range(max_M + 1):
            E_lim = self.Efp - M * kT
            density = prefactor * sum(
                (1.0 - ut.fermi_dirac(self.data_energies[m][k], self.Efp, self.T))
                * self.kpoints_weights[k]
                for m in range(self.lastVB_index + 1)
                for k in range(self.num_of_kpoints)
                if self.data_energies[m][k] >= E_lim
            )
            if density >= charge_threshold * target_h:
                E_cut_h = E_lim
                break

        CB_window = E_cut_e - self.CBM
        VB_window = self.VBM - E_cut_h

        print(f"  Energy windows ({charge_threshold*100:.0f}% of carriers):")
        print(f"    CB: {self.CBM:.4f} → {E_cut_e:.4f} eV  (width {CB_window:.4f} eV)")
        print(f"    VB: {E_cut_h:.4f} → {self.VBM:.4f} eV  (width {VB_window:.4f} eV)")
        return CB_window, VB_window

    # ------------------------------------------------------------------
    # Exact k-point list (for NSCF workflow)
    # ------------------------------------------------------------------
    def create_exact_kpoint_list(
        self,
        CB_window: float,
        VB_window: float,
        auger_type: str,
        poscar_path: str,
        search_mode: str = "Brute_Force",
        num_kpoints: Union[int, str] = "all",
    ) -> None:
        """
        Generate the list of off-grid k-points required for the ``exact_kpoint``
        approach.  Must be called *before* running NSCF calculations.
        """
        print(f"\n{'='*90}")
        print(f"{'EXACT K-POINT LIST':^90}")
        print(f"{'='*90}")

        pairs = PairGenerator(
            auger_type,
            (self, CB_window, VB_window, "exact_kpoint", False, search_mode, -1, "", poscar_path, True),
        )
        kpoints_list = pairs.generate_exact_kpoint_list(search_mode, num_kpoints)

        key = "P_134" if auger_type == "eeh" else "P_123"
        kpoints_list.sort(key=lambda x: x[key], reverse=True)
        ut.write_to_csv(
            kpoints_list,
            f"exact_kpoints_{auger_type}_{self.XX}",
            folder_to_save=self.results_folder,
        )
        print(f"  Saved {len(kpoints_list):,} k-points → exact_kpoints_{auger_type}_{self.XX}.csv")
        print(f"{'='*90}\n")

    # ------------------------------------------------------------------
    # Pair generation
    # ------------------------------------------------------------------
    def create_auger_pairs(
        self,
        CB_window: float,
        VB_window: float,
        auger_type: str,
        *,
        approach: str = "nearest_kpoint",
        is_parallel: bool = False,
        search_mode: str = "Max_Heap",
        nscf_folders: Optional[Union[str, List[str]]] = None,
        num_top_pairs: Union[int, str] = "all",
        continue_from_files: Union[str, List[str]] = [],
        multiplier: int = 1,
        table_name_suffix: str = "",
        poscar_path: Optional[str] = None,
        exact_kpoints_csv: Optional[Union[str, List[str]]] = None,
    ) -> PairGenerator:
        """
        Identify Auger scattering channels and rank them by occupation probability.

        Parameters
        ----------
        CB_window, VB_window : float
            Energy windows above CBM / below VBM (eV).
        auger_type : {'eeh', 'ehh'}
        approach : {'nearest_kpoint', 'exact_kpoint'}
        is_parallel : bool
            Use multiprocessing for ``Brute_Force`` search.
        search_mode : {'Max_Heap', 'Brute_Force'}
        nscf_folders : str or list[str]
            Required when *approach* = ``exact_kpoint``.
        num_top_pairs : int or 'all'
        continue_from_files : str or list[str]
            Resume from a previous pairs CSV.
        multiplier : int
            Heap expansion factor (only for ``Max_Heap``).
        table_name_suffix : str
            Appended to output file names.
        poscar_path : str or None
            Path to the VASP POSCAR file.  When provided, the irreducible
            k-points are expanded to the full BZ using crystal symmetry
            before pair generation.  **Required** for ``exact_kpoint``.
        exact_kpoints_csv : str, list[str], or None
            Path(s) to the ``exact_kpoints_<type>_<XX>.csv`` file(s).
            Pass a list when the k-points are split across several CSVs.
            If *None* and *approach* = ``exact_kpoint``, the path is
            auto-discovered from ``results_folder``.

        Returns
        -------
        PairGenerator
        """
        self._validate_state("pairs", auger_type, approach, search_mode)

        if approach == "exact_kpoint":
            # if poscar_path is None:
            #     raise ValueError(
            #         "poscar_path is required for the 'exact_kpoint' approach "
            #         "(needed to expand irreducible k-points to the full BZ)."
            #     )
            if nscf_folders is None:
                raise ValueError(
                    "nscf_folders is required for the 'exact_kpoint' approach."
                )

        if isinstance(continue_from_files, str):
            continue_from_files = [continue_from_files]

        num_val = -1 if num_top_pairs == "all" else num_top_pairs

        print(f"\n{'='*90}")
        print(f"{'AUGER PAIR GENERATION':^90}")
        print(f"{'='*90}")
        print(f"  Type: {auger_type.upper()}  |  Approach: {approach}  |  Search: {search_mode}")
        print(f"  CB window: {CB_window:.4f} eV  |  VB window: {VB_window:.4f} eV")
        print(f"{'─'*90}")

        t0 = time.time()
        gen = PairGenerator(
            auger_type,
            (self, CB_window, VB_window, approach, is_parallel, search_mode,
             num_val, table_name_suffix, poscar_path, False),
        )
        gen.create_pairs(
            multiplier=multiplier,
            nscf_folders=nscf_folders,
            continue_from_files=continue_from_files,
            exact_kpoints_csv=exact_kpoints_csv,
        )

        if gen.pairs:
            pairs_dicts = gen.write_pairs_to_csv(to_path=self.results_folder)
        else:
            pairs_dicts = []
            print("\n  \u26a0  No pairs generated.")
        self.auger_pairs_dicts[auger_type] = pairs_dicts
        self.is_auger_pairs_created[auger_type] = True
        self.auger_pairs_object[auger_type] = gen

        elapsed = ut.convert_seconds(time.time() - t0)
        print(f"\n  ✓ {len(gen.pairs):,} pairs  |  ⏱  {elapsed[1]:02d}h {elapsed[2]:02d}m {elapsed[3]:02d}s")
        print(f"{'='*90}\n")
        return gen

    # ------------------------------------------------------------------
    # Read-back helpers
    # ------------------------------------------------------------------
    def read_auger_pairs(
        self,
        file_paths: Union[str, List[str]],
    ) -> Tuple[List[Dict], str]:
        """Read pairs from CSV file(s).  Returns ``(pairs_list, auger_type)``."""
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        for fp in file_paths:
            if not os.path.exists(fp):
                raise FileNotFoundError(fp)

        data = ut.read_csv(file_paths)
        auger_type = data[0]["pair_type"]
        self.auger_pairs_dicts[auger_type] = data
        self.is_auger_pairs_created[auger_type] = True
        print(f"  Read {len(data):,} {auger_type.upper()} pairs from {len(file_paths)} file(s)")
        return data, auger_type

    def read_matrix_elements(self, file_path: str) -> List[Dict]:
        """Read matrix elements from a JSONL file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        with open(file_path, "r") as f:
            data = [json.loads(line.strip()) for line in f]

        # Detect auger type from pair_id pattern
        E3_idx = int(data[0]["pair_id"].split("-")[2])
        auger_type = "eeh" if E3_idx >= self.firstCB_index else "ehh"

        self.matrix_elements_dicts[auger_type] = data
        self.is_matrix_elements_calculated[auger_type] = True
        print(f"  Read {len(data):,} matrix elements ({auger_type.upper()}) from {file_path}")
        return data

    # ------------------------------------------------------------------
    # Matrix-element calculation
    # ------------------------------------------------------------------
    def calculate_matrix_elements(
        self,
        auger_type: str,
        wavecar_files: Union[str, List[str]] = "WAVECAR",
        dielectric_constant: Optional[float] = None,
        num_matrix_elements: Union[int, str] = "all",
        continue_from_files: Union[str, List[str]] = [],
        add_suffix_name: str = "",
    ) -> MatrixElements:
        """
        Compute Coulomb matrix elements |M|² in parallel.

        Parameters
        ----------
        auger_type : {'eeh', 'ehh'}
        wavecar_files : str or list[str]
            WAVECAR path(s).  For ``nearest_kpoint`` provide a single
            SCF WAVECAR.  For ``exact_kpoint`` provide
            ``['NSCF_1/WAVECAR', 'NSCF_2/WAVECAR', ...]``.
        dielectric_constant : float or None
            Falls back to the value parsed from ``vasprun.xml``.
        num_matrix_elements : int or 'all'
        continue_from_files : str or list[str]
            JSONL file(s) with previously computed elements.
        add_suffix_name : str

        Returns
        -------
        MatrixElements
        """
        if not self.is_auger_pairs_created.get(auger_type):
            raise RuntimeError(f"Create {auger_type} pairs first.")
        if dielectric_constant is None:
            dielectric_constant = self.dielectric_constant
            if dielectric_constant is None:
                raise ValueError("Provide a dielectric constant.")

        if isinstance(continue_from_files, str):
            continue_from_files = [continue_from_files]

        t0 = time.time()
        me = MatrixElements(self, auger_type, dielectric_constant, wavecar_files)
        results = me.calculate_matrix_elements_parallel(
            wavecar_files, num_matrix_elements, add_suffix_name, continue_from_files,
        )
        self.matrix_elements_dicts[auger_type] = results
        self.is_matrix_elements_calculated[auger_type] = True

        elapsed = ut.convert_seconds(time.time() - t0)
        print(f"  ⏱  {elapsed[1]:02d}h {elapsed[2]:02d}m {elapsed[3]:02d}s")
        return me

    # ------------------------------------------------------------------
    # Auger rate / coefficient
    # ------------------------------------------------------------------
    def calculate_auger_rates(
        self,
        auger_type: str,
        delta_function: Sequence[str] = ("Gaussian", "Lorentzian", "Rectangular"),
        FWHM: Sequence[float] = (0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2),
        add_suffix_name: str = "",
    ) -> List[Dict]:
        """
        Evaluate Auger recombination coefficients C_n / C_p.

        Parameters
        ----------
        auger_type : {'eeh', 'ehh'}
        delta_function : sequence of str
            Names of delta-function approximations.
        FWHM : sequence of float
            Broadening widths (eV) for each delta function.
        add_suffix_name : str
            Suffix for output file names.

        Returns
        -------
        list[dict]
            Rows with keys ``'Delta function'``, ``'FWHM'``, ``'Auger coefficient'``.
        """
        if not self.is_matrix_elements_calculated.get(auger_type):
            raise RuntimeError(f"Calculate {auger_type} matrix elements first.")

        suffix = f"_{add_suffix_name}" if add_suffix_name else ""
        t0 = time.time()

        print(f"\n{'='*90}")
        print(f"{'AUGER COEFFICIENT CALCULATION':^90}")
        print(f"{'='*90}")
        print(f"  Type: {auger_type.upper()}  |  T = {self.T} K")
        print(f"{'─'*90}")

        pairs = self.auger_pairs_dicts[auger_type]
        me_list = self.matrix_elements_dicts[auger_type]
        V_m3 = self.volume * 1e-30
        h_bar = HBAR
        auger_factor = (4 * np.pi / h_bar) * (1.0 / V_m3**3) * (1.0 / eV) * 1e12

        if auger_type == "eeh":
            carrier = self.n**2 * self.p - self.ni**2 * self.n
        else:
            carrier = self.p**2 * self.n - self.ni**2 * self.p
        carrier *= (1e6) ** 3
        auger_factor *= (1.0 / carrier) if carrier != 0 else 0.0

        # Resolve delta functions
        delta_fns = {}
        for name in delta_function:
            if name not in ut.DELTA_FUNCTIONS:
                raise ValueError(f"Unknown delta function '{name}'")
            delta_fns[name] = ut.DELTA_FUNCTIONS[name]

        results = {d: {f: 0.0 for f in FWHM} for d in delta_function}
        me_map = {m["pair_id"]: m for m in me_list}

        for pair in pairs:
            pid = pair["pair_id"]
            if pid not in me_map:
                continue
            M2 = me_map[pid]["|M|^2"] * eV**2  # J²
            E1, E2, E3, E4 = pair["E1"], pair["E2"], pair["E3"], pair["E4"]
            kw1, kw2, kw3, kw4 = pair["kw1"], pair["kw2"], pair["kw3"], pair["kw4"]
            P = pair["probability"]
            dE = (E2 - E1) - (E3 - E4) if auger_type == "eeh" else (E1 - E2) - (E3 - E4)

            for dname in delta_function:
                for fwhm in FWHM:
                    dval = delta_fns[dname](dE, fwhm)
                    if auger_type == "eeh":
                        contrib = P * M2 * V_m3**2 * dval * auger_factor * kw1 * kw3 * kw4
                    else:
                        contrib = P * M2 * V_m3**2 * dval * auger_factor * kw1 * kw2 * kw3
                    results[dname][fwhm] += contrib
                    # Add columns :
                    pair[f"C_{dname}_FWHM_{fwhm:.3f}"] = contrib # cm⁶/s

        # Flatten to CSV-friendly list
        rows = []
        for d in delta_function:
            for f in FWHM:
                rows.append({
                    "Delta function": d,
                    "FWHM": f,
                    "Auger coefficient": results[d][f],
                })

        ut.write_to_csv(rows, f"Auger_coefficients_{auger_type}_{self.XX}{suffix}",
                        folder_to_save=self.results_folder)
        ut.write_to_csv(pairs, f"auger_{auger_type}_pairs_{self.XX}_completed{suffix}",
                        folder_to_save=self.results_folder)
        self.auger_coefficients[auger_type] = rows

        elapsed = ut.convert_seconds(time.time() - t0)
        print(f"\n  ✓ Saved Auger_coefficients_{auger_type}_{self.XX}{suffix}.csv")
        mid = len(FWHM) // 2
        for d in delta_function:
            print(f"    {d} (FWHM={FWHM[mid]}):  C = {results[d][FWHM[mid]]:.4e} cm⁶/s")
        print(f"  ⏱  {elapsed[1]:02d}h {elapsed[2]:02d}m {elapsed[3]:02d}s")
        print(f"{'='*90}\n")
        return rows

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------
    def _validate_state(self, stage, auger_type, approach=None, search_mode=None):
        if not self.parms_imported:
            raise RuntimeError("Import band-structure data first.")
        if auger_type not in ("eeh", "ehh"):
            raise ValueError(f"auger_type must be 'eeh' or 'ehh', got '{auger_type}'")
        if approach and approach not in (
            "nearest_kpoint", "exact_kpoint"
        ):
            raise ValueError(f"Invalid approach: '{approach}'")
        if search_mode and search_mode not in ("Max_Heap", "Brute_Force"):
            raise ValueError(f"Invalid search_mode: '{search_mode}'")
