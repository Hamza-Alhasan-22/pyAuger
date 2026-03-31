# pyAuger — Ab-Initio Direct Auger Recombination Calculator

![Tests](https://img.shields.io/badge/tests-164%20passed-brightgreen?style=flat-square&logo=pytest&logoColor=white)
![Python](https://img.shields.io/badge/python-3.x-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-TBD-lightgrey?style=flat-square)

A Python package for calculating direct Auger recombination coefficients
(**C_n** and **C_p**) for semiconductors using first-principles VASP data.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Hamza-Alhasan-22/pyAuger
cd pyAuger

# Install in editable mode
pip install -e .
```

Or download easily and directly using pip:

```bash
pip install git+https://github.com/Hamza-Alhasan-22/pyAuger
```

**Dependencies:** `numpy`, `scipy`, `pandas`, `matplotlib`, `pymatgen`, `pyvaspwfc`

The code was created and tested with the following versions:
- python 3.12.12
- numpy 2.4.2
- scipy 1.16.3
- pandas 2.2.3
- matplotlib 3.10.7
- pymatgen 2025.10.7
- [pyvaspwfc](https://github.com/QijingZheng/VaspBandUnfolding/tree/master) 1.0

---

## Package structure

```
auger/
├── __init__.py           # Package interface & exports
├── constants.py          # Physical constants (eV, ε₀, ℏ, …)
├── utilities.py          # I/O helpers, delta functions, BZ folding
├── calculator.py         # AugerCalculator — main driver class
├── pairs.py              # PairGenerator + Pair — scattering channel identification
├── matrix_elements.py    # MatrixElements — Coulomb |M|² from wavefunctions
├── analysis.py           # AugerAnalyzer — plotting & convergence analysis
```

---

## Concepts

### Auger types

| Type | Notation | Physical process | Output |
|------|----------|-----------------|--------|
| **eeh** | electron–electron–hole | Two CB electrons scatter; one recombines with a VB hole | C_n |
| **ehh** | electron–hole–hole | A CB electron recombines with two VB holes scattering | C_p |

The total Auger coefficient is: **C_Auger = C_n + C_p**.

### Auger coefficient equation

The Auger coefficient is evaluated via Fermi's Golden Rule:

$$C_n = \frac{4\pi}{\hbar} \frac{1}{n^2 p - n_i^2 n} \sum_{\text{pairs}} P \cdot |M|^2 \cdot \delta(\Delta E)$$

where **P** is the Fermi–Dirac occupation-weighted probability, **|M|²** is the screened Coulomb matrix element, and **δ(ΔE)** enforces energy conservation (approximated by Gaussian, Lorentzian, or Rectangular broadening).

### Two approaches for the 4th state

When three states (1, 2, 3) are chosen, the 4th state must satisfy
**momentum conservation**: **k₁ + k₂ = k₃ + k₄**.

| Approach | Description | NSCF needed? |
|----------|-------------|-------------|
| `nearest_kpoint` | Finds the nearest k-point in the SCF grid to the exact k₄ vector | No |
| `exact_kpoint` | Runs NSCF calculations at the exact required k-points | Yes |

> **Note:** The `nearest_kpoint` approach requires `ISYM = -1` in the VASP INCAR so that the full BZ k-mesh is available. The `exact_kpoint` approach does not have this requirement.

### Two search modes

| Mode | Description | When to use |
|------|-------------|------------|
| `Max_Heap` | Priority queue — extracts the top-N pairs efficiently | Default — fast and memory-efficient |
| `Brute_Force` | Exhaustive triple loop over all state combinations | Reference calculations or small grids |

### Delta function approximations

Energy conservation is enforced via broadened delta functions. Three are available:

| Name | Character |
|------|-----------|
| `Gaussian` | Smooth, most commonly used |
| `Lorentzian` | Longer tails |
| `Rectangular` | Sharp cutoff |

---

## Quick start

```python
from auger import AugerCalculator

# 1) Initialise
calc = AugerCalculator(T=300, nd=0)         # 300 K, intrinsic case

# 2) Import band-structure data (pre-parsed from VASP)
calc.import_parsed_BS_data("./parsed_data")

# 3) Carrier concentrations
calc.calculate_carrier_concentrations(delta_n=1e17)

# 4) Generate Auger pairs (scattering channels)
calc.create_auger_pairs(
    CB_window=0.3, VB_window=0.3,
    auger_type="eeh",
    approach="nearest_kpoint",
    search_mode="Max_Heap",
)

# 5) Compute matrix elements
calc.calculate_matrix_elements(
    auger_type="eeh",
    wavecar_files="WAVECAR",
    dielectric_constant=16.8,
)

# 6) Evaluate Auger coefficient
results = calc.calculate_auger_rates(auger_type="eeh")
```

---

## Workflow details

### Step 0 — Manual band-edge assignment (optional)

For materials where pymatgen detects a zero or incorrect band gap (e.g. semimetals or narrow-gap systems), the automatic identification of the conduction band minimum (CBM) and valence band maximum (VBM) may fail. In such cases, manually specify the band indices **before** parsing:

```python
calc = AugerCalculator(T=300, nd=1e17)

# 0-based indices: firstCB is the first conduction band,
# lastVB is the last valence band.
calc.assign_firstCB_and_lastVB(
    firstCB_index=25,
    lastVB_index=24,
)

# Then proceed with parsing as usual using parse_BS_data ...
```

This assigning must be done **before** `parse_BS_data` or `import_parsed_BS_data`.

### Step 1 — Parse VASP data

```python
calc.parse_BS_data(
    folder_path="./vasp_scf",     # directory with EIGENVAL, vasprun.xml, KPOINTS, POSCAR
    write_path="./results",       # output directory (created if needed)
    scissor_shift=0.0,            # rigid shift applied to CB bands (eV)
    force_gap=None,               # if set, overrides scissor_shift to enforce this exact gap
)
```

This reads the eigenvalues, k-points, dielectric tensor, and reciprocal lattice from the VASP outputs, then saves `Egrid_X_XX.npy`, `kgrid_X_XX.npy`, `kw_X_XX.npy`, and `band_info.txt` to the output directory.

### Step 2 — Import the parsed data

To reload previously parsed data:

```python
calc.import_parsed_BS_data("./results")
```

### Step 3 — Carrier concentrations

```python
calc.calculate_carrier_concentrations(
    delta_n=1e17,        # excess carrier concentration (cm⁻³); 0 for equilibrium
    start_Ef=None,       # lower bound for Fermi-level search (eV); auto-determined if None
    end_Ef=None,         # upper bound for Fermi-level search (eV); auto-determined if None
    Nsteps_Ef=500,       # resolution of the Fermi-level grid
)
```

Solves charge neutrality self-consistently to find **E_F**, then computes:
- **n**, **p** — electron and hole concentrations
- **n_i** — intrinsic carrier concentration
- Quasi-Fermi levels **E_Fn** and **E_Fp** (when `delta_n > 0`)
- Fermi–Dirac occupations for every state

### Step 4 — Energy cutoffs (optional)

```python
CB_auto, VB_auto = calc.calculate_energy_cutoffs(
    charge_threshold=0.99,   # fraction of carrier density to capture
    max_M=30,                # maximum integer M for the cut-off E_cut = Ef + M*k_B*T
)
```

Automatically determines the smallest energy window around the band edges that contains the specified fraction of the total carrier density. Alternatively, set `CB_window` and `VB_window` manually.

### Step 5 — Pair generation

```python
gen = calc.create_auger_pairs(
    CB_window=0.3, # or CB_auto
    VB_window=0.3, # or VB_auto
    auger_type="eeh",                 # "eeh" for C_n, "ehh" for C_p
    approach="nearest_kpoint",        # or "exact_kpoint"
    search_mode="Max_Heap",           # or "Brute_Force"
    num_top_pairs=1000,               # keep top 1000 pairs, or "all"
)
```

For the `exact_kpoint` approach, the workflow is:

1. Generate the required off-grid k-points:
   ```python
   calc.create_exact_kpoint_list(
       CB_window=0.3, VB_window=0.3,
       auger_type="eeh",
       poscar_path="./vasp_scf/POSCAR",
   )
   ```
2. Create NSCF input folders for VASP:
   ```python
   from auger import utilities
   utilities.create_nscf_inputs(
       scf_folder="./vasp_scf",
       nscf_folder="./nscf_inputs",
       exact_kpoints_table="./results/exact_kpoints_eeh_XX.csv",
       auger_type="eeh",
   )
   ```
3. Run the NSCF VASP jobs externally.
4. Create pairs using the NSCF results:
   ```python
   gen = calc.create_auger_pairs(
       CB_window=0.3, VB_window=0.3,
       auger_type="eeh",
       approach="exact_kpoint",
       nscf_folders=["./nscf_outputs"],
       exact_kpoints_csv="./results/exact_kpoints_eeh_XX.csv",
   )
   ```

### Step 6 — Matrix elements

```python
me = calc.calculate_matrix_elements(
    auger_type="eeh",
    wavecar_files="WAVECAR",            # path(s) to WAVECAR file(s)
    dielectric_constant=16.8,           # static dielectric constant
    num_matrix_elements="all",          # or an integer to limit
)
```

Computes the screened Coulomb matrix element |M|² for each pair using the wavefunctions from the WAVECAR. This step runs in parallel using Python multiprocessing.

To resume an interrupted calculation, pass the partial results file:

```python
me = calc.calculate_matrix_elements(
    auger_type="eeh",
    wavecar_files="WAVECAR",
    dielectric_constant=16.8,
    continue_from_files="./results/eeh_matrix_elements_partial.jsonl",
)
```

### Step 7 — Auger coefficients

```python
results = calc.calculate_auger_rates(
    auger_type="eeh",
    delta_function=("Gaussian", "Lorentzian", "Rectangular"),
    FWHM=(0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2),
)
```

Outputs a CSV with one row per (delta function, FWHM) combination.

---

## File outputs

| File | Description |
|------|-------------|
| `band_info.txt` | Material metadata: name, k-grid, band gap, lattice vectors, etc. |
| `Egrid_X_XX.npy` | Eigenvalues referenced to VBM = 0 eV |
| `kgrid_X_XX.npy` | k-point Cartesian coordinates |
| `kw_X_XX.npy` | k-point weights |
| `auger_{type}_pairs_{XX}.csv` | Generated Auger pairs with probabilities |
| `{type}_matrix_elements_{XX}.jsonl` | Matrix elements for each pair |
| `Auger_coefficients_{type}_{XX}.csv` | Final Auger coefficients |
| `auger_{type}_pairs_{XX}_completed.csv` | Pairs with attached matrix elements (for analysis) |

---

## Units

### Input (VASP data)

| Quantity | Unit |
|----------|------|
| Eigenvalues (energies) | eV |
| k-points (Cartesian) | Å⁻¹ |
| Dielectric constant | Dimensionless (relative permittivity, i.e. ε/ε₀) |

### Input + Output (pyAuger)

| Quantity | Unit |
|----------|------|
| Energies (eigenvalues, band gap, Fermi levels, energy windows) | eV |
| k-points (Cartesian coordinates in saved `.npy` files) | Å⁻¹ |
| Carrier concentrations (n, p, n_i, delta_n, doping) | cm⁻³ |
| Matrix elements M | eV |
| Matrix elements M² | eV² |
| Auger coefficients (C_n, C_p) | cm⁶/s |

---

## How to cite pyAuger

<!-- TODO -->

---

## License

<!-- TODO -->
