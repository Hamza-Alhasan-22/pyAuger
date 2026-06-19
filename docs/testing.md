# Testing pyAuger

This guide describes the supported test and validation levels in the repository.
Run commands from the repository root unless noted otherwise.

## 1. Lightweight tests

The default pytest suite is the normal reviewer-facing test suite. It uses
synthetic band-structure fixtures in `tests/conftest.py` and does not require
VASP, compiled C++ executables, large `WAVECAR` files, or external compute
systems.

Install the package dependencies first:

```bash
python -m pip install -r requirements.txt
```

Install the test runner:

```bash
python -m pip install pytest
```

Then run:

```bash
python -m pytest
```

If you got some access denied problems, use this fall back, and add (--basetemp .pytest-tmp):

```bash
python -m pytest --basetemp .pytest-tmp
```

For a shorter local check during development, run selected modules:

```bash
python -m pytest tests/test_constants.py tests/test_utilities.py
```

The default suite is configured in `pyproject.toml` to collect tests from the
`tests/` directory using files named `test_*.py`.

## 2. C++ comparison and validation

The repository includes a manual C++ matrix-element comparison script:

```bash
python tests/validate_cpp_matrix_elements.py --help
```

This validation compares Python matrix elements with the standalone C++ matrix
element executable for a selected set of pairs.

Requirements:

- the Python package dependencies from `requirements.txt`;
- a compiled `auger/cpp/matrix_element_calc` executable
  (`auger/cpp/matrix_element_calc.exe` on Windows);
- parsed pyAuger results containing `band_info.txt`, `Egrid_*.npy`,
  `kgrid_*.npy`, and `kw_*.npy`;
- one or more pair CSV files;
- the corresponding `WAVECAR` file or files.

Build the C++ executables with a C++17 compiler and OpenMP support:

```bash
cd auger/cpp
make
cd ../..
```

Example validation command using the provided InAs files:

```bash
python tests/validate_cpp_matrix_elements.py \
  --results_folder test-files/InAs/results/nearest_kpoint \
  --wavecar_files test-files/InAs/nearest-kpoint-scf-3/WAVECAR \
  --pairs_csv test-files/InAs/results/nearest_kpoint/auger_eeh_pairs_27_1.csv \
  --auger_type eeh \
  --dielectric 12.3 \
  --firstCB_index 9 \
  --lastVB_index 8 \
  --cpp_executable auger/cpp/matrix_element_calc \
  --output_dir test-output/cpp_validation \
  --num_matrix_elements 5
```

On Windows, use `auger/cpp/matrix_element_calc.exe` for `--cpp_executable`.
The script writes validation outputs under `--output_dir`.

The `cpp` pytest marker is reserved for optional C++ validation tests if they
are added later. The current C++ comparison is a standalone script, not part of
the default pytest suite.

## 3. Heavyweight VASP-file tests

Tests that read real VASP outputs are optional heavyweight checks. They are not
part of the default lightweight pytest suite.

The repository provides example VASP-derived inputs under:

```text
test-files/InAs/
```

These files can be used for manual workflow checks through the tutorial scripts.
Because these scripts write output files, use a scratch copy of `test-files` or
edit the output paths before running them if you want to preserve the reference
files unchanged.

Nearest-kpoint workflow example:

```bash
cd tutorials
python example_nearest_kpoint.py
```

Exact-kpoint workflow example:

```bash
cd tutorials
python example_exact_kpoint_step1.py
python example_exact_kpoint_step2.py
```

The exact-kpoint step 1 script prepares NSCF input folders. Step 2 expects the
corresponding NSCF calculations to be complete and available in the configured
folder.

The `heavy` pytest marker is reserved for future tests that require real VASP
outputs. No current default pytest test requires those files.
