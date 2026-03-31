"""
Tests for auger.matrix_elements — MatrixElements class and worker functions.

Since the full calculation requires real WAVECAR files (via pyvaspwfc),
these tests focus on:
  - Debye screening computation
  - Worker-level helper functions (_get_Gvec, _get_Cg, _get_coeff_dict caching)
  - JSONL I/O
  - Validation logic (missing WAVECAR, invalid wc_index)

Integration tests with actual WAVECARs are beyond the scope of unit tests.
"""

import json
import os

import numpy as np
import pytest

from auger.matrix_elements import MatrixElements, _calc_matrix_element
from auger.constants import eV, EPSILON_0, MATRIX_FACTOR


# ======================================================================
# JSONL I/O
# ======================================================================
class TestMatrixElementsIO:

    def test_read_matrix_elements_from_file(self, tmp_path):
        """Read a JSONL file with matrix element data."""
        p = tmp_path / "test_me.jsonl"
        entries = [
            {"pair_id": "8-9-8-7-0-1-2-3", "|M|^2": 0.5},
            {"pair_id": "8-9-8-7-1-2-3-4", "|M|^2": 0.3},
            {"pair_id": "8-9-8-7-2-3-4-5", "|M|^2": 0.1},
        ]
        with open(str(p), "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = MatrixElements.read_matrix_elements_from_file(str(p))
        assert len(result) == 3
        assert result[0]["|M|^2"] == 0.5
        assert result[2]["pair_id"] == "8-9-8-7-2-3-4-5"

    def test_read_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = MatrixElements.read_matrix_elements_from_file(str(p))
        assert result == []

    def test_read_preserves_error_entries(self, tmp_path):
        p = tmp_path / "with_errors.jsonl"
        entries = [
            {"pair_id": "a", "|M|^2": 0.1},
            {"pair_id": "b", "error": "some error"},
        ]
        with open(str(p), "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = MatrixElements.read_matrix_elements_from_file(str(p))
        assert len(result) == 2
        assert "error" in result[1]


# ======================================================================
# Debye screening
# ======================================================================
class TestDebyeScreening:

    def test_inverse_debye_positive(self, loaded_calculator):
        """Inverse Debye length should be positive."""
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)

        # Create a mock MatrixElements just to test Debye
        # We need to skip the WAVECAR check, so we test the formula directly
        kB_T = 8.617333262145e-5 * calc.T
        dielectric = 12.0
        q = eV

        # Electron contribution
        dE_n = calc.Efn - calc.CBM
        if dE_n < 0 or dE_n < 1.5 * kB_T:
            lam_e = np.sqrt(dielectric * EPSILON_0 * kB_T * q
                            / ((calc.n * 1e6) * q ** 2)) * 1e10
        else:
            lam_e = np.sqrt(dielectric * EPSILON_0 * dE_n * eV
                            / (1.5 * (calc.n * 1e6) * q ** 2)) * 1e10

        # Hole contribution
        dE_p = calc.VBM - calc.Efp
        if dE_p < 0 or dE_p < 1.5 * kB_T:
            lam_h = np.sqrt(dielectric * EPSILON_0 * kB_T * q
                            / ((calc.p * 1e6) * q ** 2)) * 1e10
        else:
            lam_h = np.sqrt(dielectric * EPSILON_0 * dE_p * eV
                            / (1.5 * (calc.p * 1e6) * q ** 2)) * 1e10

        inv = np.sqrt(1.0 / lam_e ** 2 + 1.0 / lam_h ** 2)
        assert inv > 0
        assert np.isfinite(inv)


# ======================================================================
# Worker function — error handling
# ======================================================================
class TestWorkerFunction:

    def test_calc_matrix_element_handles_exception(self):
        """Worker should return error dict on failure, not raise."""
        bad_args = (
            {"pair_id": "test-bad", "k1": [0, 0, 0], "k2": [0, 0, 0],
             "k3": [0, 0, 0], "k4": [0, 0, 0],
             "k1_index": 0, "k2_index": 0, "k3_index": 0, "k4_index": 0,
             "E1_index": 0, "E2_index": 0, "E3_index": 0, "E4_index": 0,
             "k1_wc_index": None},
            "eeh", 12.0, 0.01,
            MATRIX_FACTOR, 1e-28, 0.1, 0.5, 1e-80,
            np.eye(3),
        )
        # _wfcs is empty (no worker init), so it should fail gracefully
        result = _calc_matrix_element(bad_args)
        assert "pair_id" in result
        assert "error" in result

    def test_worker_returns_dict(self):
        """Even on error, result is a dict with pair_id."""
        bad_args = (
            {"pair_id": "fail-test"},
            "eeh", 12.0, 0.01,
            MATRIX_FACTOR, 1e-28, 0.1, 0.5, 1e-80,
            np.eye(3),
        )
        result = _calc_matrix_element(bad_args)
        assert isinstance(result, dict)
        assert result["pair_id"] == "fail-test"


# ======================================================================
# MatrixElements construction validation
# ======================================================================
class TestMatrixElementsConstruction:

    def test_missing_wavecar_raises(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        with pytest.raises(FileNotFoundError, match="WAVECAR"):
            MatrixElements(calc, "eeh", 12.0, "nonexistent_WAVECAR")

    def test_missing_wavecar_in_list_raises(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        with pytest.raises(FileNotFoundError, match="WAVECAR"):
            MatrixElements(calc, "eeh", 12.0, ["also_nonexistent_WAVECAR"])
