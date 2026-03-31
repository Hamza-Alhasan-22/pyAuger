"""
Tests for auger.calculator — AugerCalculator class.

These tests exercise the calculator's state management, band-structure
import, carrier concentration calculation, energy cutoffs, pair/matrix
element read-back, Auger rate summation, and validation logic.

Heavy VASP I/O (parse_BS_data) is not tested here since it requires
real VASP output files.  We test everything downstream of import.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from auger.calculator import AugerCalculator
from auger import utilities as ut


# ======================================================================
# Construction & initialisation
# ======================================================================
class TestAugerCalculatorInit:

    def test_create_intrinsic(self, capsys):
        calc = AugerCalculator(T=300, nd=0)
        assert calc.T == 300
        assert calc.nd == 0
        assert calc.parms_imported is False
        out = capsys.readouterr().out
        assert "intrinsic" in out

    def test_create_n_type(self, capsys):
        calc = AugerCalculator(T=300, nd=1e17)
        out = capsys.readouterr().out
        assert "n-type" in out

    def test_create_p_type(self, capsys):
        calc = AugerCalculator(T=300, nd=-1e17)
        out = capsys.readouterr().out
        assert "p-type" in out

    def test_initial_containers(self):
        calc = AugerCalculator(T=300, nd=0)
        for typ in ("eeh", "ehh"):
            assert calc.is_auger_pairs_created[typ] is False
            assert calc.auger_pairs_dicts[typ] == []
            assert calc.matrix_elements_dicts[typ] == []
            assert calc.is_matrix_elements_calculated[typ] is False


# ======================================================================
# Band assignment
# ======================================================================
class TestBandAssignment:

    def test_assign_CB_VB(self):
        calc = AugerCalculator(T=300, nd=0)
        calc.assign_firstCB_and_lastVB(10, 9)
        assert calc.firstCB_index == 10
        assert calc.lastVB_index == 9
        assert calc.is_assigned_manually is True


# ======================================================================
# Import parsed data
# ======================================================================
class TestImportParsedData:

    def test_import_sets_flag(self, loaded_calculator):
        assert loaded_calculator.parms_imported is True

    def test_import_material_name(self, loaded_calculator):
        assert loaded_calculator.material_name == "TestMat"

    def test_import_band_gap(self, loaded_calculator):
        assert loaded_calculator.CBM - loaded_calculator.VBM > 0

    def test_import_energies_shape(self, loaded_calculator):
        assert loaded_calculator.data_energies.shape == (16, 64)

    def test_import_kpoints_shape(self, loaded_calculator):
        assert loaded_calculator.kpoints.shape == (64, 3)

    def test_import_weights(self, loaded_calculator):
        assert len(loaded_calculator.kpoints_weights) == 64
        assert np.sum(loaded_calculator.kpoints_weights) == pytest.approx(1.0, abs=0.01)

    def test_import_reciprocal_lattice(self, loaded_calculator):
        assert loaded_calculator.reciprocal_lattice.shape == (3, 3)

    def test_import_band_indices(self, loaded_calculator):
        assert loaded_calculator.firstCB_index == 8
        assert loaded_calculator.lastVB_index == 7

    def test_import_loads_carrier_data(self, fake_parsed_data):
        """If band_info has carrier data, it should be loaded."""
        from tests.conftest import BAND_INFO_TEMPLATE
        # Append carrier data to existing band_info.txt
        bi_path = os.path.join(fake_parsed_data, "band_info.txt")
        with open(bi_path, "a") as f:
            f.write("nd 1e17\n")
            f.write("Ef_eq 0.5\n")
            f.write("ni 1e10\n")
            f.write("n 1e17\n")
            f.write("p 1e17\n")
            f.write("delta_n 1e17\n")
            f.write("Efn 0.55\n")
            f.write("Efp 0.45\n")
        calc = AugerCalculator(T=300, nd=0)
        calc.import_parsed_BS_data(fake_parsed_data)
        assert hasattr(calc, "Ef_eq")
        assert hasattr(calc, "ni")


# ======================================================================
# Carrier concentrations
# ======================================================================
class TestCarrierConcentrations:

    def test_equilibrium(self, loaded_calculator):
        fn, fp = loaded_calculator.calculate_carrier_concentrations(delta_n=0.0)
        assert loaded_calculator.n > 0
        assert loaded_calculator.p > 0
        assert loaded_calculator.ni > 0
        assert loaded_calculator.Efn == loaded_calculator.Efp

    def test_non_equilibrium(self, loaded_calculator):
        fn, fp = loaded_calculator.calculate_carrier_concentrations(delta_n=1e17)
        assert loaded_calculator.delta_n == 1e17
        # n should be larger than equilibrium
        assert loaded_calculator.n > 0

    def test_raises_without_import(self):
        calc = AugerCalculator(T=300, nd=0)
        with pytest.raises(RuntimeError, match="Import"):
            calc.calculate_carrier_concentrations()

    def test_returns_interpolators(self, loaded_calculator):
        fn, fp = loaded_calculator.calculate_carrier_concentrations(delta_n=0.0)
        # fn and fp should be callable
        mid = (loaded_calculator.VBM + loaded_calculator.CBM) / 2
        assert fn(mid) >= 0
        assert fp(mid) >= 0

    def test_charge_neutrality_intrinsic(self, loaded_calculator):
        """For intrinsic (nd=0), n ≈ p at equilibrium."""
        loaded_calculator.nd = 0
        loaded_calculator.calculate_carrier_concentrations(delta_n=0.0)
        # Should be approximately equal for intrinsic
        ratio = loaded_calculator.n / loaded_calculator.p
        assert 0.5 < ratio < 2.0  # loose bound for discretised k-grid


# ======================================================================
# Energy cutoffs
# ======================================================================
class TestEnergyCutoffs:

    def test_returns_positive_windows(self, loaded_calculator):
        loaded_calculator.calculate_carrier_concentrations(delta_n=1e17)
        cb_w, vb_w = loaded_calculator.calculate_energy_cutoffs(charge_threshold=0.99)
        assert cb_w >= 0
        assert vb_w >= 0

    def test_larger_threshold_wider_window(self, loaded_calculator):
        loaded_calculator.calculate_carrier_concentrations(delta_n=1e17)
        cb_90, vb_90 = loaded_calculator.calculate_energy_cutoffs(charge_threshold=0.90)
        cb_99, vb_99 = loaded_calculator.calculate_energy_cutoffs(charge_threshold=0.99)
        assert cb_99 >= cb_90
        assert vb_99 >= vb_90


# ======================================================================
# Read-back helpers
# ======================================================================
class TestReadBack:

    def test_read_auger_pairs(self, loaded_calculator, sample_pairs_csv):
        data, auger_type = loaded_calculator.read_auger_pairs(sample_pairs_csv)
        assert auger_type == "eeh"
        assert len(data) == 5
        assert loaded_calculator.is_auger_pairs_created["eeh"] is True

    def test_read_auger_pairs_list(self, loaded_calculator, sample_pairs_csv):
        data, _ = loaded_calculator.read_auger_pairs([sample_pairs_csv])
        assert len(data) == 5

    def test_read_auger_pairs_not_found(self, loaded_calculator):
        with pytest.raises(FileNotFoundError):
            loaded_calculator.read_auger_pairs("nonexistent.csv")

    def test_read_matrix_elements(self, loaded_calculator, sample_pairs_csv,
                                  sample_matrix_elements_jsonl):
        # First load pairs so firstCB_index is known
        loaded_calculator.read_auger_pairs(sample_pairs_csv)
        data = loaded_calculator.read_matrix_elements(sample_matrix_elements_jsonl)
        assert len(data) == 5
        assert all("|M|^2" in d for d in data)

    def test_read_matrix_elements_not_found(self, loaded_calculator):
        with pytest.raises(FileNotFoundError):
            loaded_calculator.read_matrix_elements("nonexistent.jsonl")


# ======================================================================
# Auger rate calculation
# ======================================================================
class TestAugerRates:

    @pytest.fixture
    def calc_with_pairs_and_me(self, loaded_calculator, sample_pairs_csv,
                                sample_matrix_elements_jsonl):
        """Calculator loaded with pairs and matrix elements."""
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        calc.read_auger_pairs(sample_pairs_csv)
        calc.read_matrix_elements(sample_matrix_elements_jsonl)
        return calc

    def test_calculate_auger_rates(self, calc_with_pairs_and_me):
        results = calc_with_pairs_and_me.calculate_auger_rates(
            auger_type="eeh",
            delta_function=("Gaussian",),
            FWHM=(0.05,),
        )
        assert len(results) == 1
        assert "Auger coefficient" in results[0]
        assert "Delta function" in results[0]
        assert "FWHM" in results[0]

    def test_multiple_delta_fwhm(self, calc_with_pairs_and_me):
        results = calc_with_pairs_and_me.calculate_auger_rates(
            auger_type="eeh",
            delta_function=("Gaussian", "Lorentzian"),
            FWHM=(0.01, 0.05, 0.1),
        )
        assert len(results) == 6  # 2 deltas × 3 FWHMs

    def test_raises_without_matrix_elements(self, loaded_calculator, sample_pairs_csv):
        loaded_calculator.calculate_carrier_concentrations(delta_n=1e17)
        loaded_calculator.read_auger_pairs(sample_pairs_csv)
        with pytest.raises(RuntimeError, match="matrix elements"):
            loaded_calculator.calculate_auger_rates(auger_type="eeh")

    def test_raises_without_pairs(self, loaded_calculator):
        loaded_calculator.calculate_carrier_concentrations(delta_n=1e17)
        with pytest.raises(RuntimeError):
            loaded_calculator.calculate_auger_rates(auger_type="eeh")

    def test_output_csv_created(self, calc_with_pairs_and_me):
        calc = calc_with_pairs_and_me
        calc.calculate_auger_rates(
            auger_type="eeh",
            delta_function=("Gaussian",),
            FWHM=(0.05,),
        )
        # Check that output files exist in results_folder
        folder = calc.results_folder
        csv_files = [f for f in os.listdir(folder)
                     if f.startswith("Auger_coefficients_eeh")]
        assert len(csv_files) >= 1

    def test_unknown_delta_raises(self, calc_with_pairs_and_me):
        with pytest.raises(ValueError, match="Unknown delta"):
            calc_with_pairs_and_me.calculate_auger_rates(
                auger_type="eeh",
                delta_function=("FakeDelta",),
                FWHM=(0.05,),
            )


# ======================================================================
# Validation
# ======================================================================
class TestValidation:

    def test_invalid_auger_type(self, loaded_calculator):
        with pytest.raises(ValueError, match="auger_type"):
            loaded_calculator._validate_state("pairs", "xxx")

    def test_invalid_approach(self, loaded_calculator):
        with pytest.raises(ValueError, match="approach"):
            loaded_calculator._validate_state("pairs", "eeh", approach="invalid")

    def test_invalid_search_mode(self, loaded_calculator):
        with pytest.raises(ValueError, match="search_mode"):
            loaded_calculator._validate_state("pairs", "eeh", search_mode="invalid")

    def test_not_imported(self):
        calc = AugerCalculator(T=300, nd=0)
        with pytest.raises(RuntimeError, match="Import"):
            calc._validate_state("pairs", "eeh")
