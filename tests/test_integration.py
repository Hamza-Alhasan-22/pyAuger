"""
Integration tests — end-to-end workflows using synthetic data.

Tests the full pipeline:
  import → carrier concentrations → pair generation → Auger rates
  (matrix elements are mocked since we don't have real WAVECARs)

Also tests cross-module interactions and resumption workflows.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from auger.calculator import AugerCalculator
from auger.pairs import PairGenerator, Pair
from auger import utilities as ut


# ======================================================================
# Full nearest-kpoint pipeline (no WAVECAR)
# ======================================================================
class TestNearestKpointPipeline:
    """End-to-end test: import → carriers → pairs → (mock ME) → rates."""

    @pytest.fixture
    def pipeline_calc(self, fake_parsed_data, tmp_path):
        """Run the pipeline up to and including pair generation."""
        calc = AugerCalculator(T=300, nd=0)
        calc.import_parsed_BS_data(fake_parsed_data)
        calc.results_folder = str(tmp_path)
        calc.calculate_carrier_concentrations(delta_n=1e17)
        calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )
        return calc

    def test_pairs_created(self, pipeline_calc):
        assert pipeline_calc.is_auger_pairs_created["eeh"] is True
        assert len(pipeline_calc.auger_pairs_dicts["eeh"]) > 0

    def test_pairs_have_correct_type(self, pipeline_calc):
        for p in pipeline_calc.auger_pairs_dicts["eeh"]:
            assert p["pair_type"] == "eeh"

    def test_mock_me_then_rates(self, pipeline_calc, tmp_path):
        """Inject fake matrix elements and compute rates."""
        calc = pipeline_calc
        pairs = calc.auger_pairs_dicts["eeh"]

        # Create fake matrix elements
        me_list = []
        for p in pairs:
            me_list.append({"pair_id": p["pair_id"], "|M|^2": 0.5})

        calc.matrix_elements_dicts["eeh"] = me_list
        calc.is_matrix_elements_calculated["eeh"] = True

        results = calc.calculate_auger_rates(
            auger_type="eeh",
            delta_function=("Gaussian",),
            FWHM=(0.05,),
        )
        assert len(results) == 1
        assert results[0]["Auger coefficient"] != 0

    def test_output_files_created(self, pipeline_calc, tmp_path):
        """Check that pair CSV files exist after pipeline."""
        csv_files = [f for f in os.listdir(str(tmp_path))
                     if f.endswith(".csv") and "pairs" in f]
        assert len(csv_files) >= 1


class TestEhhPipeline:
    """Same pipeline for ehh type."""

    def test_ehh_pipeline(self, fake_parsed_data, tmp_path):
        calc = AugerCalculator(T=300, nd=0)
        calc.import_parsed_BS_data(fake_parsed_data)
        calc.results_folder = str(tmp_path)
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="ehh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )
        assert len(gen.pairs) > 0
        assert calc.is_auger_pairs_created["ehh"] is True


# ======================================================================
# Max_Heap vs Brute_Force consistency
# ======================================================================
class TestSearchModeConsistency:
    """Verify that Max_Heap and Brute_Force produce overlapping top pairs."""

    def test_top_pair_in_both(self, fake_parsed_data, tmp_path):
        calc_bf = AugerCalculator(T=300, nd=0)
        calc_bf.import_parsed_BS_data(fake_parsed_data)
        calc_bf.results_folder = str(tmp_path / "bf")
        os.makedirs(calc_bf.results_folder, exist_ok=True)
        calc_bf.calculate_carrier_concentrations(delta_n=1e17)
        gen_bf = calc_bf.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )

        calc_mh = AugerCalculator(T=300, nd=0)
        calc_mh.import_parsed_BS_data(fake_parsed_data)
        calc_mh.results_folder = str(tmp_path / "mh")
        os.makedirs(calc_mh.results_folder, exist_ok=True)
        calc_mh.calculate_carrier_concentrations(delta_n=1e17)
        gen_mh = calc_mh.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Max_Heap",
            num_top_pairs="all",
        )

        if len(gen_bf.pairs) > 0 and len(gen_mh.pairs) > 0:
            # The highest-probability pair from brute force should appear
            # in the max-heap results
            bf_sorted = sorted(gen_bf.pairs, key=lambda p: p.probability, reverse=True)
            mh_ids = {p.pair_id for p in gen_mh.pairs}
            # Top pair should match
            assert bf_sorted[0].pair_id in mh_ids


# ======================================================================
# Continuation / resume workflow
# ======================================================================
class TestContinuationWorkflow:
    """Test that continue_from_files skips already-computed pairs."""

    def test_continue_pairs(self, fake_parsed_data, tmp_path):
        calc = AugerCalculator(T=300, nd=0)
        calc.import_parsed_BS_data(fake_parsed_data)
        calc.results_folder = str(tmp_path)
        calc.calculate_carrier_concentrations(delta_n=1e17)

        # First run
        gen1 = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )
        n1 = len(gen1.pairs)

        if n1 == 0:
            pytest.skip("No pairs generated with this synthetic data")

        # Get the CSV file(s) that were written
        csv_files = sorted([
            os.path.join(str(tmp_path), f)
            for f in os.listdir(str(tmp_path))
            if f.startswith("auger_eeh_pairs") and f.endswith(".csv")
        ])

        # Second run continuing from first
        calc2 = AugerCalculator(T=300, nd=0)
        calc2.import_parsed_BS_data(fake_parsed_data)
        calc2.results_folder = str(tmp_path / "run2")
        os.makedirs(calc2.results_folder, exist_ok=True)
        calc2.calculate_carrier_concentrations(delta_n=1e17)
        gen2 = calc2.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
            continue_from_files=csv_files,
        )
        # With continuation, all pairs are already done, so no new ones are
        # generated (only old ones are loaded)
        assert len(gen2.pairs) == n1


# ======================================================================
# Multiple auger types on same calculator
# ======================================================================
class TestMultipleTypes:

    def test_eeh_then_ehh(self, fake_parsed_data, tmp_path):
        calc = AugerCalculator(T=300, nd=0)
        calc.import_parsed_BS_data(fake_parsed_data)
        calc.results_folder = str(tmp_path)
        calc.calculate_carrier_concentrations(delta_n=1e17)

        gen_eeh = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )
        gen_ehh = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="ehh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )

        assert calc.is_auger_pairs_created["eeh"] is True
        assert calc.is_auger_pairs_created["ehh"] is True
        # Both should have independent pair lists
        eeh_ids = {p["pair_id"] for p in calc.auger_pairs_dicts["eeh"]}
        ehh_ids = {p["pair_id"] for p in calc.auger_pairs_dicts["ehh"]}
        # They could overlap in pair_id format but pair_type differs
        if calc.auger_pairs_dicts["eeh"]:
            assert calc.auger_pairs_dicts["eeh"][0]["pair_type"] == "eeh"
        if calc.auger_pairs_dicts["ehh"]:
            assert calc.auger_pairs_dicts["ehh"][0]["pair_type"] == "ehh"


# ======================================================================
# Auger rate with different delta functions
# ======================================================================
class TestAugerRateSensitivity:
    """Test that different delta/FWHM give different coefficients."""

    @pytest.fixture
    def calc_with_me(self, fake_parsed_data, tmp_path):
        calc = AugerCalculator(T=300, nd=0)
        calc.import_parsed_BS_data(fake_parsed_data)
        calc.results_folder = str(tmp_path)
        calc.calculate_carrier_concentrations(delta_n=1e17)
        calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )
        # Inject fake matrix elements
        me_list = [{"pair_id": p["pair_id"], "|M|^2": 0.5}
                   for p in calc.auger_pairs_dicts["eeh"]]
        calc.matrix_elements_dicts["eeh"] = me_list
        calc.is_matrix_elements_calculated["eeh"] = True
        return calc

    def test_different_fwhm(self, calc_with_me):
        results = calc_with_me.calculate_auger_rates(
            auger_type="eeh",
            delta_function=("Gaussian",),
            FWHM=(0.01, 0.1),
        )
        c1 = results[0]["Auger coefficient"]
        c2 = results[1]["Auger coefficient"]
        # They should be different (unless all dE=0 exactly)
        if c1 != 0 and c2 != 0:
            assert c1 != c2

    def test_all_three_deltas(self, calc_with_me):
        results = calc_with_me.calculate_auger_rates(
            auger_type="eeh",
            delta_function=("Gaussian", "Lorentzian", "Rectangular"),
            FWHM=(0.05,),
        )
        assert len(results) == 3
        deltas = {r["Delta function"] for r in results}
        assert deltas == {"Gaussian", "Lorentzian", "Rectangular"}


# ======================================================================
# CSV round-trip for completed pairs
# ======================================================================
class TestCompletedPairsCSV:
    """After rates calculation, completed pairs CSV should exist."""

    def test_completed_csv(self, fake_parsed_data, tmp_path):
        calc = AugerCalculator(T=300, nd=0)
        calc.import_parsed_BS_data(fake_parsed_data)
        calc.results_folder = str(tmp_path)
        calc.calculate_carrier_concentrations(delta_n=1e17)
        calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )
        me_list = [{"pair_id": p["pair_id"], "|M|^2": 0.5}
                   for p in calc.auger_pairs_dicts["eeh"]]
        calc.matrix_elements_dicts["eeh"] = me_list
        calc.is_matrix_elements_calculated["eeh"] = True
        calc.calculate_auger_rates(auger_type="eeh",
                                    delta_function=("Gaussian",),
                                    FWHM=(0.05,))

        completed = [f for f in os.listdir(str(tmp_path))
                     if "completed" in f and f.endswith(".csv")]
        assert len(completed) >= 1

        # Read back and verify
        data = ut.read_csv([os.path.join(str(tmp_path), completed[0])])
        assert len(data) > 0
        # Should have the delta-function contribution columns
        has_contrib = any("C_Gaussian" in k for k in data[0].keys())
        assert has_contrib
