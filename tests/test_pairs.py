"""
Tests for auger.pairs — Pair data class and PairGenerator.

Exercises Pair construction (tuple and dict), serialisation round-trips,
PairGenerator energy-state initialisation, nearest_kpoint resolution,
brute-force and max-heap search, and CSV write/chunking logic.
"""

import os

import numpy as np
import pandas as pd
import pytest

from auger.pairs import Pair, PairGenerator, _state_dict, _to_list
from auger import utilities as ut


# ======================================================================
# Pair data class
# ======================================================================
class TestPairFromTuple:
    """Test Pair construction from a positional tuple."""

    @pytest.fixture
    def sample_pair(self):
        t = (
            "8-9-8-7-0-1-2-3",  # pair_id
            1.05, 1.15, 1.05, -0.05,  # E1..E4
            np.array([0.0, 0.0, 0.0]),  # k1
            np.array([0.25, 0.0, 0.0]),  # k2
            np.array([0.0, 0.25, 0.0]),  # k3
            np.array([0.0, 0.0, 0.25]),  # k4
            0.015625, 0.015625, 0.015625, 0.015625,  # kw1..kw4
            8, 9, 8, 7,  # E1_index..E4_index
            0, 1, 2, 3,  # k1_index..k4_index
            0.001,  # probability
            "eeh",  # pair_type
        )
        return Pair(t)

    def test_pair_id(self, sample_pair):
        assert sample_pair.pair_id == "8-9-8-7-0-1-2-3"

    def test_energies(self, sample_pair):
        assert sample_pair.E1 == 1.05
        assert sample_pair.E4 == -0.05

    def test_kpoints(self, sample_pair):
        np.testing.assert_allclose(sample_pair.k1, [0, 0, 0])

    def test_probability(self, sample_pair):
        assert sample_pair.probability == 0.001

    def test_pair_type(self, sample_pair):
        assert sample_pair.pair_type == "eeh"

    def test_matrix_element_initially_none(self, sample_pair):
        assert sample_pair.matrix_element is None

    def test_set_matrix_element(self, sample_pair):
        sample_pair.set_matrix_element(0.42)
        assert sample_pair.matrix_element == 0.42

    def test_set_mapped_kpoints(self, sample_pair):
        sample_pair.set_mapped_kpoints(None, np.array([0.1, 0.2, 0.3]), None, None)
        assert sample_pair.mapped_kpoints[1] is not None
        np.testing.assert_allclose(sample_pair.mapped_kpoints[1], [0.1, 0.2, 0.3])


class TestPairFromDict:
    """Test Pair construction from a dictionary."""

    @pytest.fixture
    def sample_dict(self):
        return {
            "pair_id": "8-9-8-7-0-1-2-3",
            "pair_type": "eeh",
            "E1": 1.05, "E2": 1.15, "E3": 1.05, "E4": -0.05,
            "k1": [0.0, 0.0, 0.0], "k2": [0.25, 0.0, 0.0],
            "k3": [0.0, 0.25, 0.0], "k4": [0.0, 0.0, 0.25],
            "kw1": 0.015625, "kw2": 0.015625,
            "kw3": 0.015625, "kw4": 0.015625,
            "E1_index": 8, "E2_index": 9, "E3_index": 8, "E4_index": 7,
            "k1_index": 0, "k2_index": 1, "k3_index": 2, "k4_index": 3,
            "probability": 0.001,
        }

    def test_from_dict(self, sample_dict):
        pair = Pair(sample_dict)
        assert pair.pair_id == "8-9-8-7-0-1-2-3"
        assert pair.E1 == 1.05
        assert pair.probability == 0.001

    def test_kpoints_as_arrays(self, sample_dict):
        pair = Pair(sample_dict)
        assert isinstance(pair.k1, np.ndarray)

    def test_round_trip(self, sample_dict):
        """dict → Pair → dict should preserve data."""
        pair = Pair(sample_dict)
        d = pair.get_pair_as_dict()
        assert d["pair_id"] == sample_dict["pair_id"]
        assert d["E1"] == sample_dict["E1"]
        assert d["probability"] == sample_dict["probability"]


class TestPairSerialization:
    """Test get_pair_as_dict output."""

    def test_dict_has_required_keys(self):
        t = (
            "test", 1.0, 1.1, 1.0, -0.1,
            np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
            0.01, 0.01, 0.01, 0.01,
            8, 9, 8, 7, 0, 1, 2, 3,
            0.001, "eeh",
        )
        d = Pair(t).get_pair_as_dict()
        required = {"pair_id", "pair_type", "E1", "E2", "E3", "E4",
                     "k1", "k2", "k3", "k4",
                     "kw1", "kw2", "kw3", "kw4",
                     "E1_index", "E2_index", "E3_index", "E4_index",
                     "k1_index", "k2_index", "k3_index", "k4_index",
                     "probability"}
        assert required.issubset(d.keys())

    def test_kpoints_serialised_as_lists(self):
        t = (
            "test", 1.0, 1.1, 1.0, -0.1,
            np.array([0.1, 0.2, 0.3]), np.zeros(3), np.zeros(3), np.zeros(3),
            0.01, 0.01, 0.01, 0.01,
            8, 9, 8, 7, 0, 1, 2, 3,
            0.001, "eeh",
        )
        d = Pair(t).get_pair_as_dict()
        assert isinstance(d["k1"], list)
        assert len(d["k1"]) == 3


# ======================================================================
# Helper functions
# ======================================================================
class TestHelpers:

    def test_state_dict(self):
        d = _state_dict(8, 5, 1.05, np.array([0.1, 0.2, 0.3]), 0.015, 0.8)
        assert d["band_index"] == 8
        assert d["k_index"] == 5
        assert d["energy"] == 1.05
        assert d["P"] == 0.8

    def test_to_list_ndarray(self):
        assert _to_list(np.array([1, 2, 3])) == [1, 2, 3]

    def test_to_list_none(self):
        assert _to_list(None) == []

    def test_to_list_plain_list(self):
        assert _to_list([1, 2]) == [1, 2]


# ======================================================================
# PairGenerator — energy state initialisation
# ======================================================================
class TestPairGeneratorInit:

    def test_init_eeh_states(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = PairGenerator("eeh", (
            calc, 0.3, 0.3, "nearest_kpoint", False, "Brute_Force",
            -1, "", None, False,
        ))
        assert len(gen.E1_energies) > 0
        assert len(gen.E3_energies) > 0
        assert len(gen.E4_energies) > 0
        # E1 and E3 should be the same for eeh (both CB states)
        assert len(gen.E1_energies) == len(gen.E3_energies)

    def test_init_ehh_states(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = PairGenerator("ehh", (
            calc, 0.3, 0.3, "nearest_kpoint", False, "Brute_Force",
            -1, "", None, False,
        ))
        assert len(gen.E1_energies) > 0
        assert len(gen.E2_energies) > 0
        assert len(gen.E3_energies) > 0
        # E2 and E3 should be the same for ehh (both VB states)
        assert len(gen.E2_energies) == len(gen.E3_energies)

    def test_states_sorted_by_probability(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = PairGenerator("eeh", (
            calc, 0.3, 0.3, "nearest_kpoint", False, "Max_Heap",
            -1, "", None, False,
        ))
        if len(gen.E1_energies) > 1:
            probs = [e["P"] for e in gen.E1_energies]
            assert probs == sorted(probs, reverse=True)


# ======================================================================
# PairGenerator — nearest_kpoint resolution
# ======================================================================
class TestNearestKpoint:

    @pytest.fixture
    def gen_eeh(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        return PairGenerator("eeh", (
            calc, 0.3, 0.3, "nearest_kpoint", False, "Brute_Force",
            -1, "", None, False,
        ))

    def test_returns_required_keys(self, gen_eeh, loaded_calculator):
        k_diff = np.array([0.1, 0.1, 0.1])
        E_diff = 1.1
        rl = loaded_calculator.reciprocal_lattice
        result = gen_eeh.nearest_kpoint(k_diff, E_diff, rl)
        required = {"kx_target_cart", "kx_target_frac", "kx_target_frac_mapped",
                     "kx_target_cart_mapped", "nearest_kx_index", "nearest_kx",
                     "nearest_kwx", "Ex_index", "Ex", "Px"}
        assert required.issubset(result.keys())

    def test_nearest_index_is_valid(self, gen_eeh, loaded_calculator):
        rl = loaded_calculator.reciprocal_lattice
        result = gen_eeh.nearest_kpoint(np.array([0.0, 0.0, 0.0]), 1.0, rl)
        assert 0 <= result["nearest_kx_index"] < loaded_calculator.num_of_kpoints

    def test_Px_between_0_and_1(self, gen_eeh, loaded_calculator):
        rl = loaded_calculator.reciprocal_lattice
        result = gen_eeh.nearest_kpoint(np.array([0.25, 0.0, 0.0]), 1.2, rl)
        assert 0.0 <= result["Px"] <= 1.0


# ======================================================================
# PairGenerator — brute force pair creation
# ======================================================================
class TestBruteForcePairs:

    def test_brute_force_creates_pairs(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
            num_top_pairs="all",
        )
        assert len(gen.pairs) > 0
        assert calc.is_auger_pairs_created["eeh"] is True

    def test_pairs_have_positive_probability(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )
        for pair in gen.pairs:
            assert pair.probability >= 0

    def test_ehh_brute_force(self, loaded_calculator):
        calc = loaded_calculator
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
# PairGenerator — max heap pair creation
# ======================================================================
class TestMaxHeapPairs:

    def test_max_heap_creates_pairs(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Max_Heap",
            num_top_pairs=10,
        )
        assert len(gen.pairs) > 0
        assert len(gen.pairs) <= 10

    def test_max_heap_sorted_by_probability(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Max_Heap",
            num_top_pairs=10,
        )
        probs = [p.probability for p in gen.pairs]
        assert probs == sorted(probs, reverse=True)

    def test_max_heap_ehh(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="ehh",
            approach="nearest_kpoint",
            search_mode="Max_Heap",
            num_top_pairs=10,
        )
        assert len(gen.pairs) > 0


# ======================================================================
# PairGenerator — CSV write/read
# ======================================================================
class TestPairsCSVWriteRead:

    def test_write_and_read_pairs(self, loaded_calculator, tmp_path):
        calc = loaded_calculator
        calc.results_folder = str(tmp_path)
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )
        # Pairs CSV should have been written
        csv_files = [f for f in os.listdir(str(tmp_path))
                     if f.startswith("auger_eeh_pairs") and f.endswith(".csv")]
        assert len(csv_files) >= 1

        # Read them back
        gen2 = PairGenerator("eeh")
        raw = gen2.read_pairs_from_csv(
            [os.path.join(str(tmp_path), f) for f in csv_files]
        )
        assert len(raw) == len(gen.pairs)

    def test_exclude_calculated_pairs(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = calc.create_auger_pairs(
            CB_window=0.1, VB_window=0.1,
            auger_type="eeh",
            approach="nearest_kpoint",
            search_mode="Brute_Force",
        )
        n_before = len(gen.pairs)
        if n_before > 0:
            to_exclude = [gen.pairs[0].get_pair_as_dict()]
            gen.exclude_calculated_pairs(to_exclude)
            assert len(gen.pairs) == n_before - 1


# ======================================================================
# PairGenerator — exact_kpoint helpers
# ======================================================================
class TestExactKpoint:

    def test_exact_kpoint_folds(self, loaded_calculator):
        calc = loaded_calculator
        calc.calculate_carrier_concentrations(delta_n=1e17)
        gen = PairGenerator("eeh", (
            calc, 0.3, 0.3, "exact_kpoint", False, "Brute_Force",
            -1, "", None, False,
        ))
        rl = calc.reciprocal_lattice
        result = gen.exact_kpoint(np.array([0.6, 0.6, 0.6]), rl)
        # Mapped fractional should be inside [-0.5, 0.5]
        for c in result["kx_target_frac_mapped"]:
            assert -0.5 - 1e-10 <= c <= 0.5 + 1e-10

    def test_find_closest_energy(self, loaded_calculator):
        calc = loaded_calculator
        data = calc.data_energies
        # Find closest to CBM at k-point 0
        bi, e = PairGenerator._find_closest_band_at_kpoint(
            calc.CBM, data, 0
        )
        assert 0 <= bi < calc.num_of_bands
        assert isinstance(e, float)
