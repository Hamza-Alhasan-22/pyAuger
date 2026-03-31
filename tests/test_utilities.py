"""
Tests for auger.utilities — I/O, Fermi–Dirac, BZ folding, delta functions,
Coulomb helpers, NSCF helpers, and CSV round-tripping.
"""

import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from auger import utilities as ut
from auger.constants import K_B_eV


# ======================================================================
# Fermi–Dirac
# ======================================================================
class TestFermiDirac:
    """Test the FD distribution under various conditions."""

    def test_at_fermi_level(self):
        """f(Ef) = 0.5 exactly."""
        assert ut.fermi_dirac(0.5, 0.5, 300) == pytest.approx(0.5, abs=1e-12)

    def test_deep_below_fermi(self):
        """Well below Ef → f ≈ 1."""
        assert ut.fermi_dirac(-1.0, 0.5, 300) == pytest.approx(1.0, abs=1e-6)

    def test_far_above_fermi(self):
        """Well above Ef → f ≈ 0."""
        assert ut.fermi_dirac(3.0, 0.5, 300) == pytest.approx(0.0, abs=1e-6)

    def test_zero_temperature_limit(self):
        """At T → 0, should be a step function (use T=0.01 K)."""
        assert ut.fermi_dirac(-0.1, 0.0, 0.01) == pytest.approx(1.0, abs=1e-6)
        assert ut.fermi_dirac(0.1, 0.0, 0.01) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        """f(Ef + x) + f(Ef - x) = 1."""
        Ef, T = 0.5, 300
        for dx in [0.01, 0.05, 0.1, 0.3]:
            total = ut.fermi_dirac(Ef + dx, Ef, T) + ut.fermi_dirac(Ef - dx, Ef, T)
            assert total == pytest.approx(1.0, abs=1e-10)

    def test_high_temperature(self):
        """At high T the distribution flattens toward 0.5."""
        val = ut.fermi_dirac(1.0, 0.0, 1e6)
        assert 0.4 < val < 0.6

    def test_return_type(self):
        assert isinstance(ut.fermi_dirac(0.0, 0.0, 300), float)


# ======================================================================
# Brillouin-zone folding
# ======================================================================
class TestFoldKpoint:
    """Test fold_kpoint_to_first_bz with various conventions."""

    @pytest.mark.parametrize("convention,expected", [
        ("zero_to_one", [0.7, 0.3, 0.0]),
        ("centered", [-0.3, 0.3, 0.0]),
        ("vasp_centered", [-0.3, 0.3, 0.0]),
    ])
    def test_basic_folding(self, convention, expected):
        k = np.array([0.7, 0.3, 0.0])
        result = ut.fold_kpoint_to_first_bz(k, convention=convention)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_vasp_centered_half_boundary(self):
        """vasp_centered maps -0.5 to +0.5."""
        k = np.array([-0.5, -0.5, -0.5])
        result = ut.fold_kpoint_to_first_bz(k, convention="vasp_centered")
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5], atol=1e-10)

    def test_centered_half_boundary(self):
        """centered maps 0.5 to -0.5."""
        k = np.array([0.5, 0.5, 0.5])
        result = ut.fold_kpoint_to_first_bz(k, convention="centered")
        np.testing.assert_allclose(result, [-0.5, -0.5, -0.5], atol=1e-10)

    def test_zero_to_one_negative(self):
        k = np.array([-0.3, -0.7, 1.2])
        result = ut.fold_kpoint_to_first_bz(k, convention="zero_to_one")
        np.testing.assert_allclose(result, [0.7, 0.3, 0.2], atol=1e-10)

    def test_large_coordinates(self):
        """K-points far outside [0,1) should still fold correctly."""
        k = np.array([3.7, -2.3, 5.0])
        result = ut.fold_kpoint_to_first_bz(k, convention="vasp_centered")
        for c in result:
            assert -0.5 - 1e-10 <= c <= 0.5 + 1e-10

    def test_already_in_zone(self):
        k = np.array([0.1, 0.2, 0.3])
        result = ut.fold_kpoint_to_first_bz(k, convention="vasp_centered")
        np.testing.assert_allclose(result, k, atol=1e-10)

    def test_invalid_convention_raises(self):
        with pytest.raises(ValueError, match="Unsupported convention"):
            ut.fold_kpoint_to_first_bz(np.array([0, 0, 0]), convention="invalid")


# ======================================================================
# Coordinate transforms
# ======================================================================
class TestCoordinateTransforms:
    """Test Cartesian ↔ fractional conversions."""

    def test_round_trip(self):
        """frac → cart → frac should be identity."""
        rl = np.array([[1.5, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 1.0]])
        k_frac = np.array([0.3, 0.4, 0.5])
        k_cart = ut.to_cartesian_coordinate(k_frac, rl)
        k_frac_back = ut.to_fractional_coordinate(k_cart, rl)
        np.testing.assert_allclose(k_frac_back, k_frac, atol=1e-10)

    def test_identity_lattice(self):
        """With identity reciprocal lattice, cart = frac."""
        rl = np.eye(3)
        k = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(ut.to_cartesian_coordinate(k, rl), k, atol=1e-12)
        np.testing.assert_allclose(ut.to_fractional_coordinate(k, rl), k, atol=1e-12)

    def test_non_orthogonal_lattice(self):
        rl = np.array([[1.0, 0.5, 0.0],
                        [0.0, 1.0, 0.5],
                        [0.5, 0.0, 1.0]])
        k_frac = np.array([1.0, 0.0, 0.0])
        k_cart = ut.to_cartesian_coordinate(k_frac, rl)
        np.testing.assert_allclose(k_cart, rl[0], atol=1e-12)


# ======================================================================
# Delta functions
# ======================================================================
class TestDeltaFunctions:
    """Test the three delta-function approximations."""

    def test_gaussian_peak(self):
        """Peak at x=0."""
        val = ut.delta_Gaussian(0.0, FWHM=0.05)
        assert val > 0

    def test_gaussian_symmetry(self):
        assert ut.delta_Gaussian(0.1, 0.05) == pytest.approx(
            ut.delta_Gaussian(-0.1, 0.05), rel=1e-10
        )

    def test_gaussian_normalisation(self):
        """Integral of Gaussian delta over a wide range ≈ 1."""
        xs = np.linspace(-5, 5, 100000)
        dx = xs[1] - xs[0]
        vals = np.array([ut.delta_Gaussian(x, 0.5) for x in xs])
        integral = np.sum(vals) * dx
        assert integral == pytest.approx(1.0, abs=0.01)

    def test_lorentzian_peak(self):
        val = ut.delta_Lorentzian(0.0, FWHM=0.03)
        assert val > 0

    def test_lorentzian_symmetry(self):
        assert ut.delta_Lorentzian(0.1, 0.05) == pytest.approx(
            ut.delta_Lorentzian(-0.1, 0.05), rel=1e-10
        )

    def test_lorentzian_normalisation(self):
        xs = np.linspace(-50, 50, 500000)
        dx = xs[1] - xs[0]
        vals = np.array([ut.delta_Lorentzian(x, 0.5) for x in xs])
        integral = np.sum(vals) * dx
        assert integral == pytest.approx(1.0, abs=0.02)

    def test_rectangular_inside(self):
        val = ut.delta_Rectangular(0.0, FWHM=0.2)
        assert val == pytest.approx(1.0 / 0.2, rel=1e-10)

    def test_rectangular_outside(self):
        val = ut.delta_Rectangular(0.5, FWHM=0.2)
        assert val == pytest.approx(0.0, abs=1e-15)

    def test_rectangular_normalisation(self):
        xs = np.linspace(-1, 1, 10000)
        dx = xs[1] - xs[0]
        vals = np.array([ut.delta_Rectangular(x, 0.4) for x in xs])
        integral = np.sum(vals) * dx
        assert integral == pytest.approx(1.0, abs=0.01)

    def test_delta_functions_dict(self):
        assert set(ut.DELTA_FUNCTIONS.keys()) == {"Gaussian", "Lorentzian", "Rectangular"}


# ======================================================================
# Coulomb / dielectric helpers
# ======================================================================
class TestCoulombHelpers:
    """Test I_ab, calculate_epsilon, and W."""

    def test_calculate_epsilon_zero_q(self):
        """At q=0, ε(0) = 1 + 1/a."""
        a, b, c = 0.1, 0.5, 1e-40
        eps = ut.calculate_epsilon(np.array([0.0, 0.0, 0.0]), a, b, c)
        assert eps == pytest.approx(1.0 + 1.0 / a, rel=1e-6)

    def test_calculate_epsilon_large_q(self):
        """For large q, ε → 1 (denominator dominates)."""
        a, b, c = 0.1, 0.5, 1e-80
        eps = ut.calculate_epsilon(np.array([100.0, 0.0, 0.0]), a, b, c)
        assert eps == pytest.approx(1.0, abs=0.1)

    def test_W_positive(self):
        """Screened Coulomb should be positive."""
        val = ut.W(0.5, 10.0, 0.01)
        assert val > 0

    def test_W_decreases_with_q(self):
        """W decreases with larger |q|."""
        w1 = ut.W(0.1, 10.0, 0.01)
        w2 = ut.W(1.0, 10.0, 0.01)
        assert w1 > w2

    def test_W_decreases_with_screening(self):
        """W decreases with larger screening."""
        w1 = ut.W(0.5, 10.0, 0.01)
        w2 = ut.W(0.5, 10.0, 0.1)
        assert w1 > w2

    def test_I_ab_identity(self):
        """Overlap of identical wavefunctions with G=0 → norm²."""
        G = np.array([0, 0, 0])
        Ga = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        coeffs = {(0, 0, 0): 0.5 + 0.1j, (1, 0, 0): 0.3 + 0.2j, (0, 1, 0): 0.4 - 0.1j}
        result = ut.I_ab(G, Ga, coeffs, coeffs)
        expected = sum(abs(c) ** 2 for c in coeffs.values())
        assert abs(result - expected) < 1e-10

    def test_I_ab_orthogonal(self):
        """Non-overlapping G-sets → 0."""
        G = np.array([0, 0, 0])
        Ga = np.array([[0, 0, 0]])
        da = {(0, 0, 0): 1.0 + 0j}
        db = {(5, 5, 5): 1.0 + 0j}
        result = ut.I_ab(G, Ga, da, db)
        assert abs(result) < 1e-15


# ======================================================================
# band_info I/O
# ======================================================================
class TestBandInfoIO:

    def test_read_band_info(self, band_info_file):
        info = ut.read_band_info(band_info_file)
        assert info["material_name"] == "TestMat"
        assert info["nbands"] == 16
        assert info["nkpoints"] == 64
        assert info["firstCB_index"] == 8
        assert info["lastVB_index"] == 7
        assert isinstance(info["volume"], float)
        assert info["volume"] == pytest.approx(100.0)

    def test_read_band_info_kgrid_is_list(self, band_info_file):
        info = ut.read_band_info(band_info_file)
        assert isinstance(info["kgrid"], list)
        assert len(info["kgrid"]) == 3

    def test_read_band_info_reciprocal_vectors(self, band_info_file):
        info = ut.read_band_info(band_info_file)
        assert len(info["b1"]) == 3
        assert info["b1"] == [1.0, 0.0, 0.0]


# ======================================================================
# CSV round-trip
# ======================================================================
class TestCSVRoundTrip:

    def test_write_and_read(self, tmp_dir):
        data = [
            {"pair_id": "1-2-3-4-0-1-2-3", "pair_type": "eeh",
             "E1": 1.0, "E2": 1.2, "E3": 1.0, "E4": -0.1,
             "k1": [0.0, 0.0, 0.0], "k2": [0.25, 0.0, 0.0],
             "k3": [0.0, 0.0, 0.0], "k4": [0.25, 0.0, 0.0],
             "probability": 0.05},
        ]
        ut.write_to_csv(data, "test_pairs", folder_to_save=tmp_dir)
        read_back = ut.read_csv([os.path.join(tmp_dir, "test_pairs.csv")])
        assert len(read_back) == 1
        assert read_back[0]["pair_id"] == "1-2-3-4-0-1-2-3"
        assert isinstance(read_back[0]["k1"], list)
        assert len(read_back[0]["k1"]) == 3

    def test_write_multiple_rows(self, tmp_dir):
        data = [{"x": i, "y": i * 2} for i in range(10)]
        ut.write_to_csv(data, "multi", folder_to_save=tmp_dir)
        read_back = ut.read_csv([os.path.join(tmp_dir, "multi.csv")])
        assert len(read_back) == 10

    def test_read_csv_multiple_files(self, tmp_dir):
        data1 = [{"pair_id": "a", "probability": 0.1}]
        data2 = [{"pair_id": "b", "probability": 0.2}]
        ut.write_to_csv(data1, "part1", folder_to_save=tmp_dir)
        ut.write_to_csv(data2, "part2", folder_to_save=tmp_dir)
        combined = ut.read_csv([
            os.path.join(tmp_dir, "part1.csv"),
            os.path.join(tmp_dir, "part2.csv"),
        ])
        assert len(combined) == 2
        ids = {r["pair_id"] for r in combined}
        assert ids == {"a", "b"}


# ======================================================================
# get_firstCB_and_lastVB
# ======================================================================
class TestGetCBVB:

    def test_clear_gap(self):
        """Simple case with a clear band gap."""
        nbands, nk = 10, 20
        data = np.zeros((nbands, nk))
        Ef = 0.0
        for b in range(5):
            data[b] = -1.0 + b * 0.1
        for b in range(5, 10):
            data[b] = 1.0 + (b - 5) * 0.1
        cb, vb = ut.get_firstCB_and_lastVB(data, Ef)
        assert cb == 5
        assert vb == 4

    def test_raises_on_failure(self):
        """All bands above Ef → should fail to find VB."""
        data = np.ones((5, 10)) * 2.0
        with pytest.raises(ValueError):
            ut.get_firstCB_and_lastVB(data, 0.0)


# ======================================================================
# convert_seconds
# ======================================================================
class TestConvertSeconds:

    def test_zero(self):
        assert ut.convert_seconds(0) == (0, 0, 0, 0)

    def test_one_hour(self):
        assert ut.convert_seconds(3600) == (0, 1, 0, 0)

    def test_complex_time(self):
        # 1 day + 2 hours + 3 minutes + 4 seconds = 93784
        d, h, m, s = ut.convert_seconds(93784)
        assert d == 1 and h == 2 and m == 3 and s == 4
