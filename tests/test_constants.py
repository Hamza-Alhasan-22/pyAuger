"""
Tests for auger.constants — verify physical constants and derived quantities.
"""

import math
import pytest

from auger.constants import (
    ALPHA_PENN,
    ANGSTROM,
    BOHR_TO_ANGSTROM,
    CM_PER_ANGSTROM,
    EPSILON_0,
    HBAR,
    K_B_J,
    K_B_eV,
    M_E,
    MATRIX_FACTOR,
    eV,
)


class TestFundamentalConstants:
    """Spot-check that constants match NIST / CODATA values."""

    def test_eV(self):
        assert eV == pytest.approx(1.602_176_634e-19, rel=1e-9)

    def test_angstrom(self):
        assert ANGSTROM == 1e-10

    def test_epsilon_0(self):
        assert EPSILON_0 == pytest.approx(8.854_187_8128e-12, rel=1e-9)

    def test_k_B_eV(self):
        assert K_B_eV == pytest.approx(8.617_333_262_145e-5, rel=1e-9)

    def test_k_B_J(self):
        assert K_B_J == pytest.approx(K_B_eV * eV, rel=1e-12)

    def test_hbar(self):
        assert HBAR == pytest.approx(1.054_571_817e-34, rel=1e-9)

    def test_electron_mass(self):
        assert M_E == pytest.approx(9.109_383_56e-31, rel=1e-6)

    def test_bohr_to_angstrom(self):
        assert BOHR_TO_ANGSTROM == pytest.approx(0.529_177_210_903, rel=1e-9)


class TestDerivedConstants:
    """Check that derived prefactors are consistent."""

    def test_matrix_factor(self):
        expected = eV**2 * ANGSTROM**2 * (1.0 / EPSILON_0)
        assert MATRIX_FACTOR == pytest.approx(expected, rel=1e-12)

    def test_cm_per_angstrom(self):
        assert CM_PER_ANGSTROM == 1e-8

    def test_alpha_penn(self):
        assert ALPHA_PENN == pytest.approx(1.563, rel=1e-6)
