"""
Physical constants used throughout the Auger recombination package.

All values are in SI units unless otherwise noted.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------
eV = 1.602_176_634e-19          # Joules per electron-volt  [J]
ANGSTROM = 1e-10                # Metres per Ångström       [m]
EPSILON_0 = 8.854_187_8128e-12  # Vacuum permittivity       [F/m]
K_B_eV = 8.617_333_262_145e-5   # Boltzmann constant        [eV/K]
K_B_J = K_B_eV * eV             # Boltzmann constant        [J/K]
HBAR = 1.054_571_817e-34        # Reduced Planck constant   [J·s]
M_E = 9.109_383_56e-31          # Electron rest mass        [kg]
BOHR_TO_ANGSTROM = 0.529_177_210_903  # Bohr radius → Å

# ---------------------------------------------------------------------------
# Derived prefactors (used in matrix element / Auger rate formulae)
# ---------------------------------------------------------------------------
# Coulomb matrix-element prefactor:  e^2 * Å^2 / ε₀
MATRIX_FACTOR = (eV ** 2) * (ANGSTROM ** 2) * (1.0 / EPSILON_0)

# Dielectric fitting parameter (Penn model α)
ALPHA_PENN = 1.563

# ---------------------------------------------------------------------------
# Unit-conversion helpers
# ---------------------------------------------------------------------------
CM_PER_ANGSTROM = 1e-8   # 1 Å = 1 × 10⁻⁸ cm
