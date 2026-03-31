"""
Ab-Initio Direct Auger Recombination Calculator
================================================

A Python package for calculating direct Auger recombination coefficients
(C_n and C_p) for semiconductors using first-principles data from VASP.

Implements both a standard "Full Grid" approach and a novel "Active Space"
strategy that pre-screens energy- and momentum-conserving scattering channels
before performing expensive wavefunction calculations.

Main classes
------------
AugerCalculator : Orchestrates the complete Auger calculation workflow.
PairGenerator   : Identifies valid 4-particle scattering channels.
MatrixElements  : Computes Coulomb matrix elements from VASP wavefunctions.
AugerAnalyzer   : Post-processing, visualization, and convergence analysis.

Quick start
-----------
>>> from auger import AugerCalculator
>>> calc = AugerCalculator(T=300, nd=1e17)
>>> calc.import_parsed_BS_data("./parsed_data")
>>> calc.calculate_carrier_concentrations(delta_n=1e17)
>>> calc.create_auger_pairs(CB_window=0.3, VB_window=0.3, auger_type='eeh')
>>> calc.calculate_matrix_elements(auger_type='eeh', wavecar_files='WAVECAR', dielectric_constant=16.8)
>>> calc.calculate_auger_rates(auger_type='eeh')
"""

from .calculator import AugerCalculator
from .pairs import PairGenerator, Pair
from .matrix_elements import MatrixElements
from .analysis import AugerAnalyzer
from . import utilities

__version__ = "2.0.0"
__all__ = [
    "AugerCalculator",
    "PairGenerator",
    "Pair",
    "MatrixElements",
    "AugerAnalyzer",
    "utilities",
]
