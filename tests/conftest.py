"""
Shared fixtures for the Auger recombination test suite.

Provides lightweight, self-contained mock data so that tests can run
without real VASP files (EIGENVAL, WAVECAR, vasprun.xml, etc.).
"""

import json
import os
import tempfile

import numpy as np
import pytest


# ── Paths ──────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_dir(tmp_path):
    """Return a temporary directory path as a string."""
    return str(tmp_path)


# ── Fake band_info.txt ─────────────────────────────────────────────────
BAND_INFO_TEMPLATE = """\
material_name TestMat
Crystal_System cubic
Space_Group Fm-3m
X 4
XX 64
E_Fermi 5.0
nbands 16
nkpoints 64
kgrid [4, 4, 4]
scissor_shift 0.0
band_gap 1.0
band_gap_after_shift 1.0
firstCB_index 8
lastVB_index 7
CBM 1.0
VBM 0.0
volume 100.0
dielectric_constant 12.0
b1 [1.0, 0.0, 0.0]
b2 [0.0, 1.0, 0.0]
b3 [0.0, 0.0, 1.0]
NELECT 16
q_TF 0.5
omega_p 1e16
"""


@pytest.fixture
def band_info_file(tmp_path):
    """Write a minimal band_info.txt and return its path."""
    p = tmp_path / "band_info.txt"
    p.write_text(BAND_INFO_TEMPLATE)
    return str(p)


@pytest.fixture
def band_info_with_carriers(tmp_path):
    """Write a band_info.txt that includes carrier concentration data."""
    text = BAND_INFO_TEMPLATE + """\
nd 1e17
Ef_eq 0.5
ni 1e10
n 1e17
p 1e17
delta_n 1e17
Efn 0.55
Efp 0.45
"""
    p = tmp_path / "band_info.txt"
    p.write_text(text)
    return str(p)


# ── Fake energy / kpoint arrays ────────────────────────────────────────
@pytest.fixture
def fake_parsed_data(tmp_path):
    """Create minimal .npy files + band_info.txt matching the template.

    The energies are designed so that only a handful of states fall inside
    a typical 0.1 eV CB/VB window, keeping brute-force pair generation
    O(N³) tractable for unit tests (< 1 second).

    Band layout (VBM = 0, CBM = 1.0):
      - Bands 0-6: deep VB, well below -0.5 eV → outside any test window
      - Band 7:    VBM band, energies in [-0.05, 0.0] (only a few states
                   within 0.1 eV of VBM)
      - Band 8:    CBM band, energies in [1.0, 1.05]
      - Bands 9-15: deep CB, above 1.5 eV → outside any test window
    """
    nbands, nkpts = 16, 64
    X, XX = 4, 64

    energies = np.zeros((nbands, nkpts))
    # Deep VB bands — far from VBM
    for b in range(7):
        energies[b] = -3.0 + b * 0.3  # flat, between -3.0 and -1.2

    # VBM band — small dispersion near 0
    energies[7] = np.linspace(-0.05, 0.0, nkpts)

    # CBM band — small dispersion near 1.0
    energies[8] = np.linspace(1.0, 1.05, nkpts)

    # Deep CB bands — far from CBM
    for b in range(9, 16):
        energies[b] = 2.0 + (b - 9) * 0.3  # flat, between 2.0 and 4.1

    # K-points: simple uniform grid in Cartesian
    kpoints = np.array([
        [i * 0.25, j * 0.25, k * 0.25]
        for i in range(4) for j in range(4) for k in range(4)
    ], dtype=float)

    weights = np.ones(nkpts) / nkpts

    np.save(str(tmp_path / f"Egrid_{X}_{XX}.npy"), energies)
    np.save(str(tmp_path / f"kgrid_{X}_{XX}.npy"), kpoints)
    np.save(str(tmp_path / f"kw_{X}_{XX}.npy"), weights)

    (tmp_path / "band_info.txt").write_text(BAND_INFO_TEMPLATE)

    return str(tmp_path)


# ── Fake pairs CSV content ─────────────────────────────────────────────
@pytest.fixture
def sample_pairs_csv(tmp_path):
    """Write a minimal pairs CSV and return its path."""
    import pandas as pd

    rows = []
    for i in range(5):
        rows.append({
            "pair_id": f"8-9-8-7-{i}-{i+1}-{i}-{i+2}",
            "pair_type": "eeh",
            "E1_index": 8, "E2_index": 9, "E3_index": 8, "E4_index": 7,
            "k1_index": i, "k2_index": i + 1, "k3_index": i, "k4_index": i + 2,
            "E1": 1.05, "E2": 1.15, "E3": 1.05, "E4": -0.05,
            "k1": [0.0, 0.0, 0.0], "k2": [0.25, 0.0, 0.0],
            "k3": [0.0, 0.0, 0.0], "k4": [0.25, 0.0, 0.0],
            "kw1": 1.0 / 64, "kw2": 1.0 / 64, "kw3": 1.0 / 64, "kw4": 1.0 / 64,
            "k2_mapped": [0.25, 0.0, 0.0],
            "probability": 0.01 * (5 - i),
        })

    p = tmp_path / "pairs.csv"
    pd.DataFrame(rows).to_csv(str(p), index=False)
    return str(p)


# ── Fake matrix elements JSONL ─────────────────────────────────────────
@pytest.fixture
def sample_matrix_elements_jsonl(tmp_path, sample_pairs_csv):
    """Write a JSONL file with fake matrix elements matching sample_pairs_csv."""
    import pandas as pd
    pairs = pd.read_csv(sample_pairs_csv).to_dict("records")

    p = tmp_path / "matrix_elements.jsonl"
    with open(str(p), "w") as f:
        for row in pairs:
            entry = {"pair_id": row["pair_id"], "|M|^2": np.random.uniform(0.01, 1.0)}
            f.write(json.dumps(entry) + "\n")
    return str(p)


# ── Fake Auger coefficients CSV ───────────────────────────────────────
@pytest.fixture
def sample_auger_coefficients_csv(tmp_path):
    """Write a minimal Auger coefficients CSV."""
    import pandas as pd

    rows = []
    for delta in ("Gaussian", "Lorentzian"):
        for fwhm in (0.01, 0.03, 0.05, 0.07, 0.1):
            rows.append({
                "Delta function": delta,
                "FWHM": fwhm,
                "Auger coefficient": np.random.uniform(1e-32, 1e-28),
            })

    p = tmp_path / "Auger_coefficients_eeh_64.csv"
    pd.DataFrame(rows).to_csv(str(p), index=False)
    return str(p)


# ── Lightweight AugerCalculator with pre-loaded data ───────────────────
@pytest.fixture
def loaded_calculator(fake_parsed_data):
    """Return an AugerCalculator with band data already imported."""
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from auger import AugerCalculator

    calc = AugerCalculator(T=300, nd=0)
    calc.import_parsed_BS_data(fake_parsed_data)
    return calc
