"""
Tests for auger.analysis — AugerAnalyzer class.

Tests entry management, combine_auger_types, convergence checking,
and basic error handling.  Plotting methods are tested for non-crash
behaviour only (no visual assertions).
"""

import os

import numpy as np
import pandas as pd
import pytest

from auger.analysis import AugerAnalyzer, _kgrid_label


# ======================================================================
# Entry management
# ======================================================================
class TestEntryManagement:

    def test_add_entry(self, band_info_file, sample_auger_coefficients_csv):
        az = AugerAnalyzer()
        az.add_result_entry("eeh", band_info_file, sample_auger_coefficients_csv)
        assert len(az.result_entries) == 1
        assert az.result_entries[0]["id"] == 1

    def test_add_multiple_entries(self, band_info_file, sample_auger_coefficients_csv):
        az = AugerAnalyzer()
        az.add_result_entry("eeh", band_info_file, sample_auger_coefficients_csv)
        az.add_result_entry("ehh", band_info_file, sample_auger_coefficients_csv)
        assert len(az.result_entries) == 2

    def test_delete_entry(self, band_info_file, sample_auger_coefficients_csv):
        az = AugerAnalyzer()
        az.add_result_entry("eeh", band_info_file, sample_auger_coefficients_csv)
        eid = az.result_entries[0]["id"]
        az.delete_result_entry(eid)
        assert len(az.result_entries) == 0

    def test_invalid_type_raises(self, band_info_file, sample_auger_coefficients_csv):
        az = AugerAnalyzer()
        with pytest.raises(ValueError, match="auger_type"):
            az.add_result_entry("xxx", band_info_file, sample_auger_coefficients_csv)

    def test_get_entry(self, band_info_file, sample_auger_coefficients_csv):
        az = AugerAnalyzer()
        az.add_result_entry("eeh", band_info_file, sample_auger_coefficients_csv)
        eid = az.result_entries[0]["id"]
        e = az._get_entry(eid)
        assert e["auger_type"] == "eeh"

    def test_get_entry_not_found(self, band_info_file, sample_auger_coefficients_csv):
        az = AugerAnalyzer()
        az.add_result_entry("eeh", band_info_file, sample_auger_coefficients_csv)
        with pytest.raises(ValueError, match="not found"):
            az._get_entry(999)

    def test_print_summary(self, band_info_file, sample_auger_coefficients_csv, capsys):
        az = AugerAnalyzer()
        az.add_result_entry("eeh", band_info_file, sample_auger_coefficients_csv)
        az.print_result_summary()
        out = capsys.readouterr().out
        assert "TestMat" in out

    def test_print_empty_summary(self, capsys):
        az = AugerAnalyzer()
        az.print_result_summary()
        out = capsys.readouterr().out
        assert "No entries" in out

    def test_get_ids_from_material(self, band_info_file, sample_auger_coefficients_csv):
        az = AugerAnalyzer()
        az.add_result_entry("eeh", band_info_file, sample_auger_coefficients_csv)
        ids = az.get_ids_from_material("TestMat")
        assert len(ids) == 1

    def test_get_ids_from_unknown_material(self, band_info_file, sample_auger_coefficients_csv):
        az = AugerAnalyzer()
        az.add_result_entry("eeh", band_info_file, sample_auger_coefficients_csv)
        ids = az.get_ids_from_material("Unknown")
        assert ids == []

    def test_print_entry_details(self, band_info_file, sample_auger_coefficients_csv, capsys):
        az = AugerAnalyzer()
        az.add_result_entry("eeh", band_info_file, sample_auger_coefficients_csv)
        eid = az.result_entries[0]["id"]
        e = az.print_entry_details(eid)
        assert e["auger_type"] == "eeh"

    def test_print_entry_not_found(self, capsys):
        az = AugerAnalyzer()
        e = az.print_entry_details(999)
        assert e == {}
        assert "not found" in capsys.readouterr().out


# ======================================================================
# Combine auger types
# ======================================================================
class TestCombine:

    @pytest.fixture
    def az_with_eeh_ehh(self, band_info_file, tmp_path):
        """Analyzer with matching eeh and ehh entries."""
        az = AugerAnalyzer()
        rows = []
        for delta in ("Gaussian", "Lorentzian"):
            for fwhm in (0.01, 0.03, 0.05):
                rows.append({
                    "Delta function": delta,
                    "FWHM": fwhm,
                    "Auger coefficient": 1e-30,
                })
        csv1 = tmp_path / "coeff_eeh.csv"
        csv2 = tmp_path / "coeff_ehh.csv"
        pd.DataFrame(rows).to_csv(str(csv1), index=False)
        pd.DataFrame(rows).to_csv(str(csv2), index=False)

        az.add_result_entry("eeh", band_info_file, str(csv1))
        az.add_result_entry("ehh", band_info_file, str(csv2))
        return az

    def test_combine_creates_csv(self, az_with_eeh_ehh, tmp_path):
        id_eeh = az_with_eeh_ehh.result_entries[0]["id"]
        id_ehh = az_with_eeh_ehh.result_entries[1]["id"]
        out_path = az_with_eeh_ehh.combine_auger_types(id_eeh, id_ehh, save_to=str(tmp_path))
        assert os.path.exists(out_path)
        df = pd.read_csv(out_path)
        assert "Auger coefficient" in df.columns
        assert len(df) == 6  # 2 deltas × 3 FWHMs

    def test_combine_sums_correctly(self, az_with_eeh_ehh, tmp_path):
        id_eeh = az_with_eeh_ehh.result_entries[0]["id"]
        id_ehh = az_with_eeh_ehh.result_entries[1]["id"]
        out = az_with_eeh_ehh.combine_auger_types(id_eeh, id_ehh, save_to=str(tmp_path))
        df = pd.read_csv(out)
        # Each should be 1e-30 + 1e-30 = 2e-30
        for c in df["Auger coefficient"]:
            assert c == pytest.approx(2e-30, rel=0.01)


# ======================================================================
# Convergence
# ======================================================================
class TestConvergence:

    def test_insufficient_kgrids(self, band_info_file, sample_auger_coefficients_csv):
        az = AugerAnalyzer()
        az.add_result_entry("eeh", band_info_file, sample_auger_coefficients_csv)
        ok, mean_c, info = az.check_convergence("TestMat", "eeh")
        assert ok is False
        assert "error" in info

    def test_no_entries_raises(self):
        az = AugerAnalyzer()
        with pytest.raises(ValueError, match="No entries"):
            az.check_convergence("NoMat", "eeh")


# ======================================================================
# kgrid_label helper
# ======================================================================
class TestKgridLabel:

    def test_cubic(self):
        label = _kgrid_label([10, 10, 10])
        assert "10" in label

    def test_non_cubic(self):
        label = _kgrid_label([10, 10, 7])
        assert "10" in label
        assert "7" in label
