"""
AugerAnalyzer — post-processing, plotting, and convergence analysis.

Manages a collection of Auger calculation results (from different k-grids,
materials, or auger types) and provides:

  * ``plot_Auger_vs_FWHM``                — coefficient vs. broadening width
  * ``plot_Auger_vs_kgrid``               — convergence with k-grid density
  * ``plot_Auger_vs_kgrid_multiple_materials`` — cross-material comparison
  * ``dE_histogram``                       — energy conservation residual
  * ``matrix_element_histogram``          — |M|² distribution
  * ``check_convergence``                 — automated convergence test
  * ``analyze_convergence``               — combined plots + convergence test
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import utilities as ut

# Lazy import matplotlib so the package can be loaded headlessly
_plt = None


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


# Default marker cycle
_MARKERS = ["^", "o", "s", "D", "v", "p", "h", "x", "+", "1", "2", "3",
            "4", "8", "H", "d", "|", "_", "*", "P", "X"]


# ======================================================================
class AugerAnalyzer:
    """
    Registry and visualisation helper for Auger coefficient results.

    Usage
    -----
    >>> az = AugerAnalyzer()
    >>> az.add_result_entry("eeh", "10x10x10/band_info.txt",
    ...                     "10x10x10/Auger_coefficients_eeh_1000.csv",
    ...                     "10x10x10/auger_eeh_pairs_1000_completed.csv")
    """

    def __init__(self):
        self.result_entries: List[dict] = []
        self._id_counter = 0

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------
    def add_result_entry(
        self,
        auger_type: str,
        band_info_file: str,
        auger_coefficients_file: str,
        auger_completed_table_file: str = "",
    ) -> None:
        """Register a calculation result for later analysis / plotting."""
        if auger_type not in ("eeh", "ehh", "combined"):
            raise ValueError("auger_type must be 'eeh', 'ehh', or 'combined'")
        self._id_counter += 1
        entry = {
            "id": self._id_counter,
            "auger_type": auger_type,
            "band_info_file": band_info_file,
            "auger_coefficients_file": auger_coefficients_file,
            "auger_completed_table_file": auger_completed_table_file,
        }
        bi = ut.read_band_info(band_info_file)
        print(f"  Entry {entry['id']}: ({auger_type}) {bi['material_name']} k={bi['kgrid']}")
        self.result_entries.append(entry)
        self._sort()

    def delete_result_entry(self, entry_id: int):
        self.result_entries = [e for e in self.result_entries if e["id"] != entry_id]
        self._sort()

    def _sort(self):
        def _key(e):
            bi = ut.read_band_info(e["band_info_file"])
            return (0 if e["auger_type"] == "eeh" else 1,
                    bi["material_name"],
                    np.prod(bi["kgrid"]))
        self.result_entries.sort(key=_key)

    def print_result_summary(self):
        if not self.result_entries:
            print("No entries.")
            return
        for e in self.result_entries:
            bi = ut.read_band_info(e["band_info_file"])
            print(f"  ID {e['id']:3d}  {e['auger_type']}  {bi['material_name']:<12s}  k={bi['kgrid']}")

    def print_entry_details(self, entry_id: int) -> dict:
        for e in self.result_entries:
            if e["id"] == entry_id:
                bi = ut.read_band_info(e["band_info_file"])
                for k, v in {**e, "material": bi["material_name"], "kgrid": bi["kgrid"]}.items():
                    print(f"  {k}: {v}")
                return e
        print(f"  Entry {entry_id} not found.")
        return {}

    def get_ids_from_material(self, material_name: str) -> List[int]:
        ids = [e["id"] for e in self.result_entries
               if ut.read_band_info(e["band_info_file"])["material_name"] == material_name]
        print(f"  '{material_name}' → IDs {ids}")
        return ids

    def _get_entry(self, eid: int) -> dict:
        for e in self.result_entries:
            if e["id"] == eid:
                return e
        raise ValueError(f"Entry {eid} not found")

    # ------------------------------------------------------------------
    # Combine C_eeh + C_ehh → C_total
    # ------------------------------------------------------------------
    def combine_auger_types(self, id_eeh: int, id_ehh: int,
                            save_to: str = "") -> str:
        e_eeh = self._get_entry(id_eeh)
        e_ehh = self._get_entry(id_ehh)
        bi_eeh = ut.read_band_info(e_eeh["band_info_file"])
        bi_ehh = ut.read_band_info(e_ehh["band_info_file"])
        if e_eeh["band_info_file"] != e_ehh["band_info_file"]:
            raise ValueError("band_info files must match to combine.")

        df_eeh = pd.read_csv(e_eeh["auger_coefficients_file"])
        df_ehh = pd.read_csv(e_ehh["auger_coefficients_file"])
        merged = pd.merge(df_eeh, df_ehh, on=["FWHM", "Delta function"],
                          suffixes=("_eeh", "_ehh"))
        merged["Auger coefficient"] = (merged["Auger coefficient_eeh"]
                                       + merged["Auger coefficient_ehh"])
        merged = merged[["FWHM", "Delta function", "Auger coefficient"]]

        XX = bi_eeh["XX"]
        if not save_to:
            save_to = f"Auger_coefficients_combined_{XX}.csv"
        elif not save_to.endswith(".csv"):
            save_to = save_to.rstrip("/") + f"/Auger_coefficients_combined_{XX}.csv"
        merged.to_csv(save_to, index=False)
        print(f"  Combined → {save_to}")
        return save_to

    # ------------------------------------------------------------------
    # Recalculate Auger coefficient with a new delta / FWHM
    # ------------------------------------------------------------------
    def calculate_Auger_with_new_FWHM(
        self, entry_id: int, new_delta: str, new_FWHM: float, norm: float = 1.0,
    ) -> float:
        """
        Recompute Auger coefficient from the completed-pairs table using an
        arbitrary delta function / FWHM combination.
        """
        e = self._get_entry(entry_id)
        pairs = ut.read_csv([e["auger_completed_table_file"]])
        bi = ut.read_band_info(e["band_info_file"])
        auger_type = pairs[0]["pair_type"]

        from .constants import HBAR as h_bar, eV as _eV
        V_m3 = bi["volume"] * 1e-30
        auger_factor = (4 * np.pi / h_bar) * (1 / V_m3 ** 3) * (1 / _eV) * 1e12

        if auger_type == "eeh":
            carrier = (bi["n"] ** 2 * bi["p"] - bi["ni"] ** 2 * bi["n"]) * (1e6) ** 3
        else:
            carrier = (bi["p"] ** 2 * bi["n"] - bi["ni"] ** 2 * bi["p"]) * (1e6) ** 3
        auger_factor *= (1.0 / carrier) if carrier != 0 else 0.0

        delta_fn = ut.DELTA_FUNCTIONS[new_delta]
        total = 0.0
        for p in pairs:
            M2 = p.get("|M|^2", -1)
            if M2 < 0:
                continue
            M2_J = M2 * _eV ** 2
            dE = ((p["E2"] - p["E1"]) - (p["E3"] - p["E4"])
                  if auger_type == "eeh"
                  else (p["E1"] - p["E2"]) - (p["E3"] - p["E4"]))
            dval = delta_fn(dE, new_FWHM)
            kw_prod = p["kw1"] * p["kw2"] * p["kw3"] * p["kw4"]
            kw_prod /= (p["kw2"] if auger_type == "eeh" else p["kw4"])
            total += (1.0 / norm) * p["probability"] * M2_J * V_m3 ** 2 * dval * auger_factor * kw_prod

        return total

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_Auger_vs_FWHM(
        self, entry_id: int, *,
        ylim=(None, None), legend_loc: str = "lower right",
        save_plot_to: str = "",
    ):
        """Auger coefficient vs FWHM for a single k-grid entry."""
        plt = _get_plt()
        e = self._get_entry(entry_id)
        bi = ut.read_band_info(e["band_info_file"])
        df = pd.read_csv(e["auger_coefficients_file"]).to_dict("records")
        deltas = sorted(set(d["Delta function"] for d in df))
        fwhms = sorted(set(d["FWHM"] for d in df))

        fig, ax = plt.subplots()
        for i, delta in enumerate(deltas):
            y = [d["Auger coefficient"] for d in df
                 if d["Delta function"] == delta and d["FWHM"] in fwhms]
            ax.semilogy([str(f) for f in fwhms], y,
                        marker=_MARKERS[i % len(_MARKERS)],
                        label=delta, mfc="w", markersize=12, linewidth=3.5)

        _style_ax(ax, xlabel="FWHM (eV)",
                  ylabel=r"Auger coefficient (cm$^{6}$s$^{-1}$)",
                  title=f"{bi['material_name']} – k={bi['kgrid'][0]}×{bi['kgrid'][1]}×{bi['kgrid'][2]}",
                  ylim=ylim, legend_loc=legend_loc)
        plt.tight_layout()
        if save_plot_to:
            fig.savefig(f"{save_plot_to}.png", dpi=300)

    def plot_expand_Auger_vs_FWHM(
        self, entry_id: int, new_FWHM: Sequence[float], *,
        ylim=(None, None), legend_loc: str = "lower right",
        save_plot_to: str = "",
    ):
        """Re-evaluate and plot Auger vs an expanded set of FWHM values."""
        plt = _get_plt()
        e = self._get_entry(entry_id)
        bi = ut.read_band_info(e["band_info_file"])
        df_orig = pd.read_csv(e["auger_coefficients_file"]).to_dict("records")
        df_pairs = pd.read_csv(e["auger_completed_table_file"]).to_dict("records")
        auger_type = df_pairs[0]["pair_type"]

        deltas = sorted(set(d["Delta function"] for d in df_orig))
        orig_fwhms = sorted(set(d["FWHM"] for d in df_orig))
        new_FWHM = sorted(new_FWHM)
        delta_fns = {n: ut.DELTA_FUNCTIONS[n] for n in deltas}

        # For each pair, compute contributions at new FWHMs
        for entry in df_pairs:
            dE = abs(
                (entry["E2"] - entry["E1"]) - (entry["E3"] - entry["E4"])
                if auger_type == "eeh"
                else (entry["E1"] - entry["E2"]) - (entry["E3"] - entry["E4"])
            )
            for d in deltas:
                for fwhm in new_FWHM:
                    if fwhm in orig_fwhms:
                        continue
                    # Try ratio scaling from an existing non-zero reference
                    ref_val = 0.0
                    ref_d, ref_f = d, orig_fwhms[0]
                    for d2 in deltas:
                        for f2 in orig_fwhms:
                            v = entry.get(f"{d2}-{f2}", 0)
                            if v != 0:
                                ref_val, ref_d, ref_f = v, d2, f2
                                break
                        if ref_val != 0:
                            break
                    if ref_val != 0 and delta_fns[ref_d](dE, ref_f) != 0:
                        entry[f"{d}-{fwhm}"] = (
                            ref_val / delta_fns[ref_d](dE, ref_f) * delta_fns[d](dE, fwhm)
                        )
                    else:
                        entry[f"{d}-{fwhm}"] = self.calculate_Auger_with_new_FWHM(
                            entry_id, d, fwhm
                        )

        rows = []
        for d in deltas:
            for f in new_FWHM:
                rows.append({
                    "Delta function": d, "FWHM": f,
                    "Auger coefficient": sum(e.get(f"{d}-{f}", 0) for e in df_pairs),
                })

        fig, ax = plt.subplots()
        fwhm_strs = [str(f) for f in new_FWHM]
        for i, d in enumerate(deltas):
            y = [r["Auger coefficient"] for r in rows if r["Delta function"] == d]
            ax.semilogy(fwhm_strs, y, marker=_MARKERS[i % len(_MARKERS)],
                        label=d, mfc="w", markersize=12, linewidth=3.5)
        _style_ax(ax, xlabel="FWHM (eV)",
                  ylabel=r"Auger coefficient (cm$^{6}$s$^{-1}$)",
                  title=f"{bi['material_name']} – k={bi['kgrid'][0]}×{bi['kgrid'][1]}×{bi['kgrid'][2]}",
                  ylim=ylim, legend_loc=legend_loc)
        ax.set_xticklabels(fwhm_strs, rotation=-50)
        plt.grid()
        plt.tight_layout()
        if save_plot_to:
            fig.savefig(f"{save_plot_to}.png", dpi=300)

    def plot_Auger_vs_kgrid(
        self, material_name: str, auger_type: str,
        chosen_fwhm_delta: list = None, *,
        ylim=(None, None), legend_loc: str = "lower right",
        save_plot_to: str = "",
    ):
        """
        Plot Auger coefficient vs k-grid for a given material.

        Parameters
        ----------
        chosen_fwhm_delta : list
            Either ``[('Gaussian', 0.05), ...]`` (same for all k-grids)
            or ``[[{id, delta, FWHM}, ...], ...]`` (per k-grid).
        """
        if chosen_fwhm_delta is None:
            chosen_fwhm_delta = [("Gaussian", 0.05)]
        plt = _get_plt()
        results = [e for e in self.result_entries
                   if e["auger_type"] == auger_type
                   and ut.read_band_info(e["band_info_file"])["material_name"] == material_name]
        if not results:
            raise ValueError(f"No entries for {material_name} / {auger_type}")

        results.sort(key=lambda e: np.prod(ut.read_band_info(e["band_info_file"])["kgrid"]))
        fig, ax = plt.subplots()

        if isinstance(chosen_fwhm_delta[0], tuple):
            for idx, (delta, fwhm) in enumerate(chosen_fwhm_delta):
                xs, ys = [], []
                for entry in results:
                    bi = ut.read_band_info(entry["band_info_file"])
                    kg = bi["kgrid"]
                    xs.append(_kgrid_label(kg))
                    df = pd.read_csv(entry["auger_coefficients_file"]).to_dict("records")
                    match = [d["Auger coefficient"] for d in df
                             if d["Delta function"] == delta and str(d["FWHM"]) == str(fwhm)]
                    if match:
                        ys.append(match[0])
                    else:
                        ys.append(self.calculate_Auger_with_new_FWHM(entry["id"], delta, float(fwhm)))
                ax.semilogy(xs, ys, marker=_MARKERS[idx % len(_MARKERS)],
                            label=f"{delta}-{fwhm} eV", mfc="w", markersize=12, linewidth=3.5)
        else:
            # List-of-dicts format (per k-grid)
            for idx, delta_list in enumerate(chosen_fwhm_delta):
                xs, ys = [], []
                for dd in delta_list:
                    eid, delta, fwhm = dd["id"], dd["delta"], dd["FWHM"]
                    entry = self._get_entry(eid)
                    bi = ut.read_band_info(entry["band_info_file"])
                    xs.append(_kgrid_label(bi["kgrid"]))
                    df = pd.read_csv(entry["auger_coefficients_file"]).to_dict("records")
                    match = [d["Auger coefficient"] for d in df
                             if d["Delta function"] == delta and str(d["FWHM"]) == str(fwhm)]
                    if match:
                        ys.append(match[0])
                    else:
                        ys.append(self.calculate_Auger_with_new_FWHM(eid, delta, float(fwhm)))
                ax.semilogy(xs, ys, marker=_MARKERS[idx % len(_MARKERS)],
                            label=f"{delta}-{fwhm} eV", mfc="w", markersize=12, linewidth=3.5)

        _style_ax(ax, xlabel="k-grid",
                  ylabel=r"Auger coefficient (cm$^{6}$s$^{-1}$)",
                  title=f"Auger coefficients – {material_name}",
                  ylim=ylim, legend_loc=legend_loc)
        plt.tight_layout()
        if save_plot_to:
            fig.savefig(f"{save_plot_to}.png", dpi=300)

    def plot_Auger_vs_kgrid_multiple_materials(
        self,
        entry_ids: List[int] = None,
        auger_type: str = "eeh",
        delta_FWHM: Dict[str, Tuple[str, float]] = None,
        delta_used: str = "Gaussian",
        FWHM_used: float = 0.05,
        *,
        ylim=(None, None),
        legend_loc: str = "lower right",
        save_plot_to: str = "",
    ):
        """Compare Auger coefficients across materials on one plot."""
        plt = _get_plt()
        if entry_ids is None:
            entry_ids = [e["id"] for e in self.result_entries]
        entries = [e for e in self.result_entries if e["id"] in entry_ids]
        if delta_FWHM is None:
            delta_FWHM = {}

        # Group by material
        materials: Dict[str, list] = {}
        for e in entries:
            bi = ut.read_band_info(e["band_info_file"])
            materials.setdefault(bi["material_name"], []).append({
                **e, "kgrid": bi["kgrid"], "band_info": bi,
            })
        for lst in materials.values():
            lst.sort(key=lambda x: np.prod(x["kgrid"]))

        fig, ax = plt.subplots()
        for idx, (mat, elist) in enumerate(materials.items()):
            d, f = delta_FWHM.get(mat, (delta_used, FWHM_used))
            xs, ys = [], []
            for i, item in enumerate(elist, 1):
                xs.append(f"k_{i}")
                df = pd.read_csv(item["auger_coefficients_file"]).to_dict("records")
                match = [r["Auger coefficient"] for r in df
                         if r["Delta function"] == d and str(r["FWHM"]) == str(f)]
                if match:
                    ys.append(match[0])
                else:
                    ys.append(self.calculate_Auger_with_new_FWHM(item["id"], d, float(f)))
            Eg = round(elist[-1]["band_info"]["band_gap"] + elist[-1]["band_info"]["scissor_shift"], 3)
            ax.semilogy(xs, ys, marker=_MARKERS[idx % len(_MARKERS)],
                        label=f"{mat} – Eg={Eg} eV", mfc="w", markersize=12, linewidth=3.5)

        title = "Auger coefficients" if delta_FWHM else f"Auger / {delta_used}, FWHM={FWHM_used} eV"
        _style_ax(ax, xlabel="k-grid",
                  ylabel=r"Auger coefficient (cm$^{6}$s$^{-1}$)",
                  title=title, ylim=ylim, legend_loc=legend_loc)
        plt.tight_layout()
        if save_plot_to:
            fig.savefig(f"{save_plot_to}.png", dpi=300)

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------
    def dE_histogram(
        self, entry_id: int = -1, material: str = "", auger_type: str = "", *,
        bins: int = 0, xlim=(None, None), ylim=(None, None),
        save_plot_to: str = "",
    ) -> np.ndarray:
        """Plot energy-conservation residual distribution."""
        plt = _get_plt()
        tables, pair_type, title_extra = self._resolve_tables(entry_id, material, auger_type)
        all_pairs = [p for t in tables for p in t]
        print(f"  {len(all_pairs):,} pairs")

        dE = np.array([
            (p["E2"] - p["E1"]) - (p["E3"] - p["E4"])
            if pair_type == "eeh" else
            (p["E1"] - p["E2"]) - (p["E3"] - p["E4"])
            for p in all_pairs
        ])

        if bins == 0:
            bins = int(np.sqrt(len(all_pairs)))
        mu, sigma = dE.mean(), dE.std()
        fwhm_suggest = 2 * np.sqrt(2 * np.log(2)) * sigma
        print(f"  mu={mu:.4f}  sigma={sigma:.4f}  suggested FWHM={fwhm_suggest:.4f} eV")

        fig, ax = plt.subplots(figsize=(8, 6))
        counts, edges, _ = ax.hist(dE, bins=bins, color="steelblue", alpha=0.7, label="Data")
        centres = (edges[:-1] + edges[1:]) / 2
        norm = len(dE) * (edges[1] - edges[0])
        gauss = norm / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((centres - mu) / sigma) ** 2)
        ax.plot(centres, gauss, "r--", lw=2, label=rf"Gaussian ($\sigma$={sigma:.3f} eV)")
        ax.set_xlabel(r"$\Delta E$ (eV)")
        ax.set_ylabel("Counts")
        ax.set_title(rf"$\Delta E$ histogram – {pair_type.upper()} {title_extra}")
        if xlim != (None, None): ax.set_xlim(xlim)
        if ylim != (None, None): ax.set_ylim(ylim)
        ax.legend(); ax.grid()
        if save_plot_to:
            fig.savefig(save_plot_to, dpi=300)
        plt.show()
        return dE

    def matrix_element_histogram(
        self, entry_id: int = -1, material: str = "", auger_type: str = "", *,
        plot_weighted: bool = False, bins: int = 0,
        xlim=(None, None), ylim=(None, None),
        save_plot_to: str = "",
    ) -> dict:
        """Plot log10(|M|) distribution, optionally weighted by probability."""
        plt = _get_plt()
        from matplotlib import cm

        tables, pair_type, title_extra = self._resolve_tables(entry_id, material, auger_type)
        print(f"  {sum(len(t) for t in tables):,} total pairs")

        log_M, weighted_log_M = [], []
        avg_M, wavg_M, counts_per = [], [], []
        final_dicts: List[List[dict]] = []

        for tbl in tables:
            prob_sum, M_sum, wM_sum, cnt = 0.0, 0.0, 0.0, 0
            wts = []
            final_dicts.append([])
            for p in tbl:
                M2 = p.get("|M|^2", -1)
                if M2 < 0:
                    continue
                M_abs = np.sqrt(M2)
                log_M.append(np.log10(M_abs))
                prob_sum += p["probability"]
                wM_sum += M_abs * p["probability"]
                wts.append(M_abs * p["probability"])
                M_sum += M_abs
                cnt += 1
                final_dicts[-1].append(p)
            if prob_sum > 0:
                weighted_log_M.extend([np.log10(w / prob_sum) for w in wts])
                wavg_M.append(wM_sum / prob_sum)
            avg_M.append(M_sum / cnt if cnt else 0)
            counts_per.append(cnt)

        if bins == 0:
            bins = max(10, int(np.sqrt(len(log_M))))

        mu, sigma = np.mean(log_M), np.std(log_M)
        fig, ax = plt.subplots(figsize=(8, 6))
        counts_arr, edges = np.histogram(log_M, bins=bins)

        # Probability-coloured bars
        all_probs = []
        for tbl in tables:
            total_P = sum(p["probability"] for p in tbl if p.get("|M|^2", -1) >= 0)
            for p in tbl:
                if p.get("|M|^2", -1) >= 0:
                    all_probs.append(p["probability"] / total_P if total_P else 0)
        bin_idx = np.clip(np.digitize(log_M, edges) - 1, 0, len(edges) - 2)
        max_p = np.zeros(len(edges) - 1)
        for i, bi in enumerate(bin_idx):
            max_p[bi] = max(max_p[bi], all_probs[i])
        if max_p.max() > 0:
            norm_p = (max_p - max_p.min()) / (max_p.max() - max_p.min())
        else:
            norm_p = np.zeros_like(max_p)
        colors = cm.RdYlGn(norm_p)
        for i in range(len(edges) - 1):
            ax.bar(edges[i], counts_arr[i], width=edges[i + 1] - edges[i],
                   align="edge", color=colors[i], alpha=0.7, edgecolor="black", lw=0.5)
        sm = cm.ScalarMappable(cmap=cm.RdYlGn,
                               norm=plt.Normalize(max_p.min(), max_p.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Max probability per bin")

        centres = (edges[:-1] + edges[1:]) / 2
        gn = len(log_M) * (edges[1] - edges[0])
        gauss = gn / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((centres - mu) / sigma) ** 2)
        ax.plot(centres, gauss, "r--", lw=2, label=rf"Gaussian ($\sigma$={sigma:.3f})")
        ax.set_xlabel(r"log$_{10}$(|M|) (eV)")
        ax.set_ylabel("Counts")
        ax.set_title(rf"log$_{{10}}$(|M|) histogram – {pair_type.upper()} {title_extra}")
        if xlim != (None, None): ax.set_xlim(xlim)
        if ylim != (None, None): ax.set_ylim(ylim)
        ax.legend(); ax.grid()
        if save_plot_to:
            fig.savefig(save_plot_to, dpi=300)
        plt.show()

        if plot_weighted and weighted_log_M:
            mu2, sigma2 = np.mean(weighted_log_M), np.std(weighted_log_M)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.hist(weighted_log_M, bins=bins, color="green", alpha=0.7)
            ax2.set_xlabel(r"Weighted log$_{10}$(|M|)")
            ax2.set_ylabel("Counts")
            ax2.set_title(f"Weighted |M| histogram – {pair_type.upper()} {title_extra}")
            ax2.legend(); ax2.grid()
            if save_plot_to:
                sp = save_plot_to.replace(".png", "_weighted.png") if save_plot_to.endswith(".png") else save_plot_to + "_weighted.png"
                fig2.savefig(sp, dpi=300)
            plt.show()

        return {
            "weighted_average_matrix_element": wavg_M,
            "average_matrix_element": avg_M,
            "num_matrix_elements": len(log_M),
            "matrix_element_dicts": final_dicts,
        }

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------
    def check_convergence(
        self, material_name: str, auger_type: str, *,
        reference_fwhm: float = 0.05,
        delta_function: str = "Gaussian",
        tol_kgrid: float = 0.05,
        tol_fwhm: float = 0.10,
    ) -> Tuple[bool, Optional[float], dict]:
        """
        Automated convergence check w.r.t. k-grid and FWHM plateau.

        Returns ``(is_converged, mean_coefficient, info_dict)``.
        """
        results = self._entries_for(material_name, auger_type)
        results.sort(key=lambda e: np.prod(ut.read_band_info(e["band_info_file"])["kgrid"]))

        print(f"\n{'='*70}")
        print(f"CONVERGENCE – {material_name} ({auger_type.upper()})")
        print(f"{'='*70}")

        # K-grid convergence
        kgrid_data = []
        for e in results:
            bi = ut.read_band_info(e["band_info_file"])
            df = pd.read_csv(e["auger_coefficients_file"])
            row = df[(df["Delta function"] == delta_function) & (df["FWHM"] == reference_fwhm)]
            if not row.empty:
                kgrid_data.append((np.prod(bi["kgrid"]), row["Auger coefficient"].values[0], bi["kgrid"]))

        if len(kgrid_data) < 2:
            print("  Need >= 2 k-grids for convergence.")
            return False, None, {"error": "insufficient k-grids"}

        rel = abs(kgrid_data[-1][1] - kgrid_data[-2][1]) / kgrid_data[-2][1]
        kg_ok = rel < tol_kgrid
        print(f"\n  K-grid ({delta_function}, FWHM={reference_fwhm}):")
        for i, (ks, c, kg) in enumerate(kgrid_data):
            change = "" if i == 0 else f"  {abs(c - kgrid_data[i-1][1]) / kgrid_data[i-1][1] * 100:.1f}%"
            print(f"    {kg[0]}x{kg[1]}x{kg[2]} → {c:.4e}{change}")
        status = "CONVERGED" if kg_ok else "NOT CONVERGED"
        print(f"  {status}  (last Δ={rel*100:.1f}%, tol={tol_kgrid*100:.0f}%)")

        # FWHM plateau (best k-grid)
        best = results[-1]
        df_best = pd.read_csv(best["auger_coefficients_file"])
        plat = df_best[(df_best["Delta function"] == delta_function) &
                       (df_best["FWHM"] >= 0.03) & (df_best["FWHM"] <= 0.1)]
        fwhm_ok = False
        mean_c = None
        cv = None
        if len(plat) > 0:
            mean_c = plat["Auger coefficient"].mean()
            std_c = plat["Auger coefficient"].std()
            cv = std_c / mean_c * 100 if mean_c else 0
            fwhm_ok = cv < tol_fwhm * 100
            print(f"\n  FWHM plateau (0.03–0.1 eV): mean={mean_c:.4e}, CV={cv:.1f}%")
            print(f"  {'STABLE' if fwhm_ok else 'UNSTABLE'}")

        converged = kg_ok and fwhm_ok
        print(f"\n  {'CONVERGED' if converged else 'NOT CONVERGED'}")
        if converged and mean_c is not None:
            print(f"  Recommended: {mean_c:.4e} cm⁶/s")
        print(f"{'='*70}\n")

        return converged, mean_c, {
            "kgrid_converged": kg_ok, "kgrid_change_pct": rel * 100,
            "fwhm_converged": fwhm_ok, "fwhm_cv_pct": cv,
            "mean_coefficient": mean_c, "kgrid_data": kgrid_data,
        }

    def analyze_convergence(
        self, material_name: str, auger_type: str, *,
        reference_fwhm: float = 0.05, save_plot_to: str = "",
    ) -> dict:
        """Combined convergence plot + numerical check."""
        plt = _get_plt()
        results = self._entries_for(material_name, auger_type)
        results.sort(key=lambda e: np.prod(ut.read_band_info(e["band_info_file"])["kgrid"]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        deltas = ["Gaussian", "Lorentzian"]

        # Panel 1: k-grid convergence
        for di, delta in enumerate(deltas):
            xs, ys, labels = [], [], []
            for e in results:
                bi = ut.read_band_info(e["band_info_file"])
                df = pd.read_csv(e["auger_coefficients_file"])
                row = df[(df["Delta function"] == delta) & (df["FWHM"] == reference_fwhm)]
                if not row.empty:
                    xs.append(len(ys))
                    ys.append(row["Auger coefficient"].values[0])
                    labels.append(_kgrid_label(bi["kgrid"]))
            if ys:
                ax1.plot(xs, ys, marker=_MARKERS[di], label=delta, ms=10, lw=2, mfc="w")
        ax1.set_yscale("log")
        ax1.set_title(f"K-grid convergence – {material_name} ({auger_type.upper()})")
        ax1.set_xlabel("K-grid"); ax1.set_ylabel(r"C (cm$^6$/s)")
        if labels:
            ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.legend(); ax1.grid(alpha=0.3)

        # Panel 2: FWHM dependence (best k-grid)
        best = results[-1]
        df_best = pd.read_csv(best["auger_coefficients_file"])
        bi_best = ut.read_band_info(best["band_info_file"])
        for di, delta in enumerate(deltas):
            sub = df_best[df_best["Delta function"] == delta].sort_values("FWHM")
            if len(sub):
                ax2.plot(sub["FWHM"], sub["Auger coefficient"],
                         marker=_MARKERS[di], label=delta, ms=10, lw=2, mfc="w")
        ax2.axvspan(0.03, 0.1, alpha=0.15, color="green", label="Physical range")
        ax2.set_yscale("log")
        ax2.set_title(f"FWHM – {_kgrid_label(bi_best['kgrid'])}")
        ax2.set_xlabel("FWHM (eV)"); ax2.set_ylabel(r"C (cm$^6$/s)")
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save_plot_to:
            sp = save_plot_to if save_plot_to.endswith(".png") else save_plot_to + ".png"
            fig.savefig(sp, dpi=300, bbox_inches="tight")
            print(f"  Plot → {sp}")
        plt.show()

        ok, mc, info = self.check_convergence(material_name, auger_type, reference_fwhm=reference_fwhm)
        return {"is_converged": ok, "mean_coefficient": mc, "convergence_info": info,
                "material": material_name, "auger_type": auger_type, "num_kgrids": len(results)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _entries_for(self, material_name: str, auger_type: str) -> List[dict]:
        out = [e for e in self.result_entries
               if e["auger_type"] == auger_type
               and ut.read_band_info(e["band_info_file"])["material_name"] == material_name]
        if not out:
            raise ValueError(f"No entries for {material_name} / {auger_type}")
        return out

    def _resolve_tables(self, entry_id, material, auger_type):
        """Return ``(list_of_pair_tables, pair_type, title_suffix)``."""
        if entry_id != -1:
            e = self._get_entry(entry_id)
            bi = ut.read_band_info(e["band_info_file"])
            tbl = pd.read_csv(e["auger_completed_table_file"]).to_dict("records")
            kg = bi["kgrid"]
            return [tbl], e["auger_type"], f"{bi['material_name']} k={kg[0]}x{kg[1]}x{kg[2]}"
        else:
            if not material or not auger_type:
                raise ValueError("Provide entry_id or both material & auger_type")
            results = self._entries_for(material, auger_type)
            tables = []
            for e in results:
                if not e["auger_completed_table_file"]:
                    raise ValueError(f"Missing completed-table for entry {e['id']}")
                tables.append(pd.read_csv(e["auger_completed_table_file"]).to_dict("records"))
            return tables, auger_type, f"{material} (multi k-grid)"


# ======================================================================
# Plotting helpers
# ======================================================================
def _kgrid_label(kg) -> str:
    if kg[0] == kg[1] == kg[2]:
        return rf"${kg[0]}^{{3}}$"
    return rf"${kg[0]}\times{kg[1]}\times{kg[2]}$"


def _style_ax(ax, *, xlabel, ylabel, title, ylim=(None, None), legend_loc="lower right"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim != (None, None):
        ax.set_ylim(ylim)
    ax.minorticks_on()
    ax.tick_params(direction="in", which="major", length=10,
                   bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.legend(loc=legend_loc)
