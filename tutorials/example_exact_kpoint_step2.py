from pathlib import Path
from auger import AugerCalculator, utilities


# Step 2 runs after VASP has completed in every NSCF folder from step 1.
# It builds exact-kpoint pairs, computes matrix elements, and calculates rates.


# Inputs:
# ── Paths ────────────────────────────────────────────────────────────────
RESULTS_DIR         = "../test-files/InAs/results/exact-kpoint"
NSCF_DIR            = RESULTS_DIR + "/eeh_NSCF"

# ── Physical parameters ─────────────────────────────────────────────────
AUGER_TYPE          = "eeh"        # "eeh" or "ehh"
TEMPERATURE         = 300          # K
DOPING              = 0            # cm^-3
EXCESS_CARRIER      = 1e17         # cm^-3

# ── Material parameters ─────────────────────────────────────────────────
DIELECTRIC          = 12.3         # unitless

NUM_PAIRS_TO_KEEP   = "all"        # "all" or an integer
NUM_MATRIX_ELEMENTS = "all"        # "all" or an integer


if __name__ == "__main__":

    calc = AugerCalculator(T=TEMPERATURE, nd=DOPING)

    calc.import_parsed_BS_data(from_folder=RESULTS_DIR)

    CB_auto, VB_auto = calc.calculate_energy_cutoffs(charge_threshold=0.99)

    # Find the completed NSCF folders and exact-kpoint CSV files from step 1:
    nscf_folders = sorted(
        Path(NSCF_DIR).glob(f"NSCF_{AUGER_TYPE}_*"),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    exact_csvs = sorted(
        Path(RESULTS_DIR).glob(f"exact_kpoints_{AUGER_TYPE}_{calc.XX}_*.csv"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not exact_csvs:
        exact_csvs = sorted(Path(RESULTS_DIR).glob(f"exact_kpoints_{AUGER_TYPE}_{calc.XX}.csv"))

    if not nscf_folders:
        raise FileNotFoundError(f"No NSCF folders found in {NSCF_DIR}")
    if not exact_csvs:
        raise FileNotFoundError("No exact-kpoint CSV files found. Run step 1 first.")

    gen_pairs = calc.create_auger_pairs(
        CB_window=CB_auto,
        VB_window=VB_auto,
        auger_type=AUGER_TYPE,
        approach="exact_kpoint",
        nscf_folders=[str(p) for p in nscf_folders],
        exact_kpoints_csv=[str(p) for p in exact_csvs],
        num_top_pairs=NUM_PAIRS_TO_KEEP,
    )

    wavecar_files = [str(folder / "WAVECAR") for folder in nscf_folders]
    me = calc.calculate_matrix_elements(
        auger_type=AUGER_TYPE,
        wavecar_files=wavecar_files,
        dielectric_constant=DIELECTRIC,
        num_matrix_elements=NUM_MATRIX_ELEMENTS,
    )

    auger_coeff = calc.calculate_auger_rates(
        auger_type=AUGER_TYPE,
    )
