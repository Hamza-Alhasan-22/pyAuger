from pathlib import Path
from auger import AugerCalculator, utilities


# Step 1 prepares the exact-kpoint NSCF folders.
# After this script finishes, run VASP inside each generated NSCF folder.


# Inputs:
# ── Paths ──────────────────────────────────────────────────────────────────
VASP_FOLDER       = "../test-files/InAs/exact-kpoint-scf-3"
RESULTS_DIR       = "../test-files/InAs/results/exact-kpoint"
NSCF_DIR          = RESULTS_DIR + "/eeh_NSCF"

# ── Physical parameters ────────────────────────────────────────────────────
AUGER_TYPE        = "eeh"          # "eeh" or "ehh"
TEMPERATURE       = 300            # K
DOPING            = 0              # cm^-3
EXCESS_CARRIER    = 1e17           # cm^-3

# ── Material parameters ────────────────────────────────────────────────────
DIELECTRIC        = 12.3           # unitless
FIRST_CB_INDEX    = 9              # 0-based index
LAST_VB_INDEX     = 8              # 0-based index
FORCE_GAP         = 0.4            # eV; set to None to use the parsed gap

NUM_PAIRS_TO_KEEP = "all"          # "all" or an integer
NKPOINTS_PER_NSCF = 20_000         # "all" or an integer

if __name__ == "__main__":

    calc = AugerCalculator(T=TEMPERATURE, nd=DOPING)
    
    calc.assign_firstCB_and_lastVB(
        firstCB_index=FIRST_CB_INDEX,
        lastVB_index=LAST_VB_INDEX,
    )

    calc.parse_BS_data(
        folder_path=VASP_FOLDER,
        write_path=RESULTS_DIR,
        force_gap=FORCE_GAP,
    )
    calc.import_parsed_BS_data(from_folder=RESULTS_DIR)

    calc.calculate_carrier_concentrations(delta_n=EXCESS_CARRIER)
    
    cb_window, vb_window = calc.calculate_energy_cutoffs(charge_threshold=0.99)

    calc.create_exact_kpoint_list(
        CB_window=cb_window,
        VB_window=vb_window,
        auger_type=AUGER_TYPE,
        poscar_path=VASP_FOLDER + "/POSCAR",
        num_kpoints=NUM_PAIRS_TO_KEEP
    )

    # Find the generated exact-kpoint CSV files to create NSCF inputs:
    exact_csvs = sorted(Path(RESULTS_DIR).glob(f"exact_kpoints_{AUGER_TYPE}_{calc.XX}_*.csv"))
    if not exact_csvs:
        exact_csvs = sorted(Path(RESULTS_DIR).glob(f"exact_kpoints_{AUGER_TYPE}_{calc.XX}.csv"))
    if not exact_csvs:
        raise FileNotFoundError("No exact-kpoint CSV files were created.")

    utilities.create_nscf_inputs(
        scf_folder=VASP_FOLDER,
        nscf_folder=NSCF_DIR,
        exact_kpoints_table=[str(p) for p in exact_csvs],
        auger_type=AUGER_TYPE,
        num_kpoints_per_file=NKPOINTS_PER_NSCF,
        efermi=calc.E_Fermi,
    )

    print("\nNSCF folders are ready. Run VASP in:")
    for folder in sorted(Path(NSCF_DIR).glob(f"NSCF_{AUGER_TYPE}_*")):
        print(f"  {folder}")

