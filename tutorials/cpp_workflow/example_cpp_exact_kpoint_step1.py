from pathlib import Path
import subprocess

from auger import AugerCalculator, utilities
from auger.cpp.prepare_pair_generation_input import prepare as prepare_pair_generation_input


# Exact-kpoint C++ workflow, step 1:
# parse SCF data -> generate exact-kpoint CSVs with C++ -> create NSCF input folders.
# After this script finishes, run VASP inside every generated NSCF folder.


# Inputs:
# Paths
VASP_FOLDER = "../../test-files/InAs/exact-kpoint-scf-3"
RESULTS_DIR = "../../test-files/InAs/results/cpp_exact_kpoint"
CPP_DIR = "../../auger/cpp"
NSCF_DIR = RESULTS_DIR + "/eeh_NSCF"

# Physical parameters
AUGER_TYPE = "eeh"
TEMPERATURE = 300
DOPING = 0
EXCESS_CARRIER = 1e17

# Material/calculation parameters
FIRST_CB_INDEX = 9
LAST_VB_INDEX = 8
FORCE_GAP = 0.4
NUM_PAIRS_TO_KEEP = "all"
NKPOINTS_PER_NSCF = 20000
IS_EXPANDED_FROM_IRREDUCIBLE = False


if __name__ == "__main__":

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    cpp_work_dir = Path(RESULTS_DIR) / "cpp_work"
    cpp_work_dir.mkdir(parents=True, exist_ok=True)

    pair_exe = CPP_DIR + "/pair_generation_calc"

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

    exact_input = cpp_work_dir / f"{AUGER_TYPE}_exact_kpoints_input.bin"
    exact_csv_base = RESULTS_DIR + f"/exact_kpoints_{AUGER_TYPE}_{calc.XX}.csv"

    prepare_pair_generation_input(
        results_folder=RESULTS_DIR,
        output=str(exact_input),
        task="exact_kpoints",
        auger_type=AUGER_TYPE,
        approach="exact_kpoint",
        CB_window=cb_window,
        VB_window=vb_window,
        num_to_keep=NUM_PAIRS_TO_KEEP,
        T=TEMPERATURE,
        delta_n=EXCESS_CARRIER,
        poscar_path=VASP_FOLDER + "/POSCAR",
        is_expanded_from_irreducible=IS_EXPANDED_FROM_IRREDUCIBLE,
    )

    subprocess.run([str(pair_exe), str(exact_input), str(exact_csv_base)], check=True)

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
