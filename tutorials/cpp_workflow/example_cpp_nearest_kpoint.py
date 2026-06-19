from pathlib import Path
import subprocess

from auger import AugerCalculator
from auger.cpp.prepare_pair_generation_input import prepare as prepare_pair_generation_input
from auger.cpp.prepare_cpp_input import prepare as prepare_cpp_input


# Nearest-kpoint C++ workflow:
# parse SCF data -> generate pairs with C++ -> generate matrix elements with C++ -> calculate rates.


# Inputs:
# Paths
VASP_FOLDER = "../../test-files/InAs/nearest-kpoint-scf-3"
RESULTS_DIR = "../../test-files/InAs/results/cpp_nearest_kpoint"
CPP_DIR = "../../auger/cpp"

# Physical parameters
AUGER_TYPE = "eeh"
TEMPERATURE = 300
DOPING = 0
EXCESS_CARRIER = 1e17

# Material/calculation parameters
DIELECTRIC = 12.3
FIRST_CB_INDEX = 9
LAST_VB_INDEX = 8
FORCE_GAP = 0.4
NUM_PAIRS_TO_KEEP = "all"
NUM_MATRIX_ELEMENTS = "all"
THREADS = 8
USE_CPP_SRUN = False
CPP_SRUN = {
    "srun_command": "srun",
    "srun_num_nodes": 1,
    "srun_cpu_bind": "cores",
    "srun_extra_args": [],
}


def cpp_matrix_command(matrix_exe, matrix_input, matrix_jsonl):
    command = [
        str(matrix_exe),
        str(matrix_input),
        str(matrix_jsonl),
        str(THREADS),
        "--overwrite",
        "--progress_interval",
        "10000",
    ]
    if not USE_CPP_SRUN:
        return command

    prefix = [
        str(CPP_SRUN.get("srun_command", "srun")),
        "-n",
        str(CPP_SRUN.get("srun_num_nodes", 1)),
        "-c",
        str(THREADS),
    ]
    cpu_bind = CPP_SRUN.get("srun_cpu_bind")
    if cpu_bind:
        prefix.append(f"--cpu-bind={cpu_bind}")
    prefix.extend(str(arg) for arg in CPP_SRUN.get("srun_extra_args", []))
    return prefix + command


if __name__ == "__main__":

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    cpp_work_dir = Path(RESULTS_DIR) / "cpp_work"
    cpp_work_dir.mkdir(parents=True, exist_ok=True)

    pair_exe = CPP_DIR + "/pair_generation_calc"
    matrix_exe = CPP_DIR + "/matrix_element_calc"

    calc = AugerCalculator(T=TEMPERATURE, nd=DOPING)
    calc.assign_firstCB_and_lastVB(
        firstCB_index=FIRST_CB_INDEX,
        lastVB_index=LAST_VB_INDEX,
    )

    calc.parse_BS_data(
        folder_path=str(VASP_FOLDER),
        write_path=str(RESULTS_DIR),
        force_gap=FORCE_GAP,
    )
    calc.import_parsed_BS_data(from_folder=str(RESULTS_DIR))
    calc.calculate_carrier_concentrations(delta_n=EXCESS_CARRIER)

    cb_window, vb_window = calc.calculate_energy_cutoffs(charge_threshold=0.99)

    pair_input = cpp_work_dir / f"{AUGER_TYPE}_nearest_kpoint_pairs_input.bin"
    pair_csv_base = Path(RESULTS_DIR) / f"auger_{AUGER_TYPE}_pairs_{calc.XX}.csv"

    prepare_pair_generation_input(
        results_folder=str(RESULTS_DIR),
        output=str(pair_input),
        task="pairs",
        auger_type=AUGER_TYPE,
        approach="nearest_kpoint",
        CB_window=cb_window,
        VB_window=vb_window,
        num_to_keep=NUM_PAIRS_TO_KEEP,
        T=TEMPERATURE,
        delta_n=EXCESS_CARRIER,
        poscar_path=str(VASP_FOLDER + "/POSCAR"),
    )

    subprocess.run([str(pair_exe), str(pair_input), str(pair_csv_base)], check=True)

    pair_csvs = sorted(Path(RESULTS_DIR).glob(f"auger_{AUGER_TYPE}_pairs_{calc.XX}_*.csv"))
    if not pair_csvs:
        pair_csvs = sorted(Path(RESULTS_DIR).glob(f"auger_{AUGER_TYPE}_pairs_{calc.XX}.csv"))
    if not pair_csvs:
        raise FileNotFoundError("No pair CSV files were created.")

    matrix_input = cpp_work_dir / f"{AUGER_TYPE}_nearest_kpoint_matrix_input.bin"
    matrix_jsonl = Path(RESULTS_DIR) / f"matrix_elements_{AUGER_TYPE}_{calc.XX}_1.jsonl"

    prepare_cpp_input(
        results_folder=RESULTS_DIR,
        wavecar_files=[VASP_FOLDER + "/WAVECAR"],
        pairs_csv=[str(p) for p in pair_csvs],
        auger_type=AUGER_TYPE,
        dielectric=DIELECTRIC,
        firstCB_index=FIRST_CB_INDEX,
        lastVB_index=LAST_VB_INDEX,
        output_path=str(matrix_input),
        num_matrix_elements=NUM_MATRIX_ELEMENTS,
    )

    subprocess.run(cpp_matrix_command(matrix_exe, matrix_input, matrix_jsonl), check=True)

    matrix_jsonls = sorted(Path(RESULTS_DIR).glob(f"matrix_elements_{AUGER_TYPE}_{calc.XX}_*.jsonl"))
    calc.read_auger_pairs([str(p) for p in pair_csvs])
    calc.read_matrix_elements([str(p) for p in matrix_jsonls])
    auger_coeff = calc.calculate_auger_rates(auger_type=AUGER_TYPE)

