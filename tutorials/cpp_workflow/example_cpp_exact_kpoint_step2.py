from pathlib import Path
import subprocess

from auger import AugerCalculator
from auger.cpp.prepare_pair_generation_input import prepare as prepare_pair_generation_input
from auger.cpp.prepare_cpp_input import prepare as prepare_cpp_input


# Exact-kpoint C++ workflow, step 2:
# after NSCF VASP jobs finish -> generate final pair CSVs with C++ ->
# generate matrix elements with C++ -> calculate rates.


# Inputs:
# Paths
RESULTS_DIR = "../../test-files/InAs/results/cpp_exact_kpoint"
CPP_DIR = "../../auger/cpp"
NSCF_DIR = RESULTS_DIR + "/eeh_NSCF"

# Physical parameters
AUGER_TYPE = "eeh"
TEMPERATURE = 300
DOPING = 0
EXCESS_CARRIER = 1e17

# Material/calculation parameters
DIELECTRIC = 12.3
FIRST_CB_INDEX = 9
LAST_VB_INDEX = 8
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


def numeric_suffix(path):
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return 0


if __name__ == "__main__":

    cpp_work_dir = RESULTS_DIR + "/cpp_work"
    Path(cpp_work_dir).mkdir(parents=True, exist_ok=True)

    pair_exe = CPP_DIR + "/pair_generation_calc"
    matrix_exe = CPP_DIR + "/matrix_element_calc"

    calc = AugerCalculator(T=TEMPERATURE, nd=DOPING)
    calc.assign_firstCB_and_lastVB(
        firstCB_index=FIRST_CB_INDEX,
        lastVB_index=LAST_VB_INDEX,
    )
    calc.import_parsed_BS_data(from_folder=str(RESULTS_DIR))
    calc.calculate_carrier_concentrations(delta_n=EXCESS_CARRIER)

    cb_window, vb_window = calc.calculate_energy_cutoffs(charge_threshold=0.99)

    nscf_folders = sorted(Path(NSCF_DIR).glob(f"NSCF_{AUGER_TYPE}_*"), key=numeric_suffix)
    exact_csvs = sorted(
        Path(RESULTS_DIR).glob(f"exact_kpoints_{AUGER_TYPE}_{calc.XX}_*.csv"),
        key=numeric_suffix,
    )
    if not exact_csvs:
        exact_csvs = sorted(Path(RESULTS_DIR).glob(f"exact_kpoints_{AUGER_TYPE}_{calc.XX}.csv"))

    if not nscf_folders:
        raise FileNotFoundError(f"No NSCF folders found in {NSCF_DIR}")
    if not exact_csvs:
        raise FileNotFoundError("No exact-kpoint CSV files found. Run step 1 first.")

    pair_input = cpp_work_dir + f"/{AUGER_TYPE}_exact_kpoint_pairs_input.bin"
    pair_csv_base = RESULTS_DIR + f"/auger_{AUGER_TYPE}_pairs_{calc.XX}.csv"

    prepare_pair_generation_input(
        results_folder=str(RESULTS_DIR),
        output=str(pair_input),
        task="pairs",
        auger_type=AUGER_TYPE,
        approach="exact_kpoint",
        CB_window=cb_window,
        VB_window=vb_window,
        num_to_keep=NUM_PAIRS_TO_KEEP,
        T=TEMPERATURE,
        delta_n=EXCESS_CARRIER,
        poscar_path=None,
        nscf_folders=[str(p) for p in nscf_folders],
        exact_kpoints_csv=[str(p) for p in exact_csvs],
    )

    subprocess.run([str(pair_exe), str(pair_input), str(pair_csv_base)], check=True)

    pair_csvs = sorted(
        Path(RESULTS_DIR).glob(f"auger_{AUGER_TYPE}_pairs_{calc.XX}_*.csv"),
        key=numeric_suffix,
    )
    if not pair_csvs:
        pair_csvs = sorted(Path(RESULTS_DIR).glob(f"auger_{AUGER_TYPE}_pairs_{calc.XX}.csv"))
    if not pair_csvs:
        raise FileNotFoundError("No pair CSV files were created.")

    wavecar_files = [str(folder / "WAVECAR") for folder in nscf_folders]
    matrix_input = cpp_work_dir + f"/{AUGER_TYPE}_exact_kpoint_matrix_input.bin"
    matrix_jsonl = RESULTS_DIR + f"/matrix_elements_{AUGER_TYPE}_{calc.XX}_1.jsonl"

    prepare_cpp_input(
        results_folder=str(RESULTS_DIR),
        wavecar_files=wavecar_files,
        pairs_csv=[str(p) for p in pair_csvs],
        auger_type=AUGER_TYPE,
        dielectric=DIELECTRIC,
        firstCB_index=FIRST_CB_INDEX,
        lastVB_index=LAST_VB_INDEX,
        output_path=str(matrix_input),
        num_matrix_elements=NUM_MATRIX_ELEMENTS,
    )

    subprocess.run(cpp_matrix_command(matrix_exe, matrix_input, matrix_jsonl), check=True)

    matrix_jsonls = sorted(
        Path(RESULTS_DIR).glob(f"matrix_elements_{AUGER_TYPE}_{calc.XX}_*.jsonl"),
        key=numeric_suffix,
    )
    calc.read_auger_pairs([str(p) for p in pair_csvs])
    calc.read_matrix_elements([str(p) for p in matrix_jsonls])
    auger_coeff = calc.calculate_auger_rates(auger_type=AUGER_TYPE)
