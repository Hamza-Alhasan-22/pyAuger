from auger import AugerCalculator, utilities

# Inputs:

# ── Paths ──────────────────────────────────────────────────────────────────
VASP_FOLDER       = "../test-files/InAs/nearest-kpoint-scf-3"
RESULTS_DIR       = "../test-files/InAs/results/nearest_kpoint"

# ── Physical parameters ────────────────────────────────────────────────────
AUGER_TYPE        = "eeh"          # Auger process type: "eeh" or "ehh"
TEMPERATURE       = 300            # Temperature (K)
DOPING            = 0              # Doping concentration (cm^-3)
EXCESS_CARRIER    = 1e17           # Excess carrier concentration (cm^-3)

# ── Material parameters ────────────────────────────────────────────────────
DIELECTRIC        = 12.3           # dielectric constant (unitless)
FIRST_CB_INDEX    = 9              # Index of the first conduction band (starting from 0)
LAST_VB_INDEX     = 8              # Index of the last valence band (starting from 0)
FORCE_GAP         = 0.4            # Optional: Force a band gap value (eV) when parsing the band structure data. Set it to None to use the raw band gap from the data.

NUM_PAIRS_TO_KEEP = "all"          # "all" or an integer specifying how many Auger pairs to keep based on their probabilities.

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
    
    fn, fp = calc.calculate_carrier_concentrations(
        delta_n=EXCESS_CARRIER,
    )
    
    CB_auto, VB_auto = calc.calculate_energy_cutoffs(charge_threshold=0.99)
    # or use manually defined energy windows
    
    gen_pairs = calc.create_auger_pairs(
        CB_window=CB_auto, 
        VB_window=VB_auto, 
        auger_type=AUGER_TYPE,
        approach="nearest_kpoint",
        num_top_pairs=NUM_PAIRS_TO_KEEP,
    )
    
    me = calc.calculate_matrix_elements(
        auger_type=AUGER_TYPE,
        wavecar_files=VASP_FOLDER + "/WAVECAR", 
        dielectric_constant=DIELECTRIC,
        num_matrix_elements="all",
    )
    
    auger_coeff = calc.calculate_auger_rates(
        auger_type=AUGER_TYPE,
    )
