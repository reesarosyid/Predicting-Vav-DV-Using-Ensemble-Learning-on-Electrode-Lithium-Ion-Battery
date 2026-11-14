ACTIVE_METALS = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Al", "Zn", "Y"]

VALENCE = {
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
    "Mg": 2, "Ca": 2, "Zn": 2,
    "Al": 3, "Y": 3,
}

FARADAY = 96485.0   # C/mol
RATIO_TOL = 1e-6

DATA_DIR = "Dataset"
RAW_MP_PATH = f"{DATA_DIR}/01_raw_mp.json"
RAW_AFLOW_PATH = f"{DATA_DIR}/02_raw_aflow.json"
ELEMENT_REF_PATH = f"{DATA_DIR}/03_element_ref.json"
COMBINED_RAW = f"{DATA_DIR}/04_combined_raw.json"
PAIRS_PATH = f"{DATA_DIR}/05_pairs.json"
PROPERTIES_PATH = f"{DATA_DIR}/06_properties.csv"
FEATURES_PATH = f"{DATA_DIR}/07_features.csv"
