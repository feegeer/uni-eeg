from pathlib import Path


DATA_ROOT = Path("data/ds006761")
RESULTS_ROOT = Path("data/results")
BIOSEMI_PATH = Path("data/biosemi64.mat")
PREPROCESSED_DIR = RESULTS_ROOT / "preprocessed_eeg"
MARKOV_DIR = RESULTS_ROOT / "markov_analysis"
LINEAR_DECODING_DIR = RESULTS_ROOT / "linear_decoding"
NON_LINEAR_DIR = RESULTS_ROOT / "non_linear_decoding"

# Dataset
SUB_IDS = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
STR_SUB_ID = [f"sub-{n:02d}" for n in SUB_IDS]


def data_structure_setup() -> None:
    """ Ensures correct data folder structure for pipeline. """
    # 1. Check correctness of original dataset structure
    dataset_paths = [DATA_ROOT]

    for sub_id in STR_SUB_ID:
        sub_path = DATA_ROOT / sub_id
        eeg_path = sub_path / "eeg"
        bdf_path = eeg_path / f"{sub_id}_task-RPS_eeg.bdf"
        tsv_path = eeg_path / f"{sub_id}_task-RPS_events.tsv"
        dataset_paths.extend([sub_path, eeg_path, bdf_path, tsv_path])

    dataset_correctness = True
    for path in dataset_paths:
        if not path.exists():
            dataset_correctness = False
            print(f"Path does not exist: {path}")

    if dataset_correctness:
        print("Dataset is correctly setup!")
    else:
        print("Dataset is incomplete!")

    # 2. Create all results folders
    RESULTS_ROOT.mkdir(exist_ok=True)
    PREPROCESSED_DIR.mkdir(exist_ok=True)
    LINEAR_DECODING_DIR.mkdir(exist_ok=True)
    NON_LINEAR_DIR.mkdir(exist_ok=True)
    MARKOV_DIR.mkdir(exist_ok=True)
    
    # 3. biosemi64.mat
    if not BIOSEMI_PATH.exists():
        print(f"{BIOSEMI_PATH} does not exist.")


def pipeline_progress_check() -> dict[str, bool]:
    """ 
    Checks which steps of the pipeline have been completed already
    
    Returns:
        dict[str, bool]: steps of the pipeline to do
    """
    to_do = lambda d: not any(p.is_file() for p in d.iterdir())

    to_do_steps = {
      "markov_analysis": to_do(MARKOV_DIR),
      "preprocessing": to_do(PREPROCESSED_DIR),
      "linear_decoding": to_do(LINEAR_DECODING_DIR),
      "non_linear_decoding": to_do(NON_LINEAR_DIR),
    }

    return to_do_steps
