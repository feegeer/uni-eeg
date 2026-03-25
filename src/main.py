import pathlib
import mne

import data_setup
import dataset
import eeg_visualization
import linear_decoding
import visualization


def main():
    # 0. Setup Check
    data_setup.data_structure_setup()
    to_do_steps = data_setup.pipeline_progress_check()

    # 1. Preprocessing
    if to_do_steps["preprocessing"]:
        bids_dataset = dataset.BidsDataset.get_from(data_setup.DATA_ROOT, data_setup.PREPROCESSED_DIR)
        bids_dataset.preprocess()

    # 2. Linear Decoding
    if to_do_steps["linear_decoding"]:
        linear_decoding.run_decoding()
        visualization.plot_all(data_setup.LINEAR_DECODING_DIR)

    # 3. Non-Linear Decoding
    if to_do_steps["non_linear_decoding"]:
        ...

    # 4. Markov Analysis
    if to_do_steps["markov_analysis"]:
        ...


if __name__ == "__main__":
    main()
