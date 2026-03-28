import pathlib
import mne

import data_setup
import dataset
import eeg_visualization
import linear_decoding
import decoding_visualization
import markov_analysis


def main():
    # 0. Setup Check
    data_setup.data_structure_setup()
    to_do_steps = data_setup.pipeline_progress_check()
    
    # 1. Preprocessing
    if to_do_steps["preprocessing"]:
        bids_dataset = dataset.BidsDataset.get_from(data_setup.DATA_ROOT, data_setup.PREPROCESSED_DIR)
        bids_dataset.preprocess()
    else:
        print("Preprocessing already completed!")

    # 2. Linear Decoding
    if to_do_steps["linear_decoding"]:
        linear_decoding.run_decoding()
        decoding_visualization.plot_all_decoding(data_setup.LINEAR_DECODING_DIR)
    else:
        print("Linear Decoding already completed!")

    # 3. Non-Linear Decoding
    if to_do_steps["non_linear_decoding"]:
        ...
    else:
        print("Non-Linear Decoding already completed!")

    # 4. Markov Analysis
    if to_do_steps["markov_analysis"]:
        markov_analysis.run_group_markov_analysis(data_setup.DATA_ROOT, data_setup.MARKOV_DIR)
    else:
        print("Markov Analysis already completed!")


if __name__ == "__main__":
    main()
