import pathlib
import mne
import torch

import data_setup
import dataset
import eeg_visualization
import linear_decoding
import decoding_visualization
import markov_analysis
import non_linear_decoding

# If you would like to re-run the pipeline multiple times whithout overwriting previous iterations
# of the data, change the following paths in data_setup.py:
# PREPROCESSED_DIR, NON_LINEAR_DIR, LINEAR_DECODING_DIR

def main():
    # 0. Setup Check
    data_setup.data_structure_setup()
    to_do_steps = data_setup.pipeline_progress_check()

    # 1. Preprocessing
    bids_dataset = dataset.BidsDataset.get_from(data_setup.DATA_ROOT, data_setup.PREPROCESSED_DIR)
    if to_do_steps["preprocessing"]:
        print("Starting Preprocessing...")
        # Change the arguments to True (filter, drop_bad_epochs) or 1 (for detrend) if you want to modify the preprocessing
        # Please do not forget to change the paths accordingly in this case, so that the previous results do not get overwritten
        bids_dataset.preprocess(filter=True, drop_bad_channels=False, detrend=None)
        print("Preprocessing finished!")
    else:
        print("Preprocessing already completed!")

    # 2. Linear Decoding
    if to_do_steps["linear_decoding"]:
        print("Starting Linear Decoding...")
        linear_decoding.run_decoding(data_setup.DATA_ROOT, data_setup.PREPROCESSED_DIR, data_setup.LINEAR_DECODING_DIR)
        decoding_visualization.plot_all_decoding(data_setup.LINEAR_DECODING_DIR)
        print("Linear Decoding finished!")
    else:
        print("Linear Decoding already completed!")

    # 3. Non-Linear Decoding
    if to_do_steps["non_linear_decoding"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda":
            print("Starting Non-Linear Decoding...")
            non_linear_decoding.run_non_linear_decoding()
            print("Non-Linear Decoding finished!")
        else:
            print("Non-Linear Decoding has not been completed, but there are no GPUs available. Skipping this step...")
    else:
        print("Non-Linear Decoding already completed!")

    # 4. Markov analysis and Statistical plots
    if to_do_steps["markov_stats"]:
        print("Starting Markov Analysis...")
        markov_analysis.run_group_markov_analysis(data_setup.DATA_ROOT, data_setup.MARKOV_STATS_DIR)
        print("Markov Analysis finished!")

        print("Starting Statistical Plots...")
        eeg_visualization.generate_plot_1c(bids_dataset.subjects, data_setup.MARKOV_STATS_DIR)
        eeg_visualization.generate_plot_1d(bids_dataset.subjects, data_setup.MARKOV_STATS_DIR)
        eeg_visualization.generate_plot_1e(bids_dataset.subjects, data_setup.MARKOV_STATS_DIR)
        print("Statistical Plots finished!")
    else:
        print("Markov Analysis and Statistical Plots already completed!")


if __name__ == "__main__":
    main()
