import os
import numpy as np
import pandas as pd
import mne
import h5py
import gc

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mne.decoding import SlidingEstimator, cross_val_multiscore

# =====================================================
# MODEL SELECTION SETTINGS
# =====================================================

MODEL_TYPE = "lda"  # Options: "lda", "logreg", "svm"

# Logistic Regression hyperparameters
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000

# LDA hyperparameters
LDA_SOLVER = "svd"  # "svd", "lsqr", "eigen"
LDA_SHRINKAGE = None  # None or "auto"

# SVM hyperparameters
SVM_C = 1.0  # Regularization strength
SVM_MAX_ITER = 1000  # Set -1 for no limit

# =====================================================
# CLASSIFIER FACTORY
# =====================================================


def get_classifier():
    if MODEL_TYPE.lower() == "lda":
        clf = LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE)
    elif MODEL_TYPE.lower() == "logreg":
        clf = LogisticRegression(penalty="l2", C=LOGREG_C, solver="liblinear", max_iter=LOGREG_MAX_ITER)
    elif MODEL_TYPE.lower() == "svm":
        clf = SVC(kernel="linear", C=SVM_C, max_iter=SVM_MAX_ITER)
    else:
        raise ValueError("MODEL_TYPE must be 'lda', 'logreg', or 'svm'")

    return make_pipeline(StandardScaler(), clf)


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def cosmo_style_average_samples(X, y, count=4, repeats=20, seed=1):
    """
    Replicates cosmo_average_samples: 
    For each class, randomly averages 'count' samples, repeated 'repeats' times.
    """
    rng = np.random.default_rng(seed)
    X_avg, y_avg = [], []
    classes = np.unique(y)

    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < count: continue

        for _ in range(repeats):
            chosen_idx = rng.choice(idx, size=count, replace=True)
            X_avg.append(np.mean(X[chosen_idx], axis=0))
            y_avg.append(c)

    return np.array(X_avg), np.array(y_avg)


def process_behavior(path_to_data, pair):
    """Extracts and formats behavioral data similar to the MATLAB logic."""
    tsv_path = os.path.join(path_to_data, f"sub-{pair:02d}", "eeg", f"sub-{pair:02d}_task-RPS_events.tsv")
    df = pd.read_csv(tsv_path, sep="\t")

    # Columns: 5=P1_resp, 7=P2_resp, 9=Outcome
    # MATLAB uses indices, Python uses names/0-indexing
    # Note: Adjust column names based on your actual TSV header
    events = df.iloc[:, [4, 6, 8]].values

    # Player 1 Perspective
    p1_behav = np.zeros((len(events), 5))
    p1_behav[:, :3] = events
    p1_behav[1:, 3:] = events[:-1, :2]  # Previous trials
    p1_behav[0, 3:] = np.nan

    # Player 2 Perspective
    p2_behav = np.zeros((len(events), 5))
    p2_behav[:, 0] = events[:, 1]  # Self = P2
    p2_behav[:, 1] = events[:, 0]  # Other = P1

    # Outcome flip: 1=Draw, 2=P1 Wins (P2 Loses), 3=P2 Wins (P2 Wins)
    p2_behav[:, 2] = events[:, 2]
    p2_behav[events[:, 2] == 2, 2] = 3
    p2_behav[events[:, 2] == 3, 2] = 2
    p2_behav[1:, 3:] = p2_behav[:-1, :2]
    p2_behav[0, 3:] = np.nan

    return p1_behav, p2_behav


# =====================================================
# MAIN SCRIPT
# =====================================================
path_to_data = os.path.join("src", "ds006761")
results_dir = os.path.join(path_to_data, "derivatives", "decoding_results")
os.makedirs(results_dir, exist_ok=True)

# pair_ids = [i for i in range(1, 35) if i not in [10, 23, 24]]
pair_ids = [1]
rem_idx = np.arange(0, 480, 40)  # Trials to remove (block starts)

for pair in pair_ids:
    print(f"Processing Pair {pair}...")
    p_behavs = process_behavior(path_to_data, pair)

    for ppt_idx, behav_data in enumerate(p_behavs):
        ppt = ppt_idx + 1

        save_path = os.path.join(results_dir, f"{MODEL_TYPE}-sub-{pair:02d}_player-{ppt}_decoding.h5")
        if os.path.exists(save_path): continue

        # Load preprocessed EEG
        fname = f"pair-{pair:02d}_player-{ppt}_task-RPS_eeg_epo.fif"
        fpath = os.path.join(path_to_data, "derivatives", fname)
        if not os.path.exists(fpath): continue

        epochs = mne.read_epochs(fpath, preload=True)
        epochs.set_eeg_reference("average")

        # Split into Parts A, B, C and baseline correct
        # MATLAB: Part A [-0.2, 2], B [1.8, 4], C [3.8, 5]
        # We crop and then shift time so each starts at '0' for the binning phase
        ep_a = epochs.copy().crop(tmin=-0.2, tmax=2.0).apply_baseline((-0.2, 0))
        ep_b = epochs.copy().crop(tmin=1.8, tmax=4.0).apply_baseline((1.8, 2.0))
        ep_c = epochs.copy().crop(tmin=3.8, tmax=5.0).apply_baseline((3.8, 4.0))

        # Reconstruct the discontinuous time vector
        # This ensures Part B starts at 1.8 in your results, not 2.2
        orig_times = np.concatenate([ep_a.times, ep_b.times, ep_c.times])

        # Recombine data arrays: [Trials, Chans, Times]
        combined_data = np.concatenate([ep_a.get_data(), ep_b.get_data(), ep_c.get_data()], axis=2)

        # # Resampling to 4Hz (creating 250ms bins)
        # info = ep_a.info
        # combined_epochs = mne.EpochsArray(combined_data, info)
        # combined_epochs.resample(4.0)  # Downsample to 4Hz

        # Create Dummy Epochs for resampling
        temp_epochs = mne.EpochsArray(combined_data, ep_a.info, tmin=0)
        # Resample data and the time vector simultaneously
        resampled_data = temp_epochs.copy().resample(4.0).get_data()

        # To resample the time labels correctly, we pick indices
        resample_factor = ep_a.info['sfreq'] / 4.0
        time_idx = np.arange(0, len(orig_times), resample_factor).astype(int)
        resampled_times = orig_times[time_idx]

        # Filter trials
        keep_mask = np.ones(len(resampled_data), dtype=bool)
        keep_mask[[i for i in rem_idx if i < len(keep_mask)]] = False
        X = resampled_data[keep_mask]
        y_labels = behav_data[keep_mask]

        # --- Decoding Loop ---
        with h5py.File(save_path, 'w') as hf:
            hf.attrs['pair'] = pair
            hf.attrs['player'] = ppt
            hf.attrs["model_type"] = MODEL_TYPE

            hf.create_dataset('times', data=resampled_times)
            hf.create_dataset('ch_names', data=[n.encode('utf-8') for n in ep_a.ch_names])

            target_names = ['self', 'other', 'selfp', 'otherp']
            target_cols = [0, 1, 3, 4]

            for name, col in zip(target_names, target_cols):
                y = y_labels[:, col]

                # Remove NaNs and no-responses (0)
                valid = ~np.isnan(y) & (y > 0)
                if not np.any(valid): continue

                # Replicate CoSMo SNR averaging
                X_v, y_v = X[valid], y[valid]
                X_avg, y_avg = cosmo_style_average_samples(X_v, y_v)

                # Define Classifier
                clf = get_classifier()
                # n_jobs=-1 will use all available CPU cores
                time_gen = SlidingEstimator(clf, scoring='accuracy', n_jobs=1, verbose=False)

                # Create a group for this target (like a MATLAB sub-struct)
                grp = hf.create_group(name)

                # --- Temporal Decoding ---
                scores_temp = cross_val_multiscore(time_gen, X_avg, y_avg, cv=10, n_jobs=1)
                grp.create_dataset('samples', data=scores_temp.mean(0))

                # --- Channel Searchlight ---
                # Replicating cosmo_meeg_chan_neighborhood (count=4)
                adjacency, ch_names = mne.channels.find_ch_adjacency(ep_a.info, ch_type='eeg')
                sl_scores = np.zeros((len(ch_names), X_avg.shape[2]))

                for i in range(len(ch_names)):
                    # Get indices of the channel and its nearest neighbors
                    neighbor_idx = adjacency[i].tocsr().indices
                    # Limit to neighbors (MATLAB count=4 usually includes self + 3 closest)
                    X_sl = np.take(X_avg, neighbor_idx, axis=1)

                    # Run sliding estimator on the spatial subset
                    s_scores = cross_val_multiscore(time_gen, X_sl, y_avg, cv=5, n_jobs=1)
                    sl_scores[i, :] = s_scores.mean(0)

                    del X_sl
                    gc.collect()

                grp.create_dataset('searchlight', data=sl_scores)

                # Store Attributes (The "sa" and "fa" from MATLAB)
                # This helps you know exactly what the labels were for this specific result
                grp.create_dataset('sa_targets', data=y_v)
                grp.attrs['n_samples_averaged'] = 4

                del X_v, y_v, X_avg, y_avg
                gc.collect()

        print(f"Finished Pair {pair} Player {ppt}")
