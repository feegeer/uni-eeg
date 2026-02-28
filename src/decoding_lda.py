"""
EEG LDA Decoding Pipeline
-------------------------
Time-resolved and spatial searchlight decoding pipeline using
Linear Discriminant Analysis (LDA) for the Rock-Paper-Scissors EEG dataset.

Main characteristics:
- Phase-specific baselining for long (5s) trials
- 250 ms boxcar binning to reduce temporal dimensionality
- Within-fold pseudotrial averaging to improve SNR
- LDA with fixed shrinkage (λ = 0.01)
- Stratified 10-fold cross-validation

Data Source: OpenNeuro ds006761
"""

import os
import gc
import numpy as np
import pandas as pd
import mne
import h5py
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ==========================================================
# CONFIGURATION
# ==========================================================

# LDA regularization parameter (matches CoSMoMVPA-style shrinkage)
LAMBDA = 0.01

# Cross-validation folds
N_SPLITS = 10

# Pseudotrial parameters (boost signal-to-noise ratio for LDA)
PSEUDO_COUNT = 4     # Raw trials averaged into one pseudotrial
PSEUDO_REPEATS = 20  # Pseudotrials generated per class per fold

RANDOM_SEED = 1
rng = np.random.default_rng(RANDOM_SEED)

# ==========================================================
# CLASSIFIER
# ==========================================================

def get_lda_classifier():
    """Returns a Linear Discriminant Analysis classifier with fixed shrinkage."""
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=LAMBDA)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def split_and_bin_epochs(epochs):
    """
    Convert MNE Epochs into temporally binned decoding features.
    """
    # ------------------------------------------------------
    # 1) Split 5s trials into 3 task phases
    # ------------------------------------------------------
    
    partA = epochs.copy().crop(tmin=-0.2, tmax=2.0)
    partB = epochs.copy().crop(tmin=1.8, tmax=4.0)
    partC = epochs.copy().crop(tmin=3.8, tmax=5.0)

    # ------------------------------------------------------
    # 2) Apply phase-local baseline correction
    # ------------------------------------------------------

    partA.apply_baseline((-0.2, 0.0), verbose=False)
    partB.apply_baseline((1.8, 2.0), verbose=False)
    partC.apply_baseline((3.8, 4.0), verbose=False)

    # ------------------------------------------------------
    # 3) 250 ms boxcar temporal binning
    # ------------------------------------------------------
    
    def bin_part(epo, n_bins, start_time):
        """Average signals into consecutive 250 ms windows."""
        data = epo.get_data()
        times = epo.times
        bins = []
        for b in range(n_bins):
            t_start = start_time + 0.25 * b
            t_end = t_start + 0.25
            mask = (times >= t_start) & (times < t_end)
            bins.append(data[:, :, mask].mean(axis=2))
        return np.stack(bins, axis=2)

    # Create 20 temporal bins for the LDA to classify
    A = bin_part(partA, 8, 0.0)  # Decision phase
    B = bin_part(partB, 8, 2.0)  # Response phase
    C = bin_part(partC, 4, 4.0)  # Feedback phase

    # ------------------------------------------------------
    # 4) Concatenate all temporal bins
    # ------------------------------------------------------

    return np.concatenate([A, B, C], axis=2)

# ==========================================================
# PSEUDOTRIAL GENERATION
# ==========================================================

def make_pseudotrials(X, y):
    """
    Generate pseudotrials within CV folds.
    Each pseudotrial is an average of PSEUDO_COUNT raw trials.
    """
    X_out, y_out = [], []
    classes = np.unique(y)

    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < PSEUDO_COUNT:
            continue

        for _ in range(PSEUDO_REPEATS):
            chosen = rng.choice(idx, size=PSEUDO_COUNT, replace=True)
            X_out.append(X[chosen].mean(axis=0))
            y_out.append(c)

    return np.array(X_out), np.array(y_out)

# ==========================================================
# TEMPORAL DECODING
# ==========================================================

def temporal_lda_decoding(X, y):
    """
    Performs time-resolved LDA decoding using all sensors as features.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)
    n_times = X.shape[2]
    acc = np.zeros(n_times)

    for train_idx, test_idx in skf.split(X, y):
        # Split before pseudotrial averaging
        X_train_raw, y_train_raw = X[train_idx], y[train_idx]
        X_test_raw, y_test_raw = X[test_idx], y[test_idx]

        # Generate independent pseudotrials
        X_train, y_train = make_pseudotrials(X_train_raw, y_train_raw)
        X_test, y_test = make_pseudotrials(X_test_raw, y_test_raw)

        clf = get_lda_classifier()

        for t in range(n_times):
            clf.fit(X_train[:, :, t], y_train)
            preds = clf.predict(X_test[:, :, t])
            acc[t] += accuracy_score(y_test, preds)

    return acc / N_SPLITS

# ==========================================================
# SPATIAL SEARCHLIGHT DECODING
# ==========================================================

def searchlight_lda_decoding(X, y, ch_coords):
    """
    Performs spatial searchlight LDA using 4-neighbor sensor clusters.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)
    n_times = X.shape[2]
    n_ch = X.shape[1]

    # Precompute 4-nearest-neighbor clusters
    dists = cdist(ch_coords, ch_coords)
    neighbors = [np.argsort(dists[i])[:4] for i in range(n_ch)]

    sl_acc = np.zeros((n_ch, n_times))

    for train_idx, test_idx in skf.split(X, y):
        X_train_raw, y_train_raw = X[train_idx], y[train_idx]
        X_test_raw, y_test_raw = X[test_idx], y[test_idx]

        X_train, y_train = make_pseudotrials(X_train_raw, y_train_raw)
        X_test, y_test = make_pseudotrials(X_test_raw, y_test_raw)

        for ch in range(n_ch):
            clf = get_lda_classifier()
            spatial_subset = neighbors[ch]

            for t in range(n_times):
                clf.fit(X_train[:, spatial_subset, t], y_train)
                preds = clf.predict(X_test[:, spatial_subset, t])
                sl_acc[ch, t] += accuracy_score(y_test, preds)

    return sl_acc / N_SPLITS

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline(path_to_data, results_dir):
    """Execute full decoding pipeline across all pairs and players."""
    os.makedirs(results_dir, exist_ok=True)

    pairs_removed = [10, 16, 23, 24, 29]
    pair_ids = [i for i in range(1, 35) if i not in pairs_removed]

    for pair in pair_ids:
        print(f"\nProcessing Pair {pair:02d}...")

        # --------------------------------------------------
        # Load behavioral data
        # --------------------------------------------------

        tsv_path = os.path.join(path_to_data, f"sub-{pair:02d}", "eeg", f"sub-{pair:02d}_task-RPS_events.tsv")
        df = pd.read_csv(tsv_path, sep="\t")
        events = df.iloc[:, [4, 6, 8]].values

        # Construct behavioral matrices
        p1_behav = np.zeros((len(events), 5))
        p1_behav[:, :3] = events
        p1_behav[1:, 3:] = events[:-1, :2]

        p2_behav = np.zeros((len(events), 5))
        p2_behav[:, 0], p2_behav[:, 1] = events[:, 1], events[:, 0]
        p2_behav[:, 2] = events[:, 2]

        # Flip win/loss coding
        p2_behav[events[:, 2] == 2, 2] = 3
        p2_behav[events[:, 2] == 3, 2] = 2
        p2_behav[1:, 3:] = p2_behav[:-1, :2]

        for ppt, behav in enumerate([p1_behav, p2_behav], start=1):
            print(f"  > Player {ppt}")

            epo_path = os.path.join(path_to_data, "derivatives", f"pair-{pair:02d}_player-{ppt}_task-RPS_eeg_epo.fif")
            if not os.path.exists(epo_path):
                continue

            # --------------------------------------------------
            # Load EEG epochs
            # --------------------------------------------------

            epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
            epochs.set_eeg_reference("average", verbose=False)

            # Remove block-boundary trials (every 40th trial; start of each block)
            rem_idx = np.arange(0, 480, 40)
            epochs.drop(rem_idx, verbose=False)
            behav = np.delete(behav, rem_idx, axis=0)

            # --------------------------------------------------
            # Feature extraction
            # --------------------------------------------------

            X = split_and_bin_epochs(epochs)
            ch_coords = np.array([ch["loc"][:3] for ch in epochs.info["chs"]])

            results_file = os.path.join(results_dir, f"pair-{pair:02d}_player-{ppt}_decoding.h5")

            # --------------------------------------------------
            # Run decoding for each behavioral target
            # --------------------------------------------------

            with h5py.File(results_file, "w") as hf:
                targets_dict = {"self": 0, "other": 1, "selfp": 3, "otherp": 4}

                for name, col in targets_dict.items():
                    y = behav[:, col]
                    valid = (~np.isnan(y)) & (y > 0)

                    if not np.any(valid):
                        continue

                    print(f"    * Running LDA on: {name}")

                    Xv, yv = X[valid], y[valid]

                    # Execute LDA decoding
                    temp_acc = temporal_lda_decoding(Xv, yv)
                    sl_acc = searchlight_lda_decoding(Xv, yv, ch_coords)

                    grp = hf.create_group(name)
                    grp.create_dataset("temporal_accuracy", data=temp_acc)
                    grp.create_dataset("searchlight_accuracy", data=sl_acc)

            # Free memory
            del epochs, X
            gc.collect()

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    PATH_TO_DATA = "src/ds006761"
    RESULTS_DIR = os.path.join(PATH_TO_DATA, "derivatives", "python_replication")

    run_pipeline(PATH_TO_DATA, RESULTS_DIR)
