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

N_SPLITS = 10
PSEUDO_COUNT = 4
PSEUDO_REPEATS = 20
RANDOM_STATE = 1

rng = np.random.default_rng(RANDOM_STATE)

# ==========================================================
# CLASSIFIER
# ==========================================================

def get_classifier():
    return LinearDiscriminantAnalysis()

# ==========================================================
# PSEUDOTRIALS (COSMO-STYLE, BALANCED)
# ==========================================================

def make_pseudotrials(X, y):
    X_out, y_out = [], []
    classes = np.unique(y)

    for _ in range(PSEUDO_REPEATS):
        for c in classes:
            idx = np.where(y == c)[0]

            if len(idx) < PSEUDO_COUNT:
                continue

            idx = rng.permutation(idx)
            n_groups = len(idx) // PSEUDO_COUNT

            for g in range(n_groups):
                sel = idx[g * PSEUDO_COUNT:(g + 1) * PSEUDO_COUNT]
                X_out.append(X[sel].mean(axis=0))
                y_out.append(c)

    return np.array(X_out), np.array(y_out)

# ==========================================================
# STRICT MATLAB PHASE SPLIT + BINNING
# ==========================================================

def split_and_bin_epochs(epochs):

    partA = epochs.copy().crop(tmin=-0.2, tmax=2.0)
    partB = epochs.copy().crop(tmin=1.8, tmax=4.0)
    partC = epochs.copy().crop(tmin=3.8, tmax=5.0)

    partA.apply_baseline((-0.2, 0.0), verbose=False)
    partB.apply_baseline((1.8, 2.0), verbose=False)
    partC.apply_baseline((3.8, 4.0), verbose=False)

    A_data, A_times = partA.get_data(), partA.times
    B_data, B_times = partB.get_data(), partB.times
    C_data, C_times = partC.get_data(), partC.times

    bin_edges = np.arange(0, 5.0001, 0.25)
    all_bins = []

    for i in range(len(bin_edges) - 1):
        t0, t1 = bin_edges[i], bin_edges[i + 1]

        if t0 < 2:
            mask = (A_times > t0) & (A_times < t1)   # strict MATLAB
            all_bins.append(A_data[:, :, mask].mean(axis=2))

        elif t0 < 4:
            mask = (B_times > t0) & (B_times < t1)
            all_bins.append(B_data[:, :, mask].mean(axis=2))

        else:
            mask = (C_times > t0) & (C_times < t1)
            all_bins.append(C_data[:, :, mask].mean(axis=2))

    return np.stack(all_bins, axis=2)

# ==========================================================
# TEMPORAL DECODING
# ==========================================================

def temporal_decoding(X, y):

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)
    n_times = X.shape[2]
    acc = np.zeros(n_times)

    for train_idx, test_idx in skf.split(X, y):

        X_train_raw, y_train_raw = X[train_idx], y[train_idx]
        X_test_raw, y_test_raw = X[test_idx], y[test_idx]

        X_train, y_train = make_pseudotrials(X_train_raw, y_train_raw)
        X_test, y_test = make_pseudotrials(X_test_raw, y_test_raw)

        for t in range(n_times):
            clf = get_classifier()
            clf.fit(X_train[:, :, t], y_train)
            preds = clf.predict(X_test[:, :, t])
            acc[t] += accuracy_score(y_test, preds)

    return acc / N_SPLITS

# ==========================================================
# SEARCHLIGHT DECODING (FIXED NEIGHBORS)
# ==========================================================

def searchlight_decoding(X, y, ch_coords):

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)
    n_times = X.shape[2]
    n_ch = X.shape[1]

    dists = cdist(ch_coords, ch_coords)

    neighbors = []
    for i in range(n_ch):
        order = np.argsort(dists[i])
        order = order[order != i]     # remove self
        neighbors.append(np.concatenate(([i], order[:4])))  # self + 4 nearest

    sl_acc = np.zeros((n_ch, n_times))

    for train_idx, test_idx in skf.split(X, y):

        X_train_raw, y_train_raw = X[train_idx], y[train_idx]
        X_test_raw, y_test_raw = X[test_idx], y[test_idx]

        X_train, y_train = make_pseudotrials(X_train_raw, y_train_raw)
        X_test, y_test = make_pseudotrials(X_test_raw, y_test_raw)

        for ch in range(n_ch):

            spatial_subset = neighbors[ch]

            for t in range(n_times):
                clf = get_classifier()
                clf.fit(X_train[:, spatial_subset, t], y_train)
                preds = clf.predict(X_test[:, spatial_subset, t])
                sl_acc[ch, t] += accuracy_score(y_test, preds)

    return sl_acc / N_SPLITS

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline(path_to_data, results_dir):

    os.makedirs(results_dir, exist_ok=True)

    pairs_removed = [10, 23, 24]
    pair_ids = [i for i in range(1, 35) if i not in pairs_removed]

    for pair in pair_ids:

        print(f"\nPair {pair:02d}")

        tsv_path = os.path.join(
            path_to_data,
            f"sub-{pair:02d}",
            "eeg",
            f"sub-{pair:02d}_task-RPS_events.tsv"
        )

        df = pd.read_csv(tsv_path, sep="\t")
        events = df.iloc[:, [4, 6, 8]].values

        # Behavioral matrices
        p1 = np.zeros((len(events), 5))
        p1[:, :3] = events
        p1[1:, 3:] = events[:-1, :2]

        p2 = np.zeros((len(events), 5))
        p2[:, 0], p2[:, 1] = events[:, 1], events[:, 0]
        p2[:, 2] = events[:, 2]
        p2[events[:, 2] == 2, 2] = 3
        p2[events[:, 2] == 3, 2] = 2
        p2[1:, 3:] = p2[:-1, :2]

        for ppt, behav in enumerate([p1, p2], start=1):

            print(f"  Player {ppt}")

            epo_path = os.path.join(
                path_to_data,
                "new_derivatives",
                f"pair-{pair:02d}_player-{ppt}_task-RPS_eeg_epo.fif"
            )

            if not os.path.exists(epo_path):
                continue

            epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
            epochs.set_eeg_reference("average", verbose=False)

            rem_idx = np.arange(0, 480, 40)
            epochs.drop(rem_idx, verbose=False)
            behav = np.delete(behav, rem_idx, axis=0)

            X = split_and_bin_epochs(epochs)
            ch_coords = np.array([ch["loc"][:3] for ch in epochs.info["chs"]])

            results_file = os.path.join(
                results_dir,
                f"pair-{pair:02d}_player-{ppt}_decoding.h5"
            )

            with h5py.File(results_file, "w") as hf:

                targets = {"self": 0, "other": 1, "selfp": 3, "otherp": 4}

                for name, col in targets.items():

                    y = behav[:, col]
                    valid = (~np.isnan(y)) & (y > 0)

                    if not np.any(valid):
                        continue

                    print(f"    Decoding {name}")

                    Xv, yv = X[valid], y[valid]

                    temp_acc = temporal_decoding(Xv, yv)
                    sl_acc = searchlight_decoding(Xv, yv, ch_coords)

                    grp = hf.create_group(name)
                    grp.create_dataset("temporal_accuracy", data=temp_acc)
                    grp.create_dataset("searchlight_accuracy", data=sl_acc)

            gc.collect()

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    PATH_TO_DATA = "src/ds006761"
    RESULTS_DIR = os.path.join(PATH_TO_DATA, "new_derivatives", "python_replication_corrected")

    run_pipeline(PATH_TO_DATA, RESULTS_DIR)