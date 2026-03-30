"""
Multi-classifier EEG decoding script for the RPS dataset.

Extends the LDA decoding pipeline to run four classifiers on identical
data, folds, and pseudo-trials:
  1. LDA (CoSMoMVPA-style) — replicates the original paper
  2. Shrinkage LDA (Ledoit-Wolf) — optimal analytical regularisation
  3. Linear SVM — margin-based classifier
  4. Logistic Regression (L2) — cross-entropy-trained classifier

Requires: mne, numpy, pandas, scikit-learn
"""

import pathlib

from typing import Callable
import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


mne.set_log_level("WARNING")

# --------------------
# Paths and constants
# --------------------

PAIR_IDS = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))

NUM_TRIALS = 480
TRIALS_PER_BLOCK = 40

# Parts A and B: 0–2 s in 250 ms bins (Decision + Response phases)
TIME_WINDOWS_AB = np.column_stack([np.arange(0, 2.0, 0.25), np.arange(0.25, 2.25, 0.25)])

# Part C: 0–1 s in 250 ms bins (Feedback)
TIME_WINDOWS_C = np.column_stack([np.arange(0, 1.0, 0.25), np.arange(0.25, 1.25, 0.25)])

# --------------------
# Classifier functions
# --------------------
# Each takes (train_data, train_labels, test_data) and returns predictions.


def classify_lda_cosmo(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray) -> np.ndarray:
    """LDA with CoSMoMVPA-style additive regularisation (λ = 0.01)."""
    classes = np.unique(train_labels)
    n_train, n_features = train_data.shape

    class_mean = np.zeros((len(classes), n_features))
    class_cov = np.zeros((n_features, n_features))

    for k, c in enumerate(classes):
        msk = train_labels == c
        class_samples = train_data[msk]
        class_mean[k] = class_samples.mean(axis=0)
        residuals = class_samples - class_mean[k]
        class_cov += residuals.T @ residuals

    class_cov /= n_train
    reg_term = np.eye(n_features) * np.trace(class_cov) / max(1, n_features)
    class_cov_reg = class_cov + reg_term * 0.01

    class_weight = np.linalg.solve(class_cov_reg.T, class_mean.T).T
    class_offset = np.sum(class_weight * class_mean, axis=1)

    scores = test_data @ class_weight.T - 0.5 * class_offset[None, :]
    return classes[np.argmax(scores, axis=1)]


def classify_shrinkage_lda(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray) -> np.ndarray:
    """LDA with Ledoit-Wolf analytical shrinkage."""
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(train_data, train_labels)
    return clf.predict(test_data)


def classify_svm(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray) -> np.ndarray:
    """Linear SVM with z-scored features."""
    train_scaled, test_scaled = zscore(train_data, test_data)
    clf = LinearSVC(C=1.0, max_iter=10000, dual="auto")
    clf.fit(train_scaled, train_labels)
    return clf.predict(test_scaled)


def classify_logreg(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray) -> np.ndarray:
    """L2-regularised multinomial logistic regression with z-scored features."""
    train_scaled, test_scaled = zscore(train_data, test_data)
    clf = LogisticRegression(C=1.0, max_iter=10000, solver="lbfgs")
    clf.fit(train_scaled, train_labels)
    return clf.predict(test_scaled)


def zscore(train_data: np.ndarray, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score test data using training statistics."""
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train_data - mean) / std, (test_data - mean) / std


# All classifiers: short_name -> (display_name, function)
CLASSIFIERS = {
  "lda_cosmo": ("LDA (CoSMoMVPA)", classify_lda_cosmo),
  "shrinkage_lda": ("Shrinkage LDA (Ledoit-Wolf)", classify_shrinkage_lda),
  "linear_svm": ("Linear SVM", classify_svm),
  "logistic_reg": ("Logistic Regression", classify_logreg)
}


# --------------------
# CoSMoMVPA-equivalent helper functions
# --------------------
def cosmo_sample_unique(k: int, n: int, count: int, seed: int | None = None) -> np.ndarray:
    """
    Balanced sampling without replacement.

    Args:
        k (int): number of samples per subset
        n (int): number of available samples
        count (int): number of subsets to generate
        seed (int, optional): random seed

    Returns:
        np.ndarray: array of shape (k, count) with sampled indices
    """
    rng = np.random.default_rng(seed)

    rs_mat = np.zeros((n, count + 1), dtype=int)
    for c in range(count + 1):
        rs_mat[:, c] = rng.permutation(n)

    rs = rs_mat.ravel(order='F')

    samples = np.zeros((k, count), dtype=int)
    visited = np.zeros(len(rs), dtype=bool)

    first_non_visited = 0
    for col in range(count):
        in_bin = np.zeros(n, dtype=bool)
        pos = first_non_visited
        for row in range(k):
            while visited[pos] or in_bin[rs[pos]]:
                pos += 1
            r = rs[pos]
            in_bin[r] = True
            samples[row, col] = r
            visited[pos] = True

        while first_non_visited < len(visited) and visited[first_non_visited]:
            first_non_visited += 1

    samples.sort(axis=0)
    return samples


def cosmo_chunkize(targets: np.ndarray, n_chunks: int) -> np.ndarray:
    """
    Assign cross-validation fold labels while keeping classes balanced.

    Args:
        targets (np.ndarray): class labels for each sample
        n_chunks (int): number of folds

    Returns:
        np.ndarray: chunk index for each sample (1..n_chunks)
    """
    chunks = np.zeros(len(targets), dtype=int)

    for t in np.unique(targets):
        idx = np.where(targets == t)[0]
        for i, ix in enumerate(idx):
            chunks[ix] = (i % n_chunks) + 1

    return chunks


def cosmo_average_samples(data: np.ndarray,
                          targets: np.ndarray,
                          chunks: np.ndarray,
                          count=4,
                          repeats=20,
                          seed=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create pseudo-trials by averaging samples within each (target, chunk).

    Args:
        data (np.ndarray): EEG data (n_trials, n_channels, n_timebins)
        targets (np.ndarray): class label for each trial
        chunks (np.ndarray): fold assignment for each trial
        count (int): number of trials to average together
        repeats (int): number of pseudo-trials per condition
        seed (int): random seed

    Returns:
        tuple:
            avg_data (np.ndarray): pseudo-trials
            avg_targets (np.ndarray): labels
            avg_chunks (np.ndarray): chunk assignments
    """
    avg_data_list, avg_targets_list, avg_chunks_list = [], [], []

    for chunk_val in np.unique(chunks):
        for target_val in np.unique(targets):
            mask = (targets == target_val) & (chunks == chunk_val)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            sample_ids = cosmo_sample_unique(count, len(indices), repeats, seed=seed)

            for rep in range(repeats):
                chosen = indices[sample_ids[:, rep]]
                avg_data_list.append(data[chosen].mean(axis=0))
                avg_targets_list.append(target_val)
                avg_chunks_list.append(chunk_val)

    return np.array(avg_data_list), np.array(avg_targets_list), np.array(avg_chunks_list)


# --------------------
# Cross-validation (generic)
# --------------------


def run_crossvalidation(data: np.ndarray,
                        targets: np.ndarray,
                        chunks: np.ndarray,
                        classify_fn: Callable,
                        n_folds=10) -> float:
    """
    Run n-fold cross-validated decoding with any classifier function.

    Args:
        data (np.ndarray): feature matrix (n_samples, n_features)
        targets (np.ndarray): class labels
        chunks (np.ndarray): fold assignments
        classify_fn: function(train_data, train_labels, test_data) -> predictions
        n_folds (int): number of folds

    Returns:
        float: classification accuracy
    """
    unique_chunks = np.unique(chunks)
    assert len(unique_chunks) == n_folds, (f"Expected {n_folds} unique chunks, got {len(unique_chunks)}")

    correct = 0
    total = 0
    for test_chunk in unique_chunks:
        test_mask = chunks == test_chunk
        train_mask = ~test_mask

        preds = classify_fn(data[train_mask], targets[train_mask], data[test_mask])
        correct += np.sum(preds == targets[test_mask])
        total += test_mask.sum()

    return correct / total


def run_searchlight_channel(data: np.ndarray,
                            targets: np.ndarray,
                            chunks: np.ndarray,
                            dist_matrix: np.ndarray,
                            classify_fn: Callable,
                            n_neighbours: int = 4,
                            n_folds: int = 10) -> np.ndarray:
    """
    Channel searchlight decoding using nearest neighbours.

    Args:
        data (np.ndarray): EEG data (n_samples, n_channels, n_timebins)
        targets (np.ndarray): class labels
        chunks (np.ndarray): cross-validation folds
        ch_names (list[str]): channel names
        dist_matrix (np.ndarray): pairwise channel distances
        classify_fn: classifier function
        n_neighbours (int): number of neighbours in searchlight
        n_folds (int): CV folds

    Returns:
        np.ndarray: accuracy (n_channels, n_timebins)
    """
    n_channels = data.shape[1]
    n_timebins = data.shape[2]
    sl_acc = np.zeros((n_channels, n_timebins))

    for c_idx in range(n_channels):
        dists = dist_matrix[c_idx].copy()
        dists[c_idx] = np.inf
        neigh_idx = np.argsort(dists)[:n_neighbours]
        all_idx = np.concatenate([[c_idx], neigh_idx])

        for t in range(n_timebins):
            sl_acc[c_idx, t] = run_crossvalidation(data[:, all_idx, t], targets, chunks, classify_fn, n_folds)

    return sl_acc


def compute_channel_distance_matrix(ch_names: list[str], montage: mne.channels.DigMontage) -> np.ndarray:
    """
    Compute Euclidean distance between EEG channels.

    Args:
        ch_names (list[str]): channel names
        montage (mne.channels.DigMontage): electrode positions

    Returns:
        np.ndarray: distance matrix (n_channels, n_channels)
    """
    ch_pos = montage.get_positions()["ch_pos"]
    coords = np.array([ch_pos[ch] for ch in ch_names])
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


# --------------------
# Data loading and preparation
# --------------------


def load_events(data_root: pathlib.Path, pair_id: int) -> pd.DataFrame:
    """Load the events TSV for a pair."""
    sub = f"sub-{pair_id:02d}"
    return pd.read_csv(data_root / sub / "eeg" / f"{sub}_task-RPS_events.tsv", sep="\t")


def build_behaviour_matrices(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build behavioural matrices for Player 1 and Player 2.

    Each matrix has one row per trial and the following columns:
        0: player's response (1=Rock, 2=Paper, 3=Scissors)
        1: opponent's response
        2: outcome relative to the player (1=draw, 2=player wins, 3=player loses)
        3: player's response on the previous trial
        4: opponent's response on the previous trial

    Args:
        events (pd.DataFrame): events table loaded from the dataset.

    Returns:
        tuple:
            player1_behav (np.ndarray): behavioural matrix for player 1
            player2_behav (np.ndarray): behavioural matrix for player 2
    """
    p1_resp = events["player1_resp"].values
    p2_resp = events["player2_resp"].values
    outcome = events["outcome"].values  # 1=draw, 2=p1 wins, 3=p2 wins

    # Player 1: self=p1, other=p2, outcome as-is
    p1_prev_self = np.concatenate([[np.nan], p1_resp[:-1]])
    p1_prev_other = np.concatenate([[np.nan], p2_resp[:-1]])
    player1_behav = np.column_stack([p1_resp, p2_resp, outcome, p1_prev_self, p1_prev_other])

    # Player 2: self=p2, other=p1, outcome recoded relative to p2
    p2_outcome = np.zeros(len(outcome), dtype=float)
    p2_outcome[outcome == 1] = 1  # draw
    p2_outcome[outcome == 2] = 3  # p1 wins -> p2 loses
    p2_outcome[outcome == 3] = 2  # p2 wins

    p2_prev_self = np.concatenate([[np.nan], p2_resp[:-1]])
    p2_prev_other = np.concatenate([[np.nan], p1_resp[:-1]])
    player2_behav = np.column_stack([p2_resp, p1_resp, p2_outcome, p2_prev_self, p2_prev_other])

    return player1_behav, player2_behav


def baseline_correct(data: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Subtract mean of the [-0.2, 0] s window from each channel and trial."""
    bl_mask = (times >= -0.2) & (times <= 0)
    if bl_mask.sum() > 0:
        bl_mean = data[:, :, bl_mask].mean(axis=2, keepdims=True)
        return data - bl_mean
    return data


def bin_data(data: np.ndarray, times: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """Average EEG data into time bins defined by windows (strict inequalities)."""
    n_trials, n_channels, _ = data.shape
    n_bins = windows.shape[0]
    binned = np.zeros((n_trials, n_channels, n_bins))

    for w in range(n_bins):
        t_mask = (times > windows[w, 0]) & (times < windows[w, 1])
        if t_mask.sum() > 0:
            binned[:, :, w] = data[:, :, t_mask].mean(axis=2)

    return binned


def epoch_to_timebinned_array(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert epochs to 250 ms time bins for decision, response and feedback phases.

    Args:
        epochs (mne.Epochs): preprocessed EEG epochs

    Returns:
        tuple:
            data (np.ndarray): binned EEG data (n_trials, n_channels, 20)
            time_labels (np.ndarray): time point for each bin
    """
    if not epochs.preload:
        epochs.load_data()
    full_data = epochs.get_data(copy=True)  # in Volts
    times = epochs.times

    # --- Split into 3 parts (with 200 ms overlap for baseline) ---
    mask_a = (times >= -0.2) & (times <= 2.0)
    mask_b = (times >= 1.8) & (times <= 4.0)
    mask_c = (times >= 3.8) & (times <= 5.0)

    data_a = baseline_correct(full_data[:, :, mask_a], times[mask_a])
    data_b = baseline_correct(full_data[:, :, mask_b], times[mask_b] - 2.0)
    data_c = baseline_correct(full_data[:, :, mask_c], times[mask_c] - 4.0)

    # Average into time bins
    binned_a = bin_data(data_a, times[mask_a], TIME_WINDOWS_AB)
    binned_b = bin_data(data_b, times[mask_b] - 2.0, TIME_WINDOWS_AB)
    binned_c = bin_data(data_c, times[mask_c] - 4.0, TIME_WINDOWS_C)

    data = np.concatenate([binned_a, binned_b, binned_c], axis=2)

    # Time labels (right edges, matching MATLAB)
    time_labels = np.concatenate([
      TIME_WINDOWS_AB[:, 1],  # Decision: 0.25, 0.50, ..., 2.00
      TIME_WINDOWS_AB[:, 1] + 2.0,  # Response: 2.25, 2.50, ..., 4.00
      TIME_WINDOWS_C[:, 1] + 4.0,  # Feedback: 4.25, 4.50, ..., 5.00
    ])

    return data, time_labels


def remove_block_first_trials(data: np.ndarray, behav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove the first trial of each block (previous-trial info undefined).

    Args:
        data (np.ndarray): EEG data
        behav (np.ndarray): behavioural matrix

    Returns:
        tuple: filtered data and behaviour arrays
    """
    keep_mask = np.ones(NUM_TRIALS, dtype=bool)
    keep_mask[np.arange(0, NUM_TRIALS, TRIALS_PER_BLOCK)] = False
    return data[keep_mask], behav[keep_mask]


def save_group_level(all_decoding: dict, time_labels: np.ndarray, output_dir: pathlib.Path) -> None:
    print("\nSaving group-level summaries...")
    for clf_key, (clf_display, _) in CLASSIFIERS.items():
        group_summary = {}
        for t in range(4):
            if all_decoding[clf_key][t]:
                group_summary[f"decoding_{t}"] = np.stack([d["accuracy"] for d in all_decoding[clf_key][t]])
                group_summary[f"decoding_{t}_pairs"] = np.array([d["pair"] for d in all_decoding[clf_key][t]])
                group_summary[f"decoding_{t}_players"] = np.array([d["player"] for d in all_decoding[clf_key][t]])

        group_summary["time_labels"] = time_labels
        group_path = output_dir / f"group_{clf_key}.npz"
        np.savez_compressed(group_path, **group_summary)
        print(f"  {clf_display} -> {group_path.name}")


def save_per_player_level(all_decoding: dict, all_searchlight: dict, time_labels: np.ndarray, ch_names: list[str],
                          pair: int, player_num: int, output_dir: pathlib.Path) -> None:
    for clf_key in CLASSIFIERS:
        out_path = (output_dir / f"pair-{pair:02d}_player-{player_num}_task-RPS_{clf_key}.npz")
        save_dict = {}
        for t in range(4):
            save_dict[f"decoding_acc_{t}"] = all_decoding[clf_key][t][-1]["accuracy"]
            save_dict[f"searchlight_acc_{t}"] = all_searchlight[clf_key][t][-1]["accuracy"]
        save_dict["time_labels"] = time_labels
        save_dict["ch_names"] = np.array(ch_names)
        save_dict["pair"] = pair
        save_dict["player"] = player_num
        np.savez_compressed(out_path, **save_dict)

    print(f"    Saved all classifiers for pair-{pair:02d}_player-{player_num}")


# --------------------
# Main pipeline
# --------------------


def run_decoding(data_root: pathlib.Path, preprocessed_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    """
    Run the full decoding pipeline for all pairs, players, and classifiers.

    For each participant, the data is loaded and prepared once. Then all
    four classifiers are run on the identical pseudo-trials and folds.
    Results are saved separately per classifier.
    """
    # Storage: clf_key -> target_idx -> list of results
    all_decoding = {clf_key: {t: [] for t in range(4)} for clf_key in CLASSIFIERS}
    all_searchlight = {clf_key: {t: [] for t in range(4)} for clf_key in CLASSIFIERS}

    time_labels = None

    for p_idx, pair in enumerate(PAIR_IDS):
        print(f"Loading pair {p_idx + 1} of {len(PAIR_IDS)} (pair ID {pair:02d})")

        # Load behavioural data
        events = load_events(data_root, pair)
        player1_behav, player2_behav = build_behaviour_matrices(events)
        all_behav = [player1_behav, player2_behav]

        for ppt in range(2):  # Player 1 and 2
            player_num = ppt + 1
            print(f"  Player {player_num}")

            # Load preprocessed epochs
            fif_path = (preprocessed_dir / f"pair-{pair:02d}_player-{player_num}_task-RPS_eeg_epo.fif")
            if not fif_path.exists():
                print(f"    Skipping: {fif_path.name} not found")
                continue

            epochs = mne.read_epochs(fif_path, preload=True, verbose=False)

            # Re-reference to average
            epochs.set_eeg_reference("average", projection=False, verbose=False)

            # Convert to time-binned array
            eeg_data, time_labels = epoch_to_timebinned_array(epochs)
            ch_names = list(epochs.ch_names)

            # Get behaviour for this player
            behav = all_behav[ppt].copy()

            # Remove first trial of each block
            eeg_data, behav = remove_block_first_trials(eeg_data, behav)

            # Compute channel distance matrix for searchlight
            dist_matrix = compute_channel_distance_matrix(ch_names, epochs.get_montage())

            # --- Loop over decode targets ---
            # 0 = own response (current), 1 = other's response (current)
            # 2 = own response (previous), 3 = other's response (previous)
            target_columns = [0, 1, 3, 4]  # columns in behav matrix
            target_names = ["self", "other", "self_prev", "other_prev"]

            for test_idx in range(4):
                col = target_columns[test_idx]
                targets = behav[:, col].copy()

                # Remove no-responses (target == 0) and NaN
                valid_mask = (~np.isnan(targets)) & (targets > 0)
                ds_data = eeg_data[valid_mask]
                ds_targets = targets[valid_mask].astype(int)

                # Assign chunks (10 balanced folds)
                chunks = cosmo_chunkize(ds_targets, n_chunks=10)

                # Average samples: 4 trials averaged, 20 repeats
                avg_data, avg_targets, avg_chunks = cosmo_average_samples(ds_data,
                                                                          ds_targets,
                                                                          chunks,
                                                                          count=4,
                                                                          repeats=20,
                                                                          seed=1)
                n_timebins = avg_data.shape[2]

                # --- Run all classifiers on identical data ---
                for clf_key, (clf_display, clf_fn) in CLASSIFIERS.items():

                    # Temporal decoding (all channels, per time bin)
                    temp_acc = np.zeros(n_timebins)
                    for t in range(n_timebins):
                        features = avg_data[:, :, t]
                        temp_acc[t] = run_crossvalidation(features,
                                                          avg_targets,
                                                          avg_chunks,
                                                          classify_fn=clf_fn,
                                                          n_folds=10)

                    print(f"    {target_names[test_idx]:>10s} | "
                          f"{clf_display:<30s} | "
                          f"mean acc = {temp_acc.mean() * 100:.1f}%")

                    # Store temporal decoding result
                    all_decoding[clf_key][test_idx].append({
                      "pair": pair,
                      "player": player_num,
                      "accuracy": temp_acc,
                      "time_labels": time_labels
                    })

                    sl_acc = run_searchlight_channel(avg_data,
                                                     avg_targets,
                                                     avg_chunks,
                                                     dist_matrix,
                                                     classify_fn=clf_fn,
                                                     n_neighbours=4,
                                                     n_folds=10)

                    all_searchlight[clf_key][test_idx].append({
                      "pair": pair,
                      "player": player_num,
                      "accuracy": sl_acc,
                      "ch_names": ch_names,
                      "time_labels": time_labels
                    })

            save_per_player_level(all_decoding, all_searchlight, time_labels, ch_names, pair, player_num, output_dir)

    save_group_level(all_decoding, time_labels, output_dir)

    print("Done.")
