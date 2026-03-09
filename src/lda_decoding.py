"""
Decoding script (Python equivalent of the MATLAB decoding pipeline):
  - Decode own & opponent's response for current & previous trial
  - Uses MNE for EEG data handling, custom LDA matching CoSMoMVPA
  - Mirrors CoSMoMVPA's decoding pipeline from the original MATLAB code

Dependencies: mne, numpy, pandas

Notes:
  - We excluded pair 10 (major CMS issues for ppt 2), 23 (no triggers),
    and 24 (major CMS issues for ppt 2 - first 32 trials only)
  - The preprocessing Python script saves MNE Epochs as .fif files
  - This script expects preprocessed data in {data_root}/new_derivatives/
"""

import pathlib

import mne
import numpy as np
import pandas as pd

# Suppress MNE verbosity
mne.set_log_level("WARNING")


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
PATH_TO_DATA = pathlib.Path("src/ds006761")
PATH_TO_DERIVATIVES = PATH_TO_DATA / "v2"

PAIR_IDS = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
NUM_PAIRS = len(PAIR_IDS)
NUM_TRIALS = 480
NUM_CHAN = 64
NUM_BLOCKS = 12
TRIALS_PER_BLOCK = 40

# Time-bin edges (in seconds) used to average EEG within each phase
# Parts A & B: 0-2 s in 250 ms bins -> 8 bins each
TIME_WINDOWS_AB = np.column_stack(
    [np.arange(0, 2.0, 0.25), np.arange(0.25, 2.25, 0.25)]
)  # shape (8, 2)

# Part C: 0-1 s in 250 ms bins -> 4 bins
TIME_WINDOWS_C = np.column_stack(
    [np.arange(0, 1.0, 0.25), np.arange(0.25, 1.25, 0.25)]
)  # shape (4, 2)

# Total number of time bins: 8 (Decision) + 8 (Response) + 4 (Feedback) = 20
NUM_TIME_BINS = TIME_WINDOWS_AB.shape[0] * 2 + TIME_WINDOWS_C.shape[0]


# ---------------------------------------------------------------------------
# CoSMoMVPA-equivalent helper functions
# ---------------------------------------------------------------------------

def cosmo_sample_unique(
    k: int, n: int, count: int, seed: int | None = None
) -> np.ndarray:
    """
    Sample without replacement from 0:n-1 in a balanced manner.

    Mirrors CoSMoMVPA's cosmo_sample_unique exactly:
      - Returns a (k, count) array of indices in range [0, n-1]
      - Each column has no repeated values
      - Across the entire matrix, each value in 0:n-1 occurs
        approximately equally often

    The algorithm:
      1. Generate (count+1) random permutations of 0:n-1, concatenate
         into a flat vector.
      2. Walk through this vector, filling column by column, skipping
         values already used in the current column or already visited
         globally.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Generate (count+1) random permutations of 0:n-1, stacked as columns
    # CoSMoMVPA does: v = cosmo_rand(n, count+1, 'seed', s);
    #                 [~, r] = sort(v, 1);
    # This produces random permutations of 1:n in each column.
    rs_mat = np.zeros((n, count + 1), dtype=int)
    for c in range(count + 1):
        rs_mat[:, c] = rng.permutation(n)

    # Flatten column-major (Fortran order) to match MATLAB's rs_mat(:)
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

        # Update first_non_visited
        while first_non_visited < len(visited) and visited[first_non_visited]:
            first_non_visited += 1

    # Sort each column (matching MATLAB: samples = sort(samples, 1))
    samples.sort(axis=0)

    return samples


def cosmo_chunkize(targets: np.ndarray, n_chunks: int) -> np.ndarray:
    """
    Assign chunk labels (1..n_chunks) so that each chunk has roughly
    balanced class counts. Mirrors cosmo_chunkize in CoSMoMVPA.

    Since input chunks are all unique (one per trial), the function
    groups individual trials into n_chunks balanced folds. CoSMoMVPA
    does this by building a histogram of target counts per input chunk,
    then finding the best balanced partition.

    For single-trial input chunks (the case in this pipeline), this
    simplifies to: for each target class, distribute trials as evenly
    as possible across output chunks in sequential order (no shuffling,
    matching CoSMoMVPA's deterministic optimization).
    """
    n = len(targets)
    chunks = np.zeros(n, dtype=int)
    unique_targets = np.unique(targets)

    for t in unique_targets:
        idx = np.where(targets == t)[0]
        # Distribute sequentially across chunks (matching CoSMoMVPA's
        # find_best_assignment which assigns input chunks to output chunks
        # in order when each input chunk has exactly 1 sample)
        for i, ix in enumerate(idx):
            chunks[ix] = (i % n_chunks) + 1

    return chunks


def cosmo_average_samples(
    data: np.ndarray,
    targets: np.ndarray,
    chunks: np.ndarray,
    count: int = 4,
    repeats: int = 20,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create pseudo-trials by averaging `count` random samples of the same
    target and chunk, repeated `repeats` times.

    Mirrors cosmo_average_samples(ds, 'count', 4, 'repeats', 20, 'seed', 1).

    Uses cosmo_sample_unique internally to ensure balanced sampling:
    across all repeats, each original trial is used approximately
    the same number of times.

    Parameters
    ----------
    data : (n_trials, n_channels, n_timebins)
    targets : (n_trials,)
    chunks : (n_trials,)
    count : number of trials to average together
    repeats : how many pseudo-trials per (target, chunk) combination
    seed : random seed

    Returns
    -------
    avg_data : (n_pseudo, n_channels, n_timebins)
    avg_targets : (n_pseudo,)
    avg_chunks : (n_pseudo,)
    """
    unique_targets = np.unique(targets)
    unique_chunks = np.unique(chunks)

    # CoSMoMVPA splits by (targets, chunks), finds min bin count,
    # then determines nselect = count, nrepeat = repeats.
    # For each split, it calls cosmo_sample_unique(nselect, bin_count, nrepeat, 'seed', seed)

    avg_data_list = []
    avg_targets_list = []
    avg_chunks_list = []

    for chunk_val in unique_chunks:
        for target_val in unique_targets:
            mask = (targets == target_val) & (chunks == chunk_val)
            indices = np.where(mask)[0]
            bin_count = len(indices)
            if bin_count == 0:
                continue

            # Get balanced sampling indices: (count, repeats) array
            # Each column is a subset of count indices from 0:bin_count-1
            sample_ids = cosmo_sample_unique(count, bin_count, repeats, seed=seed)

            for rep in range(repeats):
                chosen = indices[sample_ids[:, rep]]
                avg_data_list.append(data[chosen].mean(axis=0))
                avg_targets_list.append(target_val)
                avg_chunks_list.append(chunk_val)

    return (
        np.array(avg_data_list),
        np.array(avg_targets_list),
        np.array(avg_chunks_list),
    )


def regularised_lda_classify(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    reg: float = 0.01,
) -> np.ndarray:
    """
    Regularised LDA classifier matching CoSMoMVPA's cosmo_classify_lda.

    CoSMoMVPA computes:
      1. Pooled within-class covariance: class_cov = sum_k (X_k - mu_k)' * (X_k - mu_k)
      2. Normalise: class_cov = class_cov / N_train
      3. Regularisation: reg_term = eye(p) * trace(class_cov) / max(1, p)
      4. class_cov_reg = class_cov + reg_term * regularization
      5. class_weight = class_mean / class_cov_reg  (i.e. class_mean @ inv(class_cov_reg))
      6. class_offset = sum(class_weight .* class_mean, axis=1)
      7. prediction = argmax(test @ class_weight' - 0.5 * class_offset)

    NOTE: This differs from scikit-learn's shrinkage LDA which uses a convex
    combination (1-lambda)*S + lambda*trace(S)/p*I. CoSMoMVPA instead *adds*
    the regularisation term: S + lambda*trace(S)/p*I.

    Parameters
    ----------
    train_data : (n_train, n_features)
    train_labels : (n_train,)
    test_data : (n_test, n_features)
    reg : regularisation parameter (default 0.01 matching MATLAB)

    Returns
    -------
    predictions : (n_test,) predicted class labels
    """
    classes = np.unique(train_labels)
    n_classes = len(classes)
    n_train, n_features = train_data.shape

    class_mean = np.zeros((n_classes, n_features))
    class_cov = np.zeros((n_features, n_features))

    for k, c in enumerate(classes):
        msk = train_labels == c
        class_samples = train_data[msk]
        class_mean[k] = class_samples.mean(axis=0)
        residuals = class_samples - class_mean[k]
        class_cov += residuals.T @ residuals

    # Normalise by total number of training samples (matching CoSMoMVPA)
    class_cov /= n_train

    # Regularisation: add lambda * trace(S)/p * I  (NOT convex combination)
    reg_term = np.eye(n_features) * np.trace(class_cov) / max(1, n_features)
    class_cov_reg = class_cov + reg_term * reg

    # Compute weights: class_weight = class_mean @ inv(class_cov_reg)
    class_weight = np.linalg.solve(class_cov_reg.T, class_mean.T).T

    # Offset (bias term)
    class_offset = np.sum(class_weight * class_mean, axis=1)

    # Predict: score = test @ weight' - 0.5 * offset
    scores = test_data @ class_weight.T - 0.5 * class_offset[None, :]
    predictions = classes[np.argmax(scores, axis=1)]

    return predictions


def run_crossvalidation(
    data: np.ndarray,
    targets: np.ndarray,
    chunks: np.ndarray,
    n_folds: int = 10,
    reg: float = 0.01,
) -> float:
    """
    N-fold cross-validation using regularised LDA on a single time-bin.

    Mirrors cosmo_crossvalidation_measure with cosmo_nfold_partitioner.

    Parameters
    ----------
    data : (n_samples, n_features) - features for one time bin
    targets : (n_samples,)
    chunks : (n_samples,) - fold assignments

    Returns
    -------
    accuracy : float in [0, 1]
    """
    unique_chunks = np.unique(chunks)
    assert len(unique_chunks) == n_folds, (
        f"Expected {n_folds} unique chunks, got {len(unique_chunks)}"
    )

    correct = 0
    total = 0

    for test_chunk in unique_chunks:
        test_mask = chunks == test_chunk
        train_mask = ~test_mask

        preds = regularised_lda_classify(
            data[train_mask], targets[train_mask],
            data[test_mask], reg=reg,
        )
        correct += np.sum(preds == targets[test_mask])
        total += test_mask.sum()

    return correct / total


def run_searchlight_channel(
    data: np.ndarray,
    targets: np.ndarray,
    chunks: np.ndarray,
    ch_names: list[str],
    dist_matrix: np.ndarray,
    n_neighbours: int = 4,
    n_folds: int = 10,
    reg: float = 0.01,
) -> np.ndarray:
    """
    Channel searchlight: for each channel, select it + its n_neighbours
    nearest neighbours and run cross-validated decoding.

    Mirrors cosmo_meeg_chan_neighborhood(ds, 'count', 4) combined with
    cosmo_searchlight. The 'count' parameter in CoSMoMVPA selects exactly
    the N nearest neighbours for each channel (not including self).

    Parameters
    ----------
    data : (n_samples, n_channels, n_timebins)
    targets, chunks : (n_samples,)
    ch_names : list of channel names
    dist_matrix : (n_channels, n_channels) pairwise distance matrix
    n_neighbours : number of nearest neighbours to include (default 4)
    n_folds : number of cross-validation folds
    reg : LDA regularisation parameter

    Returns
    -------
    sl_acc : (n_channels, n_timebins) - accuracy per channel per time bin
    """
    n_channels = len(ch_names)
    n_timebins = data.shape[2]
    sl_acc = np.zeros((n_channels, n_timebins))

    for c_idx in range(n_channels):
        # Get distances from this channel to all others
        dists = dist_matrix[c_idx].copy()
        dists[c_idx] = np.inf  # exclude self from neighbour ranking

        # Select exactly n_neighbours nearest channels
        neigh_sorted = np.argsort(dists)
        neigh_idx = neigh_sorted[:n_neighbours]

        # Include self + neighbours (matching CoSMoMVPA: self is always included)
        all_idx = np.concatenate([[c_idx], neigh_idx])

        for t in range(n_timebins):
            features = data[:, all_idx, t]
            sl_acc[c_idx, t] = run_crossvalidation(
                features, targets, chunks, n_folds=n_folds, reg=reg,
            )

    return sl_acc


# ---------------------------------------------------------------------------
# Distance matrix computation
# ---------------------------------------------------------------------------

def compute_channel_distance_matrix(
    ch_names: list[str],
    montage: mne.channels.DigMontage,
) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix between channels.

    Parameters
    ----------
    ch_names : list of channel names
    montage : MNE montage with 3D positions

    Returns
    -------
    dist_matrix : (n_channels, n_channels) symmetric distance matrix
    """
    pos = montage.get_positions()
    ch_pos = pos["ch_pos"]
    coords = np.array([ch_pos[ch] for ch in ch_names])
    dist_matrix = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :], axis=2
    )
    return dist_matrix


# ---------------------------------------------------------------------------
# Main data loading and epoching helpers
# ---------------------------------------------------------------------------

def load_events(data_root: pathlib.Path, pair_id: int) -> pd.DataFrame:
    """Load the events TSV for a pair."""
    sub = f"sub-{pair_id:02d}"
    events_path = data_root / sub / "eeg" / f"{sub}_task-RPS_events.tsv"
    return pd.read_csv(events_path, sep="\t")


def build_behaviour_matrices(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build behavioural data matrices for Player 1 and Player 2,
    matching the MATLAB code's format:

    For each player:
      Column 0: this player's response (1=Rock, 2=Paper, 3=Scissors)
      Column 1: other player's response
      Column 2: outcome (1=draw, 2=this player wins, 3=other player wins)
      Column 3: this player's previous response
      Column 4: other player's previous response
    """
    p1_resp = events["player1_resp"].values
    p2_resp = events["player2_resp"].values
    outcome = events["outcome"].values  # 1=draw, 2=p1 wins, 3=p2 wins

    n = len(p1_resp)

    # Player 1: self=p1, other=p2, outcome as-is
    p1_prev_self = np.concatenate([[np.nan], p1_resp[:-1]])
    p1_prev_other = np.concatenate([[np.nan], p2_resp[:-1]])
    player1_behav = np.column_stack([
        p1_resp, p2_resp, outcome, p1_prev_self, p1_prev_other
    ])

    # Player 2: self=p2, other=p1, outcome recoded relative to p2
    p2_outcome = np.zeros(n, dtype=float)
    p2_outcome[outcome == 1] = 1  # draw
    p2_outcome[outcome == 2] = 3  # p1 wins -> p2 loses
    p2_outcome[outcome == 3] = 2  # p2 wins

    p2_prev_self = np.concatenate([[np.nan], p2_resp[:-1]])
    p2_prev_other = np.concatenate([[np.nan], p1_resp[:-1]])
    player2_behav = np.column_stack([
        p2_resp, p1_resp, p2_outcome, p2_prev_self, p2_prev_other
    ])

    return player1_behav, player2_behav


def epoch_to_timebinned_array(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """
    Split epochs into Decision/Response/Feedback phases, apply baseline
    correction, and average into 250 ms time bins.

    Mirrors the MATLAB code's phase-splitting + time-binning logic.

    Parameters
    ----------
    epochs : MNE Epochs, tmin ~ -0.2005 s, tmax ~ 5.0 s, 256 Hz

    Returns
    -------
    data : (n_trials, n_channels, 20) - time-binned data
    time_labels : (20,) - right edge of each time bin (for reference)
    """
    if not epochs.preload:
        epochs.load_data()
    full_data = epochs.get_data(copy=True)  # in Volts
    times = epochs.times

    n_trials, n_channels, _ = full_data.shape

    # --- Split into 3 parts (with 200 ms overlap for baseline) ---
    # Part A (Decision): -0.2 to 2.0 s
    mask_a = (times >= -0.2) & (times <= 2.0)
    times_a = times[mask_a]
    data_a = full_data[:, :, mask_a]

    # Part B (Response): 1.8 to 4.0 s
    mask_b = (times >= 1.8) & (times <= 4.0)
    times_b = times[mask_b]
    data_b = full_data[:, :, mask_b]

    # Part C (Feedback): 3.8 to 5.0 s
    mask_c = (times >= 3.8) & (times <= 5.0)
    times_c = times[mask_c]
    data_c = full_data[:, :, mask_c]

    # --- Shift time labels so 0 = start of each phase ---
    times_b_shifted = times_b - 2.0  # 1.8->-0.2, 4.0->2.0
    times_c_shifted = times_c - 4.0  # 3.8->-0.2, 5.0->1.0

    # --- Baseline correction: subtract mean of [-0.2, 0] from each part ---
    def baseline_correct(data_part, times_part):
        bl_mask = (times_part >= -0.2) & (times_part <= 0)
        if bl_mask.sum() > 0:
            bl_mean = data_part[:, :, bl_mask].mean(axis=2, keepdims=True)
            return data_part - bl_mean
        return data_part

    data_a = baseline_correct(data_a, times_a)
    data_b = baseline_correct(data_b, times_b_shifted)
    data_c = baseline_correct(data_c, times_c_shifted)

    # --- Average into time bins ---
    def bin_data(data_part, times_part, windows):
        n_t, n_ch, _ = data_part.shape
        n_bins = windows.shape[0]
        binned = np.zeros((n_t, n_ch, n_bins))
        for w in range(n_bins):
            # Use strict inequalities to match MATLAB's > and <
            t_mask = (times_part > windows[w, 0]) & (times_part < windows[w, 1])
            if t_mask.sum() > 0:
                binned[:, :, w] = data_part[:, :, t_mask].mean(axis=2)
        return binned

    binned_a = bin_data(data_a, times_a, TIME_WINDOWS_AB)           # Decision
    binned_b = bin_data(data_b, times_b_shifted, TIME_WINDOWS_AB)   # Response
    binned_c = bin_data(data_c, times_c_shifted, TIME_WINDOWS_C)    # Feedback

    # Concatenate: 8 + 8 + 4 = 20 time bins
    data = np.concatenate([binned_a, binned_b, binned_c], axis=2)

    # Time labels (right edges, matching MATLAB)
    time_labels = np.concatenate([
        TIME_WINDOWS_AB[:, 1],           # Decision: 0.25, 0.50, ..., 2.00
        TIME_WINDOWS_AB[:, 1] + 2.0,     # Response: 2.25, 2.50, ..., 4.00
        TIME_WINDOWS_C[:, 1] + 4.0,      # Feedback: 4.25, 4.50, ..., 5.00
    ])

    return data, time_labels


def remove_block_first_trials(
    data: np.ndarray,
    behav: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove the first trial of each block (since previous-trial info
    is undefined there). Blocks are 40 trials each.

    Mirrors the MATLAB code: rem_idx = 1:40:480
    """
    rem_idx = np.arange(0, NUM_TRIALS, TRIALS_PER_BLOCK)  # 0-indexed
    keep_mask = np.ones(NUM_TRIALS, dtype=bool)
    keep_mask[rem_idx] = False

    return data[keep_mask], behav[keep_mask]


# ---------------------------------------------------------------------------
# Main decoding pipeline
# ---------------------------------------------------------------------------

def run_decoding() -> None:
    """
    Main decoding loop, mirroring the MATLAB script.
    Uses PATH_TO_DATA and PATH_TO_DERIVATIVES global constants.
    """
    PATH_TO_DERIVATIVES.mkdir(parents=True, exist_ok=True)
    path_to_lda_output = PATH_TO_DERIVATIVES / "lda"
    path_to_lda_output.mkdir(parents=True, exist_ok=True)

    # We'll collect all results across pairs/players
    all_decoding = {t: [] for t in range(4)}       # 4 decode targets
    all_searchlight = {t: [] for t in range(4)}

    for p_idx, pair in enumerate(PAIR_IDS):
        print(f"Loading pair {p_idx + 1} of {NUM_PAIRS} (pair ID {pair:02d})")

        # Load behavioural data
        events = load_events(PATH_TO_DATA, pair)
        player1_behav, player2_behav = build_behaviour_matrices(events)
        all_behav = [player1_behav, player2_behav]

        # Set random seed per pair (matching MATLAB: rng(p))
        rng = np.random.default_rng(p_idx + 1)

        for ppt in range(2):  # Player 1 and 2
            player_num = ppt + 1
            print(f"  Player {player_num}")

            # Load preprocessed epochs
            fif_path = (
                PATH_TO_DERIVATIVES
                / "preprocessed"
                / f"pair-{pair:02d}_player-{player_num}_task-RPS_eeg_epo.fif"
            )
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
            dist_matrix = compute_channel_distance_matrix(
                ch_names, epochs.get_montage()
            )

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
                # MATLAB: ds_sel.sa.chunks = (1:numel(ds_sel.sa.targets))';
                #         ds_sel.sa.chunks = cosmo_chunkize(ds_sel, 10);
                chunks = cosmo_chunkize(ds_targets, n_chunks=10)

                # Average samples: 4 trials averaged, 20 repeats
                # MATLAB: cosmo_average_samples(ds_sel, 'count', 4, 'repeats', 20, 'seed', 1)
                avg_data, avg_targets, avg_chunks = cosmo_average_samples(
                    ds_data, ds_targets, chunks,
                    count=4, repeats=20, seed=1,
                )

                # --- Temporal decoding (all channels, per time bin) ---
                # MATLAB: cosmo_interval_neighborhood(ds_sel, 'time', 'radius', 0)
                #         + cosmo_searchlight with cosmo_crossvalidation_measure
                n_timebins = avg_data.shape[2]
                temp_acc = np.zeros(n_timebins)

                for t in range(n_timebins):
                    features = avg_data[:, :, t]  # (n_pseudo, n_channels)
                    temp_acc[t] = run_crossvalidation(
                        features, avg_targets, avg_chunks,
                        n_folds=10, reg=0.01,
                    )

                print(
                    f"    Decode {target_names[test_idx]}: "
                    f"mean acc = {temp_acc.mean() * 100:.1f}%"
                )

                # Store temporal decoding result
                all_decoding[test_idx].append({
                    "pair": pair,
                    "player": player_num,
                    "accuracy": temp_acc,  # (20,)
                    "time_labels": time_labels,
                })

                # --- Channel searchlight ---
                # MATLAB: cosmo_meeg_chan_neighborhood(ds_sel, 'count', 4, ...)
                #         + cosmo_cross_neighborhood + cosmo_searchlight
                sl_acc = run_searchlight_channel(
                    avg_data, avg_targets, avg_chunks,
                    ch_names, dist_matrix,
                    n_neighbours=4,  # matching MATLAB 'count', 4
                    n_folds=10, reg=0.01,
                )

                all_searchlight[test_idx].append({
                    "pair": pair,
                    "player": player_num,
                    "accuracy": sl_acc,  # (n_channels, 20)
                    "ch_names": ch_names,
                    "time_labels": time_labels,
                })

            # Save per-player results
            # MATLAB: save(sprintf(...), 'decoding_accuracy', 'searchlight_acc')
            out_path = (
                path_to_lda_output
                / f"pair-{pair:02d}_player-{player_num}_task-RPS_decoding.npz"
            )
            save_dict = {}
            for t in range(4):
                save_dict[f"decoding_acc_{t}"] = all_decoding[t][-1]["accuracy"]
                save_dict[f"searchlight_acc_{t}"] = all_searchlight[t][-1]["accuracy"]
            save_dict["time_labels"] = time_labels
            save_dict["ch_names"] = np.array(ch_names)
            save_dict["pair"] = pair
            save_dict["player"] = player_num
            np.savez_compressed(out_path, **save_dict)
            print(f"    Saved -> {out_path.name}")

    # --- Save group-level summary ---
    print("\nSaving group-level summary...")
    group_summary = {}
    for t in range(4):
        if all_decoding[t]:
            group_summary[f"decoding_{t}"] = np.stack(
                [d["accuracy"] for d in all_decoding[t]]
            )
            group_summary[f"decoding_{t}_pairs"] = np.array(
                [d["pair"] for d in all_decoding[t]]
            )
            group_summary[f"decoding_{t}_players"] = np.array(
                [d["player"] for d in all_decoding[t]]
            )

    group_summary["time_labels"] = time_labels
    group_path = path_to_lda_output / "group_decoding_results.npz"
    np.savez_compressed(group_path, **group_summary)
    print(f"Saved group results -> {group_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_decoding()