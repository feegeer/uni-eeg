import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Tuple


def compute_markov_predictability(responses: np.ndarray,
                                  num_trials: int = 480,
                                  window_sizes: Iterable[int] = range(5, 101)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute first-order Markov prediction accuracy using sliding windows.
    Returns mean accuracy and trial-wise predictions.
    """
    responses = np.asarray(responses)

    # Build cumulative transition counts (vectorized)
    prob_data = np.zeros((num_trials, 13))
    prob_data[0] = [1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1]

    prev = responses[:-1]
    curr = responses[1:]
    valid = (prev > 0) & (curr > 0)

    trans_counts = np.zeros((num_trials, 3, 3))

    for p in range(3):
        mask_p = (prev == (p + 1)) & valid
        for c in range(3):
            trans_counts[1:, p, c][mask_p & (curr == (c + 1))] = 1

    trans_cumsum = np.cumsum(trans_counts, axis=0)
    prev_counts = np.sum(trans_counts, axis=2)
    prev_cumsum = np.cumsum(prev_counts, axis=0)

    prob_data[:, 1] = prev_cumsum[:, 0]
    prob_data[:, 5] = prev_cumsum[:, 1]
    prob_data[:, 9] = prev_cumsum[:, 2]

    prob_data[:, 2:5] = trans_cumsum[:, 0, :]
    prob_data[:, 6:9] = trans_cumsum[:, 1, :]
    prob_data[:, 10:13] = trans_cumsum[:, 2, :]

    # Add prior
    prob_data += prob_data[0] - prob_data[1]

    # Prediction phase
    n_windows = len(window_sizes)
    mean_accuracy = np.zeros(n_windows)
    predictions = np.full((n_windows, num_trials, 4), np.nan)

    for w_idx, window_size in enumerate(window_sizes):
        m_prob = np.full((3, 3), 1 / 3)
        prob_res = np.full((num_trials, 4), np.nan)

        for i in range(2, num_trials):
            if i < window_size + 1:
                inter = prob_data[i - 1]
            else:
                inter = prob_data[i - 1] - prob_data[i - window_size]

            # Transition probabilities
            m_prob[0] = inter[2:5] / inter[1] if inter[1] > 0 else 1 / 3
            m_prob[1] = inter[6:9] / inter[5] if inter[5] > 0 else 1 / 3
            m_prob[2] = inter[10:13] / inter[9] if inter[9] > 0 else 1 / 3

            prob_res[i, 0] = responses[i]

            # Handle missing responses
            if responses[i - 1] > 0:
                idx = i
            elif responses[i - 2] > 0:
                idx = i - 1
            else:
                idx = i - 2

            if idx > 1:
                last_resp = int(responses[idx - 1]) - 1
                probs = m_prob[last_resp]
                pred_move = np.argmax(probs)

                prob_res[i, 1] = pred_move + 1
                prob_res[i, 2] = probs[pred_move]

            # Accuracy
            if np.isnan(prob_res[i, 2]):
                prob_res[i, 3] = np.nan
            else:
                prob_res[i, 3] = float(prob_res[i, 0] == prob_res[i, 1])

        mean_accuracy[w_idx] = np.nanmean(prob_res[2:, 3])
        predictions[w_idx] = prob_res

    return mean_accuracy, predictions


def run_group_markov_analysis(data_path: Path, output_path: Path, window_sizes: Iterable[int] = range(5, 101)) -> None:
    """
    Run group-level Markov analysis and save results + plot.

    - Loads all subjects
    - Pools both players
    - Computes group mean + SEM
    - Saves plot (no display)
    """
    all_participants = []

    # Load all subjects
    for sub_folder in data_path.glob("sub-*"):
        tsv_files = list(sub_folder.glob("eeg/*events*.tsv"))
        if not tsv_files:
            continue

        try:
            events = pd.read_csv(tsv_files[0], sep="\t")
        except Exception as e:
            print(f"Skipping {sub_folder.name}: {e}")
            continue

        # Player 1
        resp1 = events["player1_resp"].to_numpy()
        mean_acc1, _ = compute_markov_predictability(resp1, window_sizes=window_sizes)
        all_participants.append(mean_acc1)

        # Player 2
        resp2 = events["player2_resp"].to_numpy()
        mean_acc2, _ = compute_markov_predictability(resp2, window_sizes=window_sizes)
        all_participants.append(mean_acc2)

    if len(all_participants) == 0:
        raise RuntimeError("No valid participants found.")

    all_participants = np.array(all_participants)

    # Group statistics
    group_mean = np.mean(all_participants, axis=0)
    group_sem = np.std(all_participants, axis=0) / np.sqrt(len(all_participants))
    chance = 1 / 3

    print("Total players:", len(all_participants))
    print("Grand mean predictability:", np.mean(group_mean))

    #  Plot (saved, not shown)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / "group_markov_predictability.png"

    plt.figure(figsize=(8, 5))

    # Individual subjects
    for subj in all_participants:
        plt.plot(window_sizes, subj, color="grey", alpha=0.2, linewidth=1)

    # Group mean
    plt.plot(window_sizes, group_mean, linewidth=3)

    # SEM shading
    plt.fill_between(window_sizes, group_mean - group_sem, group_mean + group_sem, alpha=0.15)

    # Chance line
    plt.axhline(chance, linestyle="--", linewidth=1)

    plt.xlabel("Window Size")
    plt.ylabel("Prediction Accuracy")
    plt.title("Group Markov Predictability")
    plt.tight_layout()

    plt.savefig(plot_file, dpi=300)
    plt.close()

    print(f"Plot saved to: {plot_file}")
