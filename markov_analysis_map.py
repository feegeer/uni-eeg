import numpy as np
import pandas as pd
from pathlib import Path


def compute_markov_predictability(
    responses: np.ndarray,
    num_trials: int = 480,
    window_sizes=range(5, 101),
):
    """
    Compute first-order Markov Chain based or a single participant.
    
    """

    prob_data = np.full((num_trials, 13), np.nan)

    # Initial values 
    prob_data[0, :] = [1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1]

    for i in range(1, num_trials):
        prob_data[i, :] = prob_data[i - 1, :]
        prob_data[i, 0] = i + 1  # Trial number (1-based equivalent)

        prev_resp = responses[i - 1]
        curr_resp = responses[i]

        if prev_resp == 1:  # Rock
            prob_data[i, 1] += 1
            prob_data[i, 2 + (curr_resp - 1)] += 1

        elif prev_resp == 2:  # Paper
            prob_data[i, 5] += 1
            prob_data[i, 6 + (curr_resp - 1)] += 1

        elif prev_resp == 3:  # Scissors
            prob_data[i, 9] += 1
            prob_data[i, 10 + (curr_resp - 1)] += 1



    mean_accuracy = np.zeros(len(window_sizes))
    predictions = np.full((len(window_sizes), num_trials, 4), np.nan)

    for w_idx, window_size in enumerate(window_sizes):

        m_prob = np.full((3, 3), 1 / 3)
        prob_res = np.full((num_trials, 4), np.nan)

        for i in range(2, num_trials):  

            
            if i < window_size + 1:
                inter = prob_data[i - 1, :]
            else:
                inter = prob_data[i - 1, :] - prob_data[i - window_size, :]

            #  Compute conditional probabilities computed here

            # After Rock probability
            if inter[1] > 0:
                m_prob[0, :] = inter[2:5] / inter[1]
            else:
                m_prob[0, :] = 1 / 3

            # After Paper probabilty
            if inter[5] > 0:
                m_prob[1, :] = inter[6:9] / inter[5]
            else:
                m_prob[1, :] = 1 / 3

            # After Scissors
            if inter[9] > 0:
                m_prob[2, :] = inter[10:13] / inter[9]
            else:
                m_prob[2, :] = 1 / 3

            prob_res[i, 0] = responses[i]

            #  Missing response 
            if responses[i - 1] > 0:
                idx = i
            elif responses[i - 2] > 0:
                idx = i - 1
            else:
                idx = i - 2

            #  next step Prediction, get the max prabability to predct
            if idx > 1:
                last_resp = responses[idx - 1] - 1
                probs = m_prob[last_resp]

                pred_move = np.argmax(probs)
                prob_res[i, 1] = pred_move + 1
                prob_res[i, 2] = probs[pred_move]

            
            if np.isnan(prob_res[i, 2]):
                prob_res[i, 3] = np.nan
            elif prob_res[i, 0] == prob_res[i, 1]:
                prob_res[i, 3] = 1
            else:
                prob_res[i, 3] = 0

        # Mean accuracy (MATLAB 3:480)
        valid = prob_res[2:, 3]
        valid = valid[np.isfinite(valid)]
        mean_accuracy[w_idx] = np.mean(valid)

        predictions[w_idx, :, :] = prob_res

    return mean_accuracy, predictions



# Run on full data

def run_markov_analysis(data_path: Path):

    pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
    num_pairs = len(pair_ids)
    num_trials = 480
    window_sizes = range(5, 101)

    Mean_Accuracy = np.zeros((num_pairs, 2, len(window_sizes)))
    M_pred = np.zeros((num_pairs, 2, len(window_sizes), num_trials, 4))

    for p, pair in enumerate(pair_ids):
        print(f"Loading pair {p + 1} of {num_pairs}")

        events_path = (
            data_path
            / f"sub-{pair:02d}"
            / "eeg"
            / f"sub-{pair:02d}_task-RPS_events.tsv"
        )

        events = pd.read_csv(events_path, sep="\t")

        responses = np.column_stack(
            [
                events["player1_resp"].to_numpy(),
                events["player2_resp"].to_numpy(),
            ]
        )

        for ppt in range(2):

            mean_acc, pred = compute_markov_predictability(
                responses[:, ppt],
                num_trials=num_trials,
                window_sizes=window_sizes,
            )

            Mean_Accuracy[p, ppt, :] = mean_acc
            M_pred[p, ppt, :, :, :] = pred

    out_path = data_path / "derivatives"
    out_path.mkdir(exist_ok=True)

    np.savez(
        out_path / "markov_chain_pred.npz",
        Mean_Accuracy=Mean_Accuracy,
        M_pred=M_pred,
    )


if __name__ == "__main__":
    run_markov_analysis(Path("../data"))