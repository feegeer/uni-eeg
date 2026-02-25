import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_1samp

from markov_analysis_map import compute_markov_predictability



def validate_single_subject(tsv_path: Path, player: int = 1):


    print(f"\nLoading file: {tsv_path.name}")

    # Load TSV
    events = pd.read_csv(tsv_path, sep="\t")

    # Select player
    if player == 1:
        responses = events["player1_resp"].to_numpy()
    else:
        responses = events["player2_resp"].to_numpy()

    print("Number of trials:", len(responses))
    print("Unique responses:", np.unique(responses))

    # Run Markov analysis
    mean_acc, predictions = compute_markov_predictability(responses)

    print("\nAccuracy per window (5–100):")
    print(mean_acc)

    avg_acc = np.mean(mean_acc)
    print("\nAverage predictability:", round(avg_acc, 4))

    # Statistical test vs chance for the move(33.3%)
    chance = 1 / 3
    t_stat, p_val = ttest_1samp(mean_acc, chance)

    print("\nTest vs chance (33.3%)")
    print("t =", round(t_stat, 4))
    print("p =", round(p_val, 6))

    # Plot the res 
    window_sizes = range(5, 101)

    plt.figure(figsize=(8, 5))
    plt.plot(window_sizes, mean_acc, linewidth=2)
    plt.axhline(chance, linestyle="--")
    plt.xlabel("Window Size")
    plt.ylabel("Prediction Accuracy")
    plt.title(f"Markov Predictability ({tsv_path.name}, Player {player})")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    # Adjust if needed but i fistly test with one file
    tsv_file = Path("sub-01_task-RPS_events.tsv")

    validate_single_subject(tsv_file, player=1)