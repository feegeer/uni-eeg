import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from markov_analysis_map import compute_markov_predictability

# =====================================================
# DATA PATH
# =====================================================

DATA_PATH = Path("data/ds006761")

window_sizes = range(5, 101)
all_participants = []

# =====================================================
# LOAD ALL SUBJECTS (POOL PLAYER 1 + 2)
# =====================================================

for sub_folder in DATA_PATH.glob("sub-*"):

    tsv_files = list(sub_folder.glob("eeg/*events*.tsv"))
    if len(tsv_files) == 0:
        continue

    events = pd.read_csv(tsv_files[0], sep="\t")

    # Player 1
    resp1 = events["player1_resp"].to_numpy()
    mean_acc1, _ = compute_markov_predictability(resp1)
    all_participants.append(mean_acc1)

    # Player 2
    resp2 = events["player2_resp"].to_numpy()
    mean_acc2, _ = compute_markov_predictability(resp2)
    all_participants.append(mean_acc2)

all_participants = np.array(all_participants)

# =====================================================
# GROUP STATISTICS
# =====================================================

group_mean = np.mean(all_participants, axis=0)
group_sem = np.std(all_participants, axis=0) / np.sqrt(len(all_participants))

chance = 1 / 3

print("Total players:", len(all_participants))
print("Grand mean predictability:", np.mean(group_mean))

# =====================================================
# PLOT (Paper Style)
# =====================================================

plt.figure(figsize=(8, 5))

# --- Individual participants (grey, transparent) ---
for subj in all_participants:
    plt.plot(window_sizes, subj, color="grey", alpha=0.2, linewidth=1)

# --- Group mean (blue, bold) ---
plt.plot(window_sizes, group_mean, color="blue", linewidth=3)

# --- SEM shading (light blue) ---
plt.fill_between(window_sizes, group_mean - group_sem, group_mean + group_sem, color="blue", alpha=0.15)

# --- Chance line ---
plt.axhline(chance, linestyle="--", color="black", linewidth=1)

plt.xlabel("Window Size")
plt.ylabel("Prediction Accuracy")
plt.title("Group Markov Predictability")
plt.tight_layout()
plt.show()
