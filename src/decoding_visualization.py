"""
EEG LDA Decoding Visualization
------------------------------
Group-level visualization of time-resolved decoding accuracy
for the Rock-Paper-Scissors EEG dataset.

Main characteristics:
- Aggregates temporal decoding results across all pairs
- Computes group mean and standard error (SEM)
- Phase-segmented visualization (Decision / Response / Feedback)
- Phase-colored confidence bands
- Chance-level reference line (33.33%)

Data Source: OpenNeuro ds006761
Input: HDF5 decoding outputs from python_replication pipeline
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# CONFIGURATION
# ==========================================================

FILE_PREFIX = "pair-"  # matches new pipeline
chance = 100 / 3

times_reference = np.array([
  0.125, 0.375, 0.625, 0.875, # Decision
  1.125, 1.375, 1.625, 1.875, # Decision
  2.125, 2.375, 2.625, 2.875, # Response
  3.125, 3.375, 3.625, 3.875, # Response
  4.125, 4.375, 4.625, 4.875  # Feedback
])

PHASES = {'Decision': (0, 2), 'Response': (2, 4), 'Feedback': (4, 5)}

PHASE_COLORS = {'Decision': '#F5A623', 'Response': '#D0021B', 'Feedback': '#9013FE'}

results_dir = os.path.join("src", "ds006761", "derivatives", "python_replication")

# ==========================================================
# LOAD DECODING FILES
# ==========================================================

h5_files = [
  os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.startswith(FILE_PREFIX) and f.endswith(".h5")
]

print(f"Found {len(h5_files)} result files.")

# ==========================================================
# EXTRACT TEMPORAL DECODING SCORES
# ==========================================================

targets = ['self', 'other', 'selfp', 'otherp']

titles = {
  'self': "(a) Own response",
  'other': "(b) Opponent's response",
  'selfp': "(c) Own previous response",
  'otherp': "(d) Opponent's previous response"
}

all_scores = {t: [] for t in targets}

for file_path in h5_files:
    try:
        with h5py.File(file_path, 'r') as hf:
            for target in targets:
                if target not in hf:
                    continue
                
                grp = hf[target]
                if 'temporal_accuracy' in grp:
                    scores = grp['temporal_accuracy'][:]
                elif 'samples' in grp:
                    scores = grp['samples'][:]
                else:
                    continue

                all_scores[target].append(scores)

    except Exception as e:
        print(f"Skipping {file_path}: {e}")

# ==========================================================
# COMPUTE GROUP STATISTICS
# ==========================================================

all_means, all_sems = [], []

for target in targets:
    if len(all_scores[target]) > 0:
        data = np.vstack(all_scores[target]) * 100
        mean = np.mean(data, axis=0)
        sem = np.std(data, axis=0) / np.sqrt(data.shape[0])
        all_means.append(mean)
        all_sems.append(sem)

if all_means:
    global_max = max([(m + s).max() for m, s in zip(all_means, all_sems)])
    global_min = min([(m - s).min() for m, s in zip(all_means, all_sems)])

    y_max = max(global_max + 2, 45)
    y_min = min(global_min - 2, 30)
else:
    y_min, y_max = 30, 40

# Optional fixed scaling for consistency across figures
y_min, y_max = 30, 40

# ==========================================================
# VISUALIZATION
# ==========================================================

sns.set_style("ticks")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, target in zip(axes, targets):
    if len(all_scores[target]) == 0:
        continue

    data = np.vstack(all_scores[target]) * 100
    mean = np.mean(data, axis=0)
    sem = np.std(data, axis=0) / np.sqrt(data.shape[0])

    for phase_name, (t_start, t_end) in PHASES.items():

        mask = (times_reference >= t_start) & (times_reference <= t_end)

        t_phase = times_reference[mask]
        m_phase = mean[mask]
        s_phase = sem[mask]
        color = PHASE_COLORS[phase_name]

        ax.fill_between(
            t_phase, 
            m_phase - s_phase, 
            m_phase + s_phase, 
            color=color, 
            alpha=0.2, 
            edgecolor='none'
        )

        ax.plot(
            t_phase,
            m_phase,
            color=color,
            lw=2.5,
            marker='o',
            markersize=6,
            markerfacecolor='white',
            markeredgewidth=1.5)

        ax.text(
            (t_start + t_end) / 2,
            y_max - (y_max * 0.02),
            phase_name,
            ha='center',
            va='top',
            fontsize=11,
            fontweight='bold',
            color=color
        )

    ax.axhline(chance, linestyle="--", color="#444444", lw=1.2, zorder=0)

    ax.set_title(titles[target], loc='left', fontweight='bold', fontsize=14, pad=20)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.1, 5.1)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=12)

    ax.set_yticks(np.arange(np.floor(y_min / 5) * 5, np.ceil(y_max / 5) * 5 + 5, 5))

    sns.despine(ax=ax, offset=10, trim=True)

plt.tight_layout()

# ==========================================================
# SAVE AND SHOW FIGURE
# ==========================================================

output_path = os.path.join(results_dir, "group_temporal_decoding.png")

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Figure saved to: {output_path}")

plt.show()
