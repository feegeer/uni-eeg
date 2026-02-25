import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

# -------------------------------------------------
# SETTINGS & STYLING
# -------------------------------------------------
MODEL_TYPE = "lda"
chance = 100 / 3  # Reference shows ~33.3% (1/3)

# Define the three phases based on the image
PHASES = {
  'Decision': (0, 2),  # Ended at 1.9 previously
  'Response': (2, 4),  # Started at 2.1 previously
  'Feedback': (4, 5)  # Started at 4.1 previously
}

PHASE_COLORS = {'Decision': '#F5A623', 'Response': '#D0021B', 'Feedback': '#9013FE'}

# Match colors from the reference
PHASE_COLORS = {
  'Decision': '#F5A623',  # Orange-ish
  'Response': '#D0021B',  # Red-ish
  'Feedback': '#9013FE'  # Purple-ish
}

results_dir = os.path.join("src", "ds006761", "derivatives", "decoding_results")

h5_files = [
  os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.startswith(MODEL_TYPE) and f.endswith(".h5")
]

print(f"Found {len(h5_files)} files for model: {MODEL_TYPE}")

targets = ['self', 'other', 'selfp', 'otherp']
titles = {
  'self': "(a) Own response",
  'other': "(b) Opponent's response",
  'selfp': "(c) Own previous response",
  'otherp': "(d) Opponent's previous response"
}

all_scores = {t: [] for t in targets}
times_reference = None

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
for file_path in h5_files:
    with h5py.File(file_path, 'r') as hf:
        times = hf['times'][:]
        for target in targets:
            if target in hf:
                scores = hf[target]['samples'][:]
                min_len = min(len(times), len(scores))
                scores = scores[:min_len]
                if times_reference is None:
                    times_reference = times[:min_len]
                else:
                    scores = scores[:len(times_reference)]
                all_scores[target].append(scores)

# -------------------------------------------------
# DETERMINE DYNAMIC Y-LIMITS
# -------------------------------------------------
# Convert all data to percentage and find global min/max
all_data = []
for target in targets:
    if len(all_scores[target]) == 0:
        continue
    data = np.vstack(all_scores[target]) * 100
    all_data.append(data)

if all_data:
    combined = np.vstack(all_data)
    y_min, y_max = combined.min(), combined.max()
    padding = (y_max - y_min) * 0.1  # 10% padding
    y_min -= padding
    y_max += padding
else:
    y_min, y_max = 25, 60  # fallback

# -------------------------------------------------
# PLOTTING
# -------------------------------------------------
sns.set_style("ticks")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for ax, target in zip(axes, targets):
    if len(all_scores[target]) == 0:
        continue

    data = np.vstack(all_scores[target]) * 100
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    for phase_name, (t_start, t_end) in PHASES.items():
        mask = (times_reference >= t_start) & (times_reference <= t_end)
        t_phase = times_reference[mask]
        m_phase = mean[mask]
        s_phase = std[mask]
        color = PHASE_COLORS[phase_name]

        # Background shading and label
        ax.axvspan(t_start, t_end, color='gray', alpha=0.05)
        ax.text((t_start + t_end) / 2,
                y_max - (y_max - y_min) * 0.05,
                phase_name,
                ha='center',
                fontsize=14,
                fontweight='bold',
                alpha=1.0)

        # Shaded error and mean line
        ax.fill_between(t_phase, m_phase - s_phase, m_phase + s_phase, color=color, alpha=0.2)
        ax.plot(t_phase,
                m_phase,
                color=color,
                lw=1.5,
                marker='o',
                markersize=5,
                markerfacecolor='none',
                markeredgewidth=1)

    # Chance line
    ax.axhline(chance, linestyle="--", color="black", lw=1, zorder=1)

    # Axes styling
    ax.set_title(titles[target], loc='left', fontweight='bold', fontsize=18)
    ax.set_ylabel("Decoding accuracy (%)")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([0, 1, 2, 3, 4, 5])

    sns.despine(ax=ax, offset=10)

plt.tight_layout()

# -------------------------------------------------
# SAVE AND SHOW FIGURE
# -------------------------------------------------
output_dir = os.path.join(results_dir, "figures")
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"temporal_decoding_{MODEL_TYPE}.png")
fig.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figure saved as: {save_path}")

plt.show()
