"""
EEG LDA Decoding Visualization
------------------------------
Group-level visualization of time-resolved decoding accuracy
for the Rock-Paper-Scissors EEG dataset.

Reads output from the Python LDA decoding pipeline (decoding.py),
which saves results as .npz files.

Two loading modes:
  1. Group file: group_decoding_results.npz (preferred, single file)
  2. Per-player files: pair-XX_player-X_task-RPS_decoding.npz

Main characteristics:
  - Phase-segmented visualization (Decision / Response / Feedback)
  - Phase-colored confidence bands (mean +/- SEM)
  - Chance-level reference line (33.33%)
"""

import pathlib
import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# CONFIGURATION
# ==========================================================

DECODING_METHOD = "lda" # lda or eegnet
DATA_VERSION = "v2"

# Path to derivatives directory containing decoding results
RESULTS_DIR = pathlib.Path(f"src/ds006761") / DATA_VERSION / DECODING_METHOD

# Chance level for 3-class decoding (Rock / Paper / Scissors)
CHANCE = 100 / 3

# Phase definitions: name -> (start_s, end_s) in the 0-5 s epoch
PHASES = {
    "Decision": (0, 2),
    "Response": (2, 4),
    "Feedback": (4, 5),
}

# Phase colours matching the paper's Figure 2
PHASE_COLORS = {
    "Decision": "#F5A623",  # orange
    "Response": "#D0021B",  # red
    "Feedback": "#9013FE",  # purple
}

# Decode target indices -> labels and subplot titles
TARGET_KEYS = [0, 1, 2, 3]
TARGET_LABELS = ["self", "other", "self_prev", "other_prev"]
TARGET_TITLES = {
    0: "(a) Own response",
    1: "(b) Opponent's response",
    2: "(c) Own previous response",
    3: "(d) Opponent's previous response",
}

# Y-axis limits (percentage)
Y_MIN, Y_MAX = 30, 40


# ==========================================================
# LOAD DECODING RESULTS
# ==========================================================

def load_from_group_file(results_dir: pathlib.Path) -> tuple[dict, np.ndarray]:
    """
    Load from the single group_decoding_results.npz file.

    Returns
    -------
    all_scores : dict mapping target index -> (n_participants, n_timebins) array in [0, 1]
    time_labels : (n_timebins,) array of time bin centres
    """
    group_path = results_dir / "group_decoding_results.npz"
    data = np.load(group_path, allow_pickle=True)

    time_labels = data["time_labels"]
    all_scores = {}
    for t in TARGET_KEYS:
        key = f"decoding_{t}"
        if key in data:
            all_scores[t] = data[key]  # (n_participants, 20), values in [0, 1]

    return all_scores, time_labels


def load_from_per_player_files(results_dir: pathlib.Path) -> tuple[dict, np.ndarray]:
    """
    Load from individual pair-XX_player-X_task-RPS_decoding.npz files.

    Returns
    -------
    all_scores : dict mapping target index -> (n_participants, n_timebins) array in [0, 1]
    time_labels : (n_timebins,) array of time bin centres
    """
    pattern = str(results_dir / "pair-*_player-*_task-RPS_decoding.npz")
    npz_files = sorted(glob.glob(pattern))
    print(f"Found {len(npz_files)} per-player result files.")

    all_scores = {t: [] for t in TARGET_KEYS}
    time_labels = None

    for fpath in npz_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            if time_labels is None:
                time_labels = data["time_labels"]
            for t in TARGET_KEYS:
                key = f"decoding_acc_{t}"
                if key in data:
                    all_scores[t].append(data[key])
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

    # Stack into arrays
    for t in TARGET_KEYS:
        if all_scores[t]:
            all_scores[t] = np.vstack(all_scores[t])
        else:
            all_scores[t] = np.empty((0, 20))

    return all_scores, time_labels


def load_results(results_dir: pathlib.Path) -> tuple[dict, np.ndarray]:
    """Try group file first, fall back to per-player files."""
    group_path = results_dir / "group_decoding_results.npz"
    if group_path.exists():
        print(f"Loading group file: {group_path}")
        return load_from_group_file(results_dir)
    else:
        print("Group file not found, loading per-player files...")
        return load_from_per_player_files(results_dir)


# ==========================================================
# COMPUTE TIME BIN CENTRES
# ==========================================================

def right_edges_to_centres(time_labels: np.ndarray) -> np.ndarray:
    """
    Convert right-edge time labels to bin centres.

    The decoding script stores right edges of 250 ms bins:
      [0.25, 0.50, ..., 2.00, 2.25, ..., 4.00, 4.25, ..., 5.00]

    Bin centres are 125 ms earlier:
      [0.125, 0.375, ..., 1.875, 2.125, ..., 3.875, 4.125, ..., 4.875]
    """
    return time_labels - 0.125


# ==========================================================
# VISUALIZATION
# ==========================================================

def plot_decoding_results(
    all_scores: dict,
    time_centres: np.ndarray,
    output_path: pathlib.Path | None = None,
) -> None:
    """
    Create a 2x2 figure of temporal decoding accuracy, matching the
    style of the paper's Figure 2.
    """
    sns.set_style("ticks")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, t in zip(axes, TARGET_KEYS):
        if t not in all_scores or all_scores[t].shape[0] == 0:
            ax.set_visible(False)
            continue

        # Data in percentage (decoding script stores proportions in [0, 1])
        data_pct = all_scores[t] * 100
        mean = np.mean(data_pct, axis=0)
        sem = np.std(data_pct, axis=0) / np.sqrt(data_pct.shape[0])

        n_participants = data_pct.shape[0]

        # Plot each phase with its own colour
        for phase_name, (t_start, t_end) in PHASES.items():
            mask = (time_centres >= t_start) & (time_centres <= t_end)
            t_phase = time_centres[mask]
            m_phase = mean[mask]
            s_phase = sem[mask]
            color = PHASE_COLORS[phase_name]

            # Confidence band
            ax.fill_between(
                t_phase,
                m_phase - s_phase,
                m_phase + s_phase,
                color=color,
                alpha=0.2,
                edgecolor="none",
            )

            # Mean line with markers
            ax.plot(
                t_phase,
                m_phase,
                color=color,
                lw=2.5,
                marker="o",
                markersize=6,
                markerfacecolor="white",
                markeredgewidth=1.5,
            )

            # Phase label at top
            ax.text(
                (t_start + t_end) / 2,
                Y_MAX - (Y_MAX * 0.02),
                phase_name,
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                color=color,
            )

        # Chance line
        ax.axhline(CHANCE, linestyle="--", color="#444444", lw=1.2, zorder=0)

        # Labels and formatting
        ax.set_title(
            TARGET_TITLES[t], loc="left", fontweight="bold", fontsize=14, pad=20
        )
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlim(-0.1, 5.1)
        ax.set_ylabel("Decoding accuracy (%)", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_yticks(
            np.arange(np.floor(Y_MIN / 5) * 5, np.ceil(Y_MAX / 5) * 5 + 5, 5)
        )

        sns.despine(ax=ax, offset=10, trim=True)

    # Add participant count to suptitle
    if all_scores and next(iter(all_scores.values())).shape[0] > 0:
        n = next(iter(all_scores.values())).shape[0]
        fig.suptitle(
            f"{DECODING_METHOD.upper()} temporal decoding accuracy (N = {n})",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

    plt.tight_layout()

    # Save
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")

    plt.show()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    all_scores, time_labels = load_results(RESULTS_DIR)

    if time_labels is None:
        raise FileNotFoundError(
            f"No decoding results found in {RESULTS_DIR}. "
            "Run decoding.py first."
        )

    time_centres = right_edges_to_centres(time_labels)

    n_loaded = {
        TARGET_LABELS[t]: all_scores[t].shape[0]
        for t in TARGET_KEYS
        if t in all_scores
    }
    print(f"Loaded participants per target: {n_loaded}")

    output_path = RESULTS_DIR / f"{DECODING_METHOD}_group_temporal_decoding.png"
    plot_decoding_results(all_scores, time_centres, output_path)