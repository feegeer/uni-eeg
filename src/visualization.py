"""
EEG LDA decoding visualization.

Plots group-level temporal decoding accuracy from the RPS EEG dataset.
The script reads results produced by the decoding pipeline and generates
a 2x2 figure showing accuracy over time for four decoding targets.

Input files:
    - group_decoding_results.npz (preferred), or
    - pair-XX_player-X_task-RPS_decoding.npz files.

The figure shows mean accuracy across participants with SEM confidence
bands, separated into the Decision, Response and Feedback phases.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# Configuration
# --------------------

DECODING_METHODS = {
  "LDA": "lda_cosmo",
  "Shrinkage LDA": "shrinkage_lda",
  "Linear SVM": "linear_svm",
  "Logistic Regression": "logistic_reg"
}

CHANCE = 100 / 3  # 3-class chance level

PHASES = {
  "Decision": (0, 2),
  "Response": (2, 4),
  "Feedback": (4, 5),
}

PHASE_COLORS = {
  "Decision": "#F5A623",
  "Response": "#D0021B",
  "Feedback": "#9013FE",
}

TARGET_TITLES = {
  0: "(a) Own response",
  1: "(b) Opponent's response",
  2: "(c) Own previous response",
  3: "(d) Opponent's previous response",
}

Y_MIN, Y_MAX = 30, 40

# --------------------
# Data Loading
# --------------------


def load_from_group_file(results_dir: pathlib.Path, method_fname: str) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Load decoding results from a single group summary .npz."""
    data = np.load(results_dir / f"group_{method_fname}.npz", allow_pickle=True)

    all_scores = {}
    for t in TARGET_TITLES:
        key = f"decoding_{t}"
        if key in data:
            all_scores[t] = data[key]

    return all_scores, data["time_labels"]


def load_from_per_player_files(results_dir: pathlib.Path,
                               method_fname: str) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Load and stack decoding results from individual pair-player .npz files."""
    npz_files = sorted(results_dir.glob(f"pair-*_player-*_task-RPS_{method_fname}.npz"))
    print(f"Found {len(npz_files)} per-player result files.")

    all_scores = {t: [] for t in TARGET_TITLES}
    time_labels = None

    for fpath in npz_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            if time_labels is None:
                time_labels = data["time_labels"]
            for t in TARGET_TITLES:
                key = f"decoding_acc_{t}"
                if key in data:
                    all_scores[t].append(data[key])
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

    stacked = {}
    for t in TARGET_TITLES:
        stacked[t] = np.vstack(all_scores[t]) if all_scores[t] else np.empty((0, 20))

    return stacked, time_labels


def load_results(results_dir: pathlib.Path, method_name: str) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Try group file first, fall back to per-player files."""
    method_fname = DECODING_METHODS[method_name]
    group_path = results_dir / f"group_{method_fname}.npz"

    if group_path.exists():
        print(f"Loading group file: {group_path}")
        return load_from_group_file(results_dir, method_fname)

    print("Group file not found, loading per-player files...")
    return load_from_per_player_files(results_dir, method_fname)


# --------------------
# Plotting
# --------------------


def plot_phase(ax: plt.Axes, time_centres: np.ndarray, mean: np.ndarray, sem: np.ndarray, phase_name: str,
               t_start: float, t_end: float) -> None:
    """Plot a single phase (Decision/Response/Feedback) on an axis."""
    mask = (time_centres >= t_start) & (time_centres <= t_end)
    t_phase = time_centres[mask]
    color = PHASE_COLORS[phase_name]

    ax.fill_between(t_phase, mean[mask] - sem[mask], mean[mask] + sem[mask], color=color, alpha=0.2, edgecolor="none")

    ax.plot(t_phase,
            mean[mask],
            color=color,
            lw=2.5,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.5)

    ax.text((t_start + t_end) / 2,
            Y_MAX - (Y_MAX * 0.02),
            phase_name,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=color)


def format_axis(ax: plt.Axes, target_idx: int) -> None:
    """Apply shared formatting to a single subplot."""
    ax.axhline(CHANCE, linestyle="--", color="#444444", lw=1.2, zorder=0)
    ax.set_title(TARGET_TITLES[target_idx], loc="left", fontweight="bold", fontsize=14, pad=20)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(-0.1, 5.1)
    ax.set_ylabel("Decoding accuracy (%)", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_yticks(np.arange(Y_MIN, Y_MAX + 1, 5))
    sns.despine(ax=ax, offset=10, trim=True)


def plot_decoding_results(all_scores: dict[int, np.ndarray],
                          time_centres: np.ndarray,
                          method_name: str,
                          output_path: pathlib.Path | None = None) -> None:
    """Create a 2x2 figure of temporal decoding accuracy (paper Figure 2 style)."""
    sns.set_style("ticks")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, t in zip(axes.flatten(), TARGET_TITLES):
        if t not in all_scores or all_scores[t].shape[0] == 0:
            ax.set_visible(False)
            continue

        data_pct = all_scores[t] * 100
        mean = data_pct.mean(axis=0)
        sem = data_pct.std(axis=0) / np.sqrt(data_pct.shape[0])

        for phase_name, (t_start, t_end) in PHASES.items():
            plot_phase(ax, time_centres, mean, sem, phase_name, t_start, t_end)

        format_axis(ax, t)

    n = next((s.shape[0] for s in all_scores.values() if s.shape[0] > 0), 0)
    if n > 0:
        fig.suptitle(
          f"{method_name} temporal decoding accuracy (N = {n})",
          fontsize=16,
          fontweight="bold",
          y=1.02,
        )

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")


def plot_all(results_dir: pathlib.Path, plots_dir: pathlib.Path | None = None) -> None:
    """
    Load and plot temporal decoding results for all classifier methods.
 
    Args:
        results_dir: directory containing the .npz decoding outputs.
        plots_dir: directory to save figures. Defaults to results_dir / "plots".
    """
    if plots_dir is None:
        plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for method_name, method_fname in DECODING_METHODS.items():
        all_scores, time_labels = load_results(results_dir, method_name)

        if time_labels is None:
            raise FileNotFoundError(f"No decoding results found in {results_dir}. Run decoding script first.")

        time_centres = time_labels - 0.125
        output_path = plots_dir / f"group_{method_fname}.png"
        plot_decoding_results(all_scores, time_centres, method_name, output_path)
