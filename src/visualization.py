"""
EEG decoding visualization.
 
Plots group-level temporal decoding accuracy (2x2 figure with SEM bands)
for four decode targets across Decision, Response, and Feedback phases.
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

CHANCE = 100 / 3
PHASES = {"Decision": (0, 2), "Response": (2, 4), "Feedback": (4, 5)}
PHASE_COLORS = {"Decision": "#F5A623", "Response": "#D0021B", "Feedback": "#9013FE"}
TARGET_TITLES = {
  0: "(a) Own response",
  1: "(b) Opponent's response",
  2: "(c) Own previous response",
  3: "(d) Opponent's previous response"
}
Y_MIN, Y_MAX = 30, 40

# --------------------
# Data Loading
# --------------------


def load_decoding_results(results_dir: pathlib.Path, method_name: str) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Load decoding results — tries group file first, falls back to per-player files."""
    fname = DECODING_METHODS[method_name]
    group_path = results_dir / f"group_{fname}.npz"

    if group_path.exists():
        data = np.load(group_path, allow_pickle=True)
        scores = {t: data[f"decoding_{t}"] for t in TARGET_TITLES if f"decoding_{t}" in data}
        return scores, data["time_labels"]

    npz_files = sorted(results_dir.glob(f"pair-*_player-*_task-RPS_{fname}.npz"))
    per_target: dict[int, list] = {t: [] for t in TARGET_TITLES}
    time_labels = None

    for fpath in npz_files:
        data = np.load(fpath, allow_pickle=True)
        if time_labels is None:
            time_labels = data["time_labels"]
        for t in TARGET_TITLES:
            if f"decoding_acc_{t}" in data:
                per_target[t].append(data[f"decoding_acc_{t}"])

    scores = {t: np.vstack(v) if v else np.empty((0, 20)) for t, v in per_target.items()}
    return scores, time_labels


# --------------------
# Plotting
# --------------------


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

        for name, (t0, t1) in PHASES.items():
            m = (time_centres >= t0) & (time_centres <= t1)
            c = PHASE_COLORS[name]
            ax.fill_between(time_centres[m], mean[m] - sem[m], mean[m] + sem[m], color=c, alpha=0.2, edgecolor="none")
            ax.plot(time_centres[m],
                    mean[m],
                    color=c,
                    lw=2.5,
                    marker="o",
                    markersize=6,
                    markerfacecolor="white",
                    markeredgewidth=1.5)
            ax.text((t0 + t1) / 2, Y_MAX * 0.98, name, ha="center", va="top", fontsize=11, fontweight="bold", color=c)

        ax.axhline(CHANCE, ls="--", color="#444", lw=1.2, zorder=0)
        ax.set_title(TARGET_TITLES[t], loc="left", fontweight="bold", fontsize=14, pad=20)
        ax.set(ylim=(Y_MIN, Y_MAX),
               xlim=(-0.1, 5.1),
               ylabel="Decoding accuracy (%)",
               xlabel="Time (s)",
               yticks=np.arange(Y_MIN, Y_MAX + 1, 5))
        sns.despine(ax=ax, offset=10, trim=True)

    n = next((s.shape[0] for s in all_scores.values() if s.shape[0] > 0), 0)
    if n > 0:
        fig.suptitle(f"{method_name} temporal decoding accuracy (N = {n})", fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_all_decoding(results_dir: pathlib.Path, plots_dir: pathlib.Path | None = None) -> None:
    """Load and plot temporal decoding results for all classifier methods."""
    if plots_dir is None:
        plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for method_name, method_fname in DECODING_METHODS.items():
        all_scores, time_labels = load_decoding_results(results_dir, method_name)
        if time_labels is None:
            raise FileNotFoundError(f"No results in {results_dir}. Run decoding first.")
        plot_decoding_results(all_scores, time_labels - 0.125, method_name, plots_dir / f"group_{method_fname}.png")
