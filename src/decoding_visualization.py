"""
EEG decoding visualization — temporal line plots + searchlight topomaps.

Reproduces the layout of Figure 2 from Moerel et al. (2025):
  - 2x2 grid for four decode targets (own, opponent, own prev, opponent prev)
  - Each panel: temporal decoding accuracy line plot (top) with SEM bands,
    plus a row of five 1-second-window searchlight topomaps (bottom).

The temporal line plot shows group-mean accuracy ± SEM across participants,
colour-coded by task phase (Decision, Response, Feedback).  The topomaps
show channel-level decoding accuracy averaged across participants and
collapsed from 250 ms bins into 1-second windows.

Requires: mne, numpy, matplotlib, seaborn
"""

import pathlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
import numpy as np
import seaborn as sns

mne.set_log_level("ERROR")

# Configuration:

DECODING_METHODS = {
    "LDA": "lda_cosmo",
    "Shrinkage LDA": "shrinkage_lda",
    "Linear SVM": "linear_svm",
    "Logistic Regression": "logistic_reg",
}

PAIR_IDS = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))

CHANCE = 100 / 3  # 3-class chance level (%)

PHASES = {"Decision": (0, 2), "Response": (2, 4), "Feedback": (4, 5)}
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

WINDOW_LABELS = ["0-1 s", "1-2 s", "2-3 s", "3-4 s", "4-5 s"]

Y_MIN, Y_MAX = 30, 40


#  Data loading


def load_decoding_results(
    results_dir: pathlib.Path,
    method_name: str,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Load temporal decoding results — group file first, per-player fallback.

    Returns:
        scores: {target_idx: array (n_participants, 20)}
        time_labels: array of 20 time-bin right-edge labels
    """
    fname = DECODING_METHODS[method_name]
    group_path = results_dir / f"group_{fname}.npz"

    if group_path.exists():
        data = np.load(group_path, allow_pickle=True)
        scores = {
            t: data[f"decoding_{t}"]
            for t in TARGET_TITLES
            if f"decoding_{t}" in data
        }
        return scores, data["time_labels"]

    # Fallback: aggregate from per-player files
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

    scores = {
        t: np.vstack(v) if v else np.empty((0, 20))
        for t, v in per_target.items()
    }
    return scores, time_labels


def load_searchlight_results(
    results_dir: pathlib.Path,
    method_name: str,
) -> tuple[dict[int, np.ndarray], list[str]]:
    """
    Load searchlight accuracies from per-player .npz files.

    Returns:
        sl_scores: {target_idx: array (n_participants, n_channels, 20)}
        ch_names:  channel name list from the first file found.
    """
    fname = DECODING_METHODS[method_name]
    npz_files = sorted(results_dir.glob(f"pair-*_player-*_task-RPS_{fname}.npz"))

    per_target: dict[int, list] = {t: [] for t in TARGET_TITLES}
    ch_names = None

    for fpath in npz_files:
        data = np.load(fpath, allow_pickle=True)
        if ch_names is None:
            ch_names = list(data["ch_names"])
        for t in TARGET_TITLES:
            key = f"searchlight_acc_{t}"
            if key in data:
                per_target[t].append(data[key])

    sl_scores = {
        t: np.stack(v, axis=0) if v else None
        for t, v in per_target.items()
    }
    return sl_scores, ch_names


def collapse_to_1s_windows(acc: np.ndarray) -> np.ndarray:
    """
    Average 20 x 250 ms time bins into 5 x 1-second windows.

    Bin mapping:
        0-3   → Decision  0-1 s
        4-7   → Decision  1-2 s
        8-11  → Response  2-3 s
        12-15 → Response  3-4 s
        16-19 → Feedback  4-5 s
    """
    original_shape = acc.shape[:-1]
    return acc.reshape(*original_shape, 5, 4).mean(axis=-1)


def make_mne_info(ch_names: list[str]) -> mne.Info:
    """Create MNE Info with BioSemi-64 standard positions."""
    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types="eeg")
    montage = mne.channels.make_standard_montage("biosemi64")
    info.set_montage(montage, on_missing="warn")
    return info


#  Plotting ─


def plot_decoding_with_topomaps(
    temporal_scores: dict[int, np.ndarray],
    sl_scores: dict[int, np.ndarray | None],
    time_centres: np.ndarray,
    ch_names: list[str] | None,
    method_name: str,
    output_path: pathlib.Path | None = None,
) -> plt.Figure:
    """
    Create Figure-2-style 2x2 plot: line plots + searchlight topomaps.

    Each of the four panels (one per decode target) consists of:
      • Top row:    temporal decoding accuracy ± SEM, colour-coded by phase.
      • Bottom row: five topographic maps (1-s windows) of searchlight accuracy.

    Args:
        temporal_scores: {target: (n_ppt, 20)} temporal decoding accuracies.
        sl_scores:       {target: (n_ppt, n_ch, 20)} searchlight accuracies,
                         or None per target if unavailable.
        time_centres:    time axis for the line plot (20 values).
        ch_names:        EEG channel names for topomap plotting.
        method_name:     display name of the classifier (for suptitle).
        output_path:     optional path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    sns.set_style("ticks")
    has_sl = ch_names is not None and any(v is not None for v in sl_scores.values())

    #  Build the MNE Info once 
    info = make_mne_info(ch_names) if has_sl else None

    #  Compute shared colour limits across all targets 
    # This ensures the topomaps are comparable across panels.
    sl_vmin = CHANCE
    sl_vmax = CHANCE + 1.0  # fallback
    if has_sl:
        all_maxes = []
        for t in TARGET_TITLES:
            arr = sl_scores.get(t)
            if arr is not None:
                group_1s = collapse_to_1s_windows(arr.mean(axis=0))
                all_maxes.append(group_1s.max() * 100)
        if all_maxes:
            sl_vmax = max(max(all_maxes), sl_vmin + 1.0)

    #  Figure geometry 
    # Each panel: line plot (tall) + topomap row (short).
    # Outer grid: 2 rows x 2 columns of panels.
    # Inner grid per panel: 2 rows (line plot height ≈ 3x topo height).

    line_h, topo_h = 3, 1.2  # relative heights
    panel_h = line_h + topo_h
    fig = plt.figure(figsize=(14, 2 * panel_h + 3.5))

    outer = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.65,
        wspace=0.30,
    )

    target_list = list(TARGET_TITLES.keys())

    for panel_idx, t in enumerate(target_list):
        row, col = divmod(panel_idx, 2)

        # Inner grid: line plot on top, topomaps on bottom
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[row, col],
            height_ratios=[line_h, topo_h],
            hspace=0.55,
        )

        #  Line plot 
        ax_line = fig.add_subplot(inner[0])
        _draw_temporal_line(ax_line, temporal_scores.get(t), time_centres, t)

        #  Topomaps 
        topo_inner = gridspec.GridSpecFromSubplotSpec(
            1, 5,
            subplot_spec=inner[1],
            wspace=0.08,
        )
        topo_axes = [fig.add_subplot(topo_inner[0, c]) for c in range(5)]

        arr = sl_scores.get(t) if has_sl else None
        if arr is not None:
            group_mean_1s = collapse_to_1s_windows(arr.mean(axis=0))  # (n_ch, 5)
            _draw_topomaps(topo_axes, group_mean_1s, info, sl_vmin, sl_vmax)
        else:
            for ax in topo_axes:
                ax.axis("off")
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        fontsize=8, color="grey", transform=ax.transAxes)

    #  Shared colourbar for all topomaps 
    if has_sl:
        cbar_ax = fig.add_axes([0.25, 0.015, 0.50, 0.015])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(sl_vmin, sl_vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Searchlight decoding accuracy (%)", fontsize=10)

    #  Super-title 
    n = next((s.shape[0] for s in temporal_scores.values() if s.shape[0] > 0), 0)
    fig.suptitle(
        f"{method_name} — temporal decoding & channel searchlight  (N = {n})",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def _draw_temporal_line(
    ax: plt.Axes,
    scores: np.ndarray | None,
    time_centres: np.ndarray,
    target_idx: int,
) -> None:
    """Draw the temporal decoding line plot on a single Axes."""
    if scores is None or scores.shape[0] == 0:
        ax.set_visible(False)
        return

    data_pct = scores * 100
    mean = data_pct.mean(axis=0)
    sem = data_pct.std(axis=0) / np.sqrt(data_pct.shape[0])

    for name, (t0, t1) in PHASES.items():
        m = (time_centres >= t0) & (time_centres <= t1)
        c = PHASE_COLORS[name]
        ax.fill_between(
            time_centres[m],
            mean[m] - sem[m],
            mean[m] + sem[m],
            color=c, alpha=0.2, edgecolor="none",
        )
        ax.plot(
            time_centres[m], mean[m],
            color=c, lw=2.5,
            marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.5,
        )
        ax.text(
            (t0 + t1) / 2, Y_MAX * 0.98, name,
            ha="center", va="top",
            fontsize=10, fontweight="bold", color=c,
        )

    ax.axhline(CHANCE, ls="--", color="#444", lw=1.2, zorder=0)
    ax.set_title(TARGET_TITLES[target_idx], loc="left", fontweight="bold", fontsize=13, pad=12)
    ax.set(
        ylim=(Y_MIN, Y_MAX),
        xlim=(-0.1, 5.1),
        ylabel="Decoding accuracy (%)",
        xlabel="",
        yticks=np.arange(Y_MIN, Y_MAX + 1, 5),
    )
    sns.despine(ax=ax, offset=8, trim=True)


def _draw_topomaps(
    axes: list[plt.Axes],
    group_mean_1s: np.ndarray,
    info: mne.Info,
    vmin: float,
    vmax: float,
) -> None:
    """Draw five searchlight topomaps (one per 1-s window)."""
    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(
            group_mean_1s[:, i] * 100,
            info,
            axes=ax,
            cmap="inferno",
            vlim=(vmin, vmax),
            show=False,
            contours=0,
            sensors=False,
        )
        ax.set_title(WINDOW_LABELS[i], fontsize=8, pad=2)


#  Public API ─


def plot_all_decoding(
    results_dir: pathlib.Path,
    plots_dir: pathlib.Path | None = None,
) -> None:
    """Load and plot temporal + searchlight results for every classifier."""
    if plots_dir is None:
        plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for method_name, method_fname in DECODING_METHODS.items():
        temporal_scores, time_labels = load_decoding_results(results_dir, method_name)
        sl_scores, ch_names = load_searchlight_results(results_dir, method_name)

        if time_labels is None:
            print(f"[skip] No results for {method_name} in {results_dir}")
            continue

        # time_labels are right edges of 250 ms bins; shift to bin centres
        time_centres = time_labels - 0.125

        plot_decoding_with_topomaps(
            temporal_scores,
            sl_scores,
            time_centres,
            ch_names,
            method_name,
            output_path=plots_dir / f"group_{method_fname}.png",
        )
        plt.close("all")

    print("All plots saved.")
