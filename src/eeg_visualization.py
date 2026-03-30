"""
Raincloud plots for behavioural analysis of the RPS dataset.
 
Generates Figures 1c, 1d, and 1e from the paper: game outcome
distributions, response biases, and game-to-game response changes.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def draw_raincloud_plot(data: list[list[float]],
                        feature_names: list[str],
                        plot_title: str,
                        colors: list[str],
                        yticks: list[float],
                        hline_height: float,
                        filepath: pathlib.Path | None = None) -> None:
    """
    Draw a vertical raincloud plot (box + half-violin + jittered scatter).
 
    Args:
        data:          list of distributions, one per category.
        feature_names: x-axis labels for each category.
        plot_title:    figure title.
        colors:        fill colour per category.
        yticks:        explicit y-axis tick positions.
        hline_height:  y-value for the dashed chance line.
        filepath:      if given, save and close; otherwise plt.show().
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Boxplot
    boxplot = ax.boxplot(data, sym="", widths=0.15, patch_artist=True, orientation="vertical")
    for patch, median in zip(boxplot['boxes'], boxplot['medians']):
        patch.set_facecolor("black")
        median.set_color("white")

    # Violin plot
    for index, feature in enumerate(data):
        kde = gaussian_kde(feature, bw_method=0.33)
        center = index + 1
        # extra margin for tails
        tail_extension = (np.max(feature) - np.min(feature)) * 0.2
        y = np.linspace(np.min(feature) - tail_extension, np.max(feature) + tail_extension, 500)
        density = kde(y)
        ax.fill_betweenx(y,
                         center,
                         center + density / density.max() * 0.4,
                         facecolor=colors[index],
                         alpha=0.5,
                         edgecolor="black")

    # Scatterplot
    rng = np.random.default_rng(seed=42)

    for idx, points in enumerate(data):
        n = len(points)
        jitter = rng.uniform(-0.03, 0.03, size=n)
        x = np.full(n, idx + 0.8) + jitter
        ax.scatter(x, points, s=10, c=colors[idx], edgecolors="black", linewidths=0.2)

    # Plots setup
    ax.axhline(hline_height, linestyle='--', color="black", linewidth=0.5)
    plt.xticks(range(1, len(feature_names) + 1), feature_names)
    plt.yticks(yticks)
    ax.set_ylim(bottom=yticks[0] - 2.5)
    plt.ylabel('Percentage')
    plt.title(plot_title)

    if filepath is not None:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def generate_plot_1c(subjects: list, output_path: pathlib.Path) -> None:
    """Figure 1c — distribution of game outcomes (winner wins / loses / draw)."""
    data = [[], [], []]

    for subject in subjects:
        pct_won, pct_lost, pct_drawn = subject.get_winners_outcome_distribution()
        data[0].append(pct_won)
        data[1].append(pct_lost)
        data[2].append(pct_drawn)

    draw_raincloud_plot(
      data,
      feature_names=["Winner wins", "Winner loses", "Draw"],
      plot_title="Game outcome",
      colors=["#2d708e", "#404788", "#481567"],
      yticks=list(range(20, 50, 5)),
      hline_height=100 / 3,
      filepath=output_path / "1c.png",
    )


def generate_plot_1d(subjects: list, output_path: pathlib.Path) -> None:
    """Figure 1d — response bias (most / mid / least played)."""
    data = [[], [], []]

    for subject in subjects:
        for player_counts in subject.get_most_mid_least_played_responses():
            counts = list(player_counts.values())
            total = sum(counts)
            pcts = [c * 100 / total for c in counts]
            data[0].append(pcts[0])
            data[1].append(pcts[1])
            data[2].append(pcts[2])

    draw_raincloud_plot(
      data,
      feature_names=["Most played", "Mid played", "Least played"],
      plot_title="Response played",
      colors=["#cb4149", "#f5dc4e", "#f78310"],
      yticks=list(range(20, 50, 5)),
      hline_height=100 / 3,
      filepath=output_path / "1d.png",
    )


def generate_plot_1e(subjects: list, output_path: pathlib.Path) -> None:
    """Figure 1e — game-to-game response change rate by prior outcome."""
    data = [[], [], []]

    for subject in subjects:
        after_win, after_loss, after_draw = subject.get_response_changes_distributions()
        data[0] += after_win
        data[1] += after_loss
        data[2] += after_draw

    draw_raincloud_plot(
      data,
      feature_names=["After win", "After loss", "After draw"],
      plot_title="Game-to-game response change",
      colors=["#2d708e", "#404788", "#481567"],
      yticks=list(range(20, 120, 20)),
      hline_height=200 / 3,
      filepath=output_path / "1e.png",
    )
