import pathlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def draw_raincloud_plot(data: list[list[float]],
                        feature_names: list[str],
                        plot_title: str,
                        colors: list[str],
                        yticks: list[float],
                        hline_height: float,
                        filepath: None | pathlib.Path = None) -> None:

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
    for index, points in enumerate(data):
        x = np.full(len(points), index + .8)
        idxs = np.arange(len(x))
        out = x.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.03, high=.03, size=len(idxs))
        x = out
        plt.scatter(x, points, s=10, c=colors[index], edgecolors="black", linewidths=0.2)
    ax.axhline(hline_height, linestyle='--', color="black", linewidth=0.5)

    plt.xticks([1, 2, 3], feature_names)
    plt.yticks(yticks)
    ax.set_ylim(bottom=yticks[0] - 2.5)
    plt.ylabel('Percentage')
    plt.title(plot_title)
    
    if filepath is not None:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def generate_plot_1c(subjects, output_path):
    output_file = output_path / "1c.png"
    data = [[], [], []]
    for subject in subjects:
        percent_won, percent_lost, percent_drawn = subject.get_winners_outcome_distribution()
        data[0].append(percent_won)
        data[1].append(percent_lost)
        data[2].append(percent_drawn)

    labels = ["Winner wins", "Winner looses", "Draw"]
    colors = ["#2d708e", "#404788", "#481567"]
    yticks = list(range(20, 50, 5))

    draw_raincloud_plot(data, labels, "Game outcome", colors, yticks, 100 / 3, output_file)

def generate_plot_1d(subjects, output_path):
    output_file = output_path / "1d.png"
    most_mid_least_played = [
      list(player.values()) for subject in subjects
      for player in subject.get_most_mid_least_played_responses()
    ]
    data = [[], [], []]
    for player in most_mid_least_played:
        player = [i * (100 / sum(player)) for i in player]
        data[0].append(player[0])
        data[1].append(player[1])
        data[2].append(player[2])

    labels = ["Most Played", "Mid Played", "Least Played"]
    colors = ["#cb4149", "#f5dc4e", "#f78310"]
    yticks = list(range(20, 50, 5))
    
    draw_raincloud_plot(data, labels, "Response played", colors, yticks, 100 / 3, output_file)

def generate_plot_1e(subjects, output_path):
    output_file = output_path / "1e.png"
    data = [[], [], []]
    for subject in subjects:
        response_changes_after_win, response_changes_after_loss, response_changes_after_draw = subject.get_response_changes_distributions(
        )
        data[0] += response_changes_after_win
        data[1] += response_changes_after_loss
        data[2] += response_changes_after_draw
    labels = ["After win", "After loss", "After draw"]
    colors = ["#2d708e", "#404788", "#481567"]
    yticks = list(range(20, 120, 20))

    draw_raincloud_plot(data, labels, "Game-to-game response change", colors, yticks, 200 / 3, output_file)
