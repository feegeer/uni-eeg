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
    plt.show()
    if filepath is not None:
        ...
