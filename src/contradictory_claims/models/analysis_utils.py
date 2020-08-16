"""Utilities for analyzing and visualizing results based on trained model."""

# -*- coding: utf-8 -*-

import datetime
import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Example inputs to make_timeline_plot
# Graph data
# names = ['A', 'B', 'C', 'D', 'E']  # Should already come in order!
# dates = ["02/28/20", "03/28/20", "04/14/20", "06/01/20", "06/04/20"]
# paper_titles = ["title1", "title2", "title3", "title4", "title5"]
# edges_comparisons = [('A', 'C'), ('A', 'E'), ('B', 'D')]
# edges_comparisons_colors = ["green", "red", "green"]
# drug = "chloroquine"


def dates_to_proportions(dates: list[str], date_format: str = '%m/%d/%y'):
    """
    Convert list of dates to positions on a timeline based on proportion of time passed

    :param dates: list of dates as strings
    :param date_format: format of the dates, to be parsed by datetime
    :return: list of proportions of time passed from 0 to 1
    """

    # Read into datetime format
    date_objs = [datetime.datetime.strptime(date, date_format) for date in dates]
    time_diffs = [(d2 - d1).days for d1, d2 in zip(date_objs[:-1], date_objs[1:])]
    time_diffs.insert(0, 0)  # The first time point is t=0
    proportions = list(np.cumsum(time_diffs) / np.sum(time_diffs))

    return proportions


def make_timeline_plot(names: list[str],
                       dates: list[str],
                       paper_titles: list[str],
                       edges_comparisons: list[(str,str)],
                       edges_comparisons_colors: list[str],
                       drug: str,
                       out_plot_dir: str,
                       date_format: str = '%m/%d/%y'):
    """
    Create a timeline plot depicting paper contradictions and agreements for a specific drug

    :param names: Names of nodes (NOTE: This may become obsolete later)
    :param dates: List of dates as strings
    :param paper_titles: List of paper titles/identifiers
    :param edges_comparisons: List of tuples, where each tuple is an edge suggesting comparable papers
    :param edges_comparisons_colors: List of strings (colors) indicating the type of comparison (contra, entail)
    :param drug: Drug relevant for this timeline
    :param out_plot_dir: Output directory
    :param date_format: Format of the dates strings
    """
    x_positions = dates_to_proportions(dates, date_format)
    y_positions = list(np.zeros(len(dates)))
    positions = list(zip(x_positions, y_positions))

    edges_time = []
    for i in range(len(names) - 1):
        edges_time.append((names[i], names[i + 1]))

    # Matplotlib figure
    plt.figure('Timeline plot')

    # Create graph
    G = nx.MultiDiGraph(format='png', directed=True)

    for index, name in enumerate(names):
        G.add_node(name, pos=positions[index])

    labels = {}

    layout = dict((n, G.nodes[n]["pos"]) for n in G.nodes())
    nx.draw(G, pos=layout, with_labels=True, node_size=300)
    ax = plt.gca()
    for edge, e_color in zip(edges_comparisons, edges_comparisons_colors):
        ax.annotate("",
                    xy=layout[edge[0]], xycoords='data',
                    xytext=layout[edge[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=e_color,
                                    shrinkA=10, shrinkB=10,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=0.5", mutation_scale=20
                                    ),
                    )
    for edge in edges_time:
        ax.annotate("",
                    xy=layout[edge[0]], xycoords='data',
                    xytext=layout[edge[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="0.5",
                                    shrinkA=10, shrinkB=10,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc,rad=0.5",
                                    ),
                    )

    for pos, date, title in zip(positions, dates, paper_titles):
        # NOTE the offsets here are hard coded. This will likely cause errors as we scale to real data. This should be proportional instead probably
        # ax.text(positions[i][0], -.021, s=dates[i], bbox=dict(facecolor='#DDDDDD', edgecolor="#DDDDDD", alpha=1),rotation=90,horizontalalignment='center')
        ax.text(pos[0], -.021, s=date, rotation=90, horizontalalignment='center')
        ax.text(pos[0] + .12, -.021, s=title, rotation=90, horizontalalignment='center')

    plt.title(f"Timeline of comparable claims for {drug}", size=20)
    plt.show()
    plot_path = os.path.join(out_plot_dir, f"{drug}_comparisons_timeline.png")
    plt.savefig(plot_path)
