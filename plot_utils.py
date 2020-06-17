import numpy as np
import matplotlib.pyplot as plt

def plot_stacked_bar(ax, data, series_labels, category_labels=None,
                     show_values=False, value_format="{}", y_label=None,
                     colors=None, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(ax.bar(ind, row_data, bottom=cum_size,
                            label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        ax.set_xticks(ind)
        ax.set_xticklabels(category_labels)

    if y_label:
        ax.set_ylabel(y_label)

    ax.legend()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                ax.text(bar.get_x() + w/2, bar.get_y() + h/2, value_format.format(h), ha="center", va="center")


def plot_ax(ax, data):
    series_labels = ['Pellets', 'Chow']
    category_labels = ['control', 'depleted']

    plot_stacked_bar(ax, data, series_labels, category_labels=category_labels, show_values=True, value_format="{:.1f}",
                     colors=['tab:blue', 'tab:grey'], y_label="Choices [#]")