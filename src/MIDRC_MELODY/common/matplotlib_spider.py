#  Copyright (c) 2025 Medical Imaging and Data Resource Center (MIDRC).
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
#

from typing import List, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt
import mplcursors

from MIDRC_MELODY.common.plot_tools import SpiderPlotData, prepare_and_sort, get_full_theta, \
    compute_angles


def plot_spider_chart(plot_data: SpiderPlotData) -> plt.Figure:
    """
    Plot a spider chart for the given groups, values, and bounds.

    :arg plot_data: SpiderPlotData object containing the following fields:
        - `model_name`: Name of the model
        - `groups` (List[str]): List of group names
        - `values`: List of values for each group
        - `lower_bounds`: List of lower bounds for each group
        - `upper_bounds`: List of upper bounds for each group
        - `ylim_min`: Dict of metric and Minimum value for the y-axis
        - `ylim_max`: Dict of metric and Maximum value for the y-axis
        - `metric`: Metric to display on the plot
        - `plot_config`: Optional configuration dictionary for the plot

    :returns: Matplotlib figure object
    """
    groups, values, lower_bounds, upper_bounds = prepare_and_sort(plot_data)
    angles = compute_angles(len(groups), plot_data.plot_config)
    fig, ax = _init_spider_axes(plot_data.ylim_min[plot_data.metric],
                                plot_data.ylim_max[plot_data.metric])
    sc = _draw_main_series(ax, angles, values)
    _add_cursor_to_spider_plot(sc, fig.canvas, groups, values, lower_bounds, upper_bounds)
    _fill_bounds(ax, angles, lower_bounds, upper_bounds)
    _configure_axes(ax, angles, groups, plot_data.model_name)
    if plot_data.metric:
        _apply_metric_overlay(ax, angles, plot_data.metric, values, lower_bounds, upper_bounds)
    plt.tight_layout()
    return fig


def _init_spider_axes(ymin: float, ymax: float) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    ax.set_ylim(ymin, ymax)
    return fig, ax


def _add_cursor_to_spider_plot(sc, canvas, groups, values, lower_bounds, upper_bounds) -> mplcursors.Cursor:
    """
    Attach a hover cursor to the scatter points in the spider plot.

    :arg sc: Scatter collection object from the spider plot
    :arg groups: List of group names
    :arg values: List of values for each group
    :arg lower_bounds: List of lower bounds for each group
    :arg upper_bounds: List of upper bounds for each group
    """
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        sel.annotation.set_text(
            f"{groups[i]}\n"
            f"Median: {values[i]:.3f} "
            f"[{lower_bounds[i]:.3f}, {upper_bounds[i]:.3f}]"
        )

    def _hide_all_annotations(event):
        for sel in list(cursor.selections):
            cursor.remove_selection(sel)

    canvas.mpl_connect('button_press_event', _hide_all_annotations)
    return cursor


def _draw_main_series(ax: plt.Axes, angles: List[float], values: List[float]) -> plt.PathCollection:
    ax.plot(angles, values, color='steelblue', linestyle='-', linewidth=2)
    return ax.scatter(angles, values, marker='o', color='b')


def _apply_metric_overlay(
    ax: plt.Axes,
    angles: List[float],
    metric: str,
    values: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float],
) -> None:
    metric = metric.upper()
    full_theta = get_full_theta()
    overlay_config = {
        'QWK': {
            'baseline': {'type': 'line', 'y': 0, 'style': '--', 'color': 'seagreen', 'linewidth': 3, 'alpha': 0.8},
            'thresholds': [
                (lower_bounds, lambda v: v > 0, 'maroon'),
                (upper_bounds, lambda v: v < 0, 'red'),
            ],
        },
        'EOD': {
            'fill': {'lo': -0.1, 'hi': 0.1, 'color': 'lightgreen', 'alpha': 0.5},
            'thresholds': [
                (values, lambda v: v > 0.1, 'maroon'),
                (values, lambda v: v < -0.1, 'red'),
            ],
        },
        'AAOD': {
            'fill': {'lo': 0, 'hi': 0.1, 'color': 'lightgreen', 'alpha': 0.5},
            'baseline': {'type': 'ylim', 'lo': 0},
            'thresholds': [
                (values, lambda v: v > 0.1, 'maroon'),
            ],
        },
    }
    cfg = overlay_config.get(metric)
    if not cfg:
        return

    # Baseline rendering
    if 'baseline' in cfg:
        base = cfg['baseline']
        if base['type'] == 'line':
            ax.plot(full_theta, np.full_like(full_theta, base['y']), base['style'],
                    linewidth=base['linewidth'], alpha=base['alpha'], color=base['color'])
        elif base['type'] == 'ylim':
            _, ymax = ax.get_ylim()
            ax.set_ylim(base['lo'], ymax)

    # Fill region if specified
    if 'fill' in cfg:
        f = cfg['fill']
        ax.fill_between(full_theta, f['lo'], f['hi'], color=f['color'], alpha=f['alpha'])

    # Annotate thresholds
    for data, cond, color in cfg['thresholds']:
        _annotate(ax, angles, data, cond, color)


def _annotate(
    ax: plt.Axes,
    angles: List[float],
    data: List[float],
    condition: Any,
    color: str,
) -> None:
    for i, label in enumerate(ax.get_xticklabels()[:-1]):
        if condition(data[i]):
            angle = angles[i]
            rot = 90 + np.degrees(angle)
            label.set_fontweight('bold')
            label.set_color(color)
            ax.text(angle, data[i], '—', rotation=rot, color=color, ha='center', va='center', fontsize=24)


def _fill_bounds(
    ax: plt.Axes,
    angles: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float],
) -> None:
    ax.fill_between(angles, lower_bounds, upper_bounds, color='steelblue', alpha=0.2)


def _configure_axes(
    ax: plt.Axes,
    angles: List[float],
    groups: List[str],
    title: str,
) -> None:
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(groups[:-1])
    ax.set_title(title, size=14, weight='bold')


def figure_to_image(fig: plt.Figure) -> np.ndarray:
    """
    Convert a Matplotlib figure to a numpy array.

    :arg fig: Matplotlib figure object

    :returns: Numpy array representing the figure
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(h, w, 4)
    return buf[:, :, 1:4]


def display_figures_grid(figures: List[plt.Figure], n_cols: int = 3) -> None:
    """
    Display a grid of: figures in a single plot.

    :arg figures: List of Matplotlib figure objects
    :arg n_cols: Number of columns in the grid
    """
    n_figs = len(figures)
    n_rows = int(np.ceil(n_figs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 8 * n_rows))
    axes = np.array(axes).flatten()
    for i, f in enumerate(figures):
        img = figure_to_image(f)
        axes[i].imshow(img)
        axes[i].axis('off')
    for ax in axes[n_figs:]:
        ax.remove()
    plt.tight_layout()
