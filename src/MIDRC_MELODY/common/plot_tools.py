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

""" Plotting tools for visualizing model performance metrics. """
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SpiderPlotData:
    """ Data class for spider plot data. """
    model_name: str = ""
    groups: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    lower_bounds: List[float] = field(default_factory=list)
    upper_bounds: List[float] = field(default_factory=list)
    ylim_min: float = 0.0
    ylim_max: float = 1.0
    metric: str = ""
    plot_config: Dict[str, Any] = field(default_factory=dict)


def get_angle_rot(start_loc: str) -> float:
    """
    Get the angle rotation based on the starting location.

    :arg start_loc: Starting location string

    :returns: Angle rotation in radians
    """
    if start_loc.startswith('t'):
        return np.pi / 2
    if start_loc.startswith('l'):
        return np.pi
    if start_loc.startswith('b'):
        return 3 * np.pi / 2
    return 0.0


def get_angles(num_axes: int, plot_config: dict) -> List[float]:
    """
    Get the angles for the spider chart axes.

    :arg num_axes: Number of axes
    :arg plot_config: Plot configuration dictionary

    :returns: List of angles in radians
    """
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    if plot_config.get('clockwise', False):
        angles.reverse()
    rot = get_angle_rot(plot_config.get('start', 'right'))
    return [(a + rot) % (2 * np.pi) for a in angles]


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
    groups, values, lower_bounds, upper_bounds = _prepare_and_sort(plot_data)
    angles = _compute_angles(len(groups), plot_data.plot_config)
    fig, ax = _init_spider_axes(plot_data.ylim_min[plot_data.metric],
                                plot_data.ylim_max[plot_data.metric])
    _draw_main_series(ax, angles, values)
    _fill_bounds(ax, angles, lower_bounds, upper_bounds)
    _configure_axes(ax, angles, groups, plot_data.model_name)
    if plot_data.metric:
        _apply_metric_overlay(ax, angles, plot_data.metric, values, lower_bounds, upper_bounds)
    plt.tight_layout()
    return fig


def _prepare_and_sort(plot_data: SpiderPlotData) -> Tuple[List[str], List[float], List[float], List[float]]:
    custom_orders = plot_data.plot_config.get('custom_orders') or {
        'age_binned': ['18-29', '30-39', '40-49', '50-64', '65-74', '75-84', '85+'],
        'sex': ['Male', 'Female'],
        'race': ['White', 'Asian', 'Black or African American', 'Other'],
        'ethnicity': ['Hispanic or Latino', 'Not Hispanic or Latino'],
        'intersectional_race_ethnicity': ['White', 'Not White or Hispanic or Latino'],
    }

    def sort_key(label: str) -> Any:
        attr, grp = label.split(': ', 1)
        if attr in custom_orders and grp in custom_orders[attr]:
            # Tuple (0, index) ensures custom-ordered items come first by numeric key
            return (0, custom_orders[attr].index(grp))
        # Other items sort after custom-ordered, by string label
        return (1, label)

    zipped = list(zip(
        plot_data.groups,
        plot_data.values,
        plot_data.lower_bounds,
        plot_data.upper_bounds
    ))
    sorted_zipped = sorted(zipped, key=lambda x: sort_key(x[0]))
    groups, values, lower_bounds, upper_bounds = map(list, zip(*sorted_zipped))

    # Close the loop for spider plot
    groups.append(groups[0])
    values.append(values[0])
    lower_bounds.append(lower_bounds[0])
    upper_bounds.append(upper_bounds[0])

    return groups, values, lower_bounds, upper_bounds


def _compute_angles(num_axes_with_close: int, plot_config: dict) -> List[float]:
    """
    Compute angles for spider plot, accounting for loop closure.

    :arg num_axes_with_close: Number of items in groups list (including duplicated first at end).
    :arg plot_config: Configuration dict for angle ordering.

    :returns: Angles list matching the length of groups list.
    """
    # The groups list already closes the loop by duplicating the first entry.
    # Compute based on original number of axes (excluding the closure element).
    original_count = num_axes_with_close - 1
    angles = get_angles(original_count, plot_config)
    # Close the loop by appending the first angle
    angles.append(angles[0])
    return angles


def _init_spider_axes(ymin: float, ymax: float) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    ax.set_ylim(ymin, ymax)
    return fig, ax


def _draw_main_series(ax: plt.Axes, angles: List[float], values: List[float]) -> None:
    ax.plot(angles, values, color='steelblue', linestyle='-', linewidth=2)
    ax.scatter(angles, values, marker='o', color='b')


def _apply_metric_overlay(
    ax: plt.Axes,
    angles: List[float],
    metric: str,
    values: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float],
) -> None:
    metric = metric.upper()
    full_theta = _get_full_theta()
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


def _get_full_theta() -> np.ndarray:
    return np.linspace(0, 2 * np.pi, 100)


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
            ax.text(angle, data[i], '—', rotation=rot, color=color, ha='center', va='center', fontsize=12)


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
