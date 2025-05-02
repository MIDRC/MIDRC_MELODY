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
from typing import Any, Dict, List

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
    return 0


def get_angles(num_axes: int, plot_config: dict) -> list:
    """
    Get the angles for the spider chart axes.

    :arg num_axes: Number of axes
    :arg plot_config: Plot configuration dictionary

    :returns: List of angles in radians
    """
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    if plot_config.get('clockwise', False):  # matplotlib default is counter-clockwise
        angles.reverse()
    angle_rot = get_angle_rot(plot_config.get('start', 'right'))  # matplotlib default is right)
    angles = [(angle + angle_rot) % (2 * np.pi) for angle in angles]
    return angles


def plot_spider_chart(plot_data: SpiderPlotData) -> plt.Figure:
    """
    Plot a spider chart for the given groups, values, and bounds.

    :arg plot_data: SpiderPlotData object containing the following fields:
        - `model_name`: Name of the model
        - `groups` (List[str]): List of group names
        - `values`: List of values for each group
        - `lower_bounds`: List of lower bounds for each group
        - `upper_bounds`: List of upper bounds for each group
        - `ylim_min`: Minimum value for the y-axis
        - `ylim_max`: Maximum value for the y-axis
        - `metric`: Metric to display on the plot
        - `plot_config`: Optional configuration dictionary for the plot

    :returns: Matplotlib figure object
    """
    # Sort groups so that within each attribute they appear in order
    custom_orders = plot_data.plot_config.get('custom_orders', None)
    if custom_orders is None:
        custom_orders = {
            'age_binned': ['18-29', '30-39', '40-49', '50-64', '65-74', '75-84', '85+'],
            'sex': ['Male', 'Female'],
            'race': ['White', 'Asian', 'Black or African American', 'Other'],
            'ethnicity': ['Hispanic or Latino', 'Not Hispanic or Latino'],
            'intersectional_race_ethnicity': ['White', 'Not White or Hispanic or Latino'],
        }

    def group_sort_key(label, custom_orders):
        attr, group = label.split(': ', 1)
        if attr in custom_orders:
            attr_order = list(custom_orders.keys()).index(attr)
            order = custom_orders[attr]
            return (attr_order, order.index(group)) if group in order else (attr_order, len(order))
        return len(custom_orders), group

    combined = list(zip(plot_data.groups, plot_data.values, plot_data.lower_bounds, plot_data.upper_bounds))
    combined.sort(key=lambda x: group_sort_key(x[0], custom_orders))
    groups, values, lower_bounds, upper_bounds = zip(*combined)

    angles = get_angles(len(groups), plot_data.plot_config)

    # Close the loop for the plotted series
    values, lower_bounds, upper_bounds = map(lambda x: list(x) + [x[0]], [values, lower_bounds, upper_bounds])
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    # Plot the main line and markers
    ax.plot(angles, values, color='steelblue', linestyle='-', linewidth=2)
    ax.scatter(angles, values, marker='o', color='b')
    ax.set_ylim(plot_data.ylim_min, plot_data.ylim_max)

    if plot_data.metric is not None:
        metric = plot_data.metric.upper()
        if metric == 'QWK':
            # Instead of drawing a line between the vertices for baseline,
            # generate a smooth circle at the 0-level.
            theta_full = np.linspace(0, 2 * np.pi, 100)
            baseline_circle = np.full_like(theta_full, 0)
            ax.plot(theta_full, baseline_circle, color='seagreen', linestyle='--', linewidth=3, alpha=0.8)
        elif metric == 'EOD':
            ax.fill_between(angles, -0.1, 0.1, color='lightgreen', alpha=0.5)
        elif metric == 'AAOD':
            ax.fill_between(angles, 0, 0.1, color='lightgreen', alpha=0.5)
            ax.set_ylim(0, plot_data.ylim_max)
    else:
        metric = None

    ax.fill_between(angles, lower_bounds, upper_bounds, color='steelblue', alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(groups, fontsize=8, ha='center')
    title = ('' if metric is None else f'{metric} ') + f'Spider Plot for {plot_data.model_name}'
    ax.set_title(title, size=14, weight='bold')

    if metric == 'QWK':  # This is a QWK plot
        # Emphasize tick labels based on bounds.
        # Iterate over the tick labels (skipping the last repeated label)
        for i, label in enumerate(ax.get_xticklabels()):
            if i >= len(lower_bounds) - 1:
                break
            angle = angles[i]
            # Calculate rotation so that the dash is perpendicular to the corresponding radius.
            # For angle=0, we want 90 degrees; for angle=pi/2, we want 0 degrees, etc.
            rotation_deg = 90 + np.degrees(angle)
            # If lower bound is greater than zero, mark with blue bold text
            if lower_bounds[i] > 0:
                label.set_fontweight('bold')
                label.set_color('maroon')
                # Add a maroon point at the lower bound for emphasis.
                ax.text(angle, lower_bounds[i], '—', color='maroon', fontsize=12,
                        rotation=rotation_deg, ha='center', va='center')
            # If upper bound is less than zero, mark with red bold text
            elif upper_bounds[i] < 0:
                label.set_fontweight('bold')
                label.set_color('red')
                # Add a red point at the upper bound for emphasis.
                ax.text(angle, upper_bounds[i], '—', color='red', fontsize=12,
                        rotation=rotation_deg, ha='center', va='center')

        plt.tight_layout()
    return fig


def figure_to_image(fig) -> np.ndarray:
    """
    Convert a Matplotlib figure to a numpy array.

    :arg fig: Matplotlib figure object

    :returns: Numpy array representing the figure
    """
    # Draw the renderer
    fig.canvas.draw()
    # Get width and height in pixels
    width, height = fig.canvas.get_width_height()
    # Convert canvas to a numpy array (ARGB)
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(height, width, 4)
    # Convert ARGB to RGB by discarding the alpha channel
    # The buffer order is: [A, R, G, B]
    rgb_img = img[:, :, 1:4]
    return rgb_img


def display_figures_grid(figures, n_cols=3):
    """
    Display a grid of: figures in a single plot.

    :arg figures: List of Matplotlib figure objects
    :arg n_cols: Number of columns in the grid
    """
    # Calculate the number of rows needed
    n_figs = len(figures)
    n_rows = (n_figs + n_cols - 1) // n_cols

    # Create a figure with subplots for the grid display
    grid_fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 8))

    # In case there's a single row, make axes iterable
    if n_rows == 1:
        axes = np.array(axes)
    axes = axes.flatten()

    for i, fig in enumerate(figures):
        img = figure_to_image(fig)
        axes[i].imshow(img)
        axes[i].axis('off')  # Turn off axis for clarity
    # Remove any unused subplots
    for ax in axes[n_figs:]:
        ax.remove()

    plt.tight_layout()
