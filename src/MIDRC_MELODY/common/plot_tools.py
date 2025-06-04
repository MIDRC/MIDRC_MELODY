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

import numpy as np


@dataclass
class SpiderPlotData:
    """ Data class for spider plot data. """
    model_name: str = ""
    groups: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    lower_bounds: List[float] = field(default_factory=list)
    upper_bounds: List[float] = field(default_factory=list)
    ylim_min: Dict[str, float] = field(default_factory=dict)
    ylim_max: Dict[str, float] = field(default_factory=dict)
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


def prepare_and_sort(plot_data: SpiderPlotData) -> Tuple[List[str], List[float], List[float], List[float]]:
    custom_orders = plot_data.plot_config.get('custom_orders') or {
        'age_binned': ['18-29', '30-39', '40-49', '50-64', '65-74', '75-84', '85+'],
        'sex': ['Male', 'Female'],
        'race': ['White', 'Asian', 'Black or African American', 'Other'],
        'ethnicity': ['Hispanic or Latino', 'Not Hispanic or Latino'],
        'intersectional_race_ethnicity': ['White', 'Not White or Hispanic or Latino'],
    }
    attributes = list(custom_orders.keys())

    def sort_key(label: str) -> Any:
        attr, grp = label.split(': ', 1)
        if attr in attributes:
            if grp in custom_orders[attr]:
                return (attributes.index(attr), custom_orders[attr].index(grp))
            else:
                return (attributes.index(attr), len(custom_orders[attr]))
        # Other items sort after custom-ordered, by string label
        return (len(attributes), label)

    print('groups unsorted:', plot_data.groups)
    print('groups sorted:', sorted(plot_data.groups, key=lambda x: sort_key(x)))

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


def get_full_theta() -> np.ndarray:
    return np.linspace(0, 2 * np.pi, 100)


def compute_angles(num_axes_with_close: int, plot_config: dict) -> List[float]:
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
