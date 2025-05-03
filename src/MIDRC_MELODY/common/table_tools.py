#  Copyright (c) 2025 Medical Imaging and Data Resource Center (MIDRC).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Final, List, Tuple

from tabulate import tabulate

# ANSI color codes
_MAROON: Final = "\033[38;2;128;0;0m"
_GREEN: Final = "\033[38;2;0;128;0m"
_ORANGE: Final = "\033[38;2;255;165;0m"
_RESET: Final = "\033[0m"


def _format_value(value: float, qualifies: bool, color: str) -> str:
    """
    Format a numeric value with ANSI color if it qualifies.
    """
    formatted = f"{value:.4f}"
    if qualifies:
        return f"{color}{formatted}{_RESET}"
    return formatted


def _sort_rows(rows: List[List[str]]) -> List[List[str]]:
    """Sort rows by Model, Category, Group, then Metric if present."""
    return sorted(rows, key=lambda r: tuple(r[:4]))


def _print_section(title: str, rows: List[List[str]], headers: List[str], tablefmt: str) -> None:
    print(title)
    print(tabulate(rows, headers=headers, tablefmt=tablefmt))
    print()


def _build_eod_aaod_tables(
    eod_aaod: Dict[str, Dict[str, Dict[Any, Dict[str, Tuple[float, Tuple[float, float]]]]]]
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    all_eod, all_aaod, filtered = [], [], []

    for category, model_data in eod_aaod.items():
        for model, groups in model_data.items():
            for group, metrics in groups.items():
                for metric in ('eod', 'aaod'):
                    if metric not in metrics:
                        continue
                    median, (ci_lo, ci_hi) = metrics[metric]
                    # Determine criteria and color
                    if metric == 'eod':
                        qualifies = abs(median) > 0.1
                        color = _MAROON if median < 0 else _GREEN
                        target_list = all_eod
                    else:
                        qualifies = median > 0.1
                        color = _ORANGE
                        target_list = all_aaod

                    # Format cells
                    val_str = _format_value(median, qualifies, color)
                    lo_str = _format_value(ci_lo, qualifies, color)
                    hi_str = _format_value(ci_hi, qualifies, color)
                    row = [model, category, group, val_str, lo_str, hi_str]
                    target_list.append(row)

                    if qualifies:
                        filtered.append([
                            model,
                            category,
                            group,
                            metric.upper(),
                            val_str,
                            lo_str,
                            hi_str,
                        ])

    return _sort_rows(all_eod), _sort_rows(all_aaod), _sort_rows(filtered)


def print_table_of_nonzero_eod_aaod(
    eod_aaod: Dict[str, Dict[str, Dict[Any, Dict[str, Tuple[float, Tuple[float, float]]]]]],
    tablefmt: str = 'grid'
) -> None:
    """
    Print tables for EOD and AAOD medians, highlighting values meeting criteria.
    """
    all_eod, all_aaod, filtered = _build_eod_aaod_tables(eod_aaod)

    headers_all = ['Model', 'Category', 'Group', 'Median', 'Lower CI', 'Upper CI']
    headers_filtered = ['Model', 'Category', 'Group', 'Metric', 'Median', 'Lower CI', 'Upper CI']

    _print_section('All EOD median values:', all_eod, headers_all, tablefmt)
    _print_section('All AAOD median values:', all_aaod, headers_all, tablefmt)

    if filtered:
        _print_section('EOD/AAOD median values meeting criteria:', filtered, headers_filtered, tablefmt)
    else:
        print('No model/group combinations meeting the specified criteria for EOD/AAOD.')


def _build_delta_tables(
    delta_kappas: Dict[str, Dict[str, Dict[Any, Tuple[float, Tuple[float, float]]]]]
) -> Tuple[List[List[str]], List[List[str]]]:
    all_deltas, filtered = [], []

    for category, model_data in delta_kappas.items():
        for model, groups in model_data.items():
            for group, (delta, (ci_lo, ci_hi)) in groups.items():
                qualifies = ci_lo > 0 or ci_hi < 0
                color = _MAROON if delta < 0 else _GREEN

                delta_str = _format_value(delta, qualifies, color)
                lo_str = _format_value(ci_lo, qualifies, color)
                hi_str = _format_value(ci_hi, qualifies, color)
                row = [model, category, group, delta_str, lo_str, hi_str]
                all_deltas.append(row)

                if qualifies:
                    filtered.append(row)

    return _sort_rows(all_deltas), _sort_rows(filtered)


def print_table_of_nonzero_deltas(
    delta_kappas: Dict[str, Dict[str, Dict[Any, Tuple[float, Tuple[float, float]]]]],
    tablefmt: str = 'grid'
) -> None:
    """
    Print tables for Delta Kappa values, highlighting those with 95% CI excluding zero.
    """
    all_deltas, filtered = _build_delta_tables(delta_kappas)
    headers = ['Model', 'Category', 'Group', 'Delta Kappa', 'Lower CI', 'Upper CI']

    _print_section('All Delta Kappa Values:', all_deltas, headers, tablefmt)

    if filtered:
        _print_section('Delta Kappa values with 95% CI excluding zero:', filtered, headers, tablefmt)
    else:
        print('No model/group combinations meeting the specified criteria for Delta Kappa.')
