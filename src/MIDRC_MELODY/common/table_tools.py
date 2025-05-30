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
GLOBAL_COLORS: Final = {
    'eod_negative': (128, 0, 0),  # Maroon
    'eod_positive': (0, 128, 0),  # Green
    'aaod': (255, 165, 0),  # Orange
    'kappa_negative': (128, 0, 0),  # Maroon
    'kappa_positive': (0, 128, 0),  # Green
}

_CONSOLE_RESET: Final = "\033[0m"


def _console_color(color: str | Tuple[int, int, int]) -> str:
    """
    Convert RGB color tuple or color name to ANSI escape code string.
    """
    if isinstance(color, tuple):
        return f"\033[38;2;{color[0]};{color[1]};{color[2]}m"
    if isinstance(color, str):
        if color in GLOBAL_COLORS:
            rgb = GLOBAL_COLORS[color]
            return _console_color(rgb)
    return ""


def _format_console_string(value: str, color: str) -> str:
    """
    Format a string value with ANSI color if it qualifies.
    """
    if color is not None:
        return f"{color}{value}{_CONSOLE_RESET}"
    return value


def _format_console_value(value: float, color: str) -> str:
    """
    Format a numeric value with ANSI color if it qualifies.
    """
    formatted = f"{value:.4f}"
    return _format_console_string(formatted, color)


def _sort_rows(rows: List[List[str]]) -> List[List[str]]:
    """Sort rows by Model, Category, Group, then Metric if present."""
    return sorted(rows, key=lambda r: tuple(r[:4]))


def _build_eod_aaod_tables_generic(
    eod_aaod: Dict[str, Dict[str, Dict[Any, Dict[str, Tuple[float, Tuple[float, float]]]]]], *,
    console: bool
) -> List[Tuple[List[str], str]]:
    """
    Generate tables for EOD/AAOD metrics.
    If console is True, returns tables of list[str] rows with ANSI coloring.
    If console is False, returns tables as tuples of (row: list[str], color: QColor | None).
    """
    # Set up color formatting based on mode.
    if console:
        color_fn = _console_color
    else:
        # For GUI, use plain formatting
        try:
            from PySide6.QtGui import QColor
        except ImportError:
            raise ImportError("PySide6 is required for GUI table generation.")
        color_fn = lambda x: QColor(*GLOBAL_COLORS[x]) if x in GLOBAL_COLORS else None

    # Initialize lists; console returns lists of rows, GUI returns tuples (row, color).
    all_eod = []
    all_aaod = []
    filtered = []
    
    for category, model_data in eod_aaod.items():
        for model, groups in model_data.items():
            for group, metrics in groups.items():
                for metric in ('eod', 'aaod'):
                    if metric not in metrics:
                        continue
                    median, (ci_lo, ci_hi) = metrics[metric]
                    if metric == 'eod':
                        qualifies = abs(median) > 0.1
                        color = None if not qualifies else\
                            color_fn('eod_negative') if median < 0 else color_fn('eod_positive')
                        target_list = all_eod
                    else:
                        qualifies = median > 0.1
                        color = color_fn('aaod') if qualifies else None
                        target_list = all_aaod
                    
                    # Format each cell
                    format_fn = lambda v: f"{v:.4f}"
                    val_str = format_fn(median)
                    lo_str = format_fn(ci_lo)
                    hi_str = format_fn(ci_hi)
                    
                    row = [model, category, group, val_str, lo_str, hi_str]
                    target_list.append((row, color))
                    
                    if qualifies:
                        # For filtered rows, insert the metric name.
                        row_f = row.copy()
                        row_f.insert(3, metric.upper())
                        filtered.append((row_f, color))

    # Define a common sort key based on the first 4 cells of each row.
    sort_key = lambda x: tuple(x[0][:4])

    # Sort all lists using map for conciseness.
    sorted_all_eod, sorted_all_aaod, sorted_filtered = map(
        lambda rows: sorted(rows, key=sort_key),
        [all_eod, all_aaod, filtered]
    )

    return sorted_all_eod, sorted_all_aaod, sorted_filtered


def _print_section(title: str, rows: List[List[str]], headers: List[str], tablefmt: str) -> None:
    print(title)
    print(tabulate(rows, headers=headers, tablefmt=tablefmt))
    print()


def _build_eod_aaod_tables_console(
    eod_aaod: Dict[str, Dict[str, Dict[Any, Dict[str, Tuple[float, Tuple[float, float]]]]]]
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    # Delegate table-building to the generic function with console=True.
    sorted_all_eod, sorted_all_aaod, sorted_filtered = _build_eod_aaod_tables_generic(eod_aaod, console=True)

    def convert_fn(rows: List[Tuple[List[str], str]]) -> List[List[str]]:
        return [
            row if color is None else row[:-3] + [f"{color}{cell}{_CONSOLE_RESET}" for cell in row[-3:]]
            for row, color in rows
        ]

    sorted_all_eod, sorted_all_aaod, sorted_filtered = map(convert_fn,
                                                           [sorted_all_eod, sorted_all_aaod, sorted_filtered])
    return sorted_all_eod, sorted_all_aaod, sorted_filtered


def print_table_of_nonzero_eod_aaod(
    eod_aaod: Dict[str, Dict[str, Dict[Any, Dict[str, Tuple[float, Tuple[float, float]]]]]],
    tablefmt: str = 'grid'
) -> None:
    """
    Print tables for EOD and AAOD medians, highlighting values meeting criteria.
    """
    all_eod, all_aaod, filtered = _build_eod_aaod_tables_console(eod_aaod)

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
                color = None if not qualifies else\
                    _console_color('kappa_negative' if delta < 0 else 'kappa_positive')

                delta_str = _format_console_value(delta, color)
                lo_str = _format_console_value(ci_lo, color)
                hi_str = _format_console_value(ci_hi, color)
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


try:
    from PySide6.QtGui import QColor
except ImportError:
    build_eod_aaod_tables_gui = None
else:
    def build_eod_aaod_tables_gui(
            eod_aaod: Dict[str, Dict[str, Dict[Any, Dict[str, Tuple[float, Tuple[float, float]]]]]]
    ) -> tuple[list[tuple[list[str], "QColor | None"]], list[tuple[list[str], "QColor | None"]], list[tuple[list[str], "QColor | None"]]]:
        # Delegate table-building to the generic function with console=False.
        return _build_eod_aaod_tables_generic(eod_aaod, console=False)
