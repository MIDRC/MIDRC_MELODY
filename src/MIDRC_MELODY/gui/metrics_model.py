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

from MIDRC_MELODY.common.qwk_metrics import (
    calculate_delta_kappa,
    calculate_kappas_and_intervals,
)
from MIDRC_MELODY.common.eod_aaod_metrics import (
    binarize_scores,
    calculate_eod_aaod,
)

from PySide6.QtGui import QColor
from MIDRC_MELODY.common.table_tools import GLOBAL_COLORS, build_eod_aaod_tables_gui
from dataclasses import replace

def compute_qwk_metrics(test_data):
    # Compute QWK metrics and prepare table rows and plot args.
    delta_kappas = calculate_delta_kappa(test_data)
    all_rows = []
    filtered_rows = []
    maroon = QColor(*GLOBAL_COLORS['kappa_negative'])
    green = QColor(*GLOBAL_COLORS['kappa_positive'])
    for category, model_data in delta_kappas.items():
        for model, groups in model_data.items():
            for group, (delta, (lower_ci, upper_ci)) in groups.items():
                qualifies = (lower_ci > 0 or upper_ci < 0)
                color = green if qualifies and delta >= 0 else (maroon if qualifies and delta < 0 else None)
                row = [model, category, group, f"{delta:.4f}", f"{lower_ci:.4f}", f"{upper_ci:.4f}"]
                all_rows.append((row, color))
                if qualifies:
                    filtered_rows.append((row, color))
    kappas, intervals = calculate_kappas_and_intervals(test_data)
    kappas_rows = []
    for model in sorted(kappas.keys()):
        row = [model, f"{kappas[model]:.4f}", f"{intervals[model][0]:.4f}", f"{intervals[model][1]:.4f}"]
        kappas_rows.append((row, None))
    plot_args = (delta_kappas, test_data.test_cols, {})  # empty plot config
    return all_rows, filtered_rows, kappas_rows, plot_args

def compute_eod_aaod_metrics(test_data, threshold):
    # Binzarize the scores and compute EOD/AAOD metrics, then build tables and plot args.
    binarized = binarize_scores(test_data.matched_df, test_data.truth_col, test_data.test_cols, threshold=threshold)
    new_data = replace(test_data, matched_df=binarized)
    eod_aaod = calculate_eod_aaod(new_data)
    all_eod_rows, all_aaod_rows, filtered_rows = build_eod_aaod_tables_gui(eod_aaod)
    plot_args = (eod_aaod, new_data.test_cols, {})  # empty plot config
    return all_eod_rows, all_aaod_rows, filtered_rows, plot_args
