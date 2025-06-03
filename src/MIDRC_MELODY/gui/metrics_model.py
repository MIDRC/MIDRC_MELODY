import json
import yaml
import os
# ...existing imports for data and metrics logic...
from MIDRC_MELODY.common.data_loading import build_test_and_demographic_data as build_demo_data
from MIDRC_MELODY.common.qwk_metrics import (
    calculate_delta_kappa,
    calculate_kappas_and_intervals,
    create_spider_plot_data_qwk
)
from MIDRC_MELODY.common.eod_aaod_metrics import (
    binarize_scores,
    calculate_eod_aaod,
    create_spider_plot_data_eod_aaod,
    generate_plot_data_eod_aaod
)
# New imports for GUI colors and table construction in metric processing.
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor
from MIDRC_MELODY.common.table_tools import GLOBAL_COLORS, build_eod_aaod_tables_gui
from dataclasses import replace


def load_config_dict() -> dict:
    """
    Load config settings from QSettings or fallback to config.yaml in the repo root.
    Post-process numeric columns so that any ".inf"/"inf" strings become float("inf").
    """
    settings = QSettings("MIDRC", "MIDRC-MELODY")
    config_str = settings.value("config", "")
    if config_str:
        config = json.loads(config_str)
    else:
        # Default path: three levels up from this file
        config_path = os.path.join(os.path.dirname(__file__), ".", ".", ".", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.CLoader)
        settings.setValue("config", json.dumps(config))

    # Convert any ".inf"/"inf" bins into actual float("inf")
    if "numeric_cols" in config:
        for col, d in config["numeric_cols"].items():
            if "bins" in d:
                processed_bins = []
                for b in d["bins"]:
                    if isinstance(b, (int, float)):
                        processed_bins.append(b)
                    else:
                        try:
                            if b.strip() in [".inf", "inf"]:
                                processed_bins.append(float("inf"))
                            else:
                                processed_bins.append(float(b))
                        except Exception:
                            processed_bins.append(b)
                config["numeric_cols"][col]["bins"] = processed_bins

    return config


@staticmethod
def save_config_dict(config: dict) -> None:
    settings = QSettings("MIDRC", "MIDRC-MELODY")
    settings.setValue("config", json.dumps(config))

def build_demo_data_wrapper(config):
    # Wrap build_demo_data and return the test data
    return build_demo_data(config)

def compute_qwk_metrics(test_data):
    # Compute QWK metrics and prepare table rows and plot args.
    delta_kappas = calculate_delta_kappa(test_data)
    all_rows = []
    filtered_rows = []
    maroon = QColor(*GLOBAL_COLORS['kappa_negative'])
    green = QColor(*GLOBAL_COLORS['kappa_positive'])
    for category, model_data in delta_kappas.items():
        # ...existing iteration logic...
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
    return (all_rows, filtered_rows, kappas_rows, plot_args)

def compute_eod_aaod_metrics(test_data, threshold):
    # Binzarize the scores and compute EOD/AAOD metrics, then build tables and plot args.
    binarized = binarize_scores(test_data.matched_df, test_data.truth_col, test_data.test_cols, threshold=threshold)
    new_data = replace(test_data, matched_df=binarized)
    eod_aaod = calculate_eod_aaod(new_data)
    all_eod_rows, all_aaod_rows, filtered_rows = build_eod_aaod_tables_gui(eod_aaod)
    plot_args = (eod_aaod, new_data.test_cols, {})  # empty plot config
    return (all_eod_rows, all_aaod_rows, filtered_rows, plot_args)
