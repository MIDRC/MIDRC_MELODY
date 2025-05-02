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

""" Module for calculating quadratic weighted kappa and delta kappa values with confidence intervals. """

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from joblib import delayed, Parallel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample
from tabulate import tabulate
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from MIDRC_MELODY.common.data_loading import TestAndDemographicData
from MIDRC_MELODY.common.plot_tools import display_figures_grid, plot_spider_chart, SpiderPlotData


def calculate_kappas_and_intervals(
    test_data: TestAndDemographicData
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Calculate Cohen's quadratic weighted kappa and bootstrap confidence intervals.

    :arg test_data: TestAndDemographicData object containing the test and demographic data.

    :returns: Tuple of dictionaries containing kappa scores and 95% confidence intervals.
    """
    ai_cols = test_data.test_cols
    if not isinstance(ai_cols, list):
        ai_cols = [ai_cols]
    kappas: Dict[str, float] = {}
    intervals: Dict[str, Tuple[float, float]] = {}
    y_true = test_data.matched_df[test_data.truth_col].to_numpy(dtype=int)

    rng = np.random.default_rng(test_data.base_seed)
    print('-'*50)
    print("Overall Quadratic Weighted Kappa Scores:")
    for col in ai_cols:
        y_pred = test_data.matched_df[col].to_numpy(dtype=int)
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        kappas[col] = kappa

        kappa_scores = np.empty(test_data.n_iter)
        for i in range(test_data.n_iter):
            indices = rng.integers(0, len(y_true), size=len(y_true))
            kappa_scores[i] = cohen_kappa_score(y_true[indices], y_pred[indices], weights='quadratic')
        lower_bnd, upper_bnd = np.percentile(kappa_scores, [2.5, 97.5])
        intervals[col] = (lower_bnd, upper_bnd)
        print(f"Model: {col} | Kappa: {kappa:.4f} | 95% CI: ({lower_bnd:.4f}, {upper_bnd:.4f}) N: {len(y_true)}")
    print('-'*50)

    return kappas, intervals


def bootstrap_kappa(test_data: TestAndDemographicData, n_jobs: int = -1) -> Dict[str, List[float]]:
    """
    Perform bootstrap estimation of quadratic weighted kappa scores for each model in parallel.

    :arg test_data: TestAndDemographicData object containing the test and demographic data.
    :arg n_jobs: Number of parallel jobs.

    :returns: Dictionary of model names and their corresponding kappa scores.
    """
    models = test_data.test_cols
    if not isinstance(models, list):
        models = [models]
    rng = np.random.default_rng(test_data.base_seed)
    seeds = rng.integers(0, 1_000_000, size=test_data.n_iter)

    def resample_and_compute_kappa(df: pd.DataFrame, truth_col: str, _models: List[str], seed: int) -> List[float]:
        sampled_df = resample(df, replace=True, random_state=seed)
        return [
            cohen_kappa_score(sampled_df[truth_col].to_numpy(dtype=int),
                              sampled_df[model].to_numpy(dtype=int),
                              weights='quadratic')
            for model in _models
        ]

    with tqdm_joblib(total=test_data.n_iter, desc="Bootstrapping", leave=False):
        kappas_2d = Parallel(n_jobs=n_jobs)(
            delayed(resample_and_compute_kappa)(test_data.matched_df, test_data.truth_col, models, seed)
            for seed in seeds
        )
    kappa_dict = dict(zip(models, zip(*kappas_2d)))
    return kappa_dict


def calculate_delta_kappa(
    # df: pd.DataFrame, categories: List[str], reference_groups: Dict[str, Any], valid_groups: Dict[str, List[Any]],
    # truth_col: str, ai_columns: List[str], n_iter: int = 1000, base_seed: Optional[int] = None
    test_data: TestAndDemographicData
) -> Dict[str, Dict[str, Dict[Any, Tuple[float, Tuple[float, float]]]]]:
    """
    Calculate delta kappa (difference between group and reference) with bootstrap confidence intervals.

    :arg test_data: TestAndDemographicData object containing the test and demographic data.

    :returns: Dictionary of delta quadratic weighted kappa values with 95% confidence intervals.
    """
    delta_kappas: Dict[str, Dict[str, Dict[Any, Tuple[float, Tuple[float, float]]]]] = {}
    df = test_data.matched_df

    for category in tqdm(test_data.categories, desc="Categories", position=0):
        if category not in test_data.valid_groups:
            continue

        delta_kappas[category] = {model: {} for model in test_data.test_cols}
        unique_values = df[category].unique().tolist()

        kappa_dicts: Dict[str, Dict[str, List[float]]] = {}
        for value in tqdm(unique_values, desc=f"Category \033[1m{category}\033[0m Groups", leave=False, position=1):
            if value not in test_data.valid_groups[category]:
                continue

            filtered_df = df[df[category] == value]

            # Create a shallow copy of test_data and update matched_df with filtered_df
            filtered_test_data = replace(test_data, matched_df=filtered_df)

            kappa_dicts[value] = bootstrap_kappa(filtered_test_data, n_jobs=-1)

        # Remove and store reference bootstraps.
        ref_bootstraps = kappa_dicts.pop(test_data.reference_groups[category])

        # Now calculate the differences.
        for value, kappa_dict in kappa_dicts.items():
            for model in test_data.test_cols:
                model_boot = np.array(kappa_dict[model])
                ref_boot = np.array(ref_bootstraps[model])
                deltas = model_boot - ref_boot
                delta_median = float(np.median(deltas))
                lower_value, upper_value = np.percentile(deltas, [2.5, 97.5])
                delta_kappas[category][model][value] = (
                    delta_median,
                    (float(lower_value), float(upper_value))
                    )
    return delta_kappas


def extract_plot_data(delta_kappas: Dict[str, Dict[str, Dict[Any, Tuple[float, Tuple[float, float]]]]],
                      model_name: str) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Extract group names, delta values and confidence intervals for plotting.

    :arg delta_kappas: Dictionary of delta kappa values with 95% confidence intervals.
    :arg model_name: Name of the AI model.

    :returns: Tuple of group names, delta values, lower bounds and upper bounds.
    """
    groups: List[str] = []
    values: List[float] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    for category, model_data in delta_kappas.items():
        if model_name in model_data:
            for group, (value, (lower_ci, upper_ci)) in model_data[model_name].items():
                groups.append(f"{category}: {group}")
                values.append(value)
                lower_bounds.append(lower_ci)
                upper_bounds.append(upper_ci)
    return groups, values, lower_bounds, upper_bounds


def generate_plots_from_delta_kappas(
    delta_kappas: Dict[str, Dict[str, Dict[Any, Tuple[float, Tuple[float, float]]]]],
    ai_models: List[str],
    plot_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Generate spider plots for delta kappas using consistent scale across models.

    :arg delta_kappas: Dictionary of delta kappa values with 95% confidence intervals.
    :arg ai_models: List of test columns (AI model names).
    :arg plot_config: Optional configuration dictionary for plotting
    """
    all_values, all_lower, all_upper = [], [], []

    for model in ai_models:
        _, values, lower_bounds, upper_bounds = extract_plot_data(delta_kappas, model)
        all_values.extend(values)
        all_lower.extend(lower_bounds)
        all_upper.extend(upper_bounds)

    global_min = min(all_lower) - 0.05
    global_max = max(all_upper) + 0.05

    figures = []
    base_plot_data = SpiderPlotData(
        ylim_min=global_min,
        ylim_max=global_max,
        plot_config=plot_config,
        metric="QWK",
    )
    for model in ai_models:
        # Create a new copy based on the base instance
        plot_data = SpiderPlotData(**base_plot_data.__dict__)
        plot_data.model_name = model
        plot_data.groups, plot_data.values, plot_data.lower_bounds, plot_data.upper_bounds = \
            extract_plot_data(delta_kappas, model)
        fig = plot_spider_chart(plot_data)
        figures.append(fig)

    display_figures_grid(figures)
    plt.show()


def print_table_of_nonzero_deltas(delta_kappas: Dict[str, Dict[str, Dict[Any, Tuple[float, Tuple[float, float]]]]],
                                  tablefmt: str = "grid") -> None:
    """
    Print two tables:
      1. All delta kappa values.
      2. Filtered delta kappas that have a 95% CI excluding zero.
         Negative delta values are printed in maroon, positive in green.

    :arg delta_kappas: Dictionary of delta kappa values with 95% confidence intervals.
    :arg tablefmt: Table format string for tabulate.
    """
    # ANSI escape codes for maroon and green using 24-bit RGB colors
    maroon = "\033[38;2;128;0;0m"
    green = "\033[38;2;0;128;0m"
    reset = "\033[0m"

    all_deltas = []
    filtered = []

    for category, model_data in delta_kappas.items():
        for model, groups in model_data.items():
            for group, (delta, (lower_ci, upper_ci)) in groups.items():
                qualifies = (lower_ci > 0 or upper_ci < 0)
                if qualifies:
                    color = maroon if delta < 0 else green
                    row = [model, category, group,
                           f"{color}{delta:.4f}{reset}",
                           f"{color}{lower_ci:.4f}{reset}",
                           f"{color}{upper_ci:.4f}{reset}"]
                    filtered.append(row)
                else:
                    row = [model, category, group,
                           f"{delta:.4f}",
                           f"{lower_ci:.4f}",
                           f"{upper_ci:.4f}"]
                all_deltas.append(row)

    all_deltas.sort(key=lambda row: (row[0], row[1], row[2]))
    filtered.sort(key=lambda row: (row[0], row[1], row[2]))

    headers = ["Model", "Category", "Group", "Delta Kappa", "Lower CI", "Upper CI"]

    print("All Delta Kappa Values:")
    print(tabulate(all_deltas, headers=headers, tablefmt=tablefmt))
    print("\n")

    if filtered:
        print("Delta Kappa values with 95% CI excluding zero:")
        print(tabulate(filtered, headers=headers, tablefmt=tablefmt))
    else:
        print("No model/group combinations meeting the specified criteria for Delta Kappa.")
