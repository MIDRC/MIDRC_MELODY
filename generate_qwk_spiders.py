"""This script generates QWK spider plots for multiple models across different categories."""

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample
from tabulate import tabulate
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import yaml

from data_loading import create_matched_df_from_files, determine_valid_n_reference_groups, save_pickled_data
from plot_tools import plot_spider_chart, display_figures_grid
from typing import List, Dict, Tuple, Any, Optional, Union


def calculate_kappas_and_intervals(
    df: pd.DataFrame, truth_col: str, ai_cols: Union[List[str], str], n_iter: int = 1000, base_seed: Optional[int] = None
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Calculate Cohen's quadratic weighted kappa and bootstrap confidence intervals.

    :arg df: DataFrame containing the truth and AI columns.
    :arg truth_col: Column name of the truth labels.
    :arg ai_cols: List of AI column names.
    :arg n_iter: Number of bootstrap iterations.
    :arg base_seed: Base seed for reproducibility.

    :returns: Tuple of dictionaries containing kappa scores and 95% confidence intervals.
    """
    if not isinstance(ai_cols, list):
        ai_cols = [ai_cols]
    kappas: Dict[str, float] = {}
    intervals: Dict[str, Tuple[float, float]] = {}
    y_true = df[truth_col].to_numpy(dtype=int)

    # Using a modern random generator for reproducibility.
    rng = np.random.default_rng(base_seed)
    for col in ai_cols:
        y_pred = df[col].to_numpy(dtype=int)
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        kappas[col] = kappa

        kappa_scores = []
        for _ in range(n_iter):
            indices = rng.integers(0, len(y_true), size=len(y_true))
            kappa_bs = cohen_kappa_score(y_true[indices], y_pred[indices], weights='quadratic')
            kappa_scores.append(kappa_bs)
        kappa_scores.sort()
        lower_bnd = kappa_scores[int(0.025 * n_iter)]
        upper_bnd = kappa_scores[int(0.975 * n_iter)]
        intervals[col] = (lower_bnd, upper_bnd)
        print(f"Model: {col} | Kappa: {kappa:.4f} | 95% CI: ({lower_bnd:.4f}, {upper_bnd:.4f}) N: {len(y_true)}")

    return kappas, intervals

def bootstrap_kappa(
    df: pd.DataFrame, truth_col: str, models: Union[List[str], Any], n_iter: int = 1000,
    n_jobs: int = -1, base_seed: Optional[int] = None
) -> Dict[str, Tuple]:
    """
    Perform bootstrap estimation of quadratic weighted kappa scores for each model in parallel.

    :arg df: DataFrame containing the truth and AI columns.
    :arg truth_col: Column name of the truth labels.
    :arg models: List of test column names.
    :arg n_iter: Number of bootstrap iterations.
    :arg n_jobs: Number of parallel jobs.
    :arg base_seed: Base seed for reproducibility.

    :returns: Dictionary of model names and their corresponding kappa scores.
    """
    if not isinstance(models, list):
        try:
            models = models.tolist()
        except AttributeError:
            models = [models]
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 1_000_000, size=n_iter)

    def resample_and_compute_kappa(df: pd.DataFrame, truth_col: str, models: List[str], seed: int) -> List[float]:
        sampled_df = resample(df, replace=True, random_state=seed)
        return [
            cohen_kappa_score(sampled_df[truth_col].to_numpy(dtype=int),
                              sampled_df[model].to_numpy(dtype=int),
                              weights='quadratic')
            for model in models
        ]

    with tqdm_joblib(total=n_iter, desc=r"Bootstrapping", leave=False):
        kappas_2d = Parallel(n_jobs=n_jobs)(
            delayed(resample_and_compute_kappa)(df, truth_col, models, seed)
            for seed in seeds
        )
    kappa_dict = dict(zip(models, zip(*kappas_2d)))
    return kappa_dict

def calculate_delta_kappa(
    df: pd.DataFrame, categories: List[str], reference_groups: Dict[str, Any], valid_groups: Dict[str, List[Any]],
    truth_col: str, ai_columns: List[str], n_iter: int = 1000, base_seed: Optional[int] = None
) -> Dict[str, Dict[str, Dict[Any, Tuple[float, Tuple[float, float]]]]]:
    """
    Calculate delta kappa (difference between group and reference) with bootstrap confidence intervals.

    :arg df: DataFrame containing the truth and AI columns.
    :arg categories: List of category column names.
    :arg reference_groups: Dictionary of reference groups for each category.
    :arg valid_groups: Dictionary of valid groups for each category.
    :arg truth_col: Column name of the truth labels.
    :arg ai_columns: List of test column names.
    :arg n_iter: Number of bootstrap iterations.
    :arg base_seed: Base seed for reproducibility.

    :returns: Dictionary of delta quadratic weighted kappa values with 95% confidence intervals.
    """
    delta_kappas: Dict[str, Dict[str, Dict[Any, Tuple[float, Tuple[float, float]]]]] = {}
    rng = np.random.default_rng(base_seed)

    for category in tqdm(categories, desc=r"Categories", position=0):
        if category not in valid_groups:
            continue

        delta_kappas[category] = {model: {} for model in ai_columns}
        unique_values = df[category].unique().tolist()

        kappa_dicts = {}
        for value in tqdm(unique_values, desc=f"Category \033[1m{category}\033[0m Groups", leave=False, position=1):
            if value not in valid_groups[category]:
                continue

            filtered_df = df[df[category] == value]
            group_seed = int(rng.integers(0, 1_000_000))

            kappa_dicts[value] = bootstrap_kappa(
                filtered_df,
                truth_col,
                ai_columns,
                n_iter,
                base_seed=group_seed,
            )

        # Remove and store reference bootstraps.
        ref_bootstraps = kappa_dicts.pop(reference_groups[category])

        # Now calculate the differences.
        for value, kappa_dict in kappa_dicts.items():
            for model in ai_columns:
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
    for model in ai_models:
        groups, values, lower_bounds, upper_bounds = extract_plot_data(delta_kappas, model)
        fig = plot_spider_chart(groups, values, lower_bounds, upper_bounds, model, global_min, global_max,
                                metric=r"QWK", plot_config=plot_config)
        figures.append(fig)

    display_figures_grid(figures)
    plt.show()

def print_table_from_dict(delta_kappas: Dict[str, Dict[str, Dict[Any, Tuple[float, Tuple[float, float]]]]],
                          tablefmt: str = r"grid") -> None:
    """
    Print a table of delta kappas that have a 95% CI excluding zero.
    Negative delta values are printed in maroon, positive in green.

    :arg delta_kappas: Dictionary of delta kappa values with 95% confidence intervals.
    :arg tablefmt: Table format string for tabulate.
    """
    results = []
    # ANSI escape codes for maroon and green using 24-bit RGB colors
    maroon = "\033[38;2;128;0;0m"
    green = "\033[38;2;0;128;0m"
    reset = "\033[0m"

    for category, model_data in delta_kappas.items():
        for model, groups in model_data.items():
            for group, (delta, (lower_ci, upper_ci)) in groups.items():
                if lower_ci > 0 or upper_ci < 0:
                    # Color the delta value based on its sign
                    color = maroon if delta < 0 else green
                    results.append([
                        model,
                        category,
                        group,
                        f"{color}{delta:.4f}{reset}",
                        f"{color}{lower_ci:.4f}{reset}",
                        f"{color}{upper_ci:.4f}{reset}"
                    ])
    results.sort(key=lambda row: (row[0], row[1], row[2]))
    if results:
        print(r"Delta Kappa values with 95% CI excluding zero:")
        headers = [r"Model", r"Category", r"Group", r"Delta Kappa", r"Lower CI", r"Upper CI"]
        print(tabulate(results, headers=headers, tablefmt=tablefmt))
    else:
        print(r"No model/group combinations with a CI excluding zero.")

if __name__ == '__main__':
    with open(r'config.yaml', r'r', encoding=r'utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)

    matched_df, categories, test_cols = create_matched_df_from_files(config[r'input data'], config[r'numeric_cols'])
    reference_groups, valid_groups, _ = determine_valid_n_reference_groups(matched_df, categories)
    bootstrap_config = config.get(r'bootstrap', {})
    base_seed = bootstrap_config.get(r'seed', None)
    n_iter = bootstrap_config.get(r'iterations', 1000)
    truth_col = config[r'input data'].get(r'truth column', r'truth')

    kappas, intervals = calculate_kappas_and_intervals(matched_df, truth_col, test_cols, n_iter=n_iter, base_seed=base_seed)
    print(r"Bootstrapping delta Kappas, this may take a while", flush=True)
    delta_kappas = calculate_delta_kappa(matched_df, categories, reference_groups, valid_groups,
                                         truth_col, test_cols, n_iter=n_iter, base_seed=base_seed)
    print_table_from_dict(delta_kappas, tablefmt=r"rounded_outline")
    generate_plots_from_delta_kappas(delta_kappas, test_cols, plot_config=config[r'plot'])
    save_pickled_data(config[r'output'], r"QWK", delta_kappas)
