"""This script generates EOD and AAOD spider plots for multiple models across different categories."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import resample
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import yaml

from data_loading import determine_valid_n_reference_groups, create_matched_df_from_files, save_pickled_data, check_required_columns
from plot_tools import plot_spider_chart, display_figures_grid
from typing import List, Dict, Any, Tuple, Optional, Union

def binarize_scores(df: pd.DataFrame, truth_col: str, ai_cols: Union[List[str], str], threshold: int = 4) -> pd.DataFrame:
    """
    Binarize scores based on a threshold for truth and AI columns.
    Converts values greater than or equal to threshold to 1, else 0.

    :arg df: DataFrame containing truth and test columns.
    :arg truth_col: Name of the truth column.
    :arg ai_cols: Name of the test column or a list of test columns.
    :arg threshold: Threshold value for binarization.

    :returns: DataFrame with binarized columns.
    """
    if not isinstance(ai_cols, list):
        ai_cols = [ai_cols]
    cols = [truth_col] + ai_cols
    check_required_columns(df, cols)
    df[cols] = (df[cols] >= threshold).astype(int)
    return df

def resample_by_column(df: pd.DataFrame, col: str, seed: int) -> pd.DataFrame:
    """
    Resample each group in a DataFrame by the specified column
    using the same seed across groups.

    :arg df: DataFrame to resample.
    :arg col: Column to group by.
    :arg seed: Seed for reproducibility across groups.

    :returns: Resampled DataFrame.
    """
    sampled_groups = [
        resample(group_df, replace=True, n_samples=len(group_df), random_state=seed)
        for _, group_df in df.groupby(col)
    ]
    return pd.concat(sampled_groups)

def compute_bootstrap_eod_aaod(
    df: pd.DataFrame,
    category: str,
    ref_group: Any,
    group_value: Any,
    truth_col: str,
    ai_columns: List[str],
    seed: int
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap estimates for EOD and AAOD metrics.

    :arg df: DataFrame containing truth and test columns.
    :arg category: Column to group by.
    :arg ref_group: Reference group value.
    :arg group_value: Group value to compare against reference.
    :arg truth_col: Name of the truth column.
    :arg ai_columns: List of test columns.
    :arg seed: Seed for reproducibility.

    :returns: Dictionary of EOD and AAOD values for each model.
    """
    sample_df = resample_by_column(df, category, seed)
    ref_df = sample_df[sample_df[category] == ref_group]
    group_df = sample_df[sample_df[category] == group_value]

    # Precompute truth masks for both reference and group DataFrames
    ref_truth_pos = (ref_df[truth_col] == 1)
    ref_truth_neg = ~ref_truth_pos
    group_truth_pos = (group_df[truth_col] == 1)
    group_truth_neg = ~group_truth_pos

    results: Dict[str, Tuple[float, float]] = {}
    for model in ai_columns:
        ref_pred = (ref_df[model] == 1)
        group_pred = (group_df[model] == 1)

        tpr_ref = ref_pred[ref_truth_pos].sum() / ref_truth_pos.sum() if ref_truth_pos.sum() else np.nan
        fpr_ref = ref_pred[ref_truth_neg].sum() / ref_truth_neg.sum() if ref_truth_neg.sum() else np.nan
        tpr_group = group_pred[group_truth_pos].sum() / group_truth_pos.sum() if group_truth_pos.sum() else np.nan
        fpr_group = group_pred[group_truth_neg].sum() / group_truth_neg.sum() if group_truth_neg.sum() else np.nan

        eod = tpr_group - tpr_ref
        aaod = 0.5 * (abs(fpr_group - fpr_ref) + abs(tpr_group - tpr_ref))
        results[model] = (eod, aaod)

    return results

def calculate_eod_aaod(
    df: pd.DataFrame,
    categories: List[str],
    reference_groups: Dict[str, Any],
    valid_groups: Dict[str, List[Any]],
    truth_col: str,
    ai_columns: List[str],
    n_iter: int = 1000,
    base_seed: Optional[int] = None
) -> Dict[str, Dict[str, Dict[Any, Dict[str, Any]]]]:
    """
    Calculate EOD and AAOD metrics with bootstrap iterations for multiple categories.

    :arg df: DataFrame containing truth and test columns.
    :arg categories: List of columns to group by.
    :arg reference_groups: Dictionary of reference groups for each category.
    :arg valid_groups: Dictionary of valid groups for each category.
    :arg truth_col: Name of the truth column.
    :arg ai_columns: List of test columns.
    :arg n_iter: Number of bootstrap iterations.
    :arg base_seed: Base seed for reproducibility.

    :returns: Dictionary of EOD and AAOD values for each model.
    """
    eod_aaod: Dict[str, Dict[str, Dict[Any, Dict[str, Any]]]] = {
        category: {model: {} for model in ai_columns} for category in categories
    }
    rng = np.random.default_rng(base_seed)

    for category in tqdm(categories, desc='Categories', position=0):
        if category not in valid_groups:
            continue

        ref_group = reference_groups[category]
        unique_values = df[category].unique()

        for group_value in tqdm(unique_values, desc=f"Category \'{category}\' Groups", leave=False, position=1):
            if group_value == ref_group or group_value not in valid_groups[category]:
                continue

            eod_samples = {model: [] for model in ai_columns}
            aaod_samples = {model: [] for model in ai_columns}

            # Preassign seeds for each bootstrap iteration.
            seeds = rng.integers(0, 1_000_000, size=n_iter)

            with tqdm_joblib(total=n_iter, desc=f"Bootstrapping \'{group_value}\' Group", leave=False):
                bootstrap_results = Parallel(n_jobs=-1)(
                    delayed(compute_bootstrap_eod_aaod)(
                        df, category, ref_group, group_value, truth_col, ai_columns, seed
                    ) for seed in seeds
                )

            for result in bootstrap_results:
                for model in ai_columns:
                    eod_samples[model].append(result[model][0])
                    aaod_samples[model].append(result[model][1])

            for model in ai_columns:
                eod_median = np.median(eod_samples[model])
                aaod_median = np.median(aaod_samples[model])
                eod_ci = np.percentile(eod_samples[model], [2.5, 97.5])
                aaod_ci = np.percentile(aaod_samples[model], [2.5, 97.5])
                eod_aaod[category][model][group_value] = {
                    'eod': (eod_median, eod_ci),
                    'aaod': (aaod_median, aaod_ci)
                }
    return eod_aaod

def extract_plot_data_eod_aaod(
    eod_aaod: Dict[str, Dict[str, Dict[Any, Dict[str, Any]]]],
    model: str,
    metric: str = 'eod'
) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Extract groups, metric values and confidence intervals for plotting.

    :arg eod_aaod: Dictionary of EOD and AAOD values for each model.
    :arg model: Name of the model to extract data for.
    :arg metric: Metric to extract data for (EOD or AAOD).

    :returns: Tuple of groups, values, lower bounds and upper bounds.
    """
    groups: List[str] = []
    values: List[float] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    for category, model_data in eod_aaod.items():
        if model in model_data:
            for group, metrics in model_data[model].items():
                groups.append(f"{category}: {group}")
                value, (lower, upper) = metrics[metric]
                values.append(value)
                lower_bounds.append(lower)
                upper_bounds.append(upper)

    return groups, values, lower_bounds, upper_bounds

def generate_plot_data_eod_aaod(
    eod_aaod: Dict[str, Dict[str, Dict[Any, Dict[str, Any]]]],
    test_cols: List[str],
    metrics: List[str] = ('eod', 'aaod')
) -> Tuple[Dict[str, Dict[str, Tuple[List[str], List[float], List[float], List[float]]]], float, float]:
    """
    Generate plot data for each metric and compute global axis limits.

    :arg eod_aaod: Dictionary of EOD and AAOD values for each model.
    :arg test_cols: List of test columns.
    :arg metrics: List of metrics to plot.

    :returns: Tuple of plot data dictionary, global minimum and maximum values.
    """
    plot_data_dict: Dict[str, Dict[str, Tuple[List[str], List[float], List[float], List[float]]]] = {}
    all_values: List[float] = []

    for metric in metrics:
        plot_data_dict[metric] = {}
        for model in test_cols:
            groups, values, lower, upper = extract_plot_data_eod_aaod(eod_aaod, model, metric)
            plot_data_dict[metric][model] = (groups, values, lower, upper)
            all_values.extend(lower + upper)

    global_min, global_max = min(all_values) - 0.05, max(all_values) + 0.05
    return plot_data_dict, global_min, global_max

def plot_data_eod_aaod(
    plot_data_dict: Dict[str, Dict[str, Tuple[List[str], List[float], List[float], List[float]]]],
    test_cols: List[str],
    metrics: List[str] = ('eod', 'aaod'),
    plot_config: Optional[Dict[str, Any]] = None,
    global_min: float = 0.0,
    global_max: float = 1.0
) -> Dict[str, List[Any]]:
    """
    Plot EOD and AAOD spider charts for each model.

    :arg plot_data_dict: Dictionary of plot data for each metric and model.
    :arg test_cols: List of test columns.
    :arg metrics: List of metrics to plot.
    :arg plot_config: Optional configuration for plotting.
    :arg global_min: Global minimum value for the y-axis.
    :arg global_max: Global maximum value for the y-axis.

    :returns: Dictionary of generated figures for each metric.
    """
    figures_dict: Dict[str, List[Any]] = {metric: [] for metric in metrics}
    for metric in metrics:
        for model in test_cols:
            groups, values, lower, upper = plot_data_dict[metric][model]
            fig = plot_spider_chart(groups, values, lower, upper, model, global_min, global_max,
                                    metric=metric, plot_config=plot_config)
            figures_dict[metric].append(fig)

    for figures in figures_dict.values():
        display_figures_grid(figures)

    return figures_dict

if __name__ == '__main__':
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)

    matched_df, categories, test_cols = create_matched_df_from_files(config['input data'], config['numeric_cols'])
    reference_groups, valid_groups, _ = determine_valid_n_reference_groups(matched_df, categories)
    truth_col = config['input data'].get('truth column', 'truth')

    # Check required columns before further processing
    required_columns = [truth_col] + test_cols + categories
    check_required_columns(matched_df, required_columns)

    # Binarize scores
    matched_df = binarize_scores(matched_df, truth_col, test_cols, threshold=4)

    # EOD & AAOD calculation
    bootstrap_config = config.get('bootstrap', {})
    base_seed = bootstrap_config.get('seed', None)
    n_iter = bootstrap_config.get('iterations', 1000)
    eod_aaod = calculate_eod_aaod(matched_df, categories, reference_groups, valid_groups,
                                  truth_col, test_cols, n_iter=n_iter, base_seed=base_seed)

    metrics = ['eod', 'aaod']
    plot_data_dict, global_min, global_max = generate_plot_data_eod_aaod(eod_aaod, test_cols, metrics=metrics)
    figures_dict = plot_data_eod_aaod(plot_data_dict, test_cols, metrics=metrics, plot_config=config['plot'],
                                      global_min=global_min, global_max=global_max)

    plt.show()

    for metric in metrics:
        save_pickled_data(config['output'], metric, plot_data_dict[metric])
