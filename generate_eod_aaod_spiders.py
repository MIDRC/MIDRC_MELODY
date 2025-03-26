from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import yaml

from data_loading import determine_valid_n_reference_groups, create_matched_df_from_files, save_pickled_data
from plot_tools import plot_spider_chart, display_figures_grid


# Helper to dynamically binarize based on threshold (e.g., 4)
def binarize_scores(df, truth_col, ai_cols, threshold=4):
    if not isinstance(ai_cols, list):
        ai_cols = [ai_cols]

    for col in [truth_col] + ai_cols:
        df[col] = (df[col] >= threshold).astype(int)
    return df


def resample_by_column(df, col, seed):
    # Resample each group by column using the same seed across groups.
    sampled_groups = [
        resample(group_df, replace=True, n_samples=len(group_df), random_state=seed)
        for _, group_df in df.groupby(col)
    ]
    return pd.concat(sampled_groups)


def compute_bootstrap_eod_aaod(df, category, ref_group, value, truth_col, ai_columns, seed):
    sample_df = resample_by_column(df, category, seed)
    ref_df = sample_df[sample_df[category] == ref_group]
    group_df = sample_df[sample_df[category] == value]

    # Precompute truth masks for both reference and group DataFrames
    ref_truth_pos = (ref_df[truth_col] == 1)
    ref_truth_neg = ~ref_truth_pos  # equivalent to ref_df[truth_col] == 0
    group_truth_pos = (group_df[truth_col] == 1)
    group_truth_neg = ~group_truth_pos  # equivalent to group_df[truth_col] == 0

    results = {}
    for model in ai_columns:
        ref_pred = (ref_df[model] == 1)
        group_pred = (group_df[model] == 1)

        # Use the precomputed masks to calculate sums for numerator and denominator
        tpr_ref = ref_pred[ref_truth_pos].sum() / ref_truth_pos.sum() if ref_truth_pos.sum() else np.nan
        fpr_ref = ref_pred[ref_truth_neg].sum() / ref_truth_neg.sum() if ref_truth_neg.sum() else np.nan
        tpr_group = group_pred[group_truth_pos].sum() / group_truth_pos.sum() if group_truth_pos.sum() else np.nan
        fpr_group = group_pred[group_truth_neg].sum() / group_truth_neg.sum() if group_truth_neg.sum() else np.nan

        eod = tpr_group - tpr_ref
        aaod = 0.5 * (abs(fpr_group - fpr_ref) + abs(tpr_group - tpr_ref))
        results[model] = (eod, aaod)

    return results


def calculate_eod_aaod(df, categories, reference_groups, valid_groups, truth_col, ai_columns, n_iter=1000, base_seed=None):
    eod_aaod = {category: {model: {} for model in ai_columns} for category in categories}
    rng = np.random.RandomState(base_seed)  # For reproducibility

    for category in tqdm(categories, desc='Categories', position=0):
        if category not in valid_groups:
            continue

        ref_group = reference_groups[category]
        unique_values = df[category].unique()

        for value in tqdm(unique_values, desc=f"Category \'{category}\' Groups", leave=False, position=1):
            if value == ref_group or value not in valid_groups[category]:
                continue

            eod_samples = {model: [] for model in ai_columns}
            aaod_samples = {model: [] for model in ai_columns}

            # Preassign seeds for each bootstrap iteration.
            seeds = rng.randint(0, 1_000_000, size=n_iter)

            # Run bootstrap iterations in parallel.
            with tqdm_joblib(total=n_iter, desc=f"Bootstrapping \'{value}\' Group", leave=False):
                bootstrap_results = Parallel(n_jobs=-1)(
                    delayed(compute_bootstrap_eod_aaod)(
                        df, category, ref_group, value, truth_col, ai_columns, seed
                    ) for seed in seeds
                )

            # Collect bootstrap samples.
            for result in bootstrap_results:
                for model in ai_columns:
                    eod_samples[model].append(result[model][0])
                    aaod_samples[model].append(result[model][1])

            # Compute median and 95% confidence intervals.
            for model in ai_columns:
                eod_median = np.median(eod_samples[model])
                aaod_median = np.median(aaod_samples[model])
                eod_ci = np.percentile(eod_samples[model], [2.5, 97.5])
                aaod_ci = np.percentile(aaod_samples[model], [2.5, 97.5])
                eod_aaod[category][model][value] = {
                    'eod': (eod_median, eod_ci),
                    'aaod': (aaod_median, aaod_ci)
                }
    return eod_aaod

# Data extraction for plotting (similar to delta kappa)
def extract_plot_data_eod_aaod(eod_aaod, model, metric='eod'):
    groups, values, lower_bounds, upper_bounds = [], [], [], []
    
    for attribute, attribute_data in eod_aaod.items():
        if model in attribute_data:
            for group, metrics in attribute_data[model].items():
                groups.append(f"{attribute}: {group}")
                value, (lower, upper) = metrics[metric]
                values.append(value)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
                
    return groups, values, lower_bounds, upper_bounds

def generate_plot_data_eod_aaod(eod_aaod, test_cols, metrics=['eod', 'aaod']):
    all_values = []
    plot_data_dict = {}
    for metric in metrics:
        plot_data_dict[metric] = {}
        for model in test_cols:
            groups, values, lower, upper = extract_plot_data_eod_aaod(eod_aaod, model, metric)
            plot_data_dict[metric][model] = (groups, values, lower, upper)
            all_values.extend(lower + upper)

    global_min, global_max = min(all_values) - 0.05, max(all_values) + 0.05

    return plot_data_dict, global_min, global_max

def plot_data_eod_aaod(plot_data_dict, test_cols, metrics=['eod', 'aaod'], plot_config=None):
    figures_dict = {metric: [] for metric in metrics}
    for metric in metrics:
        metric_dict = plot_data_dict[metric]
        for model in test_cols:
            groups, values, lower, upper = metric_dict[model]
            fig = plot_spider_chart(groups, values, lower, upper, model, global_min, global_max, metric=metric, plot_config=plot_config)
            figures_dict[metric].append(fig)

    for _, figures in figures_dict.items():
        display_figures_grid(figures)

    return figures_dict

# Example pipeline (wrap this into main if you want)
if __name__ == '__main__':
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)

    matched_df, categories, test_cols = create_matched_df_from_files(config['input data'], config['numeric_cols'])

    reference_groups, valid_groups, _ = determine_valid_n_reference_groups(matched_df, categories)
    truth_col = config['input data'].get('truth column', 'truth')

    # Binarize
    matched_df = binarize_scores(matched_df, truth_col, test_cols, threshold=4)

    # EOD & AAOD Calculation
    bootstrap_config = config.get('bootstrap', {})
    rand_seed = bootstrap_config.get('seed', None)
    n_iter = bootstrap_config.get('iterations', 1000)
    eod_aaod = calculate_eod_aaod(matched_df, categories, reference_groups, valid_groups, truth_col, test_cols, n_iter=n_iter, base_seed=rand_seed)

    metrics = ['eod', 'aaod']
    plot_data_dict, global_min, global_max = generate_plot_data_eod_aaod(eod_aaod, test_cols, metrics=metrics)

    # Plot all models
    figures_dict = plot_data_eod_aaod(plot_data_dict, test_cols, metrics=metrics, plot_config=config['plot'])

    plt.show()  # Show all figures at once

    for metric in metrics:
        save_pickled_data(config['output'], metric, plot_data_dict[metric])
