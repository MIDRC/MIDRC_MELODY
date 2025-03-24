import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from setuptools.package_index import unique_values
from sklearn.utils import resample
from tqdm import tqdm
import yaml

from data_loading import determine_validNreference_groups, create_matched_df_from_files
from plot_tools import plot_spider_chart, display_figures_grid


# Helper to dynamically binarize based on threshold (e.g., 4)
def binarize_scores(df, ai_cols, threshold=4):
    for col in ai_cols + ['truth']:
        df[col] = (df[col] >= threshold).astype(int)
    return df

# EOD calculation function
def calculate_eod_aaod(df, categories, reference_groups, ai_columns, n_iter=1000):
    eod_aaod = {category: {model: {} for model in ai_columns} for category in categories}

    for category in tqdm(categories, desc='Categories', position=0):
        ref_group = reference_groups[category]
        unique_values = df[category].unique()

        for value in tqdm(unique_values, desc=f"Category '{category}' Groups", leave=False, position=1):
            if value == ref_group:
                continue
            # Bootstrapping
            eod_samples, aaod_samples = {}, {}
            for model in ai_columns:
                eod_samples[model] = []
                aaod_samples[model] = []

            def resample_by_column(df, columns=category, seed=None):
                sampled_groups = []
                for group, group_df in df.groupby(columns):
                    n_samples = len(group_df)
                    sampled_group = resample(group_df, replace=True, n_samples=n_samples, random_state=seed)
                    sampled_groups.append(sampled_group)
                sampled_df = pd.concat(sampled_groups)
                return sampled_df

            for _ in tqdm(range(n_iter), desc="Bootstraps", leave=False, position=2):
                sample_df = resample_by_column(df)

                ref_df = sample_df[sample_df[category] == ref_group]
                group_df = sample_df[sample_df[category] == value]

                for model in ai_columns:
                    # TPR & FPR for reference group
                    tpr_ref = ref_df[(ref_df['truth'] == 1) & (ref_df[model] == 1)].shape[0] / (ref_df['truth'] == 1).sum()
                    fpr_ref = ref_df[(ref_df['truth'] == 0) & (ref_df[model] == 1)].shape[0] / (ref_df['truth'] == 0).sum()

                    # TPR & FPR for group
                    tpr_group = group_df[(group_df['truth'] == 1) & (group_df[model] == 1)].shape[0] / (group_df['truth'] == 1).sum()
                    fpr_group = group_df[(group_df['truth'] == 0) & (group_df[model] == 1)].shape[0] / (group_df['truth'] == 0).sum()

                    eod_samples[model].append(tpr_group - tpr_ref)
                    aaod_samples[model].append(0.5 * (abs(fpr_group - fpr_ref) + abs(tpr_group - tpr_ref)))

            for model in ai_columns:
                # Median and 95% CI
                eod_median, aaod_median = np.median(eod_samples[model]), np.median(aaod_samples[model])
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


# Example pipeline (wrap this into main if you want)
if __name__ == '__main__':
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)

    matched_df, categories = create_matched_df_from_files(config['input data'], config['numeric_cols'])

    reference_groups, valid_groups, _ = determine_validNreference_groups(matched_df, categories)
    ai_cols = [col for col in matched_df.columns if col.startswith('ai_')]

    # Binarize
    matched_df = binarize_scores(matched_df, ai_cols, threshold=4)

    # EOD & AAOD Calculation
    eod_aaod = calculate_eod_aaod(matched_df, categories, reference_groups, ai_cols)

    # Global scaling range
    all_values = []
    plot_data_dict = {}
    for model in ai_cols:
        plot_data_dict[model] = {}
        for metric in ['eod', 'aaod']:
            groups, values, lower, upper = extract_plot_data_eod_aaod(eod_aaod, model, metric)
            plot_data_dict[model][metric] = (groups, values, lower, upper)
            all_values.extend(lower + upper)

    global_min, global_max = min(all_values) - 0.05, max(all_values) + 0.05

    # Plot all models
    figures_dict = {'eod': [], 'aaod': []}
    for model in ai_cols:
        for metric in ['eod', 'aaod']:
            groups, values, lower, upper = plot_data_dict[model][metric]
            fig = plot_spider_chart(groups, values, lower, upper, model, global_min, global_max, metric=metric)
            figures_dict[metric].append(fig)

    for _, figures in figures_dict.items():
        display_figures_grid(figures)

    plt.show()  # Show all figures at once
