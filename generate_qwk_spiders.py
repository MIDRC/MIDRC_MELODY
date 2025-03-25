from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample
from tabulate import tabulate
from tqdm import tqdm
import yaml

from data_loading import create_matched_df_from_files, determine_valid_n_reference_groups, save_pickled_data
from plot_tools import plot_spider_chart, display_figures_grid

# Step 6: Calculate kappa and bootstrap confidence intervals
def calculate_kappas_and_intervals(df, truth_col, ai_cols, n_iter=1000, base_seed=None):
    np.random.seed(base_seed)  # For reproducibility
    kappas = {}
    intervals = {}
    y_true = df[truth_col]
    y_true_np = np.array(y_true.tolist(), dtype=int)
    for col in ai_cols:
        y_pred = df[col]
        y_pred_np = np.array(y_pred.tolist(), dtype=int)
        kappa = cohen_kappa_score(y_true_np, y_pred_np, weights='quadratic')
        kappas[col] = kappa

        kappa_scores = []
        for _ in range(n_iter):
            indices = np.random.choice(len(y_true_np), len(y_true_np), replace=True)
            kappa_bs = cohen_kappa_score(y_true_np[indices], y_pred_np[indices], weights='quadratic')
            kappa_scores.append(kappa_bs)
        kappa_scores = sorted(kappa_scores)
        lower_bnd = kappa_scores[int(0.025 * n_iter)]
        upper_bnd = kappa_scores[int(0.975 * n_iter)]
        intervals[col] = (lower_bnd, upper_bnd)
        print(f"Model: {col} | Kappa: {kappa:.4f} | 95% CI: ({lower_bnd:.4f}, {upper_bnd:.4f}) N: {len(y_true_np)}")
        
    return kappas, intervals

# Custom bootstrap kappa
def bootstrap_kappa(df, truth_col, model, n_iter=1000, n_jobs=-1, base_seed=None):
    # Generate unique seeds for each iteration from the base seed
    seeds = np.random.RandomState(base_seed).randint(0, 1_000_000, size=n_iter)

    def resample_and_compute_kappa(df, model, seed):
        sampled_df = resample(df, replace=True, random_state=seed)
        return cohen_kappa_score(sampled_df[truth_col], sampled_df[model], weights='quadratic')

    # Use Parallel to run the bootstrap iterations in parallel
    kappas = Parallel(n_jobs=n_jobs)(delayed(resample_and_compute_kappa)(df, model, seed) for seed in seeds)

    return kappas

# Custom bootstrap kappa
def bootstrap_kappa_by_columns(df, truth_col, model, columns, n_iter=1000, n_jobs=-1, base_seed=None):
    # Ensure columns is a list; if not, wrap it in a list.
    if not isinstance(columns, list):
        columns = [columns]

    # Generate unique seeds for each iteration from the base seed
    seeds = np.random.RandomState(base_seed).randint(0, 1_000_000, size=n_iter)

    def resample_and_compute_kappa(df, truth_col, model, columns, seed):
        sampled_groups = []
        for group, group_df in df.groupby(columns):
            n_samples = len(group_df)
            sampled_group = resample(group_df, replace=True, n_samples=n_samples, random_state=seed)
            sampled_groups.append(sampled_group)
        sampled_df = pd.concat(sampled_groups)
        return cohen_kappa_score(sampled_df[truth_col], sampled_df[model], weights='quadratic')

    kappas = Parallel(n_jobs=n_jobs)(
        delayed(resample_and_compute_kappa)(df, truth_col, model, columns, seed)
        for seed in seeds
    )

# Step 7: Calculate delta kappa
def calculate_delta_kappa(df, categories, reference_groups, valid_groups, truth_col, ai_columns, n_iter=1000, base_seed=None):
    delta_kappas = {}

    for category in tqdm(categories, desc='Categories', position=0):
        # Skip if the category is not in the valid groups
        if category not in valid_groups:
            continue

        delta_kappas[category] = {model: {} for model in ai_columns}
        unique_values = df[category].unique()

        ref_filtered_df = df[df[category] == reference_groups[category]]

        for model in tqdm(ai_columns, desc=f"Models", leave=False, position=1):
            # for value in tqdm(unique_values, desc=f"Category '{category}' Groups", leave=False, position=1):
            #     if value == reference_groups[category]:
            #         continue
            kappas_ref = bootstrap_kappa(ref_filtered_df, truth_col, model, n_iter, base_seed=base_seed)

            for value in tqdm(unique_values, desc=f"Category '{category}' Groups", leave=False, position=2):
            # for model in tqdm(ai_columns, desc=f"Models for '{value} Group", leave=False, position=2):
                # Skip if the value is the reference group
                if value == reference_groups[category]:
                    continue
                # Skip if the value is not in the valid groups
                if value not in valid_groups[category]:
                    continue

                filtered_df = df[df[category] == value]

                kappas = bootstrap_kappa(filtered_df, truth_col, model, n_iter, base_seed=base_seed)

                deltas = [a - b for a, b in zip(kappas, kappas_ref)]
                delta_median = np.percentile(deltas, 50)
                lower_value, upper_value = np.percentile(deltas, [2.5, 97.5])
                delta_kappas[category][model][value] = (delta_median, (lower_value, upper_value))

    return delta_kappas

def extract_plot_data(delta_kappas, model_name):
    groups = []
    values = []
    lower_bounds = []
    upper_bounds = []
    
    for attribute, attribute_data in delta_kappas.items():
        if model_name in attribute_data:
            for group, (value, ci) in attribute_data[model_name].items():
                groups.append(f"{attribute}: {group}")
                values.append(value)
                lower_bounds.append(ci[0])
                upper_bounds.append(ci[1])
                
    return groups, values, lower_bounds, upper_bounds


def generate_plots_from_delta_kappas(delta_kappas, ai_models, plot_config=None):
    # Determine the global range across all models for consistent scaling
    all_values = []
    all_lower = []
    all_upper = []

    for model in ai_models:
        _, values, lower_bounds, upper_bounds = extract_plot_data(delta_kappas, model)
        all_values.extend(values)
        all_lower.extend(lower_bounds)
        all_upper.extend(upper_bounds)

    global_min = min(all_lower) - 0.05  # Padding for better visualization
    global_max = max(all_upper) + 0.05

    # Plot for each AI model
    figures = []
    for model in ai_models:
        groups, values, lower_bounds, upper_bounds = extract_plot_data(delta_kappas, model)
        fig = plot_spider_chart(groups, values, lower_bounds, upper_bounds, model, global_min, global_max, metric='QWK', plot_config=plot_config)
        figures.append(fig)

    display_figures_grid(figures)

    # Finally, show all charts with one plt.show() call
    plt.show()

def print_table_from_dict(delta_kappas, tablefmt="grid"):
    from tabulate import tabulate

    # Gather results that meet the condition that 0 is not in the CI.
    results = []
    for category, model_data in delta_kappas.items():
        for model, groups in model_data.items():
            for group, (delta, (lower_ci, upper_ci)) in groups.items():
                # Only include when the CI excludes 0.
                if lower_ci > 0 or upper_ci < 0:
                    results.append([
                        model,
                        category,
                        group,
                        round(delta, 4),
                        round(lower_ci, 4),
                        round(upper_ci, 4)
                    ])

    # Sort results by model, category, then group.
    results.sort(key=lambda row: (row[0], row[1], row[2]))

    # Print the table if there are any entries.
    if results:
        print(f"Delta Kappa values with 95% CI excluding zero:")
        headers = ["Model", "Category", "Group", "Delta Kappa", "Lower CI", "Upper CI"]
        print(tabulate(results, headers=headers, tablefmt=tablefmt))
    else:
        print("No model/group combinations with a CI excluding zero.")

# Main execution
if __name__ == '__main__':
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)

    matched_df, categories, test_cols = create_matched_df_from_files(config['input data'], config['numeric_cols'])

    reference_groups, valid_groups, _ = determine_valid_n_reference_groups(matched_df, categories)

    bootstrap_config = config.get('bootstrap', {})
    rand_seed = bootstrap_config.get('seed', None)
    n_iter = bootstrap_config.get('iterations', 1000)

    truth_col = config['input data'].get('truth column', 'truth')
    kappas, intervals = calculate_kappas_and_intervals(matched_df, truth_col, test_cols, base_seed=rand_seed)

    # Calculate delta Kappas
    print(f"Bootstrapping delta Kappas, this may take a while", flush=True)
    delta_kappas = calculate_delta_kappa(matched_df,
                                         categories,
                                         reference_groups,
                                         valid_groups,
                                         truth_col,
                                         test_cols,
                                         n_iter=n_iter,
                                         base_seed=rand_seed,
                                         )
    print_table_from_dict(delta_kappas, tablefmt="rounded_outline")

    generate_plots_from_delta_kappas(delta_kappas, test_cols, plot_config=config['plot'])

    save_pickled_data(config['output'], 'QWK', delta_kappas)
