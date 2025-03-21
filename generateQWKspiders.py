import pickle
import time

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample
from tqdm import tqdm
import yaml

from data_preprocessing import bin_dataframe_column

# Step 3: Bin numerical columns like age
def bin_data(df, bins_config):
    for column, bin_details in bins_config.items():
        df[column] = pd.cut(df[column], bins=bin_details['bins'], labels=bin_details['labels'], right=False)
    return df

# Step 4: Match case names between file1 and file2
def match_cases(df1, df2, column='case_name'):
    merged_df = df1.merge(df2, on=column, how='inner', suffixes=('_truth', '_ai'))
    return merged_df

# CDC age bin configuration
age_bins = {
    'age': {
        'bins': [0, 18, 30, 40, 50, 65, 75, 85, np.inf],
        'labels': ['<18', '18-29', '30-39', '40-49', '50-64', '65-74', '75-84', '85+']
    }
}

# Step 5: Determine reference groups
def determine_validNreference_groups(df, categories, min_count=10):
    if isinstance(categories, pd.Index):
        categories = categories.to_list()

    reference_groups = {}
    valid_groups = {}
    
    for category in categories:
        valid_groups[category] = {}
        category_counts = df[category].value_counts()

        for value in category_counts.index:
            if category_counts[value] >= min_count and value != 'Not Reported':
                valid_groups[category][value] = category_counts[value]

        if valid_groups[category]:
            reference_groups[category] = max(valid_groups[category], key=valid_groups[category].get)

    # Filter the DataFrame based on valid groups
    filtered_df = df.copy()
    for category in categories:
        valid_values = list(valid_groups[category].keys())
        filtered_df = filtered_df[filtered_df[category].isin(valid_values)]

    return reference_groups, valid_groups, filtered_df

# Step 6: Calculate kappa and bootstrap confidence intervals
def calculate_kappas_and_intervals(df, ai_cols, n_iter=1000):
    kappas = {}
    intervals = {}
    y_true = df['truth']
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
def bootstrap_kappa(df, model, n_iter=1000, n_jobs=-1, base_seed=42):
    # Generate unique seeds for each iteration from the base seed
    seeds = np.random.RandomState(base_seed).randint(0, 1_000_000, size=n_iter)

    def resample_and_compute_kappa(df, model, seed):
        sampled_df = resample(df, replace=True, random_state=seed)
        return cohen_kappa_score(sampled_df['truth'], sampled_df[model])

    # Use Parallel to run the bootstrap iterations in parallel
    kappas = Parallel(n_jobs=n_jobs)(delayed(resample_and_compute_kappa)(df, model, seed) for seed in seeds)

    return kappas

# Custom bootstrap kappa
def bootstrap_kappa_by_columns(df, model, columns, n_iter=1000, n_jobs=-1, base_seed=42):
    # Ensure columns is a list; if not, wrap it in a list.
    if not isinstance(columns, list):
        columns = [columns]

    # Generate unique seeds for each iteration from the base seed
    seeds = np.random.RandomState(base_seed).randint(0, 1_000_000, size=n_iter)

    def resample_and_compute_kappa(df, model, columns, seed):
        sampled_groups = []
        for group, group_df in df.groupby(columns):
            n_samples = len(group_df)
            sampled_group = resample(group_df, replace=True, n_samples=n_samples, random_state=seed)
            sampled_groups.append(sampled_group)
        sampled_df = pd.concat(sampled_groups)
        return cohen_kappa_score(sampled_df['truth'], sampled_df[model])

    kappas = Parallel(n_jobs=n_jobs)(
        delayed(resample_and_compute_kappa)(df, model, columns, seed)
        for seed in seeds
    )

# Step 7: Calculate delta kappa
def calculate_delta_kappa(df, categories, reference_groups, valid_groups, ai_columns, n_iter=1000):
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
            kappas_ref = bootstrap_kappa(ref_filtered_df, model, n_iter)

            for value in tqdm(unique_values, desc=f"Category '{category}' Groups", leave=False, position=2):
            # for model in tqdm(ai_columns, desc=f"Models for '{value} Group", leave=False, position=2):
                # Skip if the value is the reference group
                if value == reference_groups[category]:
                    continue
                # Skip if the value is not in the valid groups
                if value not in valid_groups[category]:
                    continue

                filtered_df = df[df[category] == value]

                kappas = bootstrap_kappa(filtered_df, model, n_iter)

                deltas = [a - b for a, b in zip(kappas, kappas_ref)]
                delta_median = np.percentile(deltas, 50)
                lower_value, upper_value = np.percentile(deltas, [2.5, 97.5])
                delta_kappas[category][model][value] = (delta_median, (lower_value, upper_value))

    return delta_kappas

def extract_ai_models(delta_kappas):
    models = set()
    for attribute_data in delta_kappas.values():
        for model_name in attribute_data:
            if model_name.startswith('ai_'):
                models.add(model_name)
    return sorted(models)

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

def plot_spider_chart(groups, values, lower_bounds, upper_bounds, model_name, global_min, global_max):
    # Sort groups so that within each attribute they appear in order
    def group_sort_key(label):
        attr, group = label.split(': ', 1)
        custom_orders = {
            'age': ['18-29', '30-39', '40-49', '50-64', '65-74', '75-84', '85+'],
            'sex': ['Male', 'Female'],
            'race': ['White', 'Black or African American', 'Asian', 'Other'],
            'ethnicity': ['Hispanic or Latino', 'Not Hispanic or Latino'],
            'white_nonhispanic': ['0', '1']
        }
        if attr in custom_orders:
            order = custom_orders[attr]
            return (attr, order.index(group)) if group in order else (attr, len(order))
        else:
            return (attr, group)

    combined = list(zip(groups, values, lower_bounds, upper_bounds))
    combined.sort(key=lambda x: group_sort_key(x[0]))
    groups, values, lower_bounds, upper_bounds = zip(*combined)

    num_axes = len(groups)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()

    # Close the loop for the plotted series
    values = list(values) + [values[0]]
    lower_bounds = list(lower_bounds) + [lower_bounds[0]]
    upper_bounds = list(upper_bounds) + [upper_bounds[0]]
    angles = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot the main line and markers
    ax.plot(angles, values, color='steelblue', linestyle='-', linewidth=2)
    ax.scatter(angles, values, marker='o', color='b')

    # Instead of drawing a line between the vertices for baseline,
    # generate a smooth circle at the 0-level.
    theta_full = np.linspace(0, 2 * np.pi, 100)
    baseline_circle = np.full_like(theta_full, 0)
    ax.plot(theta_full, baseline_circle, color='seagreen', linestyle='--', linewidth=3, alpha=0.8)

    ax.fill_between(angles, lower_bounds, upper_bounds, color='steelblue', alpha=0.2)
    ax.set_ylim(global_min, global_max)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(groups, fontsize=8, ha='center')
    ax.set_title(f'Spider Plot for {model_name}', size=14, weight='bold')

    # Emphasize tick labels based on bounds.
    # Iterate over the tick labels (skipping the last repeated label)
    for i, label in enumerate(ax.get_xticklabels()):
        if i >= len(lower_bounds) - 1:
            break
        angle = angles[i]
        # Calculate rotation so that the dash is perpendicular to the corresponding radius.
        # For angle=0, we want 90 degrees; for angle=pi/2, we want 0 degrees, etc.
        rotation_deg = 90 + np.degrees(angle)
        # If lower bound is greater than zero, mark with blue bold text
        if lower_bounds[i] > 0:
            label.set_fontweight('bold')
            label.set_color('maroon')
            # Add a maroon point at the lower bound for emphasis.
            ax.text(angle, lower_bounds[i], '—', color='maroon', fontsize=12,
                    rotation=rotation_deg, ha='center', va='center')
        # If upper bound is less than zero, mark with red bold text
        elif upper_bounds[i] < 0:
            label.set_fontweight('bold')
            label.set_color('red')
            # Add a red point at the upper bound for emphasis.
            ax.text(angle, upper_bounds[i], '—', color='red', fontsize=12,
                    rotation=rotation_deg, ha='center', va='center')

    plt.tight_layout()
    return fig


def create_matched_df_from_files(input_data, numeric_cols_dict, column='case_name'):
    df1 = pd.read_csv(input_data['file1'])
    df2 = pd.read_csv(input_data['file2'])

    categories = df1.columns[2:]

    # Bin numerical columns, specifically 'age'
    numeric_cols = config['numeric_cols']
    for str_col, col_dict in numeric_cols_dict.items():
        num_col = col_dict['raw column'] if 'raw column' in col_dict else str_col
        bins = col_dict['bins'] if 'bins' in col_dict else None
        labels = col_dict['labels'] if 'labels' in col_dict else None

        if num_col in df1.columns:
            df1 = bin_dataframe_column(df1, num_col, str_col, bins=bins, labels=labels)
            categories = categories.map(lambda x: str_col if x == num_col else x)

    return match_cases(df1, df2, column), categories


def generate_plots_from_delta_kappas(delta_kappas):
    ai_models = extract_ai_models(delta_kappas)  # in case some of the ai_cols were inconsistent

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
        fig = plot_spider_chart(groups, values, lower_bounds, upper_bounds, model, global_min, global_max)
        figures.append(fig)

    # Finally, show all charts with one plt.show() call
    plt.show()


# Main execution
if __name__ == '__main__':
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)

    matched_df, categories = create_matched_df_from_files(config['input data'], config['numeric_cols'])

    reference_groups, valid_groups, _ = determine_validNreference_groups(matched_df, categories)

    # Determine AI columns (excluding 'case_name' and 'truth')
    ai_cols = [col for col in matched_df.columns if col.startswith('ai_')]

    np.random.seed(42)  # For reproducibility
    kappas, intervals = calculate_kappas_and_intervals(matched_df, ai_cols)
    print(f"Mean Kappas: {kappas}")
    print(f"Confidence Intervals: {intervals}")

    # Calculate delta Kappas
    print(f"Bootstrapping delta Kappas, this may take a while", flush=True)
    np.random.seed(42)  # For reproducibility
    # categories = ['race', 'ethnicity']  # Speed things up during development by reducing the number of categories
    delta_kappas = calculate_delta_kappa(matched_df, categories, reference_groups, valid_groups, ai_cols)
    #print(f"Delta Kappas: {delta_kappas}")

    filename = f"delta_kappas_rob-updates_{time.strftime('%Y%m%d%H%M%S')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(delta_kappas, f)

    generate_plots_from_delta_kappas(delta_kappas)
