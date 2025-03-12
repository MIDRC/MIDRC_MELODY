import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample


# Step 1: Read data
def read_data(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    return df1, df2

# Step 2: Determine categories dynamically based on column headings
def determine_categories(df1):
    categories = df1.columns[2:]
    return categories

# Step 3: Bin numerical columns like age
def bin_data(df, bins_config):
    for column, bin_details in bins_config.items():
        df[column] = pd.cut(df[column], bins=bin_details['bins'], labels=bin_details['labels'], right=False)
    return df

# Step 4: Match case names between file1 and file2
def match_cases(df1, df2):
    merged_df = df1.merge(df2, on='case_name', how='inner', suffixes=('_truth', '_ai'))
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
    y_true_np = np.array(y_true.tolist(), dtype=np.int)
    for col in ai_cols:
        y_pred = df[col]
        y_pred_np = np.array(y_pred.tolist(), dtype=np.int)
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
        
    return kappas, intervals

# Custom bootstrap kappa
def bootstrap_kappa(df, model, n_iter=1000):
    kappas = []
    for _ in range(n_iter):
        sampled_df = resample(df, replace=True)
        kappa = cohen_kappa_score(sampled_df['truth'], sampled_df[model])
        kappas.append(kappa)
    return kappas

# Step 7: Calculate delta kappa
def calculate_delta_kappa(df, categories, reference_groups, ai_columns, n_iter=1000):
    delta_kappas = {}

    for category in categories:
        delta_kappas[category] = {model: {} for model in ai_columns}

        for value in df[category].unique():
            if value == reference_groups[category]:
                continue

            for model in ai_columns:
                condition = df[category] == value
                filtered_df = df[condition]

                ref_condition = df[category] == reference_groups[category]
                ref_filtered_df = df[ref_condition]

                kappas = bootstrap_kappa(filtered_df, model, n_iter)
                kappas_ref = bootstrap_kappa(ref_filtered_df, model, n_iter)
                deltas = [a - b for a, b in zip(kappas, kappas_ref)]
                delta_median=np.percentile(deltas, [50])
                (lower_value, upper_value) = np.percentile(deltas, [2.5, 97.5])

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
    # Sort groups (and corresponding values and bounds) so that within each attribute they appear in order
    def group_sort_key(label):
        # Split into attribute and group (e.g., 'age: 18-29' -> ('age', '18-29'))
        attr, group = label.split(': ', 1)
        # Use custom sort orders for known attributes, fallback to alphabetical
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

    # Combine all data into tuples to sort together
    combined = list(zip(groups, values, lower_bounds, upper_bounds))
    combined.sort(key=lambda x: group_sort_key(x[0]))

    # Unpack sorted data
    groups, values, lower_bounds, upper_bounds = zip(*combined)

    # Set up angles for each axis
    num_axes = len(groups)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()

    # Close the loop for the plot itself
    values = list(values) + [values[0]]
    lower_bounds = list(lower_bounds) + [lower_bounds[0]]
    upper_bounds = list(upper_bounds) + [upper_bounds[0]]
    angles = angles + [angles[0]]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot main line
    ax.plot(angles, values, color='steelblue', linestyle='-', linewidth=2)
    ax.scatter(angles, values, marker='o', c='b')

    # Add horizontal line at y=0
    baseline=np.zeros(len(values))
    ax.plot(angles, baseline, color='seagreen', linestyle='--', linewidth=1 , alpha=0.8)
    #ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

    # Confidence interval band
    ax.fill_between(angles, lower_bounds, upper_bounds, color='steelblue', alpha=0.2)

    # Set axis properties
    ax.set_ylim(global_min, global_max)
    ax.set_xticks(angles[:-1])  # the angles without the closing duplicate
    ax.set_xticklabels(groups, fontsize=8, ha='center')

    # Title
    ax.set_title(f'Spider Plot for {model_name}', size=14, weight='bold')

    plt.tight_layout()
    plt.show()


# Main execution
file1 = '../test_truthNdemographics.csv'
file2 = '../test_scores.csv'
df1, df2 = read_data(file1, file2)

categories = determine_categories(df1)

# Bin numerical columns, specifically 'age'
bins_config = {
    **age_bins,
}

df1 = bin_data(df1, bins_config)
matched_df = match_cases(df1, df2)

reference_groups, valid_groups, filtered_df = determine_validNreference_groups(matched_df, categories)

# Determine AI columns (excluding 'case_name' and 'truth')
ai_cols = [col for col in filtered_df.columns if col.startswith('ai_')]

kappas, intervals = calculate_kappas_and_intervals(filtered_df, ai_cols)
print(f"Mean Kappas: {kappas}, Intervals: {intervals}")
print(f"Bootstrapping delta Kappas, this may take a while")
delta_kappas = calculate_delta_kappa(filtered_df, categories, reference_groups, ai_cols)
#print(f"Delta Kappas: {delta_kappas}")

ai_models = extract_ai_models(delta_kappas) # in case some of the ai_cols were inconsistent

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
for model in ai_models:
    groups, values, lower_bounds, upper_bounds = extract_plot_data(delta_kappas, model)
    plot_spider_chart(groups, values, lower_bounds, upper_bounds, model, global_min, global_max)
