import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

#Bin numerical data
def bin_data(df, bins_config):
    for column, bin_details in bins_config.items():
        df[column] = pd.cut(df[column], bins=bin_details['bins'], labels=bin_details['labels'], right=False)
    return df

# CDC age bin configuration
age_bins = {
    'age': {
        'bins': [0, 18, 30, 40, 50, 65, 75, 85, np.inf],
        'labels': ['<18', '18-29', '30-39', '40-49', '50-64', '65-74', '75-84', '85+']
    }
}

# Match case names between file1 and file2
def match_cases(df1, df2):
    merged_df = df1.merge(df2, on='case_name', how='inner', suffixes=('_truth', '_ai'))
    return merged_df


# Helper to dynamically binarize based on threshold (e.g., 4)
def binarize_scores(df, ai_cols, threshold=4):
    for col in ai_cols + ['truth']:
        df[col] = (df[col] >= threshold).astype(int)
    return df

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

# EOD calculation function
def calculate_eod_aaod(df, categories, reference_groups, ai_columns, n_iter=1000):
    eod_aaod = {category: {model: {} for model in ai_columns} for category in categories}

    for category in categories:
        ref_group = reference_groups[category]

        for value in df[category].unique():
            if value == ref_group:
                continue

            for model in ai_columns:
                # Bootstrapping
                eod_samples, aaod_samples = [], []

                for _ in range(n_iter):
                    sample_df = resample(df)

                    ref_df = sample_df[sample_df[category] == ref_group]
                    group_df = sample_df[sample_df[category] == value]

                    # TPR & FPR for reference group
                    tpr_ref = ref_df[(ref_df['truth'] == 1) & (ref_df[model] == 1)].shape[0] / (ref_df['truth'] == 1).sum()
                    fpr_ref = ref_df[(ref_df['truth'] == 0) & (ref_df[model] == 1)].shape[0] / (ref_df['truth'] == 0).sum()

                    # TPR & FPR for group
                    tpr_group = group_df[(group_df['truth'] == 1) & (group_df[model] == 1)].shape[0] / (group_df['truth'] == 1).sum()
                    fpr_group = group_df[(group_df['truth'] == 0) & (group_df[model] == 1)].shape[0] / (group_df['truth'] == 0).sum()

                    eod_samples.append(tpr_group - tpr_ref)
                    aaod_samples.append(0.5 * (abs(fpr_group - fpr_ref) + abs(tpr_group - tpr_ref)))

                # Median and 95% CI
                eod_median, aaod_median = np.median(eod_samples), np.median(aaod_samples)
                eod_ci = np.percentile(eod_samples, [2.5, 97.5])
                aaod_ci = np.percentile(aaod_samples, [2.5, 97.5])

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

# Same spider plot function as before (slightly adapted title)
def plot_spider_chart(groups, values, lower_bounds, upper_bounds, model_name, global_min, global_max, metric):
    def group_sort_key(label):
        attr, group = label.split(': ', 1)
        custom_orders = {
            'age': ['18-29', '30-39', '40-49', '50-64', '65-74', '75-84', '85+'],
            'sex': ['Male', 'Female'],
            'race': ['White', 'Black or African American', 'Asian', 'Other'],
            'ethnicity': ['Hispanic or Latino', 'Not Hispanic or Latino'],
            'intersectional_race_ethnicity': ['White and Not Hispanic or Latino', 'Not White or HisapanicLatino']
        }
        return (attr, custom_orders.get(attr, []).index(group) if group in custom_orders.get(attr, []) else len(custom_orders.get(attr, [])))

    combined = list(zip(groups, values, lower_bounds, upper_bounds))
    combined.sort(key=lambda x: group_sort_key(x[0]))

    groups, values, lower_bounds, upper_bounds = zip(*combined)
    angles = np.linspace(0, 2 * np.pi, len(groups), endpoint=False).tolist()
    values, lower_bounds, upper_bounds = map(lambda x: list(x) + [x[0]], [values, lower_bounds, upper_bounds])
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='steelblue', linewidth=2)
    ax.scatter(angles, values, marker='o', c='b')

    if metric.upper() == 'EOD':
        ax.fill_between(angles, -0.1, 0.1, color='lightgreen', alpha=0.5)
        ax.set_ylim(global_min, global_max)
    elif metric.upper() == 'AAOD':
        ax.fill_between(angles, 0, 0.1, color='lightgreen', alpha=0.5)
        ax.set_ylim(0, global_max)

    ax.fill_between(angles, lower_bounds, upper_bounds, color='steelblue', alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_title(f'Spider Plot for {model_name} ({metric.upper()})', size=14)

    plt.tight_layout()
    plt.show()


# Example pipeline (wrap this into main if you want)
if __name__ == '__main__':
    file1 = '../test_truthNdemographics.csv'
    file2 = '../test.csv'

    df1, df2 = pd.read_csv(file1), pd.read_csv(file2)
    categories = df1.columns[2:]
    df1 = bin_data(df1, age_bins)  # your binning logic
    matched_df = match_cases(df1, df2)
    reference_groups, valid_groups, filtered_df = determine_validNreference_groups(matched_df, categories)
    ai_cols = [col for col in filtered_df.columns if col.startswith('ai_')]

    # Binarize
    filtered_df = binarize_scores(filtered_df, ai_cols, threshold=4)

    # EOD & AAOD Calculation
    eod_aaod = calculate_eod_aaod(filtered_df, categories, reference_groups, ai_cols)

    # Global scaling range
    all_values = []
    for model in ai_cols:
        for metric in ['eod', 'aaod']:
            _, values, lower, upper = extract_plot_data_eod_aaod(eod_aaod, model, metric)
            all_values.extend(lower + upper)

    global_min, global_max = min(all_values) - 0.05, max(all_values) + 0.05

    # Plot all models
    for model in ai_cols:
        for metric in ['eod', 'aaod']:
            groups, values, lower, upper = extract_plot_data_eod_aaod(eod_aaod, model, metric)
            plot_spider_chart(groups, values, lower, upper, model, global_min, global_max, metric)

