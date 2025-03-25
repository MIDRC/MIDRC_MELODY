import pickle
import time

import pandas as pd

from data_preprocessing import bin_dataframe_column


def create_matched_df_from_files(input_data: dict, numeric_cols_dict: dict):
    df_truth = pd.read_csv(input_data['truth file'])
    df_test = pd.read_csv(input_data['test scores'])
    uid_col = input_data.get('uid column', 'case_name')
    truth_col = input_data.get('truth column', 'truth')

    test_columns = df_test[df_test.columns.difference([uid_col])].columns
    categories =  df_truth[df_truth.columns.difference([uid_col, truth_col])].columns

    # Bin numerical columns, specifically 'age'
    for str_col, col_dict in numeric_cols_dict.items():
        num_col = col_dict['raw column'] if 'raw column' in col_dict else str_col
        bins = col_dict['bins'] if 'bins' in col_dict else None
        labels = col_dict['labels'] if 'labels' in col_dict else None

        if num_col in df_truth.columns:
            df_truth = bin_dataframe_column(df_truth, num_col, str_col, bins=bins, labels=labels)
            categories = categories.map(lambda x: str_col if x == num_col else x)

    return match_cases(df_truth, df_test, uid_col), categories, test_columns


def match_cases(df1, df2, column):
    merged_df = df1.merge(df2, on=column, how='inner') #, suffixes=('_truth', '_ai'))
    return merged_df


# Step 5: Determine reference groups
def determine_valid_n_reference_groups(df, categories, min_count=10):
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

def save_pickled_data(output_config: dict, metric: str, data: any):
    metric_config = output_config.get(metric.lower(), {})
    if metric_config.get('save', False):
        filename = f"{metric_config['file prefix']}{time.strftime('%Y%m%d%H%M%S')}.pkl"
        print(f'Saving {metric} data to filename:', filename)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
