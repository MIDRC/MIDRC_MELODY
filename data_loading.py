""" Data Loading and Preprocessing Functions """

from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
import time
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd

from data_preprocessing import bin_dataframe_column


def check_file_exists(file_path: str, key_name: str, config_file: str) -> None:
    """
    Check if a file exists and exit gracefully with an error message if it doesn't.

    :arg file_path: Path to the file to check.
    :arg key_name: Key name in the configuration file.
    :arg config_file: Name of the configuration file.
    """
    if not Path(file_path).exists():
        print(f"Error: The file specified for '{key_name}' ('{file_path}') does not exist.")
        print(f"Please update the '{key_name}' path in the '{config_file}' file to point to a valid file.")
        print("Ensure the path is correct and accessible.")
        sys.exit(1)


def create_matched_df_from_files(input_data: dict, numeric_cols_dict: dict) -> Tuple[pd.DataFrame, list, list]:
    """
    Create a matched DataFrame from the truth and test files

    :arg input_data: Dictionary containing the input data
    :arg numeric_cols_dict: Dictionary containing the numeric columns information

    :return: A tuple containing the matched DataFrame, a list of categories, and a list of test columns
    """
    truth_file = input_data['truth file']
    test_scores_file = input_data['test scores']

    # Check if files exist
    check_file_exists(truth_file, 'truth file', 'config.yaml')
    check_file_exists(test_scores_file, 'test scores', 'config.yaml')

    # Read the truth and test scores files
    df_truth = pd.read_csv(truth_file)
    df_test = pd.read_csv(test_scores_file)
    uid_col = input_data.get('uid column', 'case_name')
    truth_col = input_data.get('truth column', 'truth')

    test_columns = df_test[df_test.columns.difference([uid_col])].columns
    categories = df_truth[df_truth.columns.difference([uid_col, truth_col])].columns

    # Bin numerical columns, specifically 'age'
    for str_col, col_dict in numeric_cols_dict.items():
        num_col = col_dict['raw column'] if 'raw column' in col_dict else str_col
        bins = col_dict['bins'] if 'bins' in col_dict else None
        labels = col_dict['labels'] if 'labels' in col_dict else None

        if num_col in df_truth.columns:
            df_truth = bin_dataframe_column(df_truth, num_col, str_col, bins=bins, labels=labels)
            categories = categories.map(lambda x, col=str_col, num=num_col: col if x == num else x)

    return match_cases(df_truth, df_test, uid_col), categories.tolist(), test_columns.tolist()


def match_cases(df1, df2, column) -> pd.DataFrame:
    """
    Match cases between two DataFrames

    :arg df1: First DataFrame
    :arg df2: Second DataFrame
    :arg column: Column to match on

    :return: A DataFrame containing the matched cases
    """
    merged_df = df1.merge(df2, on=column, how='inner')  # , suffixes=('_truth', '_ai'))
    return merged_df


# Step 5: Determine reference groups
def determine_valid_n_reference_groups(df, categories, min_count=10) -> Tuple[dict, dict, pd.DataFrame]:
    """
    Determine the valid and reference groups for the given categories

    :arg df: DataFrame
    :arg categories: List of categories
    :arg min_count: Minimum count for a group to be considered valid

    :return: A tuple containing the reference groups, valid groups, and the filtered DataFrame
    """
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
    """
    Save pickled data to a file

    :arg output_config: Output configuration dictionary
    :arg metric: Metric name
    :arg data: Data to save
    """
    metric_config = output_config.get(metric.lower(), {})
    if metric_config.get('save', False):
        filename = f"{metric_config['file prefix']}{time.strftime('%Y%m%d%H%M%S')}.pkl"
        print(f'Saving {metric} data to filename:', filename)
        # Create directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, 'wb') as f:
            pickle.dump(data, f)


def check_required_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Raise an error if any required column is missing.

    :arg df: DataFrame to check for required columns.
    :arg columns: List of required columns.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


@dataclass(frozen=True)
class TestAndDemographicData:
    """
    Class to store file data
    """
    matched_df: pd.DataFrame
    truth_col: str
    categories: List[str]
    test_cols: List[str]
    reference_groups: Dict[str, Any]
    valid_groups: Dict[str, List[Any]]
    n_iter: Optional[int]
    base_seed: Optional[int]


def build_test_and_demographic_data(config: Dict[str, Any]) -> TestAndDemographicData:
    """
    Build the TestAndDemographicData object from the configuration dictionary.

    :arg config: Configuration dictionary

    :returns: TestAndDemographicData object
    """
    matched_df, categories, test_cols = create_matched_df_from_files(config['input data'], config['numeric_cols'])
    min_count = config.get('min count per category', 10)
    reference_groups, valid_groups, _ = determine_valid_n_reference_groups(matched_df, categories, min_count=min_count)
    n_iter = config.get('bootstrap', {}).get('iterations', 1000)
    base_seed = config.get('bootstrap', {}).get('seed', None)
    truth_col = config['input data'].get('truth column', 'truth')

    # Check required columns before further processing
    required_columns = [truth_col] + test_cols + categories
    check_required_columns(matched_df, required_columns)

    return TestAndDemographicData(matched_df, truth_col, categories, test_cols, reference_groups, valid_groups, n_iter,
                                  base_seed)
