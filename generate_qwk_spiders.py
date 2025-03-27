"""This script generates QWK spider plots for multiple models across different categories."""
import yaml

from data_loading import build_test_and_demographic_data, save_pickled_data
from qwk_metrics import (calculate_kappas_and_intervals, calculate_delta_kappa, generate_plots_from_delta_kappas,
                         print_table_of_nonzero_deltas)

if __name__ == '__main__':
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)

    # Load data
    test_data = build_test_and_demographic_data(config)

    # Calculate Kappas and intervals, prints the table of Kappas and intervals
    kappas, intervals = calculate_kappas_and_intervals(test_data)

    # Bootstrap delta QWKs
    print("Bootstrapping delta Kappas, this may take a while", flush=True)
    delta_kappas = calculate_delta_kappa(test_data)

    # Print the table of non-zero delta Kappas
    print_table_of_nonzero_deltas(delta_kappas, tablefmt="rounded_outline")

    # Generate and save plots
    generate_plots_from_delta_kappas(delta_kappas, test_data.test_cols, plot_config=config['plot'])

    # Save the delta Kappas
    save_pickled_data(config['output'], "QWK", delta_kappas)
