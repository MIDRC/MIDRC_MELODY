"""This script generates EOD and AAOD spider plots for multiple models across different categories."""
from dataclasses import replace

import yaml
import matplotlib.pyplot as plt

from data_loading import build_test_and_demographic_data, save_pickled_data
from eod_aaod_metrics import binarize_scores, calculate_eod_aaod, generate_plot_data_eod_aaod, plot_data_eod_aaod
from plot_tools import SpiderPlotData

if __name__ == '__main__':
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)

    # Load data
    t_data = build_test_and_demographic_data(config)

    # Binarize scores
    threshold = config['binary threshold']
    matched_df = binarize_scores(t_data.matched_df, t_data.truth_col, t_data.test_cols, threshold=threshold)
    test_data = replace(t_data, matched_df=matched_df)

    # Calculate EOD and AAOD
    eod_aaod = calculate_eod_aaod(test_data)

    # Generate and save plots
    metrics = ['eod', 'aaod']
    plot_data_dict, global_min, global_max = generate_plot_data_eod_aaod(eod_aaod, test_data.test_cols, metrics=metrics)
    base_plot_data = SpiderPlotData(ylim_min=global_min, ylim_max=global_max, plot_config=config['plot'])
    figures_dict = plot_data_eod_aaod(plot_data_dict,
                                      test_data.test_cols,
                                      metrics=metrics,
                                      base_plot_data=base_plot_data,
                                      )

    plt.show()

    # Save the EOD and AAOD data
    for metric in metrics:
        save_pickled_data(config['output'], metric, plot_data_dict[metric])
