#  Copyright (c) 2025 Medical Imaging and Data Resource Center (MIDRC).
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
#

"""This script generates EOD and AAOD spider plots for multiple models across different categories."""
from dataclasses import replace

import matplotlib.pyplot as plt
import yaml

from MIDRC_MELODY.common.data_loading import build_test_and_demographic_data, save_pickled_data
from MIDRC_MELODY.common.eod_aaod_metrics import (binarize_scores, calculate_eod_aaod, generate_plot_data_eod_aaod,
                                                  plot_data_eod_aaod)
from MIDRC_MELODY.common.plot_tools import SpiderPlotData
from MIDRC_MELODY.common.table_tools import print_table_of_nonzero_eod_aaod


def generate_eod_aaod_spiders(cfg_path: str = "config.yaml"):
    # Load configuration
    with open(cfg_path, 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)

    # Load data
    t_data = build_test_and_demographic_data(config)

    # Binarize scores
    threshold = config['binary threshold']
    matched_df = binarize_scores(t_data.matched_df, t_data.truth_col, t_data.test_cols, threshold=threshold)
    test_data = replace(t_data, matched_df=matched_df)

    # Calculate EOD and AAOD
    eod_aaod = calculate_eod_aaod(test_data)

    # Print tables for EOD and AAOD using median values
    print_table_of_nonzero_eod_aaod(eod_aaod, tablefmt="rounded_outline")

    # Generate and save plots
    metrics = ['eod', 'aaod']
    plot_data_dict, global_min, global_max = generate_plot_data_eod_aaod(eod_aaod, test_data.test_cols, metrics=metrics)

    # Save the EOD and AAOD data
    for metric in metrics:
        save_pickled_data(config['output'], metric, plot_data_dict[metric])

    base_plot_data = SpiderPlotData(ylim_min=global_min, ylim_max=global_max, plot_config=config['plot'])
    figures_dict = plot_data_eod_aaod(plot_data_dict,                    # noqa: F841
                                      test_data.test_cols,
                                      metrics=metrics,
                                      base_plot_data=base_plot_data,
                                      )

    print("\nClose all figures to continue...", flush=True)
    plt.show()


if __name__ == '__main__':
    generate_eod_aaod_spiders()
