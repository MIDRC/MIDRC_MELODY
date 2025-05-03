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

"""This script generates QWK spider plots for multiple models across different categories."""

import matplotlib.pyplot as plt
import yaml

from MIDRC_MELODY.common.data_loading import build_test_and_demographic_data, save_pickled_data
from MIDRC_MELODY.common.qwk_metrics import (calculate_delta_kappa, calculate_kappas_and_intervals,
                                             generate_plots_from_delta_kappas)
from MIDRC_MELODY.common.table_tools import print_table_of_nonzero_deltas


def generate_qwk_spiders(cfg_path: str = "config.yaml"):
    # Load configuration
    with open(cfg_path, 'r', encoding='utf-8') as stream:
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

    # Save the delta Kappas
    save_pickled_data(config['output'], "QWK", delta_kappas)

    # Generate and save plots
    generate_plots_from_delta_kappas(delta_kappas, test_data.test_cols, plot_config=config['plot'])

    print("\nClose all figures to continue...", flush=True)
    plt.show()


if __name__ == '__main__':
    generate_qwk_spiders()
