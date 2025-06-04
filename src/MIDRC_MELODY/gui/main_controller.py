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

import sys
import time
from contextlib import ExitStack, redirect_stdout, redirect_stderr

from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QTextCursor

from MIDRC_MELODY.gui.data_loading import load_config_dict, build_demo_data_wrapper
from MIDRC_MELODY.gui.metrics_model import compute_qwk_metrics, compute_eod_aaod_metrics
from MIDRC_MELODY.gui.tqdm_handler import Worker, EmittingStream


class MainController:
    def __init__(self, main_window):
        self.main_window = main_window

    def calculate_qwk(self):
        """
        Triggered by the “QWK Metrics” toolbar button.  Loads config, shows the progress view,
        and spins up a Worker that captures print() → GUI and calls compute_qwk_metrics.
        """
        try:
            config = load_config_dict()
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to load config:\n{e}")
            return

        # Ensure the “Progress Output” tab is visible before the Worker starts
        self.main_window.show_progress_view()

        # Create and start a Worker that wraps compute_qwk_metrics
        worker = self._make_worker(
            banner="QWK",
            compute_fn=compute_qwk_metrics,
            result_handler=self.main_window.update_qwk_tables,
            config=config,
        )
        self.main_window.threadpool.start(worker)

    def calculate_eod_aaod(self):
        """
        Triggered by the “EOD/AAOD Metrics” toolbar button.  Loads config, shows the progress view,
        and spins up a Worker that captures print() → GUI and calls compute_eod_aaod_metrics.
        """
        try:
            config = load_config_dict()
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to load config:\n{e}")
            return

        self.main_window.show_progress_view()

        worker = self._make_worker(
            banner="EOD/AAOD",
            compute_fn=compute_eod_aaod_metrics,
            result_handler=self.main_window.update_eod_aaod_tables,
            config=config,
        )
        self.main_window.threadpool.start(worker)

    def _make_worker(self, banner: str, compute_fn, result_handler, config: dict):
        """
        Internal helper to create a Worker that:
         1) Redirects stdout/stderr → EmittingStream → main_window.append_progress
         2) Prints “Computing {banner} metrics…”
         3) Calls the appropriate compute_fn (either compute_qwk_metrics or compute_eod_aaod_metrics)
         4) Restores stdout/stderr and returns the result

        Parameters:
        - banner:          "QWK" or "EOD/AAOD" (used in the printed banner and error dialogs)
        - compute_fn:      either compute_qwk_metrics or compute_eod_aaod_metrics
        - result_handler:  a slot on main_window (update_qwk_tables or update_eod_aaod_tables)
        - config:          the loaded configuration dict

        Returns:
        - A Worker instance that, when started, will run _task(config) in a background thread.
        """
        def _task(cfg):
            # 1) Save original stdout/stderr so we can restore later
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            # 2) Build an EmittingStream and connect its textWritten → append_progress
            stream = EmittingStream()
            stream.textWritten.connect(self.main_window.append_progress)

            # 3) Redirect stdout/stderr → EmittingStream
            with ExitStack() as es:
                es.enter_context(redirect_stdout(stream))
                es.enter_context(redirect_stderr(stream))

                # 4) Print the initial banner
                time_start = time.time()
                if not self.main_window.progress_view.document().isEmpty():
                    # Move cursor to the end of the progress view
                    self.main_window.progress_view.moveCursor(QTextCursor.End)
                    print('\n', '-'*140, '\n')
                print(f"Computing {banner} metrics... "
                      f"(Started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_start))})")

                # 5) Build test data and run the compute function
                test_data = build_demo_data_wrapper(cfg)
                plot_config = cfg.get("plot", None)
                if compute_fn is compute_eod_aaod_metrics:
                    # EOD/AAOD needs a threshold from cfg (default to 0.5)
                    threshold = cfg.get("binary threshold", 0.5)
                    result = compute_fn(test_data, threshold, plot_config)
                else:
                    # QWK just takes test_data
                    result = compute_fn(test_data, plot_config)

                print(f"Finished {banner} metrics in {time.time() - time_start:.2f} seconds.")

            # 6) Restore original stdout/stderr so further print() goes to console
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # 7) Return the computed result, e.g.
            #    - For QWK: (all_rows, filtered_rows, kappas_rows, plot_args)
            #    - For EOD/AAOD: (all_eod_rows, all_aaod_rows, filtered_rows, plot_args)
            return result

        # Instantiate the Worker around our _task function + config dict
        worker = Worker(_task, config)

        # Connect the result signal to the appropriate table‐update slot
        worker.signals.result.connect(result_handler)

        # Connect any error to a QMessageBox
        worker.signals.error.connect(
            lambda e: QMessageBox.critical(
                self.main_window,
                "Error",
                f"Error in {banner} Metrics:\n{e}"
            )
        )

        return worker
