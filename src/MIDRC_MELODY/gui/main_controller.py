import sys
import time
from contextlib import ExitStack, redirect_stdout, redirect_stderr

from PySide6.QtWidgets import QMessageBox

from MIDRC_MELODY.gui.data_loading import load_config_dict, build_demo_data_wrapper
from MIDRC_MELODY.gui.metrics_model import compute_qwk_metrics, compute_eod_aaod_metrics
from MIDRC_MELODY.gui.tqdm_handler import Worker, EmittingStream  # make sure EmittingStream is importable


class MainController:
    def __init__(self, main_window):
        self.main_window = main_window

    def calculate_qwk(self):
        """
        Entry point triggered by the toolbar button.  This will:
         1) load the config
         2) show the progress view
         3) spin up a Worker that (a) redirects print() → EmittingStream → GUI, then
            calls compute_qwk_metrics(…) and returns that result
         4) when the worker finishes, send its result into main_window.update_qwk_tables(...)
        """
        try:
            config = load_config_dict()
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to load config: {e}")
            return

        self.main_window.show_progress_view()

        # The actual “task” function that gets run in a background thread:
        def _task(config_dict):
            # 1) Save the real stdout/stderr so we can restore later
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            # 2) Create an EmittingStream and hook it up to main_window.append_progress
            stream = EmittingStream()
            stream.textWritten.connect(self.main_window.append_progress)

            # 3) Redirect stdout/stderr → EmittingStream (so print(...) goes to GUI)
            with ExitStack() as es:
                es.enter_context(redirect_stdout(stream))
                es.enter_context(redirect_stderr(stream))

                # Create a timestamp to track the time taken for the computation
                time_start = time.time()

                print('-'*120,'\n')
                print("Computing QWK metrics... "
                      f"(Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_start))})")

                # 4) Build the test_data and call the actual compute_qwk_metrics
                test_data = build_demo_data_wrapper(config_dict)
                result = compute_qwk_metrics(test_data)

                # Add blank line for better readability in the GUI output
                print("Finished computing QWK metrics in {:.2f} seconds.".format(time.time() - time_start))
                print()

            # 5) Restore the real stdout/stderr so that any further print() goes to console
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # 6) Return whatever compute_qwk_metrics returned
            #    Should be a tuple: (all_rows, filtered_rows, kappas_rows, plot_args)
            return result

        # Create a Worker around our _task function.  The Worker’s "signals.result"
        # will fire with whatever `_task(config)` returns.
        worker = Worker(_task, config)
        worker.signals.result.connect(self.main_window.update_qwk_tables)
        worker.signals.error.connect(
            lambda e: QMessageBox.critical(
                self.main_window, "Error", f"Error in QWK Metrics:\n{e}"
            )
        )

        # Finally, queue it on the threadpool
        self.main_window.threadpool.start(worker)

    def calculate_eod_aaod(self):
        """
        Entry point triggered by the toolbar button.  This will:
         1) load the config
         2) show the progress view
         3) spin up a Worker that (a) redirects print() → EmittingStream → GUI, then
            calls compute_eod_aaod_metrics(…) and returns that result
         4) when the worker finishes, send its result into main_window.update_eod_aaod_tables(...)
        """
        try:
            config = load_config_dict()
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to load config: {e}")
            return

        self.main_window.show_progress_view()

        def _task(config_dict):
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            stream = EmittingStream()
            stream.textWritten.connect(self.main_window.append_progress)

            with ExitStack() as es:
                es.enter_context(redirect_stdout(stream))
                es.enter_context(redirect_stderr(stream))

                # Create a timestamp to track the time taken for the computation
                time_start = time.time()

                print('-'*120,'\n')
                print("Computing EOD/AAOD metrics... "
                      f"(Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_start))})")

                test_data = build_demo_data_wrapper(config_dict)
                threshold = config_dict.get("binary threshold", 0.5)
                result = compute_eod_aaod_metrics(test_data, threshold)

                # Print a blank line for better readability in the GUI output
                print("Finished computing EOD/AAOD metrics in {:.2f} seconds.".format(time.time() - time_start))
                print()

            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # `compute_eod_aaod_metrics` should return:
            # (all_eod_rows, all_aaod_rows, filtered_rows, plot_args)
            return result

        worker = Worker(_task, config)
        worker.signals.result.connect(self.main_window.update_eod_aaod_tables)
        worker.signals.error.connect(
            lambda e: QMessageBox.critical(
                self.main_window, "Error", f"Error in EOD/AAOD Metrics:\n{e}"
            )
        )
        self.main_window.threadpool.start(worker)
