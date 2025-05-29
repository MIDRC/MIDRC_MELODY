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

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import json
import os

from PySide6.QtCore import QSettings, QThreadPool, Slot
from PySide6.QtGui import QAction, QBrush, QColor, QFontDatabase, QIcon
from PySide6.QtWidgets import (QDialog, QMainWindow, QMessageBox, QPlainTextEdit, QSizePolicy, QTableWidgetItem,
                               QTabWidget, QToolBar, QWidget, QFileDialog)
import yaml

# Import functions for EOD/AAOD
from MIDRC_MELODY.common.data_loading import build_test_and_demographic_data as build_demo_data
from MIDRC_MELODY.common.eod_aaod_metrics import binarize_scores, calculate_eod_aaod
from MIDRC_MELODY.common.table_tools import build_eod_aaod_tables_gui
# Import functions for QWK
from MIDRC_MELODY.common.qwk_metrics import calculate_delta_kappa, calculate_kappas_and_intervals
# Import custom classes for GUI
from MIDRC_MELODY.gui.config_editor import ConfigEditor
from MIDRC_MELODY.gui.copyabletableview import CopyableTableWidget
from MIDRC_MELODY.gui.tqdm_handler import ANSIProcessor, EmittingStream, Worker


# New: custom QTableWidgetItem subclass for numeric value sorting.
class NumericSortTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        try:
            self_data = float(self.text())
            other_data = float(other.text())
            return self_data < other_data
        except ValueError:
            return QTableWidgetItem.__lt__(self, other)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1200, 600)  # Set default size to 800x600
        self.setWindowTitle("Melody GUI")
        self.threadpool = QThreadPool()   # New thread pool for background work
        self._createMenuBar()
        self._createToolBar()
        self._createCentralWidget()
        self.progress_view = None  # Will hold a QPlainTextEdit for live progress
        self._ansi = None  # Will hold the ANSIProcessor for live progress

    def _createMenuBar(self):
        menu_bar = self.menuBar()

        # New File menu with Load Config File option
        file_menu = menu_bar.addMenu("File")
        load_config_act = QAction("Load Config File", self)
        load_config_act.triggered.connect(self.load_config_file)
        file_menu.addAction(load_config_act)

        # Existing Configuration menu
        config_menu = menu_bar.addMenu("Configuration")
        edit_config_act = QAction("Edit Config", self)
        edit_config_act.triggered.connect(self.edit_config)
        config_menu.addAction(edit_config_act)

    def _createToolBar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        eod_act = QAction("EOD/AAOD Metrics", self)
        eod_act.triggered.connect(self.calculate_eod_aaod)
        toolbar.addAction(eod_act)

        qwk_act = QAction("QWK Metrics", self)
        qwk_act.triggered.connect(self.calculate_qwk)
        toolbar.addAction(qwk_act)

        charts_act = QAction("Display Charts", self)
        charts_act.triggered.connect(self.display_charts)
        toolbar.addAction(charts_act)

        # Add spacer widget to push the configuration option to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # Add configuration option with a gear icon that triggers edit_config()
        config_icon = QIcon.fromTheme("preferences-system")
        config_act = QAction(config_icon, "Config", self)
        config_act.triggered.connect(self.edit_config)
        toolbar.addAction(config_act)

    def _createCentralWidget(self):
        # Set an empty central widget initially.
        tab_widget = QTabWidget()
        self.setCentralWidget(tab_widget)

    def create_table_widget(self, headers: list, rows: list) -> CopyableTableWidget:
        """
        rows is a list of tuples: (row_data: list of str, row_color: QColor or None)
        If row_color is provided, the entire row will be set bold and colored.
        The table will be sortable via clickable headers.
        """
        table = CopyableTableWidget()  # Using custom CopyableTableWidget
        table.setSortingEnabled(True)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for r, (row_data, row_color) in enumerate(rows):
            for c, cell in enumerate(row_data):
                # If the cell can be converted to a float, use NumericSortTableWidgetItem.
                try:
                    float(cell)
                    item = NumericSortTableWidgetItem(cell)
                except ValueError:
                    item = QTableWidgetItem(cell)

                # Update the color of the median, lower CI, and upper CI columns if row_color is provided.
                if cell in row_data[-3:] and row_color is not None:
                    item.setForeground(QBrush(row_color))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        return table

    @Slot()
    def load_config_file(self):
        try:
            _ = self.load_config_dict()
            QMessageBox.information(self, "Config Loaded", "Configuration settings loaded from QSettings.")
        except Exception as e:
            # Open file dialog to select an existing config file
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Config File", os.path.expanduser("~"),
                                                       "Config Files (*.yaml *.json);;All Files (*)")
            if file_path:
                try:
                    if file_path.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            config = yaml.load(f, Loader=yaml.CLoader)
                    settings = QSettings("MIDRC", "MIDRC-MELODY")
                    settings.setValue("config", json.dumps(config))
                    QMessageBox.information(self, "Config Loaded", "Configuration loaded from file.")
                except Exception as e2:
                    QMessageBox.critical(self, "Error", f"Failed to load selected config file: {e2}")
            else:
                QMessageBox.critical(self, "Error", "No config file selected.")

    @Slot()
    def edit_config(self):
        try:
            config = self.load_config_dict()
            editor = ConfigEditor(config, parent=self)
            if editor.exec() == QDialog.Accepted:
                self.save_config_dict(config)
        except Exception as e:
            # Ask the user whether to select a config file or use a blank config
            resp = QMessageBox.question(self, "Edit Config",
                                        f"Failed to load config: {e}\n\nWould you like to select an existing config file?\n"
                                        "Press Yes to select a file; or No to create a blank config.",
                                        QMessageBox.Yes | QMessageBox.No)
            if resp == QMessageBox.Yes:
                file_path, _ = QFileDialog.getOpenFileName(self, "Select Config File", os.path.expanduser("~"),
                                                           "Config Files (*.yaml *.json);;All Files (*)")
                if file_path:
                    try:
                        if file_path.endswith('.json'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                        else:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                config = yaml.load(f, Loader=yaml.CLoader)
                        settings = QSettings("MIDRC", "MIDRC-MELODY")
                        settings.setValue("config", json.dumps(config))
                    except Exception as e3:
                        QMessageBox.critical(self, "Error", f"Failed to load selected config file: {e3}")
                        return
                else:
                    QMessageBox.critical(self, "Error", "No config file selected.")
                    return
            else:
                config = {}  # Create a blank config dictionary
            # Open editor with the obtained config
            editor = ConfigEditor(config, parent=self)
            if editor.exec() == QDialog.Accepted:
                self.save_config_dict(config)

    def load_config_dict(self) -> dict:
        # Load config settings from QSettings or fallback to config.yaml
        settings = QSettings("MIDRC", "MIDRC-MELODY")
        config_str = settings.value("config", "")
        if config_str:
            config = json.loads(config_str)
        else:
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", '..', "config.yaml")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.CLoader)
            settings.setValue("config", json.dumps(config))

        # Post-process numeric_cols bins: convert ".inf" to float("inf")
        if "numeric_cols" in config:
            for col, d in config["numeric_cols"].items():
                if "bins" in d:
                    processed_bins = []
                    for b in d["bins"]:
                        if isinstance(b, (int, float)):
                            processed_bins.append(b)
                        else:
                            try:
                                # Convert ".inf" or "inf" strings to float("inf")
                                if b.strip() in [".inf", "inf"]:
                                    processed_bins.append(float("inf"))
                                else:
                                    processed_bins.append(float(b))
                            except Exception:
                                processed_bins.append(b)
                    config["numeric_cols"][col]["bins"] = processed_bins

        return config

    def save_config_dict(self, config: dict) -> None:
        settings = QSettings("MIDRC", "MIDRC-MELODY")
        settings.setValue("config", json.dumps(config))

    def show_progress_view(self):
        """Replace the central widget with a progress view to display live console output."""
        self.progress_view = QPlainTextEdit()
        self.progress_view.setReadOnly(True)
        fixed = QFontDatabase.systemFont(QFontDatabase.FixedFont)  # platform-native mono
        fixed.setPointSize(10)  # pick a size you like
        self.progress_view.setFont(fixed)
        self.progress_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        # No longer instantiate ANSIProcessor; we'll use its static function instead.
        progress_tabs = QTabWidget()
        progress_tabs.addTab(self.progress_view, "Progress Output")
        self.setCentralWidget(progress_tabs)

    def append_progress(self, text: str) -> None:
        # Use ANSIProcessor.process() as a static function to update the progress_view.
        ANSIProcessor.process(self.progress_view, text)

    def compute_qwk(self, config: dict):
        # Create an EmittingStream and connect its textWritten signal.
        stream = EmittingStream()
        stream.textWritten.connect(self.append_progress)  # using queued connection if needed
        with ExitStack() as es:
            es.enter_context(redirect_stdout(stream))
            es.enter_context(redirect_stderr(stream))
            # Build test data once.
            test_data = build_demo_data(config)
            # Calculate delta kappas for subgroup comparisons.
            delta_kappas = calculate_delta_kappa(test_data)
            all_rows = []
            filtered_rows = []
            maroon = QColor(128, 0, 0)
            green = QColor(0, 128, 0)
            for category, model_data in delta_kappas.items():
                for model, groups in model_data.items():
                    for group, (delta, (lower_ci, upper_ci)) in groups.items():
                        qualifies = (lower_ci > 0 or upper_ci < 0)
                        color = green if qualifies and delta >= 0 else (maroon if qualifies and delta < 0 else None)
                        row = [model, category, group, f"{delta:.4f}", f"{lower_ci:.4f}", f"{upper_ci:.4f}"]
                        all_rows.append((row, color))
                        if qualifies:
                            filtered_rows.append((row, color))
            # Additionally, calculate overall kappas and intervals.
            kappas, intervals = calculate_kappas_and_intervals(test_data)
            kappas_rows = []
            for model in sorted(kappas.keys()):
                row = [model, f"{kappas[model]:.4f}", f"{intervals[model][0]:.4f}", f"{intervals[model][1]:.4f}"]
                kappas_rows.append((row, None))
        # Return a triple containing delta values tables and kappas table.
        return (all_rows, filtered_rows, kappas_rows)

    def update_qwk_tables(self, result):
        # Unpack the results from compute_qwk().
        delta_all, delta_filtered, kappas_rows = result
        # Create table widgets for delta kappas.
        headers_delta = ["Model", "Category", "Group", "Delta Kappa", "Lower CI", "Upper CI"]
        table_all = self.create_table_widget(headers_delta, delta_all)
        table_filtered = self.create_table_widget(headers_delta, delta_filtered)
        # Create table widget for overall kappas and intervals.
        headers_kappas = ["Model", "Kappa", "Lower CI", "Upper CI"]
        table_kappas = self.create_table_widget(headers_kappas, kappas_rows)
        # Prepare a tab widget with three tabs for delta tables and one for kappas.
        result_tabs = QTabWidget()
        result_tabs.addTab(table_kappas, "Kappas & Intervals")
        result_tabs.addTab(table_all, "All Delta κ Values")
        result_tabs.addTab(table_filtered, "QWK Filtered (CI Excludes Zero)")
        # Retain progress output tab.
        result_tabs.addTab(self.progress_view, "Progress Output")
        self.setCentralWidget(result_tabs)

    def compute_eod_aaod(self, config: dict):
        # Create an EmittingStream and connect its textWritten signal using a queued connection.
        stream = EmittingStream()
        stream.textWritten.connect(self.append_progress)  # UPDATED
        with ExitStack() as es:
            es.enter_context(redirect_stdout(stream))
            es.enter_context(redirect_stderr(stream))
            t_data = build_demo_data(config)
            threshold = config['binary threshold']
            matched_df = binarize_scores(t_data.matched_df, t_data.truth_col, t_data.test_cols, threshold=threshold)
            from dataclasses import replace
            test_data = replace(t_data, matched_df=matched_df)
            eod_aaod = calculate_eod_aaod(test_data)
            return build_eod_aaod_tables_gui(eod_aaod)

    @Slot()
    def calculate_eod_aaod(self):
        try:
            config = self.load_config_dict()
            self.show_progress_view()
            worker = Worker(self.compute_eod_aaod, config)
            worker.signals.result.connect(self.update_eod_aaod_tables)
            worker.signals.error.connect(lambda e:
                                         QMessageBox.critical(self, "Error", f"Error in EOD/AAOD Metrics: {e}"))
            self.threadpool.start(worker)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in EOD/AAOD Metrics: {e}")

    def update_eod_aaod_tables(self, result):
        all_eod_rows, all_aaod_rows, filtered_rows = result
        headers = ["Model", "Category", "Group", "Median", "Lower CI", "Upper CI"]
        table_all_eod = self.create_table_widget(headers, all_eod_rows)
        table_all_aaod = self.create_table_widget(headers, all_aaod_rows)
        headers.insert(3, "Metric")  # Insert "Metric" column header
        table_filtered = self.create_table_widget(headers, filtered_rows)
        result_tabs = QTabWidget()
        result_tabs.addTab(table_all_eod, "All EOD Values")
        result_tabs.addTab(table_all_aaod, "All AAOD Values")
        result_tabs.addTab(table_filtered, "EOD/AAOD Filtered (values outside [-0.1, 0.1])")
        # Retain the progress output tab.
        result_tabs.addTab(self.progress_view, "Progress Output")
        self.setCentralWidget(result_tabs)

    @Slot()
    def calculate_qwk(self):
        try:
            config = self.load_config_dict()
            # Show progress view before starting work.
            self.show_progress_view()
            worker = Worker(self.compute_qwk, config)
            worker.signals.result.connect(self.update_qwk_tables)
            worker.signals.error.connect(lambda e: QMessageBox.critical(self, "Error", f"Error in QWK Metrics: {e}"))
            self.threadpool.start(worker)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in QWK Metrics: {e}")

    @Slot()
    def display_charts(self):
        QMessageBox.information(self, "Charts", "Displaying Charts...")
