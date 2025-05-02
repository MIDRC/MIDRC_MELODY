from PySide6.QtGui import QAction, QIcon, QBrush, QColor, QTextCursor, QFontDatabase, QFont
from PySide6.QtWidgets import QMainWindow, QMenuBar, QToolBar, QTabWidget, QMessageBox, QWidget, QSizePolicy, QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QLabel, QCheckBox, QHBoxLayout, QTableWidget, QTableWidgetItem, QDoubleSpinBox, QSpinBox, QPlainTextEdit
from PySide6.QtCore import Slot, QSettings, QRunnable, QThreadPool, QObject, Signal, Qt
import yaml
import os
import json
from contextlib import ExitStack, redirect_stdout, redirect_stderr

# Import functions for QWK
from common.data_loading import build_test_and_demographic_data
from common.qwk_metrics import calculate_kappas_and_intervals, calculate_delta_kappa
# Import functions for EOD/AAOD
from common.data_loading import build_test_and_demographic_data as build_demo_data
from common.eod_aaod_metrics import binarize_scores, calculate_eod_aaod
from gui.copyabletableview import CopyableTableWidget
from gui.tqdm_handler import EmittingStream, ANSIProcessor, Worker, WorkerSignals

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
                if row_color is not None:
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
            config = self.load_config_dict()
            QMessageBox.information(self, "Config Loaded", "Configuration settings loaded from QSettings.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

    @Slot()
    def edit_config(self):
        try:
            config = self.load_config_dict()
            editor = ConfigEditor(config, parent=self)
            if editor.exec() == QDialog.Accepted:
                # Future: Save updated config from editor back to QSettings.
                self.save_config_dict(config)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

    def load_config_dict(self) -> dict:
        # Load config settings from QSettings or fallback to config.yaml
        settings = QSettings("MIDRC", "MIDRC-MELODY")
        config_str = settings.value("config", "")
        if config_str:
            config = json.loads(config_str)
        else:
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
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
        self._ansi = ANSIProcessor(self.progress_view)

        fixed = QFontDatabase.systemFont(QFontDatabase.FixedFont)  # platform-native mono
        fixed.setPointSize(10)  # pick a size you like
        self.progress_view.setFont(fixed)

        # optional: keep long tqdm lines readable
        self.progress_view.setLineWrapMode(QPlainTextEdit.NoWrap)

        progress_tabs = QTabWidget()
        progress_tabs.addTab(self.progress_view, "Progress Output")
        self.setCentralWidget(progress_tabs)

    def append_progress(self, text: str) -> None:
        cursor = self.progress_view.textCursor()  # QPlainTextEdit
        if text.startswith('\r'):
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.movePosition(QTextCursor.MoveOperation.StartOfLine, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()  # delete the old line
            text = text.lstrip('\r')  # write fresh line below
        cursor.insertText(text)
        self.progress_view.setTextCursor(cursor)
        self.progress_view.ensureCursorVisible()

    def compute_qwk(self, config: dict):
        # Create an EmittingStream and connect its textWritten signal.
        stream = EmittingStream()
        stream.textWritten.connect(self.append_progress)  # using queued connection if needed
        with ExitStack() as es:
            es.enter_context(redirect_stdout(stream))
            es.enter_context(redirect_stderr(stream))
            # Build test data once.
            test_data = build_test_and_demographic_data(config)
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
        result_tabs.addTab(table_filtered, "Filtered (CI Excludes Zero)")
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
            all_rows = []
            filtered_rows = []
            maroon = QColor(128, 0, 0)
            green = QColor(0, 128, 0)
            for metric, groups in eod_aaod.items():
                for group, (value, lower_ci, upper_ci) in groups.items():
                    qualifies = (lower_ci > 0 or upper_ci < 0)
                    color = green if qualifies and value >= 0 else (maroon if qualifies and value < 0 else None)
                    row = [metric, group, f"{value:.4f}", f"{lower_ci:.4f}", f"{upper_ci:.4f}"]
                    all_rows.append((row, color))
                    if qualifies:
                        filtered_rows.append((row, color))
        return (all_rows, filtered_rows)

    @Slot()
    def calculate_eod_aaod(self):
        try:
            config = self.load_config_dict()
            self.show_progress_view()
            worker = Worker(self.compute_eod_aaod, config)
            worker.signals.result.connect(self.update_eod_aaod_tables)
            worker.signals.error.connect(lambda e: QMessageBox.critical(self, "Error", f"Error in EOD/AAOD Metrics: {e}"))
            self.threadpool.start(worker)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in EOD/AAOD Metrics: {e}")

    def update_eod_aaod_tables(self, result):
        all_rows, filtered_rows = result
        headers = ["Metric", "Group", "Value", "Lower CI", "Upper CI"]
        table_all = self.create_table_widget(headers, all_rows)
        table_filtered = self.create_table_widget(headers, filtered_rows)
        result_tabs = QTabWidget()
        result_tabs.addTab(table_all, "All EOD/AAOD Values")
        result_tabs.addTab(table_filtered, "Filtered (CI excludes zero)")
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

# Updated: ConfigEditor dialog for editing config options with Cancel, Apply, and Save buttons.
class ConfigEditor(QDialog):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config  # store reference to config dict
        self.setWindowTitle("Edit Config")
        self.resize(600, 400)
        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.setup_input_tab()
        self.setup_calculations_tab()
        self.setup_output_tab()
        self.setup_plots_tab()

        # Button layout: Cancel, Apply, Save
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_changes)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.on_save)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(save_btn)
        main_layout.addLayout(btn_layout)

    def setup_input_tab(self):
        self.input_edits = {}
        input_tab = QWidget()
        input_layout = QFormLayout(input_tab)
        for key, value in self.config.get("input data", {}).items():
            le = QLineEdit(str(value))
            self.input_edits[key] = le
            input_layout.addRow(QLabel(key), le)
        numeric = self.config.get("numeric_cols", {})
        numeric_str = "\n".join(f"{k}: {v}" for k, v in numeric.items())
        self.numeric_edit = QLineEdit(numeric_str)
        input_layout.addRow(QLabel("numeric_cols"), self.numeric_edit)
        self.tab_widget.addTab(input_tab, "Input")

    def setup_calculations_tab(self):
        self.calc_edits = {}
        calc_tab = QWidget()
        calc_layout = QFormLayout(calc_tab)
        # Use QDoubleSpinBox for binary threshold
        if "binary threshold" in self.config:
            threshold_value = self.config["binary threshold"]
            threshold_spin = QDoubleSpinBox()
            threshold_spin.setDecimals(2)
            threshold_spin.setRange(0, 1000)  # adjust range as needed
            threshold_spin.setValue(float(threshold_value))
            self.calc_edits["binary threshold"] = threshold_spin
            calc_layout.addRow(QLabel("binary threshold"), threshold_spin)
        # Use QSpinBox for min count per category
        if "min count per category" in self.config:
            min_count = self.config["min count per category"]
            min_count_spin = QSpinBox()
            min_count_spin.setMinimum(0)
            min_count_spin.setMaximum(10000)  # adjust range as needed
            min_count_spin.setValue(int(min_count))
            self.calc_edits["min count per category"] = min_count_spin
            calc_layout.addRow(QLabel("min count per category"), min_count_spin)
        # Bootstrap settings using QSpinBox
        bootstrap = self.config.get("bootstrap", {})
        self.bootstrap_iterations = QSpinBox()
        self.bootstrap_iterations.setMinimum(0)
        self.bootstrap_iterations.setMaximum(1000000)
        self.bootstrap_iterations.setValue(int(bootstrap.get("iterations", 0)))
        calc_layout.addRow(QLabel("Bootstrap Iterations"), self.bootstrap_iterations)
        self.bootstrap_seed = QSpinBox()
        self.bootstrap_seed.setMinimum(0)
        self.bootstrap_seed.setMaximum(1000000)
        self.bootstrap_seed.setValue(int(bootstrap.get("seed", 0)))
        calc_layout.addRow(QLabel("Bootstrap Seed"), self.bootstrap_seed)
        self.tab_widget.addTab(calc_tab, "Calculations")

    def setup_output_tab(self):
        self.output_widgets = {}
        output_tab = QWidget()
        output_layout = QFormLayout(output_tab)
        output = self.config.get("output", {})
        for subcat, settings in output.items():
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox("Save")
            cb.setChecked(settings.get("save", False))
            le = QLineEdit(str(settings.get("file prefix", "")))
            le.setEnabled(cb.isChecked())
            cb.toggled.connect(lambda checked, le=le: le.setEnabled(checked))
            row_layout.addWidget(cb)
            row_layout.addWidget(QLabel("File Prefix:"))
            row_layout.addWidget(le)
            output_layout.addRow(QLabel(subcat.capitalize()), row_widget)
            self.output_widgets[subcat] = (cb, le)
        self.tab_widget.addTab(output_tab, "Output")

    def setup_plots_tab(self):
        plots_tab = QWidget()
        plots_layout = QFormLayout(plots_tab)
        plot = self.config.get("plot", {})
        custom_orders = plot.get("custom_orders", {})
        custom_orders_str = "\n".join(f"{k}: {v}" for k, v in custom_orders.items())
        self.custom_orders_edit = QLineEdit(custom_orders_str)
        plots_layout.addRow(QLabel("Custom Orders"), self.custom_orders_edit)
        self.clockwise_checkbox = QCheckBox()
        self.clockwise_checkbox.setChecked(plot.get("clockwise", False))
        plots_layout.addRow(QLabel("Clockwise"), self.clockwise_checkbox)
        self.start_edit = QLineEdit(str(plot.get("start", "")))
        plots_layout.addRow(QLabel("Start"), self.start_edit)
        self.tab_widget.addTab(plots_tab, "Plots")

    def apply_changes(self):
        # Input Tab
        for key, widget in self.input_edits.items():
            self.config.setdefault("input data", {})[key] = widget.text()
        try:
            new_numeric = yaml.load(self.numeric_edit.text(), Loader=yaml.SafeLoader)
            if not isinstance(new_numeric, dict):
                raise ValueError("numeric_cols must be a dictionary")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Invalid format for numeric_cols: {e}")
            new_numeric = self.config.get("numeric_cols", {})
        self.config["numeric_cols"] = new_numeric

        # Calculations Tab: use spinbox values
        for key, widget in self.calc_edits.items():
            self.config[key] = widget.value()  # Use value() from spinboxes
        self.config["bootstrap"] = {
            "iterations": self.bootstrap_iterations.value(),
            "seed": self.bootstrap_seed.value()
        }
        # Output Tab
        for subcat, (cb, le) in self.output_widgets.items():
            self.config.setdefault("output", {})[subcat] = {
                "save": cb.isChecked(),
                "file prefix": le.text()
            }
        # Plots Tab: Parse custom_orders into a dictionary
        try:
            new_custom_orders = yaml.load(self.custom_orders_edit.text(), Loader=yaml.SafeLoader)
            if not isinstance(new_custom_orders, dict):
                raise ValueError("custom_orders must be a dictionary")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Invalid format for custom_orders: {e}")
            new_custom_orders = self.config.get("plot", {}).get("custom_orders", {})
        self.config.setdefault("plot", {})["custom_orders"] = new_custom_orders

        self.config["plot"]["clockwise"] = self.clockwise_checkbox.isChecked()
        self.config["plot"]["start"] = self.start_edit.text()
        QMessageBox.information(self, "Applied", "Configuration updated.")

    def on_save(self):
        self.apply_changes()
        self.accept()

