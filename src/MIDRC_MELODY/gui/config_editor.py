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

from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)
import yaml


class ConfigEditor(QDialog):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config  # store reference to config dict
        self.setWindowTitle("Edit Config")
        self.resize(600, 400)
        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Initialize class variables
        self.input_edits = {}
        self.calc_edits = {}
        self.output_widgets = {}
        self.numeric_edit = None
        self.custom_orders_edit = None
        self.clockwise_checkbox = None
        self.start_edit = None
        self.bootstrap_iterations = None
        self.bootstrap_seed = None

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
            hbox = QHBoxLayout()
            le = QLineEdit(str(value))
            self.input_edits[key] = le
            hbox.addWidget(le)
            if key in ['truth file', 'test scores']:
                browse_btn = QPushButton("Browse")
                browse_btn.clicked.connect(lambda _, le=le: self.browse_file(le))
                hbox.addWidget(browse_btn)
            input_layout.addRow(QLabel(key), hbox)
        numeric = self.config.get("numeric_cols", {})
        numeric_str = "\n".join(f"{k}: {v}" for k, v in numeric.items())
        self.numeric_edit = QLineEdit(numeric_str)
        input_layout.addRow(QLabel("numeric_cols"), self.numeric_edit)
        self.tab_widget.addTab(input_tab, "Input")

    def browse_file(self, line_edit: QLineEdit):
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        if file_path:
            line_edit.setText(file_path)

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
        self.tab_widget.addTab(output_tab, "Pickle (.pkl) Output")

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
