from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMainWindow, QMenuBar, QToolBar, QTabWidget, QMessageBox, QWidget, QSizePolicy, QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QLabel, QCheckBox, QHBoxLayout
from PySide6.QtCore import Slot, QSettings
import yaml
import os
import json

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Melody GUI")

        self._createMenuBar()
        self._createToolBar()
        self._createCentralWidget()

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
        tab_widget = QTabWidget()
        self.setCentralWidget(tab_widget)

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
        return config

    def save_config_dict(self, config: dict) -> None:
        settings = QSettings("MIDRC", "MIDRC-MELODY")
        settings.setValue("config", json.dumps(config))

    @Slot()
    def calculate_eod_aaod(self):
        QMessageBox.information(self, "EOD/AAOD", "Calculating EOD/AAOD Metrics...")

    @Slot()
    def calculate_qwk(self):
        QMessageBox.information(self, "QWK Metrics", "Calculating QWK Metrics...")

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
        for key in ["binary threshold", "min count per category"]:
            if key in self.config:
                le = QLineEdit(str(self.config[key]))
                self.calc_edits[key] = le
                calc_layout.addRow(QLabel(key), le)
        bootstrap = self.config.get("bootstrap", {})
        self.bootstrap_iterations = QLineEdit(str(bootstrap.get("iterations", "")))
        self.bootstrap_seed = QLineEdit(str(bootstrap.get("seed", "")))
        calc_layout.addRow(QLabel("Bootstrap Iterations"), self.bootstrap_iterations)
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
        # Parse numeric_cols text into a dictionary instead of saving raw string
        try:
            new_numeric = yaml.load(self.numeric_edit.text(), Loader=yaml.SafeLoader)
            if not isinstance(new_numeric, dict):
                raise ValueError("numeric_cols must be a dictionary")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Invalid format for numeric_cols: {e}")
            new_numeric = self.config.get("numeric_cols", {})
        self.config["numeric_cols"] = new_numeric
        
        # Calculations Tab
        for key, widget in self.calc_edits.items():
            self.config[key] = widget.text()
        self.config["bootstrap"] = {
            "iterations": self.bootstrap_iterations.text(),
            "seed": self.bootstrap_seed.text()
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
