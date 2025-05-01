from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QMenuBar, QToolBar, QTabWidget, QMessageBox
from PySide6.QtCore import Slot
import yaml
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Melody GUI")

        self._createMenuBar()
        self._createToolBar()
        self._createCentralWidget()

    def _createMenuBar(self):
        menu_bar = self.menuBar()
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

    def _createCentralWidget(self):
        tab_widget = QTabWidget()
        self.setCentralWidget(tab_widget)

    @Slot()
    def edit_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.CLoader)
            QMessageBox.information(self, "Config Loaded", "Config.yaml has been loaded. Add UI elements to edit options.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

    @Slot()
    def calculate_eod_aaod(self):
        QMessageBox.information(self, "EOD/AAOD", "Calculating EOD/AAOD Metrics...")

    @Slot()
    def calculate_qwk(self):
        QMessageBox.information(self, "QWK Metrics", "Calculating QWK Metrics...")

    @Slot()
    def display_charts(self):
        QMessageBox.information(self, "Charts", "Displaying Charts...")

