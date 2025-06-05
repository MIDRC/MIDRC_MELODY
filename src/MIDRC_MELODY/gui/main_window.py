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

from typing import Dict, List, Tuple

from PySide6.QtCore import Qt, QThreadPool, Slot
from PySide6.QtGui import QAction, QBrush, QFontDatabase, QIcon, QColor
from PySide6.QtWidgets import (
    QMainWindow,
    QPlainTextEdit,
    QSizePolicy,
    QTabWidget,
    QWidget,
    QTableWidgetItem,
)

# ─── Data‐loading and Metrics imports ────────────────────────────────────────────
from MIDRC_MELODY.common.eod_aaod_metrics import (
    create_spider_plot_data_eod_aaod,
    generate_plot_data_eod_aaod,
)
from MIDRC_MELODY.common.plot_tools import SpiderPlotData

from MIDRC_MELODY.common.qwk_metrics import (
    create_spider_plot_data_qwk,
)

# ─── GUI‐specific imports ────────────────────────────────────────────────────────
from MIDRC_MELODY.gui.copyabletableview import CopyableTableWidget
# from MIDRC_MELODY.gui.qchart_spider_widget import display_spider_charts_in_tabs
from MIDRC_MELODY.gui.matplotlib_spider_widget import MatplotlibSpiderWidget
from MIDRC_MELODY.gui.matplotlib_spider_widget import display_spider_charts_in_tabs_matplotlib as display_spider_charts_in_tabs
from MIDRC_MELODY.gui.tqdm_handler import ANSIProcessor

from MIDRC_MELODY.gui.data_loading import load_config_file, edit_config_file
from MIDRC_MELODY.gui.main_controller import MainController


class NumericSortTableWidgetItem(QTableWidgetItem):
    """A QTableWidgetItem that sorts numerically when its text can be parsed as float."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


def _add_ref_group(rows: List[Tuple[List[str], QColor]], ref_groups: Dict[str, str]) -> List[Tuple[List[str], QColor]]:
    """
    Add a reference group to each row based on the category.
    If the category is not found in ref_groups, use "N/A".
    """
    new_rows = []
    for row in rows:
        row_0 = row[0]
        category = row_0[1]
        ref = ref_groups.get(category, "N/A")
        row_0.insert(2, ref)  # Insert the reference group at index 3
        new_rows.append((row_0, row[1]))  # Keep the original color if any
    return new_rows


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(1200, 600)
        self.setWindowTitle("Melody GUI")
        self.threadpool = QThreadPool()

        # Track whether to show MPL toolbar in spider charts
        self._show_mpl_toolbar = False

        # ─── Instantiate controller and hand it this window ──────────────────────
        self.controller = MainController(self)

        # ─── Build menus, toolbar, and central widget ────────────────────────────
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_central_widget()

        # Prepare the QPlainTextEdit (progress_view) but don’t show it yet
        self.progress_view: QPlainTextEdit = QPlainTextEdit()
        self._ansi_processor: ANSIProcessor = None  # Lazy‐init when first appending text

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # File → Load Config
        file_menu = menu_bar.addMenu("File")
        load_config_act = QAction("Load Config File", self)
        load_config_act.triggered.connect(self.load_config_file)
        file_menu.addAction(load_config_act)

        # Configuration → Edit Config
        config_menu = menu_bar.addMenu("Configuration")
        edit_config_act = QAction("Edit Config", self)
        edit_config_act.triggered.connect(self.edit_config)
        config_menu.addAction(edit_config_act)

    def _create_tool_bar(self):
        toolbar = self.addToolBar("MainToolbar")
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        # EOD/AAOD Metrics button
        eod_icon = QIcon.fromTheme(QIcon.ThemeIcon.Computer)
        eod_act = QAction(eod_icon, "EOD/AAOD Metrics", self)
        eod_act.setToolTip("Calculate EOD/AAOD Metrics")
        # ←— Now wired to controller.calculate_eod_aaod()
        eod_act.triggered.connect(self.controller.calculate_eod_aaod)
        toolbar.addAction(eod_act)

        # QWK Metrics button
        qwk_icon = QIcon.fromTheme("accessories-calculator")
        qwk_act = QAction(qwk_icon, "QWK Metrics", self)
        qwk_act.setToolTip("Calculate QWK Metrics")
        # ←— Now wired to controller.calculate_qwk()
        qwk_act.triggered.connect(self.controller.calculate_qwk)
        toolbar.addAction(qwk_act)

        # Spacer pushes “Config” button to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # ─── Checkable “Show MPL Toolbar” action ────────────────────────────
        self._toggle_mpl_act = QAction(f"{'Hide' if self._show_mpl_toolbar else 'Show'} Plot Toolbar", self)
        self._toggle_mpl_act.setCheckable(True)
        self._toggle_mpl_act.setChecked(self._show_mpl_toolbar)
        self._toggle_mpl_act.setToolTip("Toggle the Matplotlib navigation toolbar in spider‐chart tabs")
        self._toggle_mpl_act.toggled.connect(self._on_toggle_mpl_toolbar)
        toolbar.addAction(self._toggle_mpl_act)

        # Config button (duplicate of menu option)
        config_icon = QIcon.fromTheme(QIcon.ThemeIcon.DocumentProperties)
        config_act = QAction(config_icon, "Config", self)
        config_act.setToolTip("Edit Configuration")
        config_act.triggered.connect(self.edit_config)
        toolbar.addAction(config_act)

    def _create_central_widget(self):
        """
        The central widget is a QTabWidget. Tab #0 will hold the progress view
        when computing; subsequent tabs will hold result tables and charts.
        """
        tab_widget = QTabWidget()
        tab_widget.setMovable(True)
        self.setCentralWidget(tab_widget)

    @Slot()
    def load_config_file(self):
        # Delegate to data_loading.load_config_file()
        load_config_file(self)

    @Slot()
    def edit_config(self):
        # Delegate to data_loading.edit_config_file()
        edit_config_file(self)

    @Slot(bool)
    def _on_toggle_mpl_toolbar(self, checked: bool):
        """
        Handler for the “Show MPL Toolbar” checkbox.
        Show/hide any existing spider‐chart toolbars, and update the flag
        so future tabs respect the setting.
        """
        self._show_mpl_toolbar = checked
        self._toggle_mpl_act.setText(f"{'Hide' if checked else 'Show'} Plot Toolbar")

        # Iterate through all tabs in the central QTabWidget:
        tabs: QTabWidget = self.centralWidget()
        for idx in range(tabs.count()):
            widget = tabs.widget(idx)
            # Each spider‐chart tab is a container QWidget whose layout’s first child
            # is a MatplotlibSpiderWidget.  We can walk its children to find it.
            spiders = widget.findChildren(MatplotlibSpiderWidget)
            for spider in spiders:
                spider.set_toolbar_visible(checked)

    def show_progress_view(self):
        """
        Insert (or re-insert) a read-only console tab (QPlainTextEdit) at index 0
        so that any redirected print() output appears there.
        """
        if not self.progress_view:
            self.progress_view = QPlainTextEdit()
        self.progress_view.setReadOnly(True)
        fixed = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed.setPointSize(10)
        self.progress_view.setFont(fixed)
        self.progress_view.setLineWrapMode(QPlainTextEdit.NoWrap)

        tabs: QTabWidget = self.centralWidget()
        # Remove any existing “Progress Output” tab
        for i in range(tabs.count()):
            if tabs.tabText(i) == "Progress Output":
                tabs.removeTab(i)
                break

        tabs.insertTab(0, self.progress_view, "Progress Output")
        tabs.setCurrentIndex(0)

    def append_progress(self, text: str) -> None:
        """
        Feed each chunk of emitted text through ANSIProcessor so colors/formatting appear.
        """
        if not self._ansi_processor:
            self._ansi_processor = ANSIProcessor()
        self._ansi_processor.process(self.progress_view, text)

    def update_tabs(self, tab_dict: Dict[QWidget, str], *, set_current=True):
        """
        Given a dict {widget: tab_title}, remove any existing tabs with those titles,
        then insert each new tab at index 1 (leaving index 0 for the progress view).
        """
        tab_widget: QTabWidget = self.centralWidget()
        # Remove tabs whose title matches any in tab_dict.values()
        for i in reversed(range(tab_widget.count())):
            if tab_widget.tabText(i) in tab_dict.values():
                tab_widget.removeTab(i)
        # Insert new tabs, starting at index=1
        for idx, (widget, title) in enumerate(tab_dict.items(), start=1):
            tab_widget.insertTab(idx, widget, title)
        if set_current:
            tab_widget.setCurrentIndex(1)

    @staticmethod
    def create_table_widget(headers: List[str], rows: List):
        """
        Build a CopyableTableWidget with the given headers and rows.
        rows is a List of (row_data: List[str], row_color: QColor or None).
        If row_color is not None, the final three columns in that row are colored + bold.
        """
        table = CopyableTableWidget()
        table.setSortingEnabled(True)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))

        for r, (row_data, row_color) in enumerate(rows):
            for c, cell in enumerate(row_data):
                # Numeric‐sort if convertible to float
                try:
                    float(cell)
                    item = NumericSortTableWidgetItem(cell)
                except ValueError:
                    item = QTableWidgetItem(cell)

                # If a color is provided, apply to last three columns
                if row_color is not None and c >= len(row_data) - 3:
                    item.setForeground(QBrush(row_color))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

                table.setItem(r, c, item)

        table.resizeColumnsToContents()
        return table

    def create_spider_plot_from_qwk(self, delta_kappas, test_cols, plot_config=None) -> QTabWidget:
        """
        Given (delta_kappas: dict, test_cols: List[str], plot_config: dict),
        build spider‐chart(s) for QWK and return a QTabWidget containing them.
        """
        plot_data_list = create_spider_plot_data_qwk(delta_kappas, test_cols, plot_config=plot_config)
        return display_spider_charts_in_tabs(plot_data_list, show_toolbar=self._show_mpl_toolbar)

    def create_spider_plot_from_eod_aaod(
        self, eod_aaod, test_cols, plot_config=None, *, metrics=("eod", "aaod")
    ) -> List[QTabWidget]:
        """
        Given (eod_aaod: dict, test_cols: List[str], plot_config: dict),
        build one or more spider‐chart tabs (one per metric) and return them.
        """
        plot_data_dict, global_min, global_max = generate_plot_data_eod_aaod(
            eod_aaod, test_cols, metrics=metrics
        )
        base_data = SpiderPlotData(ylim_max=global_max, ylim_min=global_min, plot_config=plot_config)
        plot_data_list = create_spider_plot_data_eod_aaod(plot_data_dict, test_cols, metrics, base_data)

        chart_tabs: List[QTabWidget] = []
        temp: Dict[str, List[SpiderPlotData]] = {}
        for pd in plot_data_list:
            temp.setdefault(pd.metric, []).append(pd)

        for metric, data_list in temp.items():
            tw = display_spider_charts_in_tabs(data_list, show_toolbar=self._show_mpl_toolbar)
            tw.setObjectName(f"{metric.upper()} Spider Charts")
            chart_tabs.append(tw)

        return chart_tabs

    def update_qwk_tables(self, result):
        """
        Called by MainController when QWK worker finishes.
        Expect result = (all_rows, filtered_rows, kappas_rows, plot_args).
        Build three tables + a spider‐chart tab, then insert into QTabWidget.
        """
        (all_rows, filtered_rows, kappas_rows, plot_args), reference_groups = result

        # Table of all delta‐κ values
        headers_delta = ["Model", "Category", "Reference", "Group", "Delta Kappa", "Lower CI", "Upper CI"]
        delta_table_data = _add_ref_group(all_rows, reference_groups)
        table_all = self.create_table_widget(headers_delta, delta_table_data)

        # Table of filtered delta‐κ values
        table_filtered = self.create_table_widget(headers_delta, filtered_rows)

        # Table of overall kappa metrics
        headers_kappas = ["Model", "Kappa", "Lower CI", "Upper CI"]
        table_kappas = self.create_table_widget(headers_kappas, kappas_rows)

        # Spider‐chart tab for QWK
        charts_tab = self.create_spider_plot_from_qwk(*plot_args)

        tabs = {
            table_kappas: "Kappas & Intervals",
            table_all: "All Delta κ Values",
            table_filtered: "QWK Filtered (CI Excludes Zero)",
            charts_tab: "QWK Spider Charts",
        }
        self.update_tabs(tabs)

    def update_eod_aaod_tables(self, result):
        """
        Called by MainController when EOD/AAOD worker finishes.
        Expect result = (all_eod_rows, all_aaod_rows, filtered_rows, plot_args).
        Build two tables + filtered table + spider‐chart tabs, then insert into QTabWidget.
        """
        table_info, reference_groups = result
        all_eod_rows, all_aaod_rows, filtered_rows, plot_args = table_info

        # Table of all EOD values
        headers = ["Model", "Category", "Reference", "Group", "Median", "Lower CI", "Upper CI"]
        eod_table_data = _add_ref_group(all_eod_rows, reference_groups)
        table_all_eod = self.create_table_widget(headers, eod_table_data)

        # Table of all AAOD values
        aaod_table_data = _add_ref_group(all_aaod_rows, reference_groups)
        table_all_aaod = self.create_table_widget(headers, aaod_table_data)

        # Filtered table with an extra “Metric” column inserted at index 3
        filt_headers = headers.copy()
        filt_headers.insert(4, "Metric")
        filtered_table_data = _add_ref_group(filtered_rows, reference_groups)
        table_filtered = self.create_table_widget(filt_headers, filtered_table_data)

        # One or more spider‐chart tabs for EOD/AAOD
        chart_tabs = self.create_spider_plot_from_eod_aaod(*plot_args)

        tabs: Dict[QWidget, str] = {
            table_all_eod: "All EOD Values",
            table_all_aaod: "All AAOD Values",
            table_filtered: r"EOD/AAOD Filtered (values outside [-0.1, 0.1])",
        }
        for ct in chart_tabs:
            tabs[ct] = ct.objectName()

        self.update_tabs(tabs)
