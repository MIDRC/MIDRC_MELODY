#  Copyright (c) 2025 Medical Imaging and Data Resource Center (MIDRC).
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
main_window.py

Main application window for MIDRC Melody GUI. Handles menus, toolbar actions,
and central tab widget, including spider-chart tabs, data tables, and progress output.
"""

from typing import Any, Dict, List, Optional, Tuple

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

from MIDRC_MELODY.common.eod_aaod_metrics import (
    create_spider_plot_data_eod_aaod,
    generate_plot_data_eod_aaod,
)
from MIDRC_MELODY.common.plot_tools import SpiderPlotData
from MIDRC_MELODY.common.qwk_metrics import create_spider_plot_data_qwk

from MIDRC_MELODY.gui.copyabletableview import CopyableTableWidget
from MIDRC_MELODY.gui.matplotlib_spider_widget import (
    MatplotlibSpiderWidget,
    display_spider_charts_in_tabs_matplotlib as display_spider_charts_in_tabs,
)
from MIDRC_MELODY.gui.tqdm_handler import ANSIProcessor
from MIDRC_MELODY.gui.data_loading import load_config_file, edit_config_file
from MIDRC_MELODY.gui.main_controller import MainController


class NumericSortTableWidgetItem(QTableWidgetItem):
    """
    A QTableWidgetItem that sorts numerically when its text can be parsed as float.

    Falls back to default string comparison on ValueError.
    """

    def __lt__(self, other: QTableWidgetItem) -> bool:
        """
        Compare two items as floats when possible.

        Parameters
        ----------
        other : QTableWidgetItem
            The other table widget item to compare.

        Returns
        -------
        bool
            True if self < other numerically, otherwise lexical comparison.
        """
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


def _add_ref_group(
    rows: List[Tuple[List[str], QColor]], ref_groups: Dict[str, str]
) -> List[Tuple[List[str], QColor]]:
    """
    Add a reference group (from ref_groups) to each row based on the category.

    If the category (row_data[1]) is not found in ref_groups, use "N/A".

    Parameters
    ----------
    rows : list of tuple(list of str, QColor)
        Each tuple contains a row's data and an optional color.
    ref_groups : dict of str to str
        Mapping from category to reference group string.

    Returns
    -------
    list of tuple(list of str, QColor)
        New list of rows with reference group inserted at index 2.
    """
    new_rows: List[Tuple[List[str], QColor]] = []
    for row_data, color in rows:
        row_copy = list(row_data)
        category = row_copy[1]
        ref = ref_groups.get(category, "N/A")
        row_copy.insert(2, ref)
        new_rows.append((row_copy, color))
    return new_rows


class MainWindow(QMainWindow):
    """
    Main application window for the Melody GUI.

    Manages the toolbar, menus, and the central QTabWidget which holds
    progress output, data tables, and spider-chart tabs. Provides actions
    for loading/editing config and toggling Matplotlib spider-chart toolbars.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the MainWindow.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget in the Qt hierarchy. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Melody GUI")
        self.resize(1200, 600)

        # Thread pool for background tasks
        self.threadpool = QThreadPool()

        # Whether to show Matplotlib toolbar in spider-chart tabs
        self._show_mpl_toolbar: bool = False

        # Instantiate controller and hand it this window
        self.controller = MainController(self)

        # Build the menu bar, toolbar, and central widget
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_central_widget()

        # Prepare the progress view (console) as a QPlainTextEdit (hidden by default)
        self.progress_view: QPlainTextEdit = QPlainTextEdit()
        self._ansi_processor: Optional[ANSIProcessor] = None  # Initialized on first use

    def _create_menu_bar(self) -> None:
        """
        Build the application's menu bar with File and Configuration menus.
        """
        menu_bar = self.menuBar()

        # File menu: Load Config File
        file_menu = menu_bar.addMenu("File")
        load_config_act = QAction("Load Config File", self)
        load_config_act.triggered.connect(self.load_config_file)
        file_menu.addAction(load_config_act)

        # Configuration menu: Edit Config File
        config_menu = menu_bar.addMenu("Configuration")
        edit_config_act = QAction("Edit Config File", self)
        edit_config_act.triggered.connect(self.edit_config)
        config_menu.addAction(edit_config_act)

    def _create_tool_bar(self) -> None:
        """
        Build the main toolbar with actions for metrics, config, and toggling spider-chart toolbar.
        """
        toolbar = self.addToolBar("MainToolbar")
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        # EOD/AAOD Metrics action
        eod_icon = QIcon.fromTheme("computer")
        eod_act = QAction(eod_icon, "EOD/AAOD Metrics", self)
        eod_act.setToolTip("Calculate EOD/AAOD Metrics")
        eod_act.triggered.connect(self.controller.calculate_eod_aaod)
        toolbar.addAction(eod_act)

        # QWK Metrics action
        qwk_icon = QIcon.fromTheme("accessories-calculator")
        qwk_act = QAction(qwk_icon, "QWK Metrics", self)
        qwk_act.setToolTip("Calculate QWK Metrics")
        qwk_act.triggered.connect(self.controller.calculate_qwk)
        toolbar.addAction(qwk_act)

        # Spacer to push next actions to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # Checkable action: Show/Hide Matplotlib spider-chart toolbar
        self._toggle_mpl_act = QAction(
            f"{'Hide' if self._show_mpl_toolbar else 'Show'} Plot Toolbar", self
        )
        self._toggle_mpl_act.setCheckable(True)
        self._toggle_mpl_act.setChecked(self._show_mpl_toolbar)
        self._toggle_mpl_act.setToolTip(
            "Toggle the Matplotlib navigation toolbar in spider-chart tabs"
        )
        self._toggle_mpl_act.toggled.connect(self._on_toggle_mpl_toolbar)
        toolbar.addAction(self._toggle_mpl_act)

        # Config action
        config_icon = QIcon.fromTheme("document-properties")
        config_act = QAction(config_icon, "Config", self)
        config_act.setToolTip("Edit Configuration")
        config_act.triggered.connect(self.edit_config)
        toolbar.addAction(config_act)

    def _create_central_widget(self) -> None:
        """
        Create and set the central widget as a QTabWidget.

        Tab index 0 will hold the progress view; subsequent tabs are for data and charts.
        """
        tab_widget = QTabWidget()
        tab_widget.setMovable(True)
        self.setCentralWidget(tab_widget)

    @Slot()
    def load_config_file(self) -> None:
        """
        Slot to load configuration file via the data_loading module.
        """
        load_config_file(self)

    @Slot()
    def edit_config(self) -> None:
        """
        Slot to edit the configuration file via the data_loading module.
        """
        edit_config_file(self)

    @Slot(bool)
    def _on_toggle_mpl_toolbar(self, checked: bool) -> None:
        """
        Show or hide the Matplotlib spider-chart toolbar across all existing tabs
        and update the flag for future tabs.

        Parameters
        ----------
        checked : bool
            True to show the toolbar, False to hide it.
        """
        self._show_mpl_toolbar = checked
        self._toggle_mpl_act.setText(f"{'Hide' if checked else 'Show'} Plot Toolbar")

        tabs: QTabWidget = self.centralWidget()  # type: ignore
        for idx in range(tabs.count()):
            page = tabs.widget(idx)
            spiders = page.findChildren(MatplotlibSpiderWidget)
            for spider in spiders:
                spider.set_toolbar_visible(checked)

    def show_progress_view(self) -> None:
        """
        Insert or reinsert the read-only console tab (QPlainTextEdit) at index 0
        so that redirected output appears there.
        """
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setPointSize(10)
        self.progress_view.setFont(fixed_font)
        self.progress_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.progress_view.setReadOnly(True)

        tabs: QTabWidget = self.centralWidget()  # type: ignore

        # Remove any existing "Progress Output" tab
        for i in range(tabs.count()):
            if tabs.tabText(i) == "Progress Output":
                tabs.removeTab(i)
                break

        # Insert new progress tab at index 0 and make it current
        tabs.insertTab(0, self.progress_view, "Progress Output")
        tabs.setCurrentIndex(0)

    def append_progress(self, text: str) -> None:
        """
        Feed each chunk of emitted text through ANSIProcessor so colors/formatting appear.

        Parameters
        ----------
        text : str
            The text chunk containing possible ANSI escape sequences.
        """
        if not self._ansi_processor:
            self._ansi_processor = ANSIProcessor()
        self._ansi_processor.process(self.progress_view, text)

    def update_tabs(
        self, tab_dict: Dict[QWidget, str], set_current: bool = True
    ) -> None:
        """
        Given a mapping {widget: tab_title}, remove existing tabs with those titles,
        then insert each new tab starting at index 1 (index 0 is reserved for progress).

        Parameters
        ----------
        tab_dict : dict
            Mapping of QWidget instances to their desired tab titles.
        set_current : bool, optional
            If True, switch to the first new tab after insertion. Defaults to True.
        """
        tabs: QTabWidget = self.centralWidget()  # type: ignore

        # Remove tabs whose title matches any in tab_dict
        for i in reversed(range(tabs.count())):
            if tabs.tabText(i) in tab_dict.values():
                tabs.removeTab(i)

        # Insert new tabs at indices starting from 1
        for idx, (widget, title) in enumerate(tab_dict.items(), start=1):
            tabs.insertTab(idx, widget, title)

        if set_current:
            tabs.setCurrentIndex(1)

    @staticmethod
    def create_table_widget(
        headers: List[str], rows: List[Tuple[List[str], Optional[QColor]]]
    ) -> CopyableTableWidget:
        """
        Build a CopyableTableWidget with the given headers and rows.

        Parameters
        ----------
        headers : list of str
            Column header names.
        rows : list of (row_data, row_color)
            - row_data: list of str for each cell.
            - row_color: QColor or None; if provided, apply to the last three columns.

        Returns
        -------
        CopyableTableWidget
            The configured and populated table widget.
        """
        table = CopyableTableWidget()
        table.setSortingEnabled(True)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))

        for r, (row_data, row_color) in enumerate(rows):
            for c, cell_text in enumerate(row_data):
                try:
                    float(cell_text)
                    item: QTableWidgetItem = NumericSortTableWidgetItem(cell_text)
                except ValueError:
                    item = QTableWidgetItem(cell_text)

                if row_color is not None and c >= len(row_data) - 3:
                    item.setForeground(QBrush(row_color))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

                table.setItem(r, c, item)

        table.resizeColumnsToContents()
        return table

    def create_spider_plot_from_qwk(
        self, delta_kappas: Any, test_cols: List[str], plot_config: Optional[Dict] = None
    ) -> QTabWidget:
        """
        Build spider-chart tabs for QWK and return a QTabWidget containing them.

        Parameters
        ----------
        delta_kappas : Any
            The computed delta kappa values from the controller.
        test_cols : list of str
            Names of test columns to include in the plot.
        plot_config : dict, optional
            Configuration dictionary for spider-plot styling.

        Returns
        -------
        QTabWidget
            A tab widget containing one spider-chart per model.
        """
        plot_data_list = create_spider_plot_data_qwk(
            delta_kappas, test_cols, plot_config=plot_config
        )
        return display_spider_charts_in_tabs(
            plot_data_list, show_toolbar=self._show_mpl_toolbar
        )

    def create_spider_plot_from_eod_aaod(
        self,
        eod_aaod: Any,
        test_cols: List[str],
        plot_config: Optional[Dict] = None,
        *,
        metrics: Tuple[str, str] = ("eod", "aaod"),
    ) -> List[QTabWidget]:
        """
        Build one or more spider-chart tabs for EOD/AAOD metrics.

        Parameters
        ----------
        eod_aaod : Any
            The computed EOD/AAOD values from the controller.
        test_cols : list of str
            Names of test columns to include in the plot.
        plot_config : dict, optional
            Configuration dictionary for spider-plot styling.
        metrics : tuple of str, optional
            Two metric names to plot (default ("eod", "aaod")).

        Returns
        -------
        list of QTabWidget
            A list of tab widgets, one for each metric group.
        """
        plot_data_dict, global_min, global_max = generate_plot_data_eod_aaod(
            eod_aaod, test_cols, metrics=metrics
        )
        base_data = SpiderPlotData(
            ylim_max=global_max, ylim_min=global_min, plot_config=plot_config
        )
        plot_data_list = create_spider_plot_data_eod_aaod(
            plot_data_dict, test_cols, metrics, base_data
        )

        chart_tabs: List[QTabWidget] = []
        grouped: Dict[str, List[SpiderPlotData]] = {}
        for pdata in plot_data_list:
            grouped.setdefault(pdata.metric, []).append(pdata)

        for metric_name, data_list in grouped.items():
            tab_widget = display_spider_charts_in_tabs(
                data_list, show_toolbar=self._show_mpl_toolbar
            )
            tab_widget.setObjectName(f"{metric_name.upper()}_Spider_Charts")
            chart_tabs.append(tab_widget)

        return chart_tabs

    def update_qwk_tables(self, result: Tuple[Any, Dict[str, str]]) -> None:
        """
        Build QWK result tables and spider-chart tab when the worker finishes.

        Parameters
        ----------
        result : tuple
            ( (all_rows, filtered_rows, kappas_rows, plot_args), reference_groups ).
        """
        (all_rows, filtered_rows, kappas_rows, plot_args), reference_groups = result

        # Table of all delta-kappa values
        headers_delta = [
            "Model",
            "Category",
            "Reference",
            "Group",
            "Delta Kappa",
            "Lower CI",
            "Upper CI",
        ]
        delta_table_data = _add_ref_group(all_rows, reference_groups)
        table_all = self.create_table_widget(headers_delta, delta_table_data)

        # Table of filtered delta-kappa values
        table_filtered = self.create_table_widget(headers_delta, filtered_rows)

        # Table of overall kappa metrics
        headers_kappas = ["Model", "Kappa", "Lower CI", "Upper CI"]
        table_kappas = self.create_table_widget(headers_kappas, kappas_rows)

        # Spider-chart tab for QWK
        charts_tab = self.create_spider_plot_from_qwk(*plot_args)

        tabs_dict: Dict[QWidget, str] = {
            table_kappas: "Kappas & Intervals",
            table_all: "All Delta κ Values",
            table_filtered: "QWK Filtered (CI Excludes Zero)",
            charts_tab: "QWK Spider Charts",
        }
        self.update_tabs(tabs_dict)

    def update_eod_aaod_tables(self, result: Tuple[Any, Dict[str, str]]) -> None:
        """
        Build EOD/AAOD result tables and spider-chart tabs when the worker finishes.

        Parameters
        ----------
        result : tuple
            ( (all_eod_rows, all_aaod_rows, filtered_rows, plot_args), reference_groups ).
        """
        (all_eod_rows, all_aaod_rows, filtered_rows, plot_args), reference_groups = result

        # Table of all EOD values
        headers = [
            "Model",
            "Category",
            "Reference",
            "Group",
            "Median",
            "Lower CI",
            "Upper CI",
        ]
        eod_table_data = _add_ref_group(all_eod_rows, reference_groups)
        table_all_eod = self.create_table_widget(headers, eod_table_data)

        # Table of all AAOD values
        aaod_table_data = _add_ref_group(all_aaod_rows, reference_groups)
        table_all_aaod = self.create_table_widget(headers, aaod_table_data)

        # Filtered table with extra "Metric" column at index 4
        filt_headers = headers.copy()
        filt_headers.insert(4, "Metric")
        filtered_table_data = _add_ref_group(filtered_rows, reference_groups)
        table_filtered = self.create_table_widget(filt_headers, filtered_table_data)

        # Spider-chart tabs for EOD/AAOD
        chart_tabs = self.create_spider_plot_from_eod_aaod(*plot_args)

        tabs_dict: Dict[QWidget, str] = {
            table_all_eod: "All EOD Values",
            table_all_aaod: "All AAOD Values",
            table_filtered: r"EOD/AAOD Filtered (values outside [-0.1, 0.1])",
        }
        for ct in chart_tabs:
            tabs_dict[ct] = ct.objectName()

        self.update_tabs(tabs_dict)
