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
"""Qt-friendly Matplotlib spider-plot widget (drop-in replacement for
``plotly_spider_widget``).

This revision exposes the NavigationToolbar so callers can show/hide it at runtime.
Cropping issues are still handled by constrained layout + tight_layout on resize.
"""
from __future__ import annotations

from typing import List, Optional
import warnings

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from MIDRC_MELODY.common.matplotlib_spider import plot_spider_chart
from MIDRC_MELODY.common.plot_tools import SpiderPlotData
from MIDRC_MELODY.gui.grabbablewidget import GrabbableWidgetMixin, GrabbableChartView

__all__ = [
    "MatplotlibSpiderWidget",
    "display_spider_charts_in_tabs_matplotlib",
]


class MatplotlibSpiderWidget(QWidget):
    """Embed a Matplotlib *spider chart* (a.k.a. radar chart) in a Qt widget.

    Parameters
    ----------
    spider_data
        Data structure consumed by :func:`plot_spider_chart`.
    show_toolbar
        If *True*, display the Matplotlib *navigation toolbar*; if *False*, hide it.
        You can also toggle visibility at runtime via `set_toolbar_visible()`.
    pad
        Extra padding (in figure‐fraction units) passed to ``tight_layout`` so
        that very large tick labels have room.
    """

    def __init__(
        self,
        spider_data: SpiderPlotData,
        parent: Optional[QWidget] = None,
        *,
        show_toolbar: bool = False,
        pad: float = 0.4,
    ) -> None:
        super().__init__(parent)
        self._grabbable_mixin = GrabbableWidgetMixin(self, "MIDRC-MELODY Spider Chart ")
        _set_spider_chart_copyable_data(self, spider_data)

        self._pad = pad

        # ───────────────────────────────────────────────── build the Figure
        with mpl.rc_context({"figure.constrained_layout.use": True}):
            # Suppress only the “The figure layout has changed to tight” warning
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "The figure layout has changed to tight",
                    category=UserWarning,
                )
                fig = plot_spider_chart(spider_data)
        fig.set_constrained_layout(True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "The figure layout has changed to tight",
                category=UserWarning,
            )
            fig.tight_layout(pad=self._pad)

        # ────────────────────────────────────────── wrap figure in a FigureCanvas
        self._canvas = FigureCanvas(fig)

        # ─────────────────────────────────── create the NavigationToolbar
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._toolbar.setVisible(show_toolbar)

        # ────────────────────────────────────── lay out canvas and (optional) toolbar
        layout = QVBoxLayout(self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        # ────────────────────────────────────── auto-adjust on resize events
        self._canvas.mpl_connect("resize_event", self._on_resize)

    # ---------------------------------------------------------------- API
    def figure(self):
        """Return the underlying :class:`matplotlib.figure.Figure`."""
        return self._canvas.figure

    def canvas(self) -> FigureCanvas:
        return self._canvas

    def set_toolbar_visible(self, visible: bool) -> None:
        """Show or hide the Matplotlib navigation toolbar."""
        self._toolbar.setVisible(visible)

    # -------------------------------------------------------------- events
    def resizeEvent(self, event):  # Qt event → keep layout fresh
        super().resizeEvent(event)
        self._on_resize()

    # --------------------------------------------------------------- intern
    def _on_resize(self, *_):
        """Re-run layout manager so annotations never get clipped."""
        if not self._canvas or self._canvas.isHidden():
            return
        fig = self._canvas.figure
        if getattr(fig, "canvas", None) is None:
            return
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "The figure layout has changed to tight",
                category=UserWarning,
            )
            fig.tight_layout(pad=self._pad)
        self._canvas.draw_idle()

    @property
    def copyable_data(self) -> str:
        """
        Get the copyable data for the chart view.

        Returns:
            str: The data to be copied to the clipboard when requested.
        """
        return self._grabbable_mixin.copyable_data

    @copyable_data.setter
    def copyable_data(self, data: str):
        """
        Set the copyable data for the chart view.

        Args:
            data (str): The data to be copied to the clipboard when requested.

        Returns:
            None
        """
        self._grabbable_mixin.copyable_data = data


# ---------------------------------------------------------------------------
# Bulk helper: one tab per model (mirrors Plotly counterpart)
# ---------------------------------------------------------------------------

def display_spider_charts_in_tabs_matplotlib(
    spider_data_list: List[SpiderPlotData], *, show_toolbar: bool = False, pad: float = 0.4
) -> QTabWidget:
    """Assemble a ``QTabWidget`` containing one spider chart per entry."""

    tab_widget = QTabWidget()
    tab_widget.setDocumentMode(True)
    tab_widget.setTabPosition(QTabWidget.North)

    for spider_data in spider_data_list:
        container = QWidget()
        layout = QVBoxLayout(container)

        mpl_widget = MatplotlibSpiderWidget(
            spider_data,
            parent=container,
            show_toolbar=show_toolbar,
            pad=pad,
        )
        layout.addWidget(mpl_widget)

        tab_widget.addTab(container, spider_data.model_name)
    return tab_widget


def _set_spider_chart_copyable_data(widget: GrabbableWidgetMixin|QWidget, spider_data: SpiderPlotData) -> None:
    """
    Set the copyable data for the spider chart.

    :arg widget: A GrabbalbeWidgetMixin or a QWidget that forwards the copyable_data property to a GrabbableWidgetMixin.
    :arg spider_data: SpiderPlotData containing the data to be displayed.
    """
    if widget and spider_data:
        headers = ['Model', 'Metric', 'Category', 'Group', 'Value']
        formatted_text = "\t".join(headers) + "\n"
        for group, value in zip(spider_data.groups, spider_data.values):
            c, g = group.split(': ', 1) if ': ' in group else (group, group)
            formatted_text += f"{spider_data.model_name}\t{spider_data.metric}\t{c}\t{g}\t{value}\n"
        widget.copyable_data = formatted_text
