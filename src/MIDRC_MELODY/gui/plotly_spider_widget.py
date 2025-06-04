from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtWebEngineWidgets import QWebEngineView
from MIDRC_MELODY.common.plot_tools import SpiderPlotData
from MIDRC_MELODY.common.plotly_spider import spider_to_html


class PlotlySpiderWidget(QWidget):
    def __init__(self, spider_data: SpiderPlotData, parent=None):
        super().__init__(parent)

        # 1) Generate the HTML <div> string from Plotly
        html_div: str = spider_to_html(spider_data)

        # 2) Create a QWebEngineView and load that HTML
        self._view = QWebEngineView(self)
        # Note: setHtml defaults to UTF-8. If you need local resources, pass baseUrl.
        self._view.setHtml(html_div)

        # 3) Put the QWebEngineView inside a vertical layout
        layout = QVBoxLayout(self)
        layout.addWidget(self._view)
        self.setLayout(layout)
