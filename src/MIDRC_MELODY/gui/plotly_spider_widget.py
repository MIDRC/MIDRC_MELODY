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
