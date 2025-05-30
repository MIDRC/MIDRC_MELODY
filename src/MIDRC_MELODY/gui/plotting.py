from typing import List
from PySide6.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from PySide6.QtCharts import QPolarChart, QChartView, QLineSeries, QValueAxis, QCategoryAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter  # added import
from MIDRC_MELODY.common.plot_tools import SpiderPlotData  # reuse data class from common module

def create_spider_chart(spider_data: SpiderPlotData) -> QPolarChart:
    """
    Create a QPolarChart based on the SpiderPlotData.
    """
    chart = QPolarChart()
    chart.setTitle(f"{spider_data.model_name} - {spider_data.metric}")
    series = QLineSeries()
    
    labels = spider_data.groups
    values = spider_data.values
    step_size: float = 360 / len(labels)
    angles: List[float] = [step_size * i for i in range(len(labels))]

    # Add points for each group and close the loop.
    for angle, value in zip(angles, values):
        series.append(angle, value)
    # Close the loop
    if series.points():
        series.append(360, series.points()[0].y())
    
    chart.addSeries(series)
    
    # Create and configure the angular axis (using categories for group labels).
    cat_axis = QCategoryAxis()
    cat_axis.setRange(0, 360)
    cat_axis.setLabelsPosition(QCategoryAxis.AxisLabelsPositionOnValue)
    for label, angle in zip(labels, angles):
        cat_axis.append(label, angle)
    chart.addAxis(cat_axis, QPolarChart.PolarOrientationAngular)
    series.attachAxis(cat_axis)
    
    # Create and configure the radial axis.
    radial_axis = QValueAxis()
    radial_axis.setRange(spider_data.ylim_min[spider_data.metric],
                         spider_data.ylim_max[spider_data.metric])
    radial_axis.setLabelFormat("%.2f")
    chart.addAxis(radial_axis, QPolarChart.PolarOrientationRadial)
    series.attachAxis(radial_axis)
    
    chart.legend().hide()
    return chart

def display_spider_charts_in_tabs(spider_data_list: List[SpiderPlotData]) -> QTabWidget:
    """
    Given a list of SpiderPlotData objects, create a QTabWidget where each tab displays
    the corresponding spider chart in a QChartView.
    """
    tab_widget = QTabWidget()
    for spider_data in spider_data_list:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        chart = create_spider_chart(spider_data)
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)  # updated to use QPainter.Antialiasing
        layout.addWidget(chart_view)
        tab_widget.addTab(widget, spider_data.model_name)
    return tab_widget
