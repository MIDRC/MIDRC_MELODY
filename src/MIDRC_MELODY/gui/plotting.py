from typing import List
import math
from PySide6.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from PySide6.QtCharts import QPolarChart, QChartView, QLineSeries, QValueAxis, QCategoryAxis, QAreaSeries
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QBrush, QColor
from MIDRC_MELODY.common.plot_tools import SpiderPlotData  # reuse data class from common module
from MIDRC_MELODY.gui.grabbablewidget import GrabbableChartView
from MIDRC_MELODY.gui.plotly_spider_widget import PlotlySpiderWidget


def _fill_bounds(
    chart: QPolarChart,
    angles: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float],
    cat_axis: QCategoryAxis,
    radial_axis: QValueAxis,
) -> None:
    """
    Draw upper and lower bound lines on the spider chart (no filled area).
    """
    # Ensure lists align; if not, skip
    if not angles or len(lower_bounds) != len(angles) or len(upper_bounds) != len(angles):
        return

    # Draw lower bound line
    lower_series = QLineSeries()
    for angle, lo in zip(angles, lower_bounds):
        lower_series.append(angle, lo)
    # Close the loop
    lower_series.append(360, lower_bounds[0])
    pen_lo = QPen(QColor('steelblue'))
    pen_lo.setStyle(Qt.DashLine)
    pen_lo.setWidth(2)
    pen_lo.setColor(QColor(70, 130, 180, 128))  # semi-transparent steelblue
    lower_series.setPen(pen_lo)
    chart.addSeries(lower_series)
    lower_series.attachAxis(cat_axis)
    lower_series.attachAxis(radial_axis)

    # Draw upper bound line
    upper_series = QLineSeries()
    for angle, hi in zip(angles, upper_bounds):
        upper_series.append(angle, hi)
    upper_series.append(360, upper_bounds[0])
    pen_hi = QPen(QColor('steelblue'))
    pen_hi.setStyle(Qt.DashLine)
    pen_hi.setWidth(2)
    pen_hi.setColor(QColor(70, 130, 180, 128))
    upper_series.setPen(pen_hi)
    chart.addSeries(upper_series)
    upper_series.attachAxis(cat_axis)
    upper_series.attachAxis(radial_axis)


def _apply_metric_overlay(
    chart: QPolarChart,
    angles: List[float],
    values: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float],
    cat_axis: QCategoryAxis,
    radial_axis: QValueAxis,
    spider_data: SpiderPlotData,
) -> None:
    """
    Apply metric-specific overlays (baselines, shaded regions, threshold markers) to the given QPolarChart.
    """
    metric = spider_data.metric.upper()
    # Precompute full-circle angles for continuous lines/fills
    full_angles_deg = [360 * i / 99 for i in range(100)]

    # Determine small angle delta for threshold line markers
    step_size = angles[1] - angles[0] if len(angles) > 1 else 360
    delta = step_size * 0.2

    # Configuration for each metric
    overlay_config = {
        'QWK': {
            'baseline': {'type': 'line', 'y': 0, 'style': Qt.DashLine, 'color': 'seagreen', 'linewidth': 3, 'alpha': 0.8},
            'thresholds': [
                (lower_bounds, lambda v: v > 0, 'maroon'),
                (upper_bounds, lambda v: v < 0, 'red'),
            ],
        },
        'EOD': {
            'fill': {'lo': -0.1, 'hi': 0.1, 'color': 'lightgreen', 'alpha': 0.5},
            'thresholds': [
                (values, lambda v: v > 0.1, 'maroon'),
                (values, lambda v: v < -0.1, 'red'),
            ],
        },
        'AAOD': {
            'fill': {'lo': 0, 'hi': 0.1, 'color': 'lightgreen', 'alpha': 0.5},
            'baseline': {'type': 'ylim', 'lo': 0},
            'thresholds': [
                (values, lambda v: v > 0.1, 'maroon'),
            ],
        },
    }
    cfg = overlay_config.get(metric)
    if not cfg:
        return

    # --- Baseline rendering ---
    if 'baseline' in cfg:
        base = cfg['baseline']
        if base['type'] == 'line':
            baseline_series = QLineSeries()
            for angle_deg in full_angles_deg:
                baseline_series.append(angle_deg, base['y'])
            pen = QPen(QColor(base['color']))
            pen.setWidth(base['linewidth'])
            pen.setStyle(base['style'])
            color = QColor(base['color'])
            color.setAlphaF(base['alpha'])
            pen.setColor(color)
            baseline_series.setPen(pen)
            chart.addSeries(baseline_series)
            baseline_series.attachAxis(cat_axis)
            baseline_series.attachAxis(radial_axis)
        elif base['type'] == 'ylim':
            # Adjust radial axis minimum
            y_max = spider_data.ylim_max[spider_data.metric]
            radial_axis.setRange(base['lo'], y_max)

    # --- Fill region if specified ---
    if 'fill' in cfg and False: # Set to False to disable fill for now, QAreaSeries is not yet supported in QPolarChart
        f = cfg['fill']
        lower_series = QLineSeries()
        upper_series = QLineSeries()
        for angle_deg in full_angles_deg:
            lower_series.append(angle_deg, f['lo'])
            upper_series.append(angle_deg, f['hi'])
        area_series = QAreaSeries(upper_series, lower_series)
        color = QColor(f['color'])
        color.setAlphaF(f['alpha'])
        brush = QBrush(color)
        area_series.setBrush(brush)
        area_series.setPen(QPen(Qt.NoPen))
        chart.addSeries(area_series)
        area_series.attachAxis(cat_axis)
        area_series.attachAxis(radial_axis)

    # --- Threshold markers as short perpendicular lines ---
    for data_list, cond, color_name in cfg.get('thresholds', []):
        for i, v in enumerate(data_list):
            if cond(v):
                angle_deg = angles[i]
                radius = v
                d = delta * (spider_data.ylim_max[spider_data.metric] - radius) / (spider_data.ylim_max[spider_data.metric] - spider_data.ylim_min[spider_data.metric])
                # Create a small line segment around the threshold point
                line_series = QLineSeries()
                line_series.append(angle_deg - d, radius)
                line_series.append(angle_deg + d, radius)
                pen = QPen(QColor(color_name))
                pen.setWidth(2)
                line_series.setPen(pen)
                chart.addSeries(line_series)
                line_series.attachAxis(cat_axis)
                line_series.attachAxis(radial_axis)


def create_spider_chart(spider_data: SpiderPlotData) -> QPolarChart:
    """
    Create a QPolarChart based on the SpiderPlotData, including metric-specific overlays.
    """
    chart = QPolarChart()
    chart.setTitle(f"{spider_data.model_name} - {spider_data.metric}")
    series = QLineSeries()

    labels = spider_data.groups
    values = spider_data.values
    lower_bounds = spider_data.lower_bounds
    upper_bounds = spider_data.upper_bounds

    # Compute angles in degrees for QtCharts
    step_size: float = 360 / len(labels)
    angles: List[float] = [step_size * i for i in range(len(labels))]

    # Add points for each group and close the loop
    for angle, value in zip(angles, values):
        series.append(angle, value)
    # Close the loop by repeating the first point at 360°
    if series.points():
        series.append(angles[0] + 360, series.points()[0].y())

    chart.addSeries(series)

    # Create and configure the angular axis (categories for group labels)
    cat_axis = QCategoryAxis()
    cat_axis.setRange(0, 360)
    cat_axis.setLabelsPosition(QCategoryAxis.AxisLabelsPositionOnValue)
    for label, angle in zip(labels, angles):
        cat_axis.append(label, angle)
    chart.addAxis(cat_axis, QPolarChart.PolarOrientationAngular)
    series.attachAxis(cat_axis)

    # Create and configure the radial axis
    radial_axis = QValueAxis()
    radial_axis.setRange(spider_data.ylim_min[spider_data.metric],
                         spider_data.ylim_max[spider_data.metric])
    radial_axis.setLabelFormat("%.2f")
    chart.addAxis(radial_axis, QPolarChart.PolarOrientationRadial)
    series.attachAxis(radial_axis)

    # Fill the area between lower and upper bounds
    _fill_bounds(
        chart,
        angles,
        lower_bounds,
        upper_bounds,
        cat_axis,
        radial_axis,
    )

    # Apply metric-specific overlay via helper function
    _apply_metric_overlay(
        chart,
        angles,
        values,
        lower_bounds,
        upper_bounds,
        cat_axis,
        radial_axis,
        spider_data,
    )

    chart.legend().hide()
    return chart


def _set_spider_chart_copyable_data(chart_view: GrabbableChartView, spider_data: SpiderPlotData) -> None:
    """
    Set the copyable data for the spider chart.

    :arg chart_view: GrabbableChartView to set the copyable data for.
    :arg spider_data: SpiderPlotData containing the data to be displayed.
    """
    if chart_view and spider_data:
        headers = ['Model', 'Metric', 'Category', 'Group', 'Value']
        formatted_text = "\t".join(headers) + "\n"
        for group, value in zip(spider_data.groups, spider_data.values):
            c, g = group.split(': ', 1) if ': ' in group else (group, group)
            formatted_text += f"{spider_data.model_name}\t{spider_data.metric}\t{c}\t{g}\t{value}\n"
        chart_view.copyable_data = formatted_text


def display_spider_charts_in_tabs_deprecated(spider_data_list: List[SpiderPlotData]) -> QTabWidget:
    """
    Given a list of SpiderPlotData objects, create a QTabWidget where each tab displays
    the corresponding spider chart in a QChartView.
    """
    tab_widget = QTabWidget()
    for spider_data in spider_data_list:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        chart = create_spider_chart(spider_data)
        chart_view: GrabbableChartView = GrabbableChartView(
            chart,
            save_file_prefix=f"MIDRC-MELODY_{spider_data.metric}_{spider_data.model_name}_spider_chart",
        )
        _set_spider_chart_copyable_data(chart_view, spider_data)
        chart_view.setRenderHint(QPainter.Antialiasing)  # ensure smooth rendering
        layout.addWidget(chart_view)
        tab_widget.addTab(widget, spider_data.model_name)
    return tab_widget

def display_spider_charts_in_tabs(spider_data_list: List[SpiderPlotData]) -> QTabWidget:
    """
    Replaces the old QtCharts polar approach with a Plotly-based QWebEngineView.
    """
    tab_widget = QTabWidget()
    for spider_data in spider_data_list:
        container = QWidget()
        layout = QVBoxLayout(container)

        # Use Plotly to draw the spider chart and embed as HTML
        plotly_widget = PlotlySpiderWidget(spider_data)
        layout.addWidget(plotly_widget)

        tab_widget.addTab(container, spider_data.model_name)
    return tab_widget
