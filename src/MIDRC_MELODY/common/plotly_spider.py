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

from numpy import pi as np_pi
import plotly.graph_objects as go
import plotly.io as pio

from MIDRC_MELODY.common.plot_tools import SpiderPlotData, get_full_theta, compute_angles, prepare_and_sort


def spider_to_html(spider_data: SpiderPlotData) -> str:
    """
    Given a SpiderPlotData, return an HTML <div> string containing a Plotly radar chart where:
      - Median 'values' are drawn as a line + circle markers at discrete category angles (in degrees).
      - The region between 'lower_bounds' and 'upper_bounds' is shaded (no boundary lines),
        computed using full_theta_deg for a smooth polygon.
      - Baseline(s) and safe-band fills also use full_theta_deg (0→360°).
      - Thresholds are drawn as short line segments at ±Δθ around each category angle,
        where Δθ = delta * (ymax – radius)/(ymax – ymin) so that the visible length
        of each tick is roughly constant in screen pixels.
      - Line thickness is 1 px, and each tick is colored correctly.
    """
    raw_metric: str = spider_data.metric
    metric_display: str = spider_data.metric.upper()
    groups, values, lower_bounds, upper_bounds = prepare_and_sort(spider_data)

    # 1) Number of categories (excluding the “closing” duplicate)
    N = len(groups)

    # 2) Build full_theta_deg: 100 points from 0° to 360° for smooth circular traces
    full_theta = get_full_theta()

    # 3) Compute discrete category angles in degrees: [0°, 360/N°, 2*360/N°, …]
    cat_angles = compute_angles(len(groups), spider_data.plot_config)
    cat_labels = [g.split(": ", 1)[-1] for g in groups]

    # 5) Determine radial axis min/max from spider_data
    radial_min = spider_data.ylim_min.get(raw_metric, None)
    radial_max = spider_data.ylim_max.get(raw_metric, None)

    # 6) Start building the Plotly figure
    fig = go.Figure()

    # 7) Shade between lower_bounds and upper_bounds (CI band)
    theta_ub = cat_angles
    theta_lb = cat_angles
    theta_ci = theta_ub + theta_lb[::-1]
    r_ci = upper_bounds + lower_bounds[::-1]
    fig.add_trace(
        go.Scatterpolar(
            r=r_ci,
            theta=theta_ci,
            thetaunit="radians",       # treat ALL numeric theta as radians
            mode="none",
            fill="toself",
            fillcolor="rgba(70,130,180,0.2)",  # semi-transparent steelblue
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 8) Median “values” trace (lines + circle markers)
    theta_vals = cat_angles
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=theta_vals,
            thetaunit="radians",
            mode="lines+markers",
            line=dict(color="steelblue", width=2),
            marker=dict(symbol="circle", size=6, color="steelblue"),
            customdata=list(zip(groups, lower_bounds, upper_bounds)),
            hovertemplate="%{customdata[0]}<br>Median: %{r:.3f} [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<extra></extra>",
            showlegend=False,
        )
    )

    # 9) Metric-specific overlay rules (matching plot_tools._apply_metric_overlay)
    overlay_config = {
        "QWK": {
            "baseline": {"type": "line", "y": 0, "color": "seagreen", "width": 3, "dash": "dash", "alpha": 0.8},
            "thresholds": [
                (lower_bounds[:N], lambda v: v > 0, "maroon"),
                (upper_bounds[:N], lambda v: v < 0, "red"),
            ],
        },
        "EOD": {
            "fill": {"lo": -0.1, "hi": 0.1, "color": "lightgreen", "alpha": 0.5},
            "thresholds": [
                (values[:N], lambda v: v > 0.1, "maroon"),
                (values[:N], lambda v: v < -0.1, "red"),
            ],
        },
        "AAOD": {
            "fill": {"lo": 0.0, "hi": 0.1, "color": "lightgreen", "alpha": 0.5},
            "baseline": {"type": "ylim", "lo": 0.0},
            "thresholds": [
                (values[:N], lambda v: v > 0.1, "maroon"),
            ],
        },
    }
    cfg = overlay_config.get(metric_display, None)
    if cfg:
        # 9a) Draw baseline if specified
        if "baseline" in cfg:
            base = cfg["baseline"]
            if base["type"] == "line":
                baseline_r = [base["y"]] * len(full_theta)
                fig.add_trace(
                    go.Scatterpolar(
                        r=baseline_r,
                        theta=list(full_theta),
                        thetaunit="radians",  # treat ALL numeric theta as radians
                        mode="lines",
                        line=dict(color=base["color"], dash=base["dash"], width=base["width"]),
                        opacity=base["alpha"],
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            elif base["type"] == "ylim":
                # Override radial_min below
                radial_min = base["lo"]

        # 9b) Draw “safe‐band” fill if specified
        if "fill" in cfg:
            f = cfg["fill"]
            hi_vals = [f["hi"]] * len(full_theta)
            lo_vals = [f["lo"]] * len(full_theta)
            theta_fill = list(full_theta) + list(full_theta[::-1])
            r_fill = hi_vals + lo_vals[::-1]
            fig.add_trace(
                go.Scatterpolar(
                    r=r_fill,
                    theta=theta_fill,
                    thetaunit="radians",  # treat ALL numeric theta as radians
                    mode="none",
                    fill="toself",
                    fillcolor=f"rgba({_hex_to_rgb(f['color'])}, {f['alpha']})",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # 9c) Draw threshold ticks as short line segments of constant pixel length
        #     Compute Δθ per‐point: Δθ = delta * (radial_max – radius)/(radial_max – radial_min)
        #     so that a fixed “delta” produces roughly uniform on‐screen length.
        delta = 0.15 # Adjust this value to change the length of the threshold ticks (in radians)
        for data_list, cond, color_name in cfg.get("thresholds", []):
            for i, v in enumerate(data_list):
                if cond(v):
                    angle = cat_angles[i]
                    radius = v
                    # Avoid division by zero
                    if radial_max == radial_min:
                        d_theta = 0
                    else:
                        d_theta = delta * (radial_max - radius) / (radial_max - radial_min)

                    theta_line = [angle - d_theta, angle + d_theta]
                    r_line = [radius, radius]
                    fig.add_trace(
                        go.Scatterpolar(
                            r=r_line,
                            theta=theta_line,
                            thetaunit="radians",  # treat ALL numeric theta as radians
                            mode="lines",
                            line=dict(color=color_name, width=1.5),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

    # 10) Final polar layout adjustments - Tick angles must be degrees for Plotly
    fig.update_layout(
        title=f"{spider_data.model_name} – {metric_display}",
        polar=dict(
            radialaxis=dict(range=[radial_min, radial_max], visible=True),
            angularaxis=dict(
                tickmode="array",
                tickvals=[ang * 180.0/np_pi for ang in cat_angles],
                ticktext=cat_labels,
            ),
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # 11) Export only the <div> (omit full HTML <head>), using CDN for Plotly.js
    html_str = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    return html_str


def _hex_to_rgb(css_color: str) -> str:
    """
    Convert a CSS color name or hex string (e.g. "lightgreen") into an "R,G,B" integer string
    so that Plotly’s fillcolor accepts "rgba(R,G,B,alpha)".
    """
    import matplotlib.colors as mcolors

    rgba = mcolors.to_rgba(css_color)
    r, g, b, _ = [int(255 * c) for c in rgba]
    return f"{r},{g},{b}"
