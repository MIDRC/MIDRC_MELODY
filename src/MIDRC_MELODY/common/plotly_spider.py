import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from MIDRC_MELODY.common.plot_tools import SpiderPlotData


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

    # 1) Number of categories (excluding the “closing” duplicate)
    N = len(spider_data.groups)

    # 2) Build full_theta_deg: 100 points from 0° to 360° for smooth circular traces
    full_theta_deg = np.linspace(0, 360, 100)

    # 3) Compute discrete category angles in degrees: [0°, 360/N°, 2*360/N°, …]
    cat_angles_deg = [(360 * i) / N for i in range(N)]
    cat_labels = [g.split(": ", 1)[-1] for g in spider_data.groups]

    # 4) Prepare closed-loop arrays for median, lower_bounds, upper_bounds (length N+1)
    vals = list(spider_data.values)
    lbs = list(spider_data.lower_bounds)
    ubs = list(spider_data.upper_bounds)

    # Append first element to close the loop
    vals.append(vals[0])
    lbs.append(lbs[0])
    ubs.append(ubs[0])

    # 5) Determine radial axis min/max from spider_data
    radial_min = spider_data.ylim_min.get(raw_metric, None)
    radial_max = spider_data.ylim_max.get(raw_metric, None)

    # 6) Start building the Plotly figure
    fig = go.Figure()

    # 7) Shade between lower_bounds and upper_bounds (CI band)
    theta_ub = cat_angles_deg + [cat_angles_deg[0]]
    theta_lb = cat_angles_deg + [cat_angles_deg[0]]
    theta_ci = theta_ub + theta_lb[::-1]
    r_ci = ubs + lbs[::-1]
    fig.add_trace(
        go.Scatterpolar(
            r=r_ci,
            theta=theta_ci,
            mode="none",
            fill="toself",
            fillcolor="rgba(70,130,180,0.2)",  # semi-transparent steelblue
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 8) Median “values” trace (lines + circle markers)
    theta_vals = cat_angles_deg + [cat_angles_deg[0]]
    fig.add_trace(
        go.Scatterpolar(
            r=vals,
            theta=theta_vals,
            mode="lines+markers",
            line=dict(color="steelblue", width=2),
            marker=dict(symbol="circle", size=6, color="steelblue"),
            hovertemplate="%{theta:.1f}°: %{r}<extra></extra>",
            showlegend=False,
        )
    )

    # 9) Metric-specific overlay rules (matching plot_tools._apply_metric_overlay)
    overlay_config = {
        "QWK": {
            "baseline": {"type": "line", "y": 0, "color": "seagreen", "width": 3, "dash": "dash", "alpha": 0.8},
            "thresholds": [
                (lbs[:N], lambda v: v > 0, "maroon"),
                (ubs[:N], lambda v: v < 0, "red"),
            ],
        },
        "EOD": {
            "fill": {"lo": -0.1, "hi": 0.1, "color": "lightgreen", "alpha": 0.5},
            "thresholds": [
                (vals[:N], lambda v: v > 0.1, "maroon"),
                (vals[:N], lambda v: v < -0.1, "red"),
            ],
        },
        "AAOD": {
            "fill": {"lo": 0.0, "hi": 0.1, "color": "lightgreen", "alpha": 0.5},
            "baseline": {"type": "ylim", "lo": 0.0},
            "thresholds": [
                (vals[:N], lambda v: v > 0.1, "maroon"),
            ],
        },
    }
    cfg = overlay_config.get(metric_display, None)
    if cfg:
        # 9a) Draw baseline if specified
        if "baseline" in cfg:
            base = cfg["baseline"]
            if base["type"] == "line":
                baseline_r = [base["y"]] * len(full_theta_deg)
                fig.add_trace(
                    go.Scatterpolar(
                        r=baseline_r,
                        theta=list(full_theta_deg),
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
            hi_vals = [f["hi"]] * len(full_theta_deg)
            lo_vals = [f["lo"]] * len(full_theta_deg)
            theta_fill = list(full_theta_deg) + list(full_theta_deg[::-1])
            r_fill = hi_vals + lo_vals[::-1]
            fig.add_trace(
                go.Scatterpolar(
                    r=r_fill,
                    theta=theta_fill,
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
        delta = 8.0  # base angular‐span in degrees at r = radial_min; adjust if you want slightly longer/shorter
        for data_list, cond, color_name in cfg.get("thresholds", []):
            for i, v in enumerate(data_list):
                if cond(v):
                    angle = cat_angles_deg[i]
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
                            mode="lines",
                            line=dict(color=color_name, width=1.5),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

    # 10) Final polar layout adjustments
    fig.update_layout(
        title=f"{spider_data.model_name} – {metric_display}",
        polar=dict(
            radialaxis=dict(range=[radial_min, radial_max], visible=True),
            angularaxis=dict(
                tickmode="array",
                tickvals=cat_angles_deg,
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
