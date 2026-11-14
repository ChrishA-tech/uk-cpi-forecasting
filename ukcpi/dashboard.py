"""Plotly dashboard: CPI vs Bank Rate and Brent, backtest accuracy, latest forecasts."""
from __future__ import annotations

import os
from typing import Any, Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def export_dashboard(df, res: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bt = res["backtest"]
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.09, row_heights=[0.4, 0.3, 0.3],
                        specs=[[{"secondary_y": True}], [{}], [{}]],
                        subplot_titles=("UK CPI inflation and Bank Rate (%), Brent (USD/bbl, right axis)",
                                        "Out-of-sample RMSE relative to naive (below 1 = better)",
                                        "Latest forecasts of CPI inflation (%)"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["cpi_yoy"], name="CPI annual rate", line=dict(color="#dc2626")), 1, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["bank_rate"], name="Bank Rate", line=dict(color="#2563eb")), 1, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["brent_usd"], name="Brent", line=dict(color="#6b7280", dash="dot")),
                  1, 1, secondary_y=True)
    for m, rows in bt["models"].items():
        hs = [h for h in bt["horizons"] if str(h) in rows and "rel_rmse_vs_naive" in rows[str(h)]]
        if hs:
            fig.add_trace(go.Bar(x=[f"h={h}" for h in hs], y=[rows[str(h)]["rel_rmse_vs_naive"] for h in hs], name=m), 2, 1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#6b7280", row=2, col=1)
    fc = res["forecast_now"]
    for m in ["Naive (no change)", "AR (CPI lags)", "ARX (+ Bank Rate, Brent)"]:
        fig.add_trace(go.Scatter(x=[v["target_month"] for v in fc.values()], y=[v[m] for v in fc.values()],
                                 mode="lines+markers", name=f"Forecast: {m}"), 3, 1)
    fig.update_layout(title="UK CPI inflation forecasting: real data, rolling backtest", template="plotly_white",
                      height=1100, barmode="group")
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path
