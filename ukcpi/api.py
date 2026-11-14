"""
FastAPI service for UK CPI forecasts from the fitted models (real data cached by `python -m ukcpi.run`).

POST /api/v1/forecast-cpi
  {"horizons": [1, 3, 6, 12], "oil_change_pct": 20, "mpc_statement_text": "..."}
-> latest Naive / AR / ARX forecasts, backtest accuracy, optional oil scenario,
   optional MPC sentiment (context only: it is not used in the forecast, because it is
   not validated on historical data).

The original endpoint returned a hard-coded formula described as a 'PPO policy'; it was
removed in the Nov 2026 review.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .data import DATA_DIR, load_macro
from .exceptions import CPIForecastValidationError
from .finbert_sentiment import FinBERTSentimentAnalyzer
from .forecast import forecast_now, oil_scenario

app = FastAPI(title="UK CPI Forecasting API", version="2.0.0",
              description="Naive / AR / ARX forecasts of UK CPI inflation fitted on ONS, Bank of England and FRED data.")
analyzer = FinBERTSentimentAnalyzer(use_pretrained_hf=False)
RESULTS = Path(__file__).resolve().parent.parent / "results" / "backtest.json"


class CPIForecastRequest(BaseModel):
    horizons: List[int] = Field(default=[1, 3, 6, 12], min_length=1, max_length=6)
    oil_change_pct: Optional[float] = Field(default=None, gt=-90, lt=500)
    mpc_statement_text: Optional[str] = Field(default=None, min_length=10, max_length=5000)

    @field_validator("horizons")
    @classmethod
    def valid_horizons(cls, v):
        if any(h < 1 or h > 24 for h in v):
            raise ValueError("horizons must be between 1 and 24 months")
        return sorted(set(v))


@app.get("/health")
def health():
    return {"status": "healthy", "service": "uk-cpi-forecasting"}


@app.post("/api/v1/forecast-cpi")
def forecast_cpi(req: CPIForecastRequest):
    if not (DATA_DIR / "uk_macro_monthly.csv").exists():
        raise HTTPException(503, "No cached data: run `python -m ukcpi.run` first.")
    try:
        df = load_macro()
    except CPIForecastValidationError as exc:
        raise HTTPException(503, f"Data unavailable or invalid: {exc}") from exc
    try:
        out = {"data_end": df["date"].iloc[-1].strftime("%Y-%m"), "forecasts": forecast_now(df, req.horizons)}
        if req.oil_change_pct is not None:
            out["oil_scenario"] = {"oil_change_pct": req.oil_change_pct,
                                   "cpi_change_pp_by_horizon": oil_scenario(df, req.oil_change_pct, req.horizons),
                                   "note": "Associational estimate from the ARX regression, not a causal effect."}
        if req.mpc_statement_text:
            out["mpc_sentiment"] = {**analyzer.score_text(req.mpc_statement_text).model_dump(),
                                    "note": "Context only; not used in the forecast."}
    except CPIForecastValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    if RESULTS.exists():
        bt = json.loads(RESULTS.read_text())["backtest"]
        out["backtest_rmse"] = {m: {h: r["rmse"] for h, r in rows.items()} for m, rows in bt["models"].items()}
    return out
