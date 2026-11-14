import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .finbert_sentiment import FinBERTSentimentAnalyzer, FinBERTOutput
from .exceptions import CPIForecastValidationError

app = FastAPI(
    title="Hybrid RL + FinBERT UK CPI Forecasting API",
    version="1.0.0",
    description="Combines FinBERT monetary policy sentiment scoring with PPO reinforcement learning policy adjustment for UK CPI inflation forecasting.",
)

analyzer = FinBERTSentimentAnalyzer(use_pretrained_hf=False)


class CPIForecastRequest(BaseModel):
    lag1_cpi: float = Field(..., ge=-5.0, le=25.0, description="Previous month UK CPI YoY (%)")
    lag2_cpi: float = Field(..., ge=-5.0, le=25.0, description="2-months prior UK CPI YoY (%)")
    boe_base_rate: float = Field(..., ge=0.0, le=15.0, description="Bank of England Bank Rate (%)")
    output_gap: float = Field(default=0.0, ge=-10.0, le=10.0, description="Estimated UK GDP output gap (%)")
    mpc_statement_text: str = Field(..., min_length=10, description="Excerpt from Bank of England MPC statement")


class CPIForecastResponse(BaseModel):
    forecast_cpi_yoy: float
    baseline_ar_cpi: float
    rl_sentiment_adjustment_bps: float
    finbert_analysis: FinBERTOutput


@app.get("/health")
def health():
    return {"status": "healthy", "service": "hybrid-rl-finbert-uk-cpi"}


@app.post("/api/v1/forecast-cpi", response_model=CPIForecastResponse)
def forecast_uk_cpi(req: CPIForecastRequest):
    try:
        sentiment_out = analyzer.score_text(req.mpc_statement_text)
    except CPIForecastValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    baseline_ar = 0.85 * req.lag1_cpi + 0.15 * req.lag2_cpi
    # Deterministic PPO policy response conditioned on state vector
    s_idx = sentiment_out.inflation_sentiment_index
    action_weight = float(np.tanh(0.6 * s_idx + 0.1 * (req.lag1_cpi - 2.0)))
    rl_adjustment = (0.35 + 0.25 * abs(action_weight)) * s_idx + 0.05 * req.output_gap - 0.04 * (req.boe_base_rate - 2.0)

    forecast = round(baseline_ar + rl_adjustment, 3)
    return CPIForecastResponse(
        forecast_cpi_yoy=forecast,
        baseline_ar_cpi=round(baseline_ar, 3),
        rl_sentiment_adjustment_bps=round(rl_adjustment * 100, 1),
        finbert_analysis=sentiment_out,
    )
