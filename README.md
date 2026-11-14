# UK CPI Inflation Forecasting: AR / ARX vs Hybrid RL + FinBERT

Forecasts the UK CPI annual inflation rate 1, 3, 6 and 12 months ahead on **real data** (ONS CPI,
Bank of England Bank Rate, Brent crude), compares an autoregressive model, an **ARX** model with Bank Rate and
oil prices, and the project's original **hybrid reinforcement-learning** forecaster against a no-change
benchmark in a **rolling-origin backtest** with **Diebold-Mariano** tests, and estimates how oil-price moves
feed into UK inflation.

## Results (real data)

<!-- RESULTS:START -->
_Not run yet: `python -m ukcpi.run` downloads the data and fills in this section._
<!-- RESULTS:END -->

## Components

| Part | File | |
|---|---|---|
| Data | `ukcpi/data.py` | ONS D7G7 CPI annual rate, BoE IUMABEDR Bank Rate, FRED POILBREUSDM Brent; fallbacks (ONS API, FRED/OECD series); validation (gaps, implausible values); cached in `data/` with the source used for each series |
| Forecasting | `ukcpi/forecast.py` | Direct h-step models: naive, AR(CPI lags), ARX(+ Bank Rate, Brent 12m and 3m log-changes); rolling-origin backtest; Diebold-Mariano; HAC regression; **oil scenario** ("Brent +20% → CPI inflation +x pp") |
| Hybrid RL | `ukcpi/rl_env.py`, `ukcpi/trainer.py` | The original Gymnasium environment and agent (Stable-Baselines3 PPO if installed, else a lightweight linear policy-gradient agent), now re-trained at every backtest origin on real data |
| MPC sentiment | `ukcpi/finbert_sentiment.py` | ProsusAI/FinBERT when available, else a keyword lexicon; output states which backend was used |
| API | `ukcpi/api.py` | `POST /api/v1/forecast-cpi`: forecasts from the fitted models, optional oil scenario, optional MPC sentiment (context only) |
| Dashboard | `ukcpi/dashboard.py` | Plotly: CPI, Bank Rate, Brent; relative RMSE by model/horizon; latest forecasts |

## Run

```bash
pip install -r requirements.txt           # optional extras: pip install ".[rl,finbert]"
python -m ukcpi.run                       # real data -> results/backtest.md, output/dashboard.html, this README
python -m pytest -q
uvicorn ukcpi.api:app --reload
```

## Review notes (Nov 2026)

The original version claimed the hybrid RL + FinBERT model "demonstrates lower RMSE/MAE" than an econometric
benchmark. Re-running it showed the opposite: on its own synthetic data it was **worse than a naive no-change
forecast for every seed tested** (e.g. RMSE 0.287 vs 0.227), and more training made it worse. The tests only
checked `RMSE > 0`. Other issues and fixes:

- **All data were synthetic.** Results now use real ONS, Bank of England and FRED series.
- **The API's "PPO policy" was a hard-coded formula.** The API now serves the fitted AR/ARX models.
- **The fallback "PPO" agent was not PPO** (no clipping or value function); it is now named a linear
  policy-gradient agent, and results state which agent was used.
- **The "FinBERT" output could come from a keyword list** without saying so; outputs now report the backend.
- **MPC sentiment cannot be backtested** (no historical archive of statements is included), so it is not used in
  the reported forecasts; the RL model receives sentiment = 0 in the backtest.

The hybrid RL model is kept and evaluated as one of the candidates; the README reports whatever the backtest
shows for it.

Data: ONS (Open Government Licence v3.0), Bank of England Statistical Interactive Database, IMF via FRED.
