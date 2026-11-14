"""
UK CPI inflation forecasting on real data: direct h-step models, rolling-origin
backtest and an oil-price scenario tool.

Models (target: CPI annual rate h months ahead, y[t+h]):
  Naive       y_hat = y[t]                                   (no-change benchmark)
  AR          OLS on y[t], y[t-1], y[t-2]
  ARX         AR + Bank Rate[t] + Brent 12m log-change[t] + Brent 3m log-change[t]
  Hybrid RL   the original project's policy-adjusted AR forecaster (h=1 only),
              re-trained at every origin; MPC sentiment and output gap are not
              available historically, so they enter as 0 (documented limitation)

Direct forecasting: a separate regression per horizon h, trained only on pairs
(features at s, target at s+h) with s+h <= t, i.e. information available at origin t.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .exceptions import CPIForecastValidationError

AR_FEATURES = ["cpi", "cpi_l1", "cpi_l2"]
ARX_FEATURES = AR_FEATURES + ["bank_rate", "brent_yoy", "brent_3m"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dated t, using only data up to and including t."""
    f = pd.DataFrame({"date": df["date"].values})
    f["cpi"] = df["cpi_yoy"].values
    f["cpi_l1"] = f["cpi"].shift(1)
    f["cpi_l2"] = f["cpi"].shift(2)
    f["bank_rate"] = df["bank_rate"].values
    logb = np.log(df["brent_usd"].values)
    f["brent_yoy"] = 100 * (logb - pd.Series(logb).shift(12).values)
    f["brent_3m"] = 100 * (logb - pd.Series(logb).shift(3).values)
    return f


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def _direct_forecast(f: pd.DataFrame, cols: List[str], origin: int, h: int, min_train: int = 60) -> Optional[float]:
    y = f["cpi"].values
    X = f[cols].values
    s_idx = np.arange(0, origin - h + 1)                      # target s+h <= origin is known
    s_idx = s_idx[~np.isnan(X[s_idx]).any(axis=1)]
    if len(s_idx) < min_train or np.isnan(X[origin]).any():
        return None
    beta = _ols(X[s_idx], y[s_idx + h])
    return float(beta[0] + X[origin] @ beta[1:])


# ------------------------------------------------------------------ hybrid RL (original model)
def _rl_forecast(df: pd.DataFrame, origin: int, timesteps: int, seed: int) -> Optional[float]:
    from .rl_env import UKCPIForecastEnv
    from .trainer import make_agent
    hist = pd.DataFrame({"cpi_yoy": df["cpi_yoy"].values[:origin + 1],
                         "boe_base_rate": df["bank_rate"].values[:origin + 1],
                         "output_gap": 0.0, "finbert_sentiment": 0.0})
    if len(hist) < 24:
        return None
    env = UKCPIForecastEnv(hist)
    agent = make_agent(env, seed=seed)
    agent.learn(total_timesteps=timesteps)
    obs = np.array([hist["cpi_yoy"].iloc[-1], hist["cpi_yoy"].iloc[-2], hist["boe_base_rate"].iloc[-1], 0.0, 0.0],
                   dtype=np.float32)
    action, _ = agent.predict(obs, deterministic=True)
    return UKCPIForecastEnv.predict_from_obs(obs, action)


# ------------------------------------------------------------------ evaluation
def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> Dict[str, float]:
    d = e_model ** 2 - e_bench ** 2
    n = len(d)
    if n < 8:
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    dm_ = d.mean()
    gamma = [np.mean((d[k:] - dm_) * (d[:n - k] - dm_)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    stat = float(dm_ / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
    return {"dm_stat": round(stat, 3), "p_value": round(float(2 * stats.t.sf(abs(stat), df=n - 1)), 4)}


def rolling_backtest(df: pd.DataFrame, horizons: Sequence[int] = (1, 3, 6, 12), n_origins: int = 60,
                     include_rl: bool = True, rl_timesteps: int = 1024, seed: int = 42) -> Dict:
    f = build_features(df)
    n = len(f)
    first = n - n_origins - 1
    if first < 72:
        raise CPIForecastValidationError(f"Need at least {n_origins + 73} months for this backtest, got {n}.")
    y = f["cpi"].values
    models = {"Naive (no change)": None, "AR (CPI lags)": AR_FEATURES, "ARX (+ Bank Rate, Brent)": ARX_FEATURES}
    errs: Dict[str, Dict[int, List[float]]] = {m: {h: [] for h in horizons} for m in models}
    if include_rl:
        errs["Hybrid RL (original, h=1)"] = {1: []}
    naive_for_rl: List[float] = []

    for origin in range(first, n - 1):
        for h in horizons:
            if origin + h >= n:
                continue
            actual = y[origin + h]
            preds = {"Naive (no change)": y[origin]}
            for m, cols in models.items():
                if cols is not None:
                    p = _direct_forecast(f, cols, origin, h)
                    preds[m] = y[origin] if p is None else p
            for m, p in preds.items():
                errs[m][h].append(actual - p)
        if include_rl and 1 in horizons and origin + 1 < n:
            p = _rl_forecast(df, origin, rl_timesteps, seed)
            errs["Hybrid RL (original, h=1)"][1].append(y[origin + 1] - (y[origin] if p is None else p))

    out = {"n_obs": n, "n_origins": n_origins, "horizons": list(horizons),
           "sample": [df["date"].iloc[0].strftime("%Y-%m"), df["date"].iloc[-1].strftime("%Y-%m")],
           "evaluated_from": df["date"].iloc[first + 1].strftime("%Y-%m"), "models": {}}
    naive = "Naive (no change)"
    for m, by_h in errs.items():
        rows = {}
        for h, e in by_h.items():
            e = np.array(e)
            en = np.array(errs[naive][h])
            rmse = float(np.sqrt(np.mean(e ** 2)))
            row = {"n": len(e), "rmse": round(rmse, 3), "mae": round(float(np.mean(np.abs(e))), 3)}
            if m != naive:
                row["rel_rmse_vs_naive"] = round(rmse / float(np.sqrt(np.mean(en ** 2))), 3)
                row.update(diebold_mariano(e, en, h))
            rows[str(h)] = row
        out["models"][m] = rows
    return out


# ------------------------------------------------------------------ fitted model + scenarios
def fit_direct(df: pd.DataFrame, h: int, cols: Sequence[str] = ARX_FEATURES):
    """Full-sample direct regression with HAC (Newey-West) standard errors."""
    f = build_features(df)
    X = f[list(cols)].iloc[:len(f) - h]
    yy = f["cpi"].shift(-h).iloc[:len(f) - h]
    ok = X.notna().all(axis=1) & yy.notna()
    return sm.OLS(yy[ok], sm.add_constant(X[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 1)})


def forecast_now(df: pd.DataFrame, horizons: Sequence[int] = (1, 3, 6, 12)) -> Dict[str, Dict]:
    """Latest-origin forecasts from each model (uses all data)."""
    f = build_features(df)
    last = f.iloc[[-1]]
    out = {}
    for h in horizons:
        row = {"Naive (no change)": float(f["cpi"].iloc[-1])}
        for name, cols in (("AR (CPI lags)", AR_FEATURES), ("ARX (+ Bank Rate, Brent)", ARX_FEATURES)):
            res = fit_direct(df, h, cols)
            row[name] = float(np.asarray(res.predict(sm.add_constant(last[cols], has_constant="add")))[0])
        target = (df["date"].iloc[-1] + pd.DateOffset(months=h)).strftime("%Y-%m")
        out[str(h)] = {"target_month": target, **{k: round(v, 2) for k, v in row.items()}}
    return out


def oil_scenario(df: pd.DataFrame, oil_change_pct: float, horizons: Sequence[int] = (1, 3, 6, 12)) -> Dict[str, Dict]:
    """
    Estimated change in UK CPI inflation (percentage points) h months ahead if Brent
    moves by `oil_change_pct` % now and stays there, from the ARX coefficients
    (both the 12-month and 3-month Brent change rise by 100*log(1+x)).
    Associational estimate from a reduced-form regression, not a causal model.
    """
    if not -90 < oil_change_pct < 500:
        raise CPIForecastValidationError("oil_change_pct must be between -90 and 500.")
    shock = 100 * np.log(1 + oil_change_pct / 100)
    out = {}
    for h in horizons:
        res = fit_direct(df, h)
        b = res.params[["brent_yoy", "brent_3m"]].values
        cov = res.cov_params().loc[["brent_yoy", "brent_3m"], ["brent_yoy", "brent_3m"]].values
        effect = float(shock * b.sum())
        se = float(abs(shock) * np.sqrt(np.ones(2) @ cov @ np.ones(2)))
        out[str(h)] = {"cpi_change_pp": round(effect, 3), "ci95": [round(effect - 1.96 * se, 3), round(effect + 1.96 * se, 3)]}
    return out
