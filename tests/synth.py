"""Synthetic UK-like macro data for unit tests only (oil changes feed into CPI with a lag)."""
import numpy as np
import pandas as pd


def synthetic_macro(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    logb = np.log(60) + np.cumsum(rng.normal(0, 0.07, n))
    cpi = np.zeros(n)
    rate = np.zeros(n)
    cpi[:13], rate[:13] = 2.0, 3.0
    for t in range(13, n):
        oil12 = 100 * (logb[t - 1] - logb[t - 13])
        cpi[t] = 0.3 + 0.88 * cpi[t - 1] + 0.006 * oil12 - 0.03 * (rate[t - 1] - 3) + rng.normal(0, 0.18)
        rate[t] = np.clip(rate[t - 1] + 0.05 * (cpi[t - 1] - 2) + rng.normal(0, 0.05), 0.1, 8)
    return pd.DataFrame({"date": pd.date_range("1999-01-01", periods=n, freq="MS"),
                         "cpi_yoy": cpi.round(2), "bank_rate": rate.round(3), "brent_usd": np.exp(logb).round(2)})
