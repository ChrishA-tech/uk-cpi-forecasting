**Data:** 1997-06 to 2025-06 (337 months). Sources: cpi_yoy: ONS D7G7 (CSV); bank_rate: Bank of England IUMABEDR; brent_usd: IMF PCPS POILBRE via DBnomics (same series as FRED POILBREUSDM).
**Backtest:** 60 rolling origins, forecasts evaluated from 2020-07 (includes the 2021-23 inflation surge), run 2026-11-14.

Out-of-sample RMSE (percentage points of CPI inflation); in brackets: relative to the naive no-change forecast (<1 = better). ✓ = significantly better than naive (Diebold-Mariano p<0.05), ✗ = significantly worse.

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Naive (no change) | 0.65 | 1.37 | 2.54 | 4.83 |
| AR (CPI lags) | 0.65 (0.99) | 1.25 (0.91) | 2.33 (0.92) | 4.33 (0.90) |
| ARX (+ Bank Rate, Brent) | 0.62 (0.95) | 1.21 (0.88) | 2.28 (0.90) | 4.19 (0.87) |
| Hybrid RL (original, h=1) | 0.66 (1.01) | – | – | – |

Hybrid RL agent: linear policy-gradient (not PPO); MPC sentiment and output gap set to 0 (no historical archive).

**Latest forecasts (CPI annual rate, %):**

| Target month | Naive | AR | ARX |
|---|---|---|---|
| 2025-07 (h=1) | 3.6 | 3.6 | 3.48 |
| 2025-09 (h=3) | 3.6 | 3.56 | 3.32 |
| 2025-12 (h=6) | 3.6 | 3.46 | 3.05 |
| 2026-06 (h=12) | 3.6 | 3.09 | 2.37 |

**Oil scenario (ARX, associational):** a sustained 20% rise in Brent is associated with CPI inflation +0.04pp at h=1 (95% CI -0.00 to +0.09), +0.09pp at h=3 (95% CI -0.00 to +0.18), +0.20pp at h=6 (95% CI +0.07 to +0.33), +0.35pp at h=12 (95% CI +0.09 to +0.61).
