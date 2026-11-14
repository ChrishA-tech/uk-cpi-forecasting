**Data:** 1997-06 to 2026-09 (352 months). Sources: cpi_yoy: ONS D7G7 (CSV); bank_rate: Bank of England IUMABEDR; brent_usd: US EIA Europe Brent Spot FOB (RBRTE).
**Backtest:** 60 rolling origins, forecasts evaluated from 2021-10 (includes the 2021-23 inflation surge), run 2026-11-14.

Out-of-sample RMSE (percentage points of CPI inflation); in brackets: relative to the naive no-change forecast (<1 = better). ✓ = significantly better than naive (Diebold-Mariano p<0.05), ✗ = significantly worse.

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Naive (no change) | 0.61 | 1.28 | 2.31 | 3.84 |
| AR (CPI lags) | 0.60 (0.98) | 1.16 (0.91) | 2.12 (0.92) | 3.29 (0.86) |
| ARX (+ Bank Rate, Brent) | 0.59 (0.96) | 1.14 (0.89) | 2.10 (0.91) | 3.22 (0.84) |
| Hybrid RL (original, h=1) | 0.62 (1.02) | – | – | – |

Hybrid RL agent: linear policy-gradient (not PPO); MPC sentiment and output gap set to 0 (no historical archive).

**Latest forecasts (CPI annual rate, %):**

| Target month | Naive | AR | ARX |
|---|---|---|---|
| 2026-10 (h=1) | 3.5 | 3.58 | 3.6 |
| 2026-12 (h=3) | 3.5 | 3.74 | 3.79 |
| 2027-03 (h=6) | 3.5 | 3.88 | 4.01 |
| 2027-09 (h=12) | 3.5 | 3.61 | 3.76 |

**Oil scenario (ARX, associational):** a sustained 20% rise in Brent is associated with CPI inflation +0.03pp at h=1 (95% CI -0.01 to +0.07), +0.07pp at h=3 (95% CI -0.01 to +0.15), +0.17pp at h=6 (95% CI +0.05 to +0.29), +0.29pp at h=12 (95% CI +0.05 to +0.53).
