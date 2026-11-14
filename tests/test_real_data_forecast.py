"""Tests for the real-data pipeline (Nov 2026 upgrade)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import ukcpi.data as data
from ukcpi.exceptions import CPIForecastValidationError
from ukcpi.finbert_sentiment import FinBERTSentimentAnalyzer
from ukcpi.forecast import (ARX_FEATURES, _direct_forecast, build_features, fit_direct, forecast_now,
                            oil_scenario, rolling_backtest)
from synth import synthetic_macro

FX = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------------ parsers
def test_parse_ons_csv_keeps_only_monthly_rows():
    df = data.parse_ons_csv((FX / "ons_d7g7_sample.csv").read_text())
    assert list(df["value"]) == [4.9, 5.0, 5.0, 5.3]
    assert df["date"].iloc[0] == pd.Timestamp("1989-01-01")


def test_parse_ons_json():
    txt = json.dumps({"months": [{"date": "1989 JAN", "value": "4.9"}, {"date": "1989 FEB", "value": ""},
                                 {"date": "1989 MAR", "value": "5.0"}]})
    assert list(data.parse_ons_json(txt)["value"]) == [4.9, 5.0]


def test_parse_boe_csv_month_end_to_month_start():
    df = data.parse_boe_csv((FX / "boe_iumabedr_sample.csv").read_text())
    assert df["date"].iloc[1] == pd.Timestamp("1989-02-01") and df["value"].iloc[0] == pytest.approx(12.9479)


def test_parse_fred_csv_and_bad_columns():
    df = data.parse_fred_csv("observation_date,X\n2020-01-01,1.5\n2020-02-01,.\n", "X")
    assert len(df) == 1
    with pytest.raises(CPIForecastValidationError):
        data.parse_fred_csv("a,b\n1,2\n", "X")


def test_fetch_series_falls_back_and_reports_errors(monkeypatch):
    good = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=30, freq="MS"), "value": 1.0})

    def boom():
        raise OSError("blocked")
    monkeypatch.setitem(data.SOURCES, "cpi_yoy", [("primary", boom), ("fallback", lambda: good)])
    df, label = data.fetch_series("cpi_yoy")
    assert label == "fallback" and len(df) == 30
    monkeypatch.setitem(data.SOURCES, "cpi_yoy", [("primary", boom)])
    with pytest.raises(CPIForecastValidationError, match="primary: OSError: blocked"):
        data.fetch_series("cpi_yoy")


def test_load_macro_downloads_merges_and_caches(monkeypatch, tmp_path):
    synth = synthetic_macro(120)
    for col in ("cpi_yoy", "bank_rate", "brent_usd"):
        frame = synth[["date", col]].rename(columns={col: "value"})
        monkeypatch.setitem(data.SOURCES, col, [(f"test-{col}", lambda f=frame: f)])
    df = data.load_macro(refresh=True, start="1999-01-01", data_dir=tmp_path)
    assert len(df) == 120 and (tmp_path / "uk_macro_monthly.csv").exists()
    assert df.attrs["sources"]["cpi_yoy"] == "test-cpi_yoy"


def test_validate_macro_rejects_gaps_and_bad_values():
    df = synthetic_macro(60)
    gap = df.copy()
    gap.loc[30, "bank_rate"] = np.nan
    with pytest.raises(CPIForecastValidationError, match="Gaps"):
        data.validate_macro(gap)
    bad = df.copy()
    bad.loc[5, "cpi_yoy"] = 99
    with pytest.raises(CPIForecastValidationError, match="Implausible"):
        data.validate_macro(bad)


# ------------------------------------------------------------------ models / backtest
def test_features_and_direct_forecast_use_no_future_data():
    df = synthetic_macro(200)
    origin, h = 150, 6
    base = _direct_forecast(build_features(df), ARX_FEATURES, origin, h)
    future = df.copy()
    future.loc[origin + 1:, ["cpi_yoy", "bank_rate", "brent_usd"]] *= 3        # change everything after t
    assert _direct_forecast(build_features(future), ARX_FEATURES, origin, h) == pytest.approx(base)


def test_backtest_structure_and_naive_benchmark():
    r = rolling_backtest(synthetic_macro(220), n_origins=36, include_rl=False)
    assert set(r["models"]) == {"Naive (no change)", "AR (CPI lags)", "ARX (+ Bank Rate, Brent)"}
    assert r["models"]["ARX (+ Bank Rate, Brent)"]["1"]["n"] == 36
    assert "rel_rmse_vs_naive" not in r["models"]["Naive (no change)"]["1"]


def test_backtest_includes_hybrid_rl_at_h1():
    r = rolling_backtest(synthetic_macro(150), n_origins=12, horizons=(1, 3), rl_timesteps=64)
    assert r["models"]["Hybrid RL (original, h=1)"]["1"]["n"] == 12


def test_backtest_needs_enough_history():
    with pytest.raises(CPIForecastValidationError):
        rolling_backtest(synthetic_macro(100), n_origins=60)


def test_oil_scenario_sign_and_validation():
    df = synthetic_macro(300)                 # synthetic data where oil raises CPI
    up = oil_scenario(df, 20)
    assert up["12"]["cpi_change_pp"] > 0 and up["12"]["ci95"][0] < up["12"]["cpi_change_pp"] < up["12"]["ci95"][1]
    assert oil_scenario(df, -20)["12"]["cpi_change_pp"] < 0
    with pytest.raises(CPIForecastValidationError):
        oil_scenario(df, -95)


def test_forecast_now_and_hac_fit():
    df = synthetic_macro(200)
    fc = forecast_now(df, (1, 12))
    assert fc["1"]["target_month"] == (df["date"].iloc[-1] + pd.DateOffset(months=1)).strftime("%Y-%m")
    assert fit_direct(df, 12).cov_type == "HAC"


# ------------------------------------------------------------------ FinBERT + API
def test_finbert_reports_backend_and_handles_pipeline_shapes():
    a = FinBERTSentimentAnalyzer(use_pretrained_hf=False)
    assert a.score_text("Inflation is cooling with disinflation").backend.startswith("lexicon")
    for shape in ([{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.2}, {"label": "neutral", "score": 0.1}],
                  [[{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.2}, {"label": "neutral", "score": 0.1}]]):
        a._pipeline = lambda text, s=shape: s
        out = a.score_text("Persistent inflation and upward pressure")
        assert out.inflation_sentiment_index == pytest.approx(0.5) and out.backend == "ProsusAI/finbert"


def test_api_forecast_with_scenario_and_sentiment(monkeypatch, tmp_path):
    import ukcpi.api as api
    (tmp_path / "uk_macro_monthly.csv").write_text("x")
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "load_macro", lambda: synthetic_macro(220))
    c = TestClient(api.app)
    r = c.post("/api/v1/forecast-cpi", json={"horizons": [1, 12], "oil_change_pct": 10,
                                             "mpc_statement_text": "persistent inflation and upward pressure"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["forecasts"]) == {"1", "12"} and "oil_scenario" in body
    assert body["mpc_sentiment"]["note"].startswith("Context only")
    assert c.post("/api/v1/forecast-cpi", json={"horizons": [0]}).status_code == 422


def test_api_without_data_returns_503(monkeypatch, tmp_path):
    import ukcpi.api as api
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    assert TestClient(api.app).post("/api/v1/forecast-cpi", json={}).status_code == 503


def test_run_end_to_end_writes_results(monkeypatch, tmp_path):
    import ukcpi.run as run
    df = synthetic_macro(220)
    df.attrs["sources"] = {"cpi_yoy": "synthetic"}
    monkeypatch.setattr(run, "load_macro", lambda refresh=False: df)
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    (tmp_path / "README.md").write_text("x <!-- RESULTS:START -->\n<!-- RESULTS:END --> y")
    res = run.main(["--origins", "24", "--no-rl"])
    md = (tmp_path / "README.md").read_text()
    assert "Latest forecasts" in md and (tmp_path / "output" / "dashboard.html").exists()
    assert set(res["oil_scenario_+20pct"]) == {"1", "3", "6", "12"}


def test_manual_files_are_used_first(monkeypatch, tmp_path):
    (tmp_path / "IUMABEDR.csv").write_text((FX / "boe_iumabedr_sample.csv").read_text())
    monkeypatch.setattr(data, "MANUAL_DIR", tmp_path)
    df = data._manual("bank_rate")
    assert len(df) == 3
    with pytest.raises(FileNotFoundError):
        data._manual("cpi_yoy")


def test_download_failure_lists_manual_instructions(monkeypatch):
    def boom():
        raise OSError("dns")
    monkeypatch.setitem(data.SOURCES, "bank_rate", [("BoE", boom)])
    with pytest.raises(CPIForecastValidationError, match="save it as"):
        data.fetch_series("bank_rate")


def test_parse_dbnomics_json():
    df = data.parse_dbnomics_json((FX / "dbnomics_brent_sample.json").read_text())
    assert list(df["value"]) == [21.25, 19.81, 18.39]


def test_brent_order_is_manual_then_dbnomics_then_fred():
    labels = [l for l, _ in data.SOURCES["brent_usd"]]
    assert labels[0].startswith("manual") and "DBnomics" in labels[1] and "FRED" in labels[2]
