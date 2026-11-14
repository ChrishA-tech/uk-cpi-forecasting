import pytest
from fastapi.testclient import TestClient
from src.finbert_sentiment import FinBERTSentimentAnalyzer
from src.rl_env import UKCPIForecastEnv, generate_uk_cpi_macro_dataset
from src.trainer import train_and_evaluate_hybrid_ppo
from src.exceptions import CPIForecastValidationError
from src.api import app

client = TestClient(app)


def test_finbert_sentiment_hawkish_vs_dovish_directionality():
    analyzer = FinBERTSentimentAnalyzer(use_pretrained_hf=False)
    hawkish_text = "MPC notes persistent inflation, elevated wage growth, and upward pressure from energy shock."
    dovish_text = "Inflation is cooling rapidly with disinflation and demand weakness below target."

    hawk_res = analyzer.score_text(hawkish_text)
    dove_res = analyzer.score_text(dovish_text)

    assert hawk_res.inflation_sentiment_index > 0.20
    assert dove_res.inflation_sentiment_index < -0.20
    assert hawk_res.positive_prob > hawk_res.negative_prob
    assert dove_res.negative_prob > dove_res.positive_prob


def test_finbert_rejects_short_invalid_input():
    analyzer = FinBERTSentimentAnalyzer(use_pretrained_hf=False)
    with pytest.raises(CPIForecastValidationError):
        analyzer.score_text("abc")


def test_gymnasium_env_step_and_ppo_training_loop():
    df = generate_uk_cpi_macro_dataset(n_months=48, seed=101)
    env = UKCPIForecastEnv(df)
    obs, _ = env.reset()
    assert obs.shape == (5,)

    next_obs, reward, terminated, truncated, info = env.step([0.1, 0.2])
    assert next_obs.shape == (5,)
    assert isinstance(reward, float) and reward <= 0.0
    assert terminated is False
    assert "y_pred" in info and "y_true" in info

    # End-to-end Stable-Baselines3 PPO training check
    metrics = train_and_evaluate_hybrid_ppo(total_timesteps=256, seed=42)
    assert metrics["hybrid_rmse"] > 0.0
    assert len(metrics["test_predictions"]) == len(metrics["test_actuals"])


def test_fastapi_cpi_forecast_endpoint():
    resp = client.post(
        "/api/v1/forecast-cpi",
        json={
            "lag1_cpi": 2.6,
            "lag2_cpi": 2.2,
            "boe_base_rate": 4.75,
            "output_gap": 0.4,
            "mpc_statement_text": "The Committee observed persistent inflation and elevated wage growth above target.",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["forecast_cpi_yoy"] > data["baseline_ar_cpi"]
    assert data["finbert_analysis"]["inflation_sentiment_index"] > 0
