import unittest
from ukcpi.finbert_sentiment import FinBERTSentimentAnalyzer
from ukcpi.rl_env import UKCPIForecastEnv, generate_uk_cpi_macro_dataset
from ukcpi.trainer import train_and_evaluate_hybrid_ppo
from ukcpi.exceptions import CPIForecastValidationError


class TestHybridRLFinBERT(unittest.TestCase):
    def test_finbert_hawkish_vs_dovish_scores(self):
        analyzer = FinBERTSentimentAnalyzer(use_pretrained_hf=False)
        hawk = analyzer.score_text("MPC notes persistent inflation, elevated wage growth, and upward pressure.")
        dove = analyzer.score_text("Inflation is cooling rapidly with disinflation and demand weakness below target.")
        self.assertGreater(hawk.inflation_sentiment_index, 0.20)
        self.assertLess(dove.inflation_sentiment_index, -0.20)

    def test_finbert_validation_error(self):
        analyzer = FinBERTSentimentAnalyzer(use_pretrained_hf=False)
        with self.assertRaises(CPIForecastValidationError):
            analyzer.score_text("bad")

    def test_rl_env_and_ppo_trainer(self):
        df = generate_uk_cpi_macro_dataset(n_months=48, seed=101)
        env = UKCPIForecastEnv(df)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (5,))
        next_obs, reward, terminated, _, info = env.step([0.1, 0.2])
        self.assertEqual(next_obs.shape, (5,))
        self.assertLessEqual(reward, 0.0)
        self.assertFalse(terminated)

        metrics = train_and_evaluate_hybrid_ppo(total_timesteps=256, seed=42)
        self.assertGreater(metrics["hybrid_rmse"], 0.0)
        self.assertEqual(len(metrics["test_predictions"]), len(metrics["test_actuals"]))


if __name__ == "__main__":
    unittest.main()
