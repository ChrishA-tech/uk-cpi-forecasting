import numpy as np
import pandas as pd
from .exceptions import CPIForecastValidationError

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _DummyEnv:
        pass
    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype
    class _GymNamespace:
        Env = _DummyEnv
    class _SpacesNamespace:
        Box = _Box
    gym = _GymNamespace()
    spaces = _SpacesNamespace()


def generate_uk_cpi_macro_dataset(n_months: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    SYNTHETIC data (demo / unit tests only - real results use ukcpi.data.load_macro).
    Generates a monthly UK-like macroeconomic dataset (CPI YoY %, BoE Base Rate %,
    Output Gap %, and FinBERT MPC Sentiment Index) calibrated to ONS & Bank of England regimes.
    """
    if n_months < 24:
        raise CPIForecastValidationError("Dataset length must be at least 24 months.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n_months, freq="MS")

    cpi = np.zeros(n_months, dtype=np.float32)
    boe_rate = np.zeros(n_months, dtype=np.float32)
    output_gap = np.zeros(n_months, dtype=np.float32)
    finbert_index = np.zeros(n_months, dtype=np.float32)

    cpi[0] = 2.0
    boe_rate[0] = 0.50
    for t in range(1, n_months):
        shock = 0.35 if 60 <= t <= 85 else (-0.15 if 86 <= t <= 105 else 0.0)
        finbert_index[t] = np.clip(0.55 * finbert_index[t - 1] + shock + rng.normal(0, 0.18), -1.0, 1.0)
        output_gap[t] = np.clip(0.7 * output_gap[t - 1] + rng.normal(0, 0.25), -4.0, 4.0)
        boe_rate[t] = np.clip(boe_rate[t - 1] + 0.12 * (cpi[t - 1] - 2.0) + rng.normal(0, 0.05), 0.1, 5.25)

        cpi[t] = np.clip(
            0.82 * cpi[t - 1]
            + 0.38 * finbert_index[t]
            + 0.09 * output_gap[t]
            - 0.06 * (boe_rate[t - 1] - 2.0)
            + 0.36
            + rng.normal(0, 0.15),
            -0.5,
            11.5,
        )

    return pd.DataFrame({
        "date": dates,
        "cpi_yoy": np.round(cpi, 3),
        "boe_base_rate": np.round(boe_rate, 3),
        "output_gap": np.round(output_gap, 3),
        "finbert_sentiment": np.round(finbert_index, 4),
    })


class UKCPIForecastEnv(gym.Env):
    """
    Gymnasium environment for Hybrid FinBERT + RL UK CPI Forecasting.
    - Observation (5-dim float32): [lag1_cpi, lag2_cpi, boe_base_rate, output_gap, finbert_sentiment]
    - Action (2-dim float32 in [-1, 1]):
        action[0]: adjustment to autoregressive persistence weight
        action[1]: weight assigned to FinBERT inflation sentiment signal
    - Reward: Negative squared error -((y_true - y_hat)^2)
    """
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        required_cols = {"cpi_yoy", "boe_base_rate", "output_gap", "finbert_sentiment"}
        if not required_cols.issubset(df.columns):
            raise CPIForecastValidationError(f"DataFrame missing required columns: {required_cols - set(df.columns)}")
        if len(df) < 10:
            raise CPIForecastValidationError("DataFrame must have at least 10 time steps.")

        self.df = df.reset_index(drop=True)
        self.observation_space = spaces.Box(low=-20.0, high=25.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.current_step = 2

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step - 1]
        prev_row = self.df.iloc[self.current_step - 2]
        return np.array([
            row["cpi_yoy"],
            prev_row["cpi_yoy"],
            row["boe_base_rate"],
            row["output_gap"],
            row["finbert_sentiment"],
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 2
        return self._get_obs(), {}

    @staticmethod
    def predict_from_obs(obs: np.ndarray, action: np.ndarray) -> float:
        """Forecast implied by an observation and a policy action (shared by step() and backtests)."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        lag1_cpi, lag2_cpi, boe_rate, output_gap, sentiment = [float(v) for v in obs]
        ar_momentum = 0.85 * lag1_cpi + 0.15 * lag2_cpi
        rl_drift_adj = 0.50 * float(action[0])
        sentiment_impact = (0.35 + 0.35 * float(action[1])) * sentiment
        macro_impact = 0.05 * output_gap - 0.04 * (boe_rate - 2.0)
        return float(ar_momentum + rl_drift_adj + sentiment_impact + macro_impact)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs = self._get_obs()
        y_pred = self.predict_from_obs(obs, action)
        y_true = float(self.df.iloc[self.current_step]["cpi_yoy"])

        error = y_true - y_pred
        reward = float(-(error ** 2))

        self.current_step += 1
        terminated = self.current_step >= len(self.df)
        truncated = False
        next_obs = self._get_obs() if not terminated else np.zeros(5, dtype=np.float32)

        info = {
            "y_true": y_true,
            "y_pred": round(y_pred, 4),
            "abs_error": round(abs(error), 4),
        }
        return next_obs, reward, terminated, truncated, info
