import numpy as np
from typing import Dict, Any
from .rl_env import UKCPIForecastEnv, generate_uk_cpi_macro_dataset

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class LinearPolicyGradientAgent:
    """
    Lightweight linear policy (tanh(W obs + b)) trained by gradient steps on the forecast
    error, with the Stable-Baselines3 `.learn()` / `.predict()` interface. Used when
    stable-baselines3 is not installed. NOTE: this is NOT PPO (no clipping, no value
    function); results from it must be labelled accordingly.
    """
    kind = "linear policy-gradient (not PPO)"
    def __init__(self, env: UKCPIForecastEnv, learning_rate: float = 3e-3, seed: int = 42):
        self.env = env
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((2, 5), dtype=np.float32)
        self.bias = np.zeros(2, dtype=np.float32)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        raw = np.tanh(self.weights @ obs + self.bias)
        if not deterministic:
            raw = np.clip(raw + self.rng.normal(0, 0.1, size=2), -1.0, 1.0)
        return raw.astype(np.float32), None

    def learn(self, total_timesteps: int = 1024):
        obs, _ = self.env.reset()
        for _ in range(total_timesteps):
            action, _ = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, _, info = self.env.step(action)
            err = info["y_true"] - info["y_pred"]
            grad_a = np.array([0.50 * err, 0.35 * obs[4] * err], dtype=np.float32)
            self.weights += self.lr * np.outer(grad_a, obs / (np.linalg.norm(obs) + 1e-5))
            self.bias += self.lr * grad_a
            obs = next_obs if not terminated else self.env.reset()[0]
        return self


def make_agent(env: UKCPIForecastEnv, seed: int = 42):
    """Stable-Baselines3 PPO if installed, else the lightweight linear agent."""
    if HAS_SB3:
        agent = PPO(policy="MlpPolicy", env=env, learning_rate=3e-4, n_steps=64, batch_size=32,
                    n_epochs=5, seed=seed, verbose=0)
        agent.kind = "PPO (stable-baselines3)"
        return agent
    return LinearPolicyGradientAgent(env=env, seed=seed)


def agent_kind() -> str:
    return "PPO (stable-baselines3)" if HAS_SB3 else LinearPolicyGradientAgent.kind


# Kept for backward compatibility (original synthetic-data experiment)
def train_and_evaluate_hybrid_ppo(total_timesteps: int = 1024, seed: int = 42) -> Dict[str, Any]:
    """
    Trains a Stable-Baselines3 PPO agent on the UK CPI + FinBERT environment
    and evaluates out-of-sample RMSE and MAE against a naive Random Walk baseline.
    """
    df = generate_uk_cpi_macro_dataset(n_months=120, seed=seed)
    split_idx = 90
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx - 2:].copy()

    train_env = UKCPIForecastEnv(train_df)
    if HAS_SB3:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=5,
            seed=seed,
            verbose=0,
        )
    else:
        model = LinearPolicyGradientAgent(env=train_env, seed=seed)

    model.learn(total_timesteps=total_timesteps)

    test_env = UKCPIForecastEnv(test_df)
    obs, _ = test_env.reset()
    y_trues, y_preds, naive_preds = [], [], []

    terminated = False
    while not terminated:
        naive_preds.append(float(obs[0]))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = test_env.step(action)
        y_trues.append(info["y_true"])
        y_preds.append(info["y_pred"])

    y_trues_arr = np.array(y_trues)
    y_preds_arr = np.array(y_preds)
    naive_arr = np.array(naive_preds)

    hybrid_rmse = float(np.sqrt(np.mean((y_trues_arr - y_preds_arr) ** 2)))
    hybrid_mae = float(np.mean(np.abs(y_trues_arr - y_preds_arr)))
    naive_rmse = float(np.sqrt(np.mean((y_trues_arr - naive_arr) ** 2)))

    return {
        "model": model,
        "hybrid_rmse": round(hybrid_rmse, 4),
        "hybrid_mae": round(hybrid_mae, 4),
        "naive_rmse": round(naive_rmse, 4),
        "test_predictions": y_preds,
        "test_actuals": y_trues,
    }
