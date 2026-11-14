# Project 2: Hybrid RL + FinBERT UK CPI Forecasting System

## Overview
Implements the hybrid macroeconomic inflation forecasting system combining:
1. **FinBERT Financial Sentiment Pipeline (`finbert_sentiment.py`)**: Scores monetary policy statements (e.g., Bank of England MPC transcripts) into continuous inflation pressure indices `[-1.0, +1.0]` and 3-class probabilities (`positive`, `negative`, `neutral`). Automatically uses HuggingFace `ProsusAI/finbert` when cached/available or falls back to a calibrated financial lexicon transformer-head mock for air-gapped CI testing.
2. **Custom Gymnasium RL Environment (`rl_env.py`)**: Models UK CPI YoY forecasting as a sequential continuous control problem (`Box` action space) where the agent dynamically blends autoregressive macro momentum, BoE bank rate adjustments, and FinBERT hawkish/dovish sentiment signals to minimize squared forecast error.
3. **Stable-Baselines3 PPO Pipeline (`trainer.py`)**: Trains and evaluates the Proximal Policy Optimization (`PPO`) agent against a baseline econometric benchmark, demonstrating lower RMSE/MAE.
4. **FastAPI Inference Microservice (`api.py`)**: REST API endpoint (`POST /api/v1/forecast-cpi`) with strict validation and error handling.
