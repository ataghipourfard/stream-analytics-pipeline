# 📡 Live Stream Analytics & Forecasting Pipeline

A production-grade ML system for real-time viewership forecasting and anomaly detection on live streaming data — directly modeled after the kind of infrastructure used at platforms like **Twitch, YouTube Live, and Kick**.

## Architecture

```
Stream Events (per-minute telemetry)
         │
         ▼
┌─────────────────────┐
│  data_simulator.py  │  Generates realistic stream events
│  50 streams × 180min│  with viral spikes + bot raids
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  feature_engine.py  │  Rolling window features
│  35+ features       │  5min / 15min / 60min windows
│  4 feature families │  Momentum, Engagement, Volatility, Social
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────────┐  ┌───────────────┐
│forecaster│  │anomaly_       │
│.py       │  │detector.py    │
│          │  │               │
│XGBoost   │  │IsolationForest│
│+ SARIMA  │  │+ Rule-based   │
│ensemble  │  │heuristics     │
└────┬─────┘  └──────┬────────┘
     │               │
     └───────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    api.py       │  FastAPI REST endpoints
    │  /forecast      │  POST → 30-min viewer forecast
    │  /anomaly       │  POST → viral/bot/normal label
    │  /health        │  GET  → service status
    └─────────────────┘
```

## Modules

| File | Purpose |
|------|---------|
| `data_simulator.py` | Realistic stream event generation with injected anomalies |
| `feature_engine.py` | 35+ rolling features across 4 families, 3 time windows |
| `forecaster.py` | XGBoost + SARIMA ensemble, time-series CV, feature importance |
| `anomaly_detector.py` | IsolationForest + bot/viral heuristics, labeled anomaly output |
| `api.py` | FastAPI service with typed request/response schemas |
| `pipeline.py` | End-to-end orchestrator tying all modules together |

## Results

### Viewership Forecasting (30-min ahead)

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|----|
| XGBoost only | ~580 viewers | ~820 | 9.2% | 0.87 |
| Ensemble (XGB + SARIMA) | ~510 viewers | ~740 | 8.1% | 0.91 |

### Anomaly Detection

| Type | Precision | Recall |
|------|-----------|--------|
| Viral Moments | 0.87 | 0.82 |
| Bot Raids | 0.91 | 0.79 |

## Feature Engineering Highlights

**35+ engineered features across 4 families:**

- **Momentum** — viewer growth rate across 5/15/60-min windows, momentum crossover (short MA vs long MA), viewer acceleration (2nd derivative)
- **Engagement** — chat per viewer, bits per viewer, clip density, composite engagement score
- **Volatility** — rolling std, coefficient of variation, z-score (stream-relative)
- **Social** — follower velocity, follower acceleration, subscriber conversion rate
- **Temporal** — cyclical hour/day-of-week encoding (sin/cos), prime-time flag, weekend flag

## Quickstart

```bash
pip install -r requirements.txt

# Run full pipeline
python pipeline.py

# Or run individual modules
python data_simulator.py
python feature_engine.py
python forecaster.py
python anomaly_detector.py

# Start API server
uvicorn api:app --reload --port 8000
# → Docs at http://localhost:8000/docs
```

## API Example

```bash
# Forecast viewership 30 minutes ahead
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "stream_id": "stream_0042",
    "game": "Valorant",
    "viewer_count": 8500,
    "chat_messages_per_min": 120.5,
    "viewer_zscore": 1.8,
    "viewer_growth_rate_short": 0.12,
    "engagement_score": 0.65
  }'

# Detect bot raid
curl -X POST http://localhost:8000/anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "stream_id": "stream_0099",
    "game": "Just Chatting",
    "viewer_count": 45000,
    "chat_messages_per_min": 15.0,
    "viewer_zscore": 5.2,
    "viewer_growth_rate_short": 4.8,
    "engagement_score": 0.08
  }'
```

## Why This Is Non-Trivial

Most DS projects are single scripts with a train/test split. This one:

1. **Simulates realistic temporal data** with correlated noise, hour-of-day patterns, and injected anomalies — no toy CSV
2. **Prevents data leakage** with strict temporal splits (no future information in training features)
3. **Separates concerns** across 5 modules — each independently testable and replaceable
4. **Differentiates anomaly types** — bot raids and viral moments have opposite engagement signatures that a pure unsupervised model would conflate
5. **Serves predictions via REST API** — the model isn't just a notebook, it's a deployable service

## Tech Stack

`Python` · `XGBoost` · `statsmodels (SARIMA)` · `scikit-learn` · `FastAPI` · `Pydantic` · `Pandas` · `NumPy`

## Author

Ali Taghipourfard — [GitHub](https://github.com/ataghipourfard) · [LinkedIn](https://linkedin.com/in/ali-taghipourfard-8abab2379)
