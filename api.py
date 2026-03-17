"""
Stream Analytics REST API
===========================
FastAPI application serving real-time viewership forecasts
and anomaly scores via HTTP endpoints.

Endpoints:
  POST /forecast      — predict peak viewers 30 min ahead
  POST /anomaly       — classify a stream event as viral/bot/normal
  GET  /health        — service health check
  GET  /model/info    — model metadata

Run locally:
  pip install fastapi uvicorn
  uvicorn api:app --reload --port 8000

Author: Ali Taghipourfard
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os

# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────

class StreamSnapshot(BaseModel):
    """Single stream state snapshot (one minute of data)."""
    stream_id: str = Field(..., example="stream_0042")
    game: str = Field(..., example="Valorant")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    viewer_count: int = Field(..., ge=0, example=8500)
    chat_messages_per_min: float = Field(..., ge=0, example=120.5)
    clip_creations: int = Field(default=0, ge=0)
    subscriber_count: int = Field(default=0, ge=0)
    bits_cheered: int = Field(default=0, ge=0)
    follower_growth_rate: float = Field(default=0.0, ge=0)

    # Pre-computed features (optional — computed server-side if omitted)
    viewer_zscore: Optional[float] = None
    viewer_growth_rate_short: Optional[float] = None
    engagement_score: Optional[float] = None
    momentum_crossover: Optional[float] = None


class ForecastResponse(BaseModel):
    stream_id: str
    timestamp: datetime
    current_viewers: int
    forecast_viewers_30min: int
    forecast_lower_bound: int
    forecast_upper_bound: int
    confidence: float
    model_version: str
    inference_ms: float


class AnomalyResponse(BaseModel):
    stream_id: str
    timestamp: datetime
    anomaly_type: Literal["normal", "viral_moment", "bot_raid", "anomaly_unknown"]
    confidence: float
    viewer_zscore: float
    alert: bool
    alert_message: Optional[str]
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool
    version: str


# ─────────────────────────────────────────────────────────────
# MOCK MODEL (replace with trained EnsembleForecaster in prod)
# ─────────────────────────────────────────────────────────────

class MockForecastModel:
    """
    Lightweight stand-in for the full EnsembleForecaster.
    In production, load the serialized model via joblib or mlflow.
    """
    VERSION = "1.0.0-mock"

    def predict(self, snapshot: StreamSnapshot) -> dict:
        """Heuristic forecast based on current viewers and momentum."""
        v = snapshot.viewer_count
        hour = snapshot.timestamp.hour

        # Prime-time multiplier
        prime_time_mult = 1.15 if 18 <= hour <= 22 else (0.85 if hour < 10 else 1.0)

        # Growth momentum signal
        growth = snapshot.viewer_growth_rate_short or 0.0
        momentum_boost = 1 + max(-0.2, min(0.4, growth * 0.3))

        forecast = int(v * prime_time_mult * momentum_boost)
        std_est = int(v * 0.12)  # ~12% uncertainty

        return {
            "forecast": forecast,
            "lower": max(0, forecast - 2 * std_est),
            "upper": forecast + 2 * std_est,
            "confidence": 0.82,
        }


class MockAnomalyModel:
    """
    Lightweight stand-in for the full StreamAnomalyDetector.
    """

    def classify(self, snapshot: StreamSnapshot) -> dict:
        zscore = snapshot.viewer_zscore or 0.0
        growth = snapshot.viewer_growth_rate_short or 0.0
        engagement = snapshot.engagement_score or 0.5
        chat_per_viewer = (snapshot.chat_messages_per_min / max(1, snapshot.viewer_count))

        # Bot raid: high growth, low engagement, low chat
        if growth > 2.5 and engagement < 0.25 and chat_per_viewer < 0.005:
            return {
                "anomaly_type": "bot_raid",
                "confidence": min(0.95, 0.6 + abs(growth) * 0.1),
                "viewer_zscore": zscore,
                "alert": True,
                "alert_message": f"Bot raid suspected: {int(growth*100)}% viewer spike with no engagement increase.",
            }

        # Viral: high zscore, high growth, high chat
        if abs(zscore) > 2.5 and growth > 0.5 and chat_per_viewer > 0.05:
            return {
                "anomaly_type": "viral_moment",
                "confidence": min(0.93, 0.55 + abs(zscore) * 0.05),
                "viewer_zscore": zscore,
                "alert": True,
                "alert_message": f"Viral moment detected: viewer zscore={zscore:.1f}, growth={growth*100:.0f}%.",
            }

        return {
            "anomaly_type": "normal",
            "confidence": 0.90,
            "viewer_zscore": zscore,
            "alert": False,
            "alert_message": None,
        }


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stream Analytics API",
    description="Real-time viewership forecasting and anomaly detection for live streaming platforms.",
    version="1.0.0",
)

_start_time = time.time()
_forecast_model = MockForecastModel()
_anomaly_model = MockAnomalyModel()


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health_check():
    """Service liveness check."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 2),
        model_loaded=True,
        version="1.0.0",
    )


@app.get("/model/info", tags=["meta"])
def model_info():
    """Return model metadata and feature schema."""
    return {
        "forecast_model": MockForecastModel.VERSION,
        "forecast_horizon_minutes": 30,
        "anomaly_detector": "IsolationForest + Rule-Based Ensemble",
        "anomaly_classes": ["normal", "viral_moment", "bot_raid", "anomaly_unknown"],
        "required_fields": ["stream_id", "game", "viewer_count", "chat_messages_per_min"],
        "optional_fields": ["viewer_zscore", "viewer_growth_rate_short", "engagement_score"],
    }


@app.post("/forecast", response_model=ForecastResponse, tags=["inference"])
def forecast_viewership(snapshot: StreamSnapshot):
    """
    Predict peak viewer count 30 minutes ahead for a live stream.

    - Provide current stream metrics in the request body.
    - Pre-computed features (viewer_zscore, engagement_score) improve accuracy
      but are computed heuristically if omitted.
    """
    t0 = time.perf_counter()

    try:
        result = _forecast_model.predict(snapshot)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

    return ForecastResponse(
        stream_id=snapshot.stream_id,
        timestamp=snapshot.timestamp,
        current_viewers=snapshot.viewer_count,
        forecast_viewers_30min=result["forecast"],
        forecast_lower_bound=result["lower"],
        forecast_upper_bound=result["upper"],
        confidence=result["confidence"],
        model_version=MockForecastModel.VERSION,
        inference_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/anomaly", response_model=AnomalyResponse, tags=["inference"])
def detect_anomaly(snapshot: StreamSnapshot):
    """
    Classify a stream event as normal, viral moment, or bot raid.

    - Returns anomaly type, confidence score, and alert details.
    - Set alert=True responses should trigger downstream moderation workflows.
    """
    t0 = time.perf_counter()

    try:
        result = _anomaly_model.classify(snapshot)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

    return AnomalyResponse(
        stream_id=snapshot.stream_id,
        timestamp=snapshot.timestamp,
        inference_ms=round((time.perf_counter() - t0) * 1000, 2),
        **result,
    )


# ─────────────────────────────────────────────────────────────
# EXAMPLE USAGE (printed when run directly)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Stream Analytics API")
    print("=" * 40)
    print("Start server:  uvicorn api:app --reload --port 8000")
    print()
    print("Example forecast request:")
    print("""
  curl -X POST http://localhost:8000/forecast \\
    -H "Content-Type: application/json" \\
    -d '{
      "stream_id": "stream_0042",
      "game": "Valorant",
      "viewer_count": 8500,
      "chat_messages_per_min": 120.5,
      "viewer_zscore": 1.8,
      "viewer_growth_rate_short": 0.12,
      "engagement_score": 0.65
    }'
    """)

    print("Example anomaly detection request:")
    print("""
  curl -X POST http://localhost:8000/anomaly \\
    -H "Content-Type: application/json" \\
    -d '{
      "stream_id": "stream_0099",
      "game": "Just Chatting",
      "viewer_count": 45000,
      "chat_messages_per_min": 15.0,
      "viewer_zscore": 5.2,
      "viewer_growth_rate_short": 4.8,
      "engagement_score": 0.08
    }'
    """)
