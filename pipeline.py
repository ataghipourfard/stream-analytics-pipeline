"""
Stream Analytics Pipeline — Main Orchestrator
===============================================
End-to-end execution of the full pipeline:
  1. Simulate stream events (or load from CSV)
  2. Engineer rolling features
  3. Train viewership forecasting ensemble
  4. Detect anomalies (viral moments + bot raids)
  5. Print full evaluation report

Author: Ali Taghipourfard
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from data_simulator import StreamSimulator
from feature_engine import FeatureEngine, get_model_features
from forecaster import EnsembleForecaster, build_targets, evaluate_forecast
from anomaly_detector import StreamAnomalyDetector


def run_pipeline(
    n_streams: int = 50,
    stream_duration_min: int = 180,
    forecast_horizon: int = 30,
    verbose: bool = True,
):
    t_start = time.time()

    if verbose:
        print("=" * 65)
        print("  Twitch-Style Stream Analytics Pipeline")
        print("  XGBoost Ensemble + Isolation Forest Anomaly Detection")
        print("=" * 65)

    # ── Stage 1: Data Simulation ──────────────────────────────
    if verbose:
        print(f"\n[1/4] Simulating {n_streams} streams × {stream_duration_min} minutes...")

    sim = StreamSimulator(n_streams=n_streams, stream_duration_min=stream_duration_min, seed=42)
    raw_df = sim.to_dataframe()
    n_viral = sum(1 for p in sim.profiles if p.has_viral_event)
    n_bots = sum(1 for p in sim.profiles if p.has_bot_raid)

    if verbose:
        print(f"  Events: {len(raw_df):,} | Streams: {raw_df['stream_id'].nunique()}")
        print(f"  Injected viral events: {n_viral} | Bot raids: {n_bots}")
        print(f"  Games: {sorted(raw_df['game'].unique())}")

    # ── Stage 2: Feature Engineering ─────────────────────────
    if verbose:
        print(f"\n[2/4] Engineering rolling features...")

    engine = FeatureEngine()
    feat_df = engine.transform(raw_df)
    feature_cols = get_model_features(feat_df)

    if verbose:
        print(f"  Raw columns: {len(raw_df.columns)} → Feature columns: {len(feature_cols)}")
        families = {
            "Momentum": [c for c in feature_cols if "growth" in c or "momentum" in c or "acceleration" in c],
            "Engagement": [c for c in feature_cols if "chat" in c or "clip" in c or "bits" in c or "engagement" in c],
            "Volatility": [c for c in feature_cols if "volatility" in c or "cv_" in c or "zscore" in c],
            "Social": [c for c in feature_cols if "follower" in c or "sub_" in c],
            "Temporal": [c for c in feature_cols if "hour" in c or "dow" in c or "prime" in c or "weekend" in c],
        }
        for name, cols in families.items():
            if cols:
                print(f"  {name:<12}: {len(cols):>2} features")

    # ── Stage 3: Forecasting ──────────────────────────────────
    if verbose:
        print(f"\n[3/4] Training viewership forecasting ensemble...")

    # Temporal train/test split (no data leakage)
    cutoff = feat_df["timestamp"].quantile(0.75)
    train_df = feat_df[feat_df["timestamp"] <= cutoff].copy()
    test_df = feat_df[feat_df["timestamp"] > cutoff].copy()

    if verbose:
        print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
        print(f"  Forecast horizon: {forecast_horizon} minutes ahead")

    ensemble = EnsembleForecaster()
    ensemble.fit(train_df, feature_cols)

    test_with_targets = build_targets(test_df, horizon=forecast_horizon)
    forecast_results = ensemble.predict(test_with_targets, feature_cols)

    y_true = pd.Series(test_with_targets["target_viewers"].values)

    if verbose:
        print(f"\n  Forecast Results ({forecast_horizon}-min ahead):")
        print(f"  {'=' * 60}")
        evaluate_forecast(y_true, forecast_results["xgb_forecast"], "XGBoost only")
        evaluate_forecast(y_true, forecast_results["ensemble_forecast"], "Ensemble (XGB+SARIMA)")

    # ── Stage 4: Anomaly Detection ────────────────────────────
    if verbose:
        print(f"\n[4/4] Running anomaly detection...")

    detector = StreamAnomalyDetector()
    detector.fit(train_df)
    anomaly_results = detector.detect(test_df)
    detector.summary(anomaly_results)

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t_start

    if verbose:
        print(f"\n{'=' * 65}")
        print(f"  Pipeline completed in {elapsed:.1f}s")
        print(f"  Events processed: {len(feat_df):,}")
        print(f"  Streams analyzed: {feat_df['stream_id'].nunique()}")
        detected = (anomaly_results["anomaly_type"] != "normal").sum()
        print(f"  Anomalies flagged: {detected} ({100*detected/len(anomaly_results):.1f}%)")
        print(f"{'=' * 65}")

    return {
        "raw_data": raw_df,
        "feature_data": feat_df,
        "forecast_model": ensemble,
        "anomaly_detector": detector,
        "forecast_results": forecast_results,
        "anomaly_results": anomaly_results,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    results = run_pipeline(
        n_streams=50,
        stream_duration_min=180,
        forecast_horizon=30,
        verbose=True,
    )
