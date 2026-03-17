"""
Rolling Feature Engine
========================
Computes windowed aggregation features from raw stream event data.
Mirrors the kind of real-time feature computation done in production
streaming pipelines (Flink, Spark Streaming, Kafka Streams).

Features are organized into 4 families:
  - Momentum    : rate-of-change, acceleration
  - Engagement  : chat intensity, clip density, bits per viewer
  - Volatility  : rolling std, coefficient of variation
  - Social      : follower velocity, subscriber conversion

Author: Ali Taghipourfard
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────
# WINDOW CONFIGS
# ─────────────────────────────────────────────────────────────

WINDOWS = {
    "short":  5,    # 5-minute micro window
    "medium": 15,   # 15-minute window
    "long":   60,   # 1-hour macro window
}


# ─────────────────────────────────────────────────────────────
# CORE FEATURE ENGINE
# ─────────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Transforms raw per-minute stream events into a rich feature matrix.
    Processes each stream independently using rolling window operations.

    Usage:
        engine = FeatureEngine()
        features = engine.transform(raw_df)
    """

    def __init__(self, windows: dict = WINDOWS, min_periods: int = 3):
        self.windows = windows
        self.min_periods = min_periods

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature pipeline.
        Input: raw StreamEvent DataFrame (one row per stream per minute)
        Output: feature-enriched DataFrame ready for model training
        """
        df = df.copy()
        df = df.sort_values(["stream_id", "timestamp"]).reset_index(drop=True)

        feature_groups = [
            self._momentum_features,
            self._engagement_features,
            self._volatility_features,
            self._social_features,
            self._cross_stream_features,
        ]

        # Process each stream independently and concatenate
        # (avoids pandas version differences in groupby.apply behavior)
        stream_ids = df["stream_id"].unique()
        for group_fn in feature_groups:
            processed = []
            for sid in stream_ids:
                group = df[df["stream_id"] == sid].copy()
                processed.append(group_fn(group))
            df = pd.concat(processed, ignore_index=True)

        df = self._add_temporal_features(df)
        df = self._add_interaction_features(df)
        df = df.fillna(0)

        return df

    # ── Momentum ───────────────────────────────────────────

    def _momentum_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """Viewer growth rate and acceleration across multiple windows."""
        v = group["viewer_count"].astype(float)

        for name, w in self.windows.items():
            # Rolling mean
            group[f"viewers_rolling_mean_{name}"] = v.rolling(w, min_periods=self.min_periods).mean()

            # Growth rate: (current - mean_prev) / mean_prev
            prev_mean = v.shift(1).rolling(w, min_periods=self.min_periods).mean()
            group[f"viewer_growth_rate_{name}"] = (v - prev_mean) / (prev_mean + 1)

        # Momentum: short-window mean vs long-window mean (signal crossover)
        short_mean = v.rolling(self.windows["short"], min_periods=2).mean()
        long_mean = v.rolling(self.windows["long"], min_periods=2).mean()
        group["momentum_crossover"] = short_mean / (long_mean + 1)

        # Acceleration: 2nd derivative of viewer count
        group["viewer_acceleration"] = v.diff().diff()

        # All-time peak ratio
        group["pct_of_session_peak"] = v / (v.cummax() + 1)

        return group

    # ── Engagement ──────────────────────────────────────────

    def _engagement_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """Chat intensity, clip rate, and monetization signals."""
        v = group["viewer_count"].astype(float).replace(0, 1)
        chat = group["chat_messages_per_min"].astype(float)
        bits = group["bits_cheered"].astype(float)

        # Chat per viewer (normalized engagement)
        group["chat_per_viewer"] = chat / v

        # Rolling chat intensity
        for name, w in self.windows.items():
            group[f"chat_intensity_{name}"] = chat.rolling(w, min_periods=self.min_periods).mean()
            group[f"bits_per_viewer_{name}"] = (bits / v).rolling(w, min_periods=self.min_periods).mean()

        # Clip density (clips per 1000 viewers)
        group["clip_density"] = (group["clip_creations"] / v) * 1000

        # Engagement score (composite)
        group["engagement_score"] = (
            0.4 * group["chat_per_viewer"] +
            0.3 * (group["clip_density"] / 10).clip(0, 1) +
            0.3 * (group["bits_cheered"] / (v * 100)).clip(0, 1)
        )

        return group

    # ── Volatility ──────────────────────────────────────────

    def _volatility_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """Rolling standard deviation and coefficient of variation."""
        v = group["viewer_count"].astype(float)

        for name, w in self.windows.items():
            roll_std = v.rolling(w, min_periods=self.min_periods).std()
            roll_mean = v.rolling(w, min_periods=self.min_periods).mean()
            group[f"viewer_volatility_{name}"] = roll_std
            group[f"viewer_cv_{name}"] = roll_std / (roll_mean + 1)  # Coefficient of variation

        # Sudden spike detection: z-score relative to recent window
        roll_mean_m = v.rolling(self.windows["medium"], min_periods=self.min_periods).mean()
        roll_std_m = v.rolling(self.windows["medium"], min_periods=self.min_periods).std()
        group["viewer_zscore"] = (v - roll_mean_m) / (roll_std_m + 1)

        return group

    # ── Social ──────────────────────────────────────────────

    def _social_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """Follower velocity and subscriber conversion rate."""
        followers = group["follower_growth_rate"].astype(float)
        subs = group["subscriber_count"].astype(float)
        v = group["viewer_count"].astype(float).replace(0, 1)

        group["follower_velocity"] = followers.rolling(
            self.windows["medium"], min_periods=self.min_periods).mean()
        group["follower_acceleration"] = followers.diff()
        group["sub_conversion_rate"] = subs / v
        group["sub_rate_trend"] = (subs / v).rolling(
            self.windows["short"], min_periods=2).mean().diff()

        return group

    # ── Cross-stream ─────────────────────────────────────────

    def _cross_stream_features(self, group: pd.DataFrame) -> pd.DataFrame:
        """Category-level competitive context (market share)."""
        v = group["viewer_count"].astype(float)
        concurrent = group["concurrent_streams_in_category"].astype(float).replace(0, 1)
        group["estimated_category_share"] = v / (v + concurrent * 3000)  # Approximate share
        return group

    # ── Temporal ────────────────────────────────────────────

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hour-of-day and day-of-week cyclical encoding."""
        hour = df["timestamp"].dt.hour
        dow = df["timestamp"].dt.dayofweek
        minute = df["timestamp"].dt.minute

        # Cyclical encoding (sin/cos) for periodicity
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["is_prime_time"] = ((hour >= 18) & (hour <= 22)).astype(int)
        df["is_weekend"] = (dow >= 5).astype(int)

        return df

    # ── Interaction ─────────────────────────────────────────

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multiplicative interaction terms between feature families."""
        df["momentum_x_engagement"] = df["momentum_crossover"] * df["engagement_score"]
        df["growth_x_primetime"] = df["viewer_growth_rate_short"] * df["is_prime_time"]
        df["volatility_x_zscore"] = df.get("viewer_volatility_short", 0) * df.get("viewer_zscore", 0)
        return df


# ─────────────────────────────────────────────────────────────
# FEATURE CATALOG
# ─────────────────────────────────────────────────────────────

def get_model_features(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns suitable for model input."""
    exclude = {
        "stream_id", "streamer_name", "game", "timestamp",
        "viewer_count", "raid_incoming", "is_bot_activity",
        # raw counts — we use engineered versions instead
        "chat_messages_per_min", "clip_creations", "subscriber_count",
        "bits_cheered", "follower_growth_rate", "concurrent_streams_in_category",
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32]]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_simulator import StreamSimulator

    print("Running feature engine...")
    sim = StreamSimulator(n_streams=10, stream_duration_min=120)
    raw_df = sim.to_dataframe()
    print(f"Raw events: {len(raw_df):,}")

    engine = FeatureEngine()
    features_df = engine.transform(raw_df)

    feature_cols = get_model_features(features_df)
    print(f"\nEngineered features: {len(feature_cols)}")
    print(f"Feature families:")
    for prefix in ["viewers_rolling", "viewer_growth", "chat_intensity", "viewer_volatility",
                   "bits_per_viewer", "follower", "sub_", "hour_", "dow_", "momentum", "engagement"]:
        cols = [c for c in feature_cols if c.startswith(prefix)]
        if cols:
            print(f"  {prefix:<25}: {len(cols)} features")

    print(f"\nSample feature row:")
    print(features_df[feature_cols].dropna().iloc[0].to_string())
