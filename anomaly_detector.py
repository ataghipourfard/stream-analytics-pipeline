"""
Stream Anomaly Detector
========================
Detects two types of anomalies in live stream data:

  1. VIRAL MOMENTS  — legitimate organic spikes (viewer surges,
                      high chat velocity, clip bursts)
  2. BOT RAIDS      — inorganic viewer inflation (sudden spike +
                      zero engagement, no chat increase)

Uses an ensemble of:
  - Isolation Forest  (unsupervised outlier detection)
  - Z-score threshold (statistical process control)
  - Rule-based heuristics (bot raid pattern matching)

Author: Ali Taghipourfard
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Literal
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# ANOMALY TYPES
# ─────────────────────────────────────────────────────────────

AnomalyType = Literal["viral_moment", "bot_raid", "normal"]

@dataclass
class Anomaly:
    stream_id: str
    timestamp: pd.Timestamp
    anomaly_type: AnomalyType
    confidence: float
    viewer_count: int
    viewer_zscore: float
    engagement_ratio: float
    description: str


# ─────────────────────────────────────────────────────────────
# ISOLATION FOREST DETECTOR
# ─────────────────────────────────────────────────────────────

class IsolationForestDetector:
    """
    Isolation Forest trained on normal stream behavior.
    Flags outlier time steps as candidate anomalies.
    """

    ANOMALY_FEATURES = [
        "viewer_growth_rate_short",
        "viewer_growth_rate_medium",
        "viewer_volatility_short",
        "viewer_zscore",
        "chat_per_viewer",
        "engagement_score",
        "clip_density",
        "follower_velocity",
        "momentum_crossover",
        "bits_per_viewer_short",
    ]

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        feature_cols = [c for c in self.ANOMALY_FEATURES if c in df.columns]
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._fitted = True
        self.feature_cols = feature_cols
        print(f"  IsolationForest fitted on {len(X):,} samples | {len(feature_cols)} features")
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (lower = more anomalous). Range: [-1, 0]."""
        X = df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return 1 (normal) or -1 (anomaly) for each row."""
        X = df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ─────────────────────────────────────────────────────────────
# RULE-BASED BOT DETECTOR
# ─────────────────────────────────────────────────────────────

class BotRaidDetector:
    """
    Heuristic rules for detecting bot raids.

    Bot raid signature:
      - Large sudden viewer spike (>3x in 1-2 minutes)
      - Chat rate does NOT increase proportionally
      - Engagement score drops despite viewer increase
      - Spike reverses almost immediately
    """

    def __init__(self,
                 spike_threshold: float = 2.5,
                 engagement_drop_threshold: float = 0.3,
                 reversal_window: int = 3):
        self.spike_threshold = spike_threshold
        self.engagement_drop_threshold = engagement_drop_threshold
        self.reversal_window = reversal_window

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series: True = suspected bot raid."""
        df = df.copy()

        # Viewer spike: short growth rate is very high
        high_spike = df.get("viewer_growth_rate_short", pd.Series(0, index=df.index)) > self.spike_threshold

        # Engagement paradox: viewers spike but engagement per viewer drops
        engagement = df.get("engagement_score", pd.Series(0.5, index=df.index))
        low_engagement = engagement < self.engagement_drop_threshold

        # Chat didn't follow: chat per viewer is well below average
        chat_per_viewer = df.get("chat_per_viewer", pd.Series(0.05, index=df.index))
        low_chat = chat_per_viewer < chat_per_viewer.quantile(0.20)

        bot_suspected = high_spike & (low_engagement | low_chat)
        return bot_suspected


# ─────────────────────────────────────────────────────────────
# VIRAL MOMENT DETECTOR
# ─────────────────────────────────────────────────────────────

class ViralMomentDetector:
    """
    Identifies genuine organic viral spikes.

    Viral signature:
      - Large viewer growth
      - High z-score (unusual relative to stream's own history)
      - Chat rate increases proportionally
      - Clip creation rate spikes
      - Follower growth accelerates
    """

    def __init__(self, zscore_threshold: float = 2.5, chat_boost_threshold: float = 1.3):
        self.zscore_threshold = zscore_threshold
        self.chat_boost_threshold = chat_boost_threshold

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series: True = likely viral moment."""
        high_zscore = df.get("viewer_zscore", pd.Series(0, index=df.index)).abs() > self.zscore_threshold
        positive_growth = df.get("viewer_growth_rate_short", pd.Series(0, index=df.index)) > 0.5

        # Chat is growing with viewers (organic signal)
        chat_intensity = df.get("chat_intensity_short", pd.Series(0, index=df.index))
        mean_chat = chat_intensity.mean()
        high_chat = chat_intensity > mean_chat * self.chat_boost_threshold

        # Follower acceleration is positive
        follower_acc = df.get("follower_acceleration", pd.Series(0, index=df.index))
        follower_up = follower_acc > 0

        # High clip density
        clip_density = df.get("clip_density", pd.Series(0, index=df.index))
        high_clips = clip_density > clip_density.quantile(0.80)

        viral = high_zscore & positive_growth & high_chat
        return viral


# ─────────────────────────────────────────────────────────────
# MAIN ANOMALY DETECTOR
# ─────────────────────────────────────────────────────────────

class StreamAnomalyDetector:
    """
    Ensemble anomaly detector combining:
    - Isolation Forest (statistical outlier detection)
    - Bot raid heuristics
    - Viral moment classification

    Produces labeled anomaly events with confidence scores.
    """

    def __init__(self):
        self.iso_forest = IsolationForestDetector(contamination=0.05)
        self.bot_detector = BotRaidDetector()
        self.viral_detector = ViralMomentDetector()

    def fit(self, df: pd.DataFrame) -> "StreamAnomalyDetector":
        """Fit the Isolation Forest on historical stream data."""
        self.iso_forest.fit(df)
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full anomaly detection pipeline.
        Returns DataFrame with anomaly labels and confidence scores.
        """
        df = df.copy()

        # Isolation Forest scores
        df["iso_score"] = self.iso_forest.score(df)
        df["iso_is_anomaly"] = self.iso_forest.predict(df) == -1

        # Rule-based classifiers
        df["suspected_bot"] = self.bot_detector.detect(df)
        df["suspected_viral"] = self.viral_detector.detect(df)

        # Ensemble label
        def classify(row):
            if row["suspected_bot"] and row["iso_is_anomaly"]:
                return "bot_raid"
            elif row["suspected_viral"] and row["iso_is_anomaly"]:
                return "viral_moment"
            elif row["iso_is_anomaly"]:
                return "anomaly_unknown"
            return "normal"

        df["anomaly_type"] = df.apply(classify, axis=1)

        # Confidence: higher iso anomaly score = more anomalous
        df["anomaly_confidence"] = 1 - ((df["iso_score"] - df["iso_score"].min()) /
                                         (df["iso_score"].max() - df["iso_score"].min() + 1e-8))

        return df

    def summary(self, results: pd.DataFrame) -> None:
        """Print anomaly detection summary."""
        total = len(results)
        counts = results["anomaly_type"].value_counts()
        print(f"\n{'=' * 50}")
        print("  ANOMALY DETECTION SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Total events analyzed : {total:,}")
        for atype, count in counts.items():
            pct = 100 * count / total
            bar = "█" * int(20 * count / total)
            label = atype.replace("_", " ").title()
            print(f"  {label:<22} : {count:>5} ({pct:.1f}%) {bar}")

        print(f"\n  Sample Detected Anomalies:")
        print("-" * 65)
        anomalies = results[results["anomaly_type"] != "normal"].sort_values(
            "anomaly_confidence", ascending=False
        ).head(10)
        if len(anomalies) == 0:
            print("  None detected.")
        else:
            cols = ["stream_id", "timestamp", "anomaly_type", "anomaly_confidence",
                    "viewer_count", "viewer_zscore"]
            cols = [c for c in cols if c in anomalies.columns]
            for _, row in anomalies[cols].iterrows():
                print(f"  [{row.get('anomaly_type','?'):<18}] "
                      f"stream={row.get('stream_id','?'):<14} "
                      f"viewers={int(row.get('viewer_count',0)):>7,} "
                      f"confidence={row.get('anomaly_confidence',0):.2f}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_simulator import StreamSimulator
    from feature_engine import FeatureEngine

    print("=" * 60)
    print(" Stream Anomaly Detector (IsolationForest + Rules)")
    print("=" * 60)

    print("\nGenerating stream data with injected anomalies...")
    sim = StreamSimulator(n_streams=40, stream_duration_min=180)
    raw_df = sim.to_dataframe()
    print(f"Streams with viral events : {sum(1 for p in sim.profiles if p.has_viral_event)}")
    print(f"Streams with bot raids    : {sum(1 for p in sim.profiles if p.has_bot_raid)}")

    print("\nEngineering features...")
    engine = FeatureEngine()
    feat_df = engine.transform(raw_df)

    # Fit on first 70% of data, detect on remainder
    cutoff = feat_df["timestamp"].quantile(0.70)
    train_df = feat_df[feat_df["timestamp"] <= cutoff]
    test_df = feat_df[feat_df["timestamp"] > cutoff]

    print(f"\nFitting anomaly detector on {len(train_df):,} training events...")
    detector = StreamAnomalyDetector()
    detector.fit(train_df)

    print(f"\nDetecting anomalies in {len(test_df):,} test events...")
    results = detector.detect(test_df)
    detector.summary(results)
