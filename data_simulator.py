"""
Stream Event Simulator
=======================
Generates realistic Twitch-style streaming event data with:
- Realistic viewer growth curves (log-normal ramp-up, plateau, decay)
- Time-correlated chat activity and engagement rates
- Synthetic viral spikes and bot raid patterns
- Multiple concurrent streams across game categories

Author: Ali Taghipourfard
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Generator
from datetime import datetime, timedelta
import random


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

GAME_CATEGORIES = {
    "Fortnite":           {"base_viewers": 8000,  "volatility": 0.25, "chat_rate": 0.08},
    "League of Legends":  {"base_viewers": 12000, "volatility": 0.20, "chat_rate": 0.06},
    "Valorant":           {"base_viewers": 9500,  "volatility": 0.22, "chat_rate": 0.07},
    "Minecraft":          {"base_viewers": 6000,  "volatility": 0.18, "chat_rate": 0.10},
    "Just Chatting":      {"base_viewers": 5000,  "volatility": 0.30, "chat_rate": 0.15},
    "Elden Ring":         {"base_viewers": 4500,  "volatility": 0.35, "chat_rate": 0.12},
    "Apex Legends":       {"base_viewers": 7000,  "volatility": 0.24, "chat_rate": 0.07},
}

HOUR_MULTIPLIERS = {
    0: 0.35, 1: 0.25, 2: 0.20, 3: 0.18, 4: 0.18, 5: 0.22,
    6: 0.30, 7: 0.40, 8: 0.50, 9: 0.60, 10: 0.70, 11: 0.78,
    12: 0.85, 13: 0.88, 14: 0.90, 15: 0.92, 16: 0.95, 17: 1.00,
    18: 1.10, 19: 1.20, 20: 1.25, 21: 1.20, 22: 1.00, 23: 0.75,
}


# ─────────────────────────────────────────────────────────────
# STREAM DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class StreamEvent:
    stream_id: str
    streamer_name: str
    game: str
    timestamp: datetime
    viewer_count: int
    chat_messages_per_min: float
    clip_creations: int
    subscriber_count: int
    bits_cheered: int
    raid_incoming: bool
    is_bot_activity: bool
    follower_growth_rate: float
    concurrent_streams_in_category: int


@dataclass
class StreamProfile:
    stream_id: str
    streamer_name: str
    game: str
    start_time: datetime
    base_viewers: int
    peak_viewers: int
    volatility: float
    chat_rate: float
    has_viral_event: bool
    viral_event_minute: int = -1
    has_bot_raid: bool = False
    bot_raid_minute: int = -1


# ─────────────────────────────────────────────────────────────
# VIEWER CURVE SIMULATION
# ─────────────────────────────────────────────────────────────

def _viewer_curve(minute: int, profile: StreamProfile, noise_seed: int) -> int:
    """
    Realistic viewer curve with 4 phases:
    1. Ramp-up (log growth first 30 min)
    2. Plateau (random walk around peak)
    3. Decay (gradual fall-off)
    4. Optional viral spike
    """
    rng = np.random.default_rng(noise_seed + minute)
    total_minutes = 240  # 4-hour stream

    # Phase weights
    ramp_phase = min(1.0, minute / 30)
    decay_phase = max(0.0, (minute - 180) / 60) if minute > 180 else 0.0

    base = profile.base_viewers * ramp_phase * (1 - 0.4 * decay_phase)

    # Hour-of-day multiplier
    stream_hour = (profile.start_time.hour + minute // 60) % 24
    base *= HOUR_MULTIPLIERS[stream_hour]

    # Random walk noise
    noise = rng.normal(0, profile.volatility * base * 0.1)

    viewers = int(max(0, base + noise))

    # Viral spike (Gaussian bump centered on viral minute)
    if profile.has_viral_event and profile.viral_event_minute > 0:
        dist = abs(minute - profile.viral_event_minute)
        spike = profile.peak_viewers * 0.6 * np.exp(-0.5 * (dist / 15) ** 2)
        viewers += int(spike)

    # Bot raid (sudden unnatural spike then instant drop)
    if profile.has_bot_raid and profile.bot_raid_minute > 0:
        if minute == profile.bot_raid_minute:
            viewers += rng.integers(5000, 20000)
        elif minute == profile.bot_raid_minute + 1:
            viewers = max(0, viewers - rng.integers(4000, 18000))

    return max(0, viewers)


# ─────────────────────────────────────────────────────────────
# STREAM GENERATOR
# ─────────────────────────────────────────────────────────────

class StreamSimulator:
    """
    Simulates concurrent Twitch streams with realistic event patterns.
    Yields StreamEvent objects at configurable time resolution.
    """

    def __init__(self, n_streams: int = 50, stream_duration_min: int = 240,
                 tick_interval_min: int = 1, seed: int = 42):
        self.n_streams = n_streams
        self.duration = stream_duration_min
        self.tick = tick_interval_min
        self.seed = seed
        self.profiles: list[StreamProfile] = []
        self._init_profiles()

    def _init_profiles(self):
        rng = np.random.default_rng(self.seed)
        games = list(GAME_CATEGORIES.keys())

        for i in range(self.n_streams):
            game = rng.choice(games)
            cfg = GAME_CATEGORIES[game]
            base = int(cfg["base_viewers"] * rng.uniform(0.3, 2.0))
            peak = int(base * rng.uniform(1.5, 4.0))
            start_offset = int(rng.integers(0, 120))  # streams start at different times
            start_time = datetime(2025, 6, 15, 17, 0) + timedelta(minutes=start_offset)

            has_viral = rng.random() < 0.15   # 15% chance of viral moment
            has_bots = rng.random() < 0.08    # 8% chance of bot raid

            profile = StreamProfile(
                stream_id=f"stream_{i:04d}",
                streamer_name=f"streamer_{i}",
                game=game,
                start_time=start_time,
                base_viewers=base,
                peak_viewers=peak,
                volatility=cfg["volatility"],
                chat_rate=cfg["chat_rate"],
                has_viral_event=has_viral,
                viral_event_minute=int(rng.integers(60, 180)) if has_viral else -1,
                has_bot_raid=has_bots,
                bot_raid_minute=int(rng.integers(30, 200)) if has_bots else -1,
            )
            self.profiles.append(profile)

    def generate(self) -> Generator[StreamEvent, None, None]:
        """Yield stream events for all streams across all time ticks."""
        category_counts = self._category_counts()
        rng = np.random.default_rng(self.seed + 1)

        for minute in range(0, self.duration, self.tick):
            for profile in self.profiles:
                viewers = _viewer_curve(minute, profile, noise_seed=hash(profile.stream_id) % 10000)
                chat_rate = profile.chat_rate * viewers / 100
                chat_rate *= rng.uniform(0.8, 1.2)

                # Derived metrics
                clips = max(0, int(rng.poisson(viewers / 5000)))
                subs = max(0, int(viewers * rng.uniform(0.01, 0.04)))
                bits = max(0, int(rng.exponential(viewers * 0.5)))
                follower_rate = rng.uniform(0.001, 0.005) * viewers

                timestamp = profile.start_time + timedelta(minutes=minute)

                event = StreamEvent(
                    stream_id=profile.stream_id,
                    streamer_name=profile.streamer_name,
                    game=profile.game,
                    timestamp=timestamp,
                    viewer_count=viewers,
                    chat_messages_per_min=round(chat_rate, 2),
                    clip_creations=clips,
                    subscriber_count=subs,
                    bits_cheered=bits,
                    raid_incoming=False,
                    is_bot_activity=profile.has_bot_raid and minute == profile.bot_raid_minute,
                    follower_growth_rate=round(follower_rate, 4),
                    concurrent_streams_in_category=category_counts.get(profile.game, 1),
                )
                yield event

    def _category_counts(self) -> dict:
        counts = {}
        for p in self.profiles:
            counts[p.game] = counts.get(p.game, 0) + 1
        return counts

    def to_dataframe(self) -> pd.DataFrame:
        """Materialize all events into a DataFrame."""
        records = [vars(e) for e in self.generate()]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values(["stream_id", "timestamp"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Simulating stream events...")
    sim = StreamSimulator(n_streams=20, stream_duration_min=120, tick_interval_min=1)
    df = sim.to_dataframe()
    print(f"Generated {len(df):,} events | {df['stream_id'].nunique()} streams | {df['game'].nunique()} games")
    print(df.head(10).to_string())
    df.to_csv("stream_events.csv", index=False)
    print("\nSaved to stream_events.csv")
