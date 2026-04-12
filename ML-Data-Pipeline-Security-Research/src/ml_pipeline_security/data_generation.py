"""Synthetic data and attack simulation utilities for ML pipeline security research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataBundle:
    """Container for train/test datasets used in experiments."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame


def generate_base_dataset(n_samples: int = 5000, seed: int = 7) -> pd.DataFrame:
    """Create a synthetic security-oriented dataset.

    Features mimic telemetry from an ML-backed security workflow. The target
    `is_malicious` is generated from a latent risk function.
    """
    rng = np.random.default_rng(seed)

    query_length = rng.normal(loc=120.0, scale=40.0, size=n_samples).clip(5, 500)
    entropy = rng.beta(a=2.0, b=5.0, size=n_samples)
    source_reputation = rng.uniform(0.0, 1.0, size=n_samples)
    failed_auth_rate = rng.beta(a=1.2, b=8.0, size=n_samples)
    geo_velocity = rng.normal(loc=1.0, scale=0.6, size=n_samples).clip(0.0, 6.0)

    risk_score = (
        0.005 * query_length
        + 2.2 * entropy
        + 2.8 * failed_auth_rate
        + 0.9 * geo_velocity
        - 2.5 * source_reputation
        + rng.normal(0.0, 0.35, size=n_samples)
    )
    prob_malicious = 1.0 / (1.0 + np.exp(-risk_score + 1.6))
    is_malicious = (rng.random(n_samples) < prob_malicious).astype(int)

    df = pd.DataFrame(
        {
            "query_length": query_length,
            "entropy": entropy,
            "source_reputation": source_reputation,
            "failed_auth_rate": failed_auth_rate,
            "geo_velocity": geo_velocity,
            "is_malicious": is_malicious,
            "record_id": [f"evt-{i:07d}" for i in range(n_samples)],
        }
    )
    return df


def inject_label_poisoning(
    df: pd.DataFrame,
    poison_fraction: float = 0.08,
    trigger_value: float = 5.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Inject backdoor-like poisoning by flipping labels on trigger-pattern rows."""
    rng = np.random.default_rng(seed)
    poisoned = df.copy()

    # Poison only originally malicious rows to simulate targeted relabeling.
    candidate_idx = poisoned.index[poisoned["is_malicious"] == 1].to_numpy()
    if len(candidate_idx) == 0:
        return poisoned

    n_poison = max(1, int(len(candidate_idx) * poison_fraction))
    selected = rng.choice(candidate_idx, size=min(n_poison, len(candidate_idx)), replace=False)

    poisoned.loc[selected, "geo_velocity"] = trigger_value
    poisoned.loc[selected, "entropy"] = np.maximum(poisoned.loc[selected, "entropy"], 0.9)
    poisoned.loc[selected, "is_malicious"] = 0
    return poisoned


def inject_sensitive_column(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Add a synthetic sensitive field to test sanitization controls."""
    rng = np.random.default_rng(seed)
    with_sensitive = df.copy()
    with_sensitive["email"] = [f"user{i}@example.org" for i in rng.integers(1000, 9999, size=len(df))]
    return with_sensitive


def split_dataset(df: pd.DataFrame, seed: int = 7) -> DataBundle:
    """Split dataset into train and test partitions."""
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=seed,
        stratify=df["is_malicious"],
    )
    return DataBundle(train_df=train_df.reset_index(drop=True), test_df=test_df.reset_index(drop=True))


def make_trigger_set(df: pd.DataFrame, trigger_value: float = 5.0) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create trigger-modified copy for attack success rate measurement."""
    trigger_df = df.copy()
    trigger_df["geo_velocity"] = trigger_value
    trigger_df["entropy"] = np.maximum(trigger_df["entropy"], 0.9)
    # Kept for future extension where only a subset is trigger-modified.
    trigger_mask = np.ones(len(trigger_df), dtype=bool)
    return trigger_df, trigger_mask
