"""Security controls for data integrity, privacy, and anomaly defense."""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import IsolationForest

EXPECTED_SCHEMA = {
    "query_length": "float",
    "entropy": "float",
    "source_reputation": "float",
    "failed_auth_rate": "float",
    "geo_velocity": "float",
    "is_malicious": "int",
}

SENSITIVE_PATTERN = re.compile(r"(ssn|email|name|address|token|secret)", re.IGNORECASE)


def validate_schema(df: pd.DataFrame, expected_schema: Dict[str, str] | None = None) -> Tuple[bool, List[str]]:
    """Validate presence and dtypes for required columns."""
    schema = expected_schema or EXPECTED_SCHEMA
    issues: List[str] = []

    for col, expected_dtype in schema.items():
        if col not in df.columns:
            issues.append(f"missing column: {col}")
            continue

        if expected_dtype == "float" and not pd.api.types.is_float_dtype(df[col]):
            issues.append(f"dtype mismatch on {col}: expected float-like, got {df[col].dtype}")
        elif expected_dtype == "int" and not pd.api.types.is_integer_dtype(df[col]):
            issues.append(f"dtype mismatch on {col}: expected int-like, got {df[col].dtype}")

    return len(issues) == 0, issues


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute a deterministic hash for dataset integrity checks."""
    normalized = df.sort_index(axis=1).sort_values(by=list(df.columns), kind="mergesort", ignore_index=True)
    payload = normalized.to_json(orient="split", index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detect_sensitive_columns(df: pd.DataFrame) -> List[str]:
    """Identify likely-sensitive columns by name pattern."""
    return [col for col in df.columns if SENSITIVE_PATTERN.search(col)]


def sanitize_sensitive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sensitive column values with salted hashes."""
    sanitized = df.copy()
    sensitive_cols = detect_sensitive_columns(sanitized)

    def _hash_value(value: object) -> str:
        return hashlib.sha256(f"laiser-salt::{value}".encode("utf-8")).hexdigest()[:16]

    for col in sensitive_cols:
        sanitized[col] = sanitized[col].astype(str).map(_hash_value)

    return sanitized


def remove_anomalies(df: pd.DataFrame, contamination: float = 0.03, seed: int = 7) -> Tuple[pd.DataFrame, int]:
    """Filter anomalous rows with IsolationForest and return count removed."""
    filtered = df.copy()
    numeric_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c]) and c != "is_malicious"]

    if not numeric_cols or len(filtered) < 50:
        return filtered, 0

    detector = IsolationForest(contamination=contamination, random_state=seed)
    predictions = detector.fit_predict(filtered[numeric_cols])
    kept = filtered[predictions == 1].reset_index(drop=True)
    removed = int((predictions == -1).sum())
    return kept, removed


def remove_suspicious_trigger_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop likely-poisoned rows matching a suspicious trigger pattern.

    Backdoor poisoning in this demo tends to create benign-labeled rows with
    unusually high entropy and geo-velocity.
    """
    filtered = df.copy()
    suspicious = (
        (filtered["is_malicious"] == 0)
        & (filtered["entropy"] >= 0.88)
        & (filtered["geo_velocity"] >= 4.8)
    )
    removed = int(suspicious.sum())
    return filtered.loc[~suspicious].reset_index(drop=True), removed


def build_control_audit_log(schema_ok: bool, schema_issues: List[str], removed_rows: int, fingerprint: str) -> str:
    """Serialize control outcomes for experiment reporting."""
    return json.dumps(
        {
            "schema_ok": schema_ok,
            "schema_issues": schema_issues,
            "rows_removed_by_anomaly_control": removed_rows,
            "dataset_fingerprint": fingerprint,
        },
        indent=2,
    )
