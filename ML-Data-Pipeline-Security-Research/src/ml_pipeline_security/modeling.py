"""Model implementations and evaluation helpers."""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

FEATURE_COLUMNS = [
    "query_length",
    "entropy",
    "source_reputation",
    "failed_auth_rate",
    "geo_velocity",
]


class NumpyLogisticRegression:
    """Minimal logistic regression via gradient descent to show ML fundamentals."""

    def __init__(
        self,
        lr: float = 0.05,
        n_iter: int = 1200,
        l2: float = 1e-3,
        tolerance: float = 1e-7,
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.tolerance = tolerance
        self.weights: np.ndarray = np.array([], dtype=float)
        self.bias = 0.0
        self.is_fitted = False
        # Stored per-feature statistics for z-score standardization.
        self.mean_: np.ndarray = np.array([], dtype=float)
        self.std_: np.ndarray = np.array([], dtype=float)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        """Apply stored z-score transform.  Called after fit() has populated mean_ / std_."""
        return (x - self.mean_) / self.std_

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        # Compute and cache per-feature statistics from the training split.
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        # Guard against zero-variance features (e.g. a constant column) to avoid division by zero.
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        x = self._standardize(x)

        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        prev_loss = np.inf

        for _ in range(self.n_iter):
            linear = x @ self.weights + self.bias
            probs = self._sigmoid(linear)

            dw = (x.T @ (probs - y)) / n_samples + self.l2 * self.weights
            db = np.mean(probs - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Early stopping keeps the educational implementation efficient.
            eps = 1e-12
            probs_clipped = np.clip(probs, eps, 1 - eps)
            loss = -np.mean(y * np.log(probs_clipped) + (1 - y) * np.log(1 - probs_clipped))
            loss += 0.5 * self.l2 * float(np.sum(self.weights**2))
            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss

        self.is_fitted = True
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before prediction.")

        # Apply the same standardization that was fitted on the training data.
        x = self._standardize(x)
        linear = x @ self.weights + self.bias
        probs = self._sigmoid(linear)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)[:, 1]
        return (probs >= 0.5).astype(int)


@dataclass
class ModelArtifacts:
    """Simple model container for experiment usage."""

    model_name: str
    model: NumpyLogisticRegression | LogisticRegression


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pick model-safe numeric features used across experiments."""
    return df[FEATURE_COLUMNS].astype(float)


def train_sklearn_logreg(train_df: pd.DataFrame, class_weight: str | None = None) -> ModelArtifacts:
    """Train a standard sklearn logistic regression model."""
    x_train = select_features(train_df).to_numpy()
    y_train = train_df["is_malicious"].astype(int).to_numpy()

    model = LogisticRegression(max_iter=500, solver="lbfgs", class_weight=class_weight)
    model.fit(x_train, y_train)
    return ModelArtifacts(model_name="sklearn_logreg", model=model)


def train_numpy_logreg(train_df: pd.DataFrame) -> ModelArtifacts:
    """Train the NumPy implementation."""
    x_train = select_features(train_df).to_numpy()
    y_train = train_df["is_malicious"].astype(int).to_numpy()

    model = NumpyLogisticRegression()
    model.fit(x_train, y_train)
    return ModelArtifacts(model_name="numpy_logreg", model=model)


def evaluate_model(artifacts: ModelArtifacts, eval_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate classifier on standard binary metrics."""
    x_eval = select_features(eval_df).to_numpy()
    y_true = eval_df["is_malicious"].astype(int).to_numpy()
    model = artifacts.model

    # Both supported models expose sklearn-style predict/predict_proba.
    y_prob = model.predict_proba(x_eval)[:, 1]
    y_pred = model.predict(x_eval)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def attack_success_rate(artifacts: ModelArtifacts, trigger_df: pd.DataFrame, target_label: int = 0) -> float:
    """Measure fraction of trigger examples predicted as attacker target label."""
    x_trigger = select_features(trigger_df).to_numpy()
    y_pred = artifacts.model.predict(x_trigger)

    return float(np.mean(y_pred == target_label))
