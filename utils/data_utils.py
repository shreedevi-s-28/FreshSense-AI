"""
utils/data_utils.py

Shared helpers for:
    - the physically-motivated spoilage simulation used by
      generate_dataset.py
    - loading the generated CSV
    - turning it into model-ready numeric arrays (used by
      utils/model_utils.py and pages/2_Train_Model.py /
      pages/3_Predict_Freshness.py)

Keeping this logic in one place means the training page and the
prediction page can never drift out of sync on how a raw row is
turned into network input.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

from constants import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DATASET_DIR,
    DATASET_FILENAME,
    NUMERIC_FEATURES,
    PRODUCE_CONFIG,
    PRODUCE_LIST,
    SPOILAGE_INDEX_THRESHOLDS,
    TARGET_CLASSES,
    TARGET_COLUMN,
)


# ---------------------------------------------------------------------------
# Spoilage simulation (used by generate_dataset.py)
# ---------------------------------------------------------------------------

def temperature_deviation(temp_c: np.ndarray, produce: np.ndarray) -> np.ndarray:
    """Degrees outside the produce's optimal temperature band (0 if inside)."""
    lo = np.array([PRODUCE_CONFIG[p]["optimal_temp_c"][0] for p in produce])
    hi = np.array([PRODUCE_CONFIG[p]["optimal_temp_c"][1] for p in produce])
    below = np.clip(lo - temp_c, 0, None)
    above = np.clip(temp_c - hi, 0, None)
    return below + above


def humidity_deviation(humidity: np.ndarray, produce: np.ndarray) -> np.ndarray:
    """Percentage points outside the produce's optimal humidity band (0 if inside)."""
    lo = np.array([PRODUCE_CONFIG[p]["optimal_humidity_pct"][0] for p in produce])
    hi = np.array([PRODUCE_CONFIG[p]["optimal_humidity_pct"][1] for p in produce])
    below = np.clip(lo - humidity, 0, None)
    above = np.clip(humidity - hi, 0, None)
    return below + above


def compute_spoilage_index(
    produce: np.ndarray,
    temp_c: np.ndarray,
    humidity: np.ndarray,
    duration_days: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    A 0-1 "how far into spoilage" index.

    Grows with time relative to the produce's normal shelf life, and
    accelerates when temperature or humidity drift outside the
    produce's recommended storage band -- loosely modeling the
    accelerated respiration/microbial growth described in postharvest
    storage guidance (e.g. Q10-style temperature sensitivity), without
    claiming to reproduce any single published curve exactly.
    """
    shelf_life = np.array([PRODUCE_CONFIG[p]["shelf_life_days"] for p in produce])
    sensitivity = np.array([PRODUCE_CONFIG[p]["sensitivity"] for p in produce])

    time_fraction = duration_days / shelf_life

    temp_dev = temperature_deviation(temp_c, produce)
    humidity_dev = humidity_deviation(humidity, produce) / 100.0

    accel = (1 + 0.18 * temp_dev) * (1 + 0.12 * humidity_dev) * sensitivity
    noise = rng.normal(loc=0.0, scale=0.05, size=len(produce))

    spoilage = time_fraction * accel + noise
    return np.clip(spoilage, 0.0, 1.0)


def spoilage_index_to_class(spoilage_index: np.ndarray) -> np.ndarray:
    fresh_max = SPOILAGE_INDEX_THRESHOLDS["fresh_max"]
    consume_soon_max = SPOILAGE_INDEX_THRESHOLDS["consume_soon_max"]
    classes = np.where(
        spoilage_index < fresh_max,
        TARGET_CLASSES[0],
        np.where(spoilage_index < consume_soon_max, TARGET_CLASSES[1], TARGET_CLASSES[2]),
    )
    return classes


def simulate_quality_indicators(
    spoilage_index: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive the five observable quality scores from the latent spoilage index."""
    n = len(spoilage_index)

    firmness = 10 * (1 - spoilage_index) + rng.normal(0, 0.5, n)
    color = 10 * (1 - 0.85 * spoilage_index) + rng.normal(0, 0.5, n)
    smell = 10 * (1 - spoilage_index) + rng.normal(0, 0.6, n)

    mold = np.clip((spoilage_index - 0.5), 0, None) * 160 + rng.normal(0, 3, n)
    mold = np.clip(mold, 0, None)

    damage = spoilage_index * 35 + rng.uniform(0, 8, n)  # some damage is independent physical bruising

    firmness = np.clip(firmness, 0, 10)
    color = np.clip(color, 0, 10)
    smell = np.clip(smell, 0, 10)
    mold = np.clip(mold, 0, 100)
    damage = np.clip(damage, 0, 100)

    return firmness, color, smell, mold, damage


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def dataset_path() -> str:
    return os.path.join(DATASET_DIR, DATASET_FILENAME)


def load_dataset() -> pd.DataFrame:
    path = dataset_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No dataset found at '{path}'. Run `python generate_dataset.py` first."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Feature encoding for the neural network
# ---------------------------------------------------------------------------

def produce_to_onehot(produce_series: pd.Series) -> np.ndarray:
    """One-hot encode the Produce column using the fixed PRODUCE_LIST order."""
    index_map = {name: i for i, name in enumerate(PRODUCE_LIST)}
    idx = produce_series.map(index_map).to_numpy()
    onehot = np.zeros((len(idx), len(PRODUCE_LIST)), dtype=np.float64)
    onehot[np.arange(len(idx)), idx] = 1.0
    return onehot


def build_feature_matrix(
    df: pd.DataFrame, numeric_mean: np.ndarray = None, numeric_std: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn a raw dataframe (same schema as the generated CSV) into a
    standardized numeric feature matrix: [one-hot produce | scaled numeric].

    If numeric_mean/numeric_std are not provided they are computed from
    this dataframe (i.e. call this on the training split first, then
    reuse the returned mean/std when encoding validation/inference rows).
    """
    onehot = produce_to_onehot(df["Produce"])
    numeric = df[NUMERIC_FEATURES].to_numpy(dtype=np.float64)

    if numeric_mean is None or numeric_std is None:
        numeric_mean = numeric.mean(axis=0)
        numeric_std = numeric.std(axis=0)
        numeric_std[numeric_std == 0] = 1.0

    numeric_scaled = (numeric - numeric_mean) / numeric_std
    X = np.hstack([onehot, numeric_scaled])
    return X, numeric_mean, numeric_std


def labels_to_indices(labels: pd.Series) -> np.ndarray:
    class_map = {name: i for i, name in enumerate(TARGET_CLASSES)}
    return labels.map(class_map).to_numpy()


def train_test_split_arrays(
    X: np.ndarray, y: np.ndarray, test_fraction: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_fraction)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
