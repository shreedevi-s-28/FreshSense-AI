"""
generate_dataset.py

Generates the FreshSense AI synthetic dataset.

Each row is a plausible storage scenario for one produce item: a
storage temperature/humidity/duration are sampled (deliberately
including both well-managed and poorly-managed cold-chain scenarios),
then a latent "spoilage index" is computed from how far those
conditions and that duration sit relative to the produce's normal
storage guidelines (see constants.PRODUCE_CONFIG). The five observable
quality indicators (firmness, color, smell, mold, surface damage) and
the final Fresh / Consume Soon / Spoiled label are then derived from
that spoilage index with realistic noise, so the features and the
label stay internally consistent rather than being assigned
independently.

Usage:
    python generate_dataset.py                 # writes DEFAULT_NUM_SAMPLES rows
    python generate_dataset.py --n 20000        # custom sample count
    python generate_dataset.py --seed 7         # custom seed
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from constants import (
    DATASET_DIR,
    DATASET_FILENAME,
    DEFAULT_NUM_SAMPLES,
    NUMERIC_FEATURES,
    PRODUCE_CONFIG,
    PRODUCE_LIST,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from utils.data_utils import (
    compute_spoilage_index,
    simulate_quality_indicators,
    spoilage_index_to_class,
)


def sample_storage_conditions(n: int, rng: np.random.Generator):
    """Sample produce + temperature/humidity/duration, mixing good and bad storage."""
    produce = rng.choice(PRODUCE_LIST, size=n)

    optimal_temp_lo = np.array([PRODUCE_CONFIG[p]["optimal_temp_c"][0] for p in produce])
    optimal_temp_hi = np.array([PRODUCE_CONFIG[p]["optimal_temp_c"][1] for p in produce])
    optimal_hum_lo = np.array([PRODUCE_CONFIG[p]["optimal_humidity_pct"][0] for p in produce])
    optimal_hum_hi = np.array([PRODUCE_CONFIG[p]["optimal_humidity_pct"][1] for p in produce])
    shelf_life = np.array([PRODUCE_CONFIG[p]["shelf_life_days"] for p in produce])

    # 60% of samples: storage reasonably close to the recommended band
    # (well-run cold chain). 40%: drifted further away (poor storage,
    # transport, retail shelf, home fridge/counter, etc.)
    well_stored = rng.random(n) < 0.6

    temp_center = (optimal_temp_lo + optimal_temp_hi) / 2
    temp_good = rng.normal(temp_center, 1.5, n)
    temp_bad = rng.uniform(-5, 40, n)
    storage_temp_c = np.where(well_stored, temp_good, temp_bad)

    hum_center = (optimal_hum_lo + optimal_hum_hi) / 2
    hum_good = rng.normal(hum_center, 3, n)
    hum_bad = rng.uniform(30, 100, n)
    relative_humidity_pct = np.clip(np.where(well_stored, hum_good, hum_bad), 20, 100)

    # duration ranges from "just stocked" to well past typical shelf life.
    # A beta distribution is used (rather than uniform) so the sample is
    # weighted toward realistic, more commonly observed durations while
    # still covering the full range from fresh to well-spoiled, keeping
    # the three target classes reasonably balanced.
    duration_fraction = rng.beta(1.1, 2.8, n) * 1.7
    storage_duration_days = np.round(duration_fraction * shelf_life, 1)

    storage_temp_c = np.round(storage_temp_c, 1)
    relative_humidity_pct = np.round(relative_humidity_pct, 1)

    return produce, storage_temp_c, relative_humidity_pct, storage_duration_days


def generate(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    produce, temp_c, humidity, duration = sample_storage_conditions(n_samples, rng)

    spoilage_index = compute_spoilage_index(produce, temp_c, humidity, duration, rng)
    firmness, color, smell, mold, damage = simulate_quality_indicators(spoilage_index, rng)
    labels = spoilage_index_to_class(spoilage_index)

    df = pd.DataFrame(
        {
            "Produce": produce,
            "Storage_Temperature_C": temp_c,
            "Relative_Humidity_pct": humidity,
            "Storage_Duration_Days": duration,
            "Firmness_Score": np.round(firmness, 2),
            "Color_Score": np.round(color, 2),
            "Smell_Score": np.round(smell, 2),
            "Visible_Mold_pct": np.round(mold, 2),
            "Surface_Damage_pct": np.round(damage, 2),
            TARGET_COLUMN: labels,
        }
    )

    # sanity check: every declared numeric feature column must exist
    missing = set(NUMERIC_FEATURES) - set(df.columns)
    assert not missing, f"Generated dataframe is missing expected columns: {missing}"

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate the FreshSense AI synthetic dataset.")
    parser.add_argument("--n", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.n < 10000:
        print(f"Note: requested {args.n} rows, which is below the 10,000-row target.")

    os.makedirs(DATASET_DIR, exist_ok=True)
    df = generate(args.n, args.seed)

    out_path = os.path.join(DATASET_DIR, DATASET_FILENAME)
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df):,} rows -> {out_path}")
    print("\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print("\nRows per produce type:")
    print(df["Produce"].value_counts())


if __name__ == "__main__":
    main()
