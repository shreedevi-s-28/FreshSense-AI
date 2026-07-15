"""
constants.py

Single source of truth for FreshSense AI.

Holds:
    - the list of supported produce
    - per-produce storage parameters (approximate postharvest
      guidelines synthesized from general food-science storage
      practice: optimal temperature/humidity bands, typical shelf
      life at optimal storage, and a relative spoilage-sensitivity
      multiplier used to drive the synthetic data generator)
    - feature / target schema shared by generate_dataset.py,
      utils/, and every page under pages/

Nothing in this file trains a model or generates data — it only
defines the numbers and names everything else builds on, so a
change here propagates everywhere consistently.
"""

# ---------------------------------------------------------------------------
# Produce catalogue
# ---------------------------------------------------------------------------

FRUITS = ["Apple", "Banana", "Orange", "Mango", "Grapes", "Tomato"]

VEGETABLES = [
    "Cucumber",
    "Carrot",
    "Spinach",
    "Potato",
    "Onion",
    "Bell Pepper",
    "Broccoli",
    "Lettuce",
    "Cauliflower",
]

PRODUCE_LIST = FRUITS + VEGETABLES

# ---------------------------------------------------------------------------
# Per-produce storage parameters
#
#   optimal_temp_c:      (min, max) recommended cold-chain temperature, °C
#   optimal_humidity_pct: (min, max) recommended relative humidity, %
#   shelf_life_days:      typical days to spoilage AT optimal storage
#   sensitivity:          relative multiplier for how fast the produce
#                          spoils once storage drifts outside the optimal
#                          band (chilling-sensitive / fast-respiring
#                          produce such as leafy greens and berries score
#                          higher; hardy roots/bulbs score lower)
# ---------------------------------------------------------------------------

PRODUCE_CONFIG = {
    "Apple":        {"optimal_temp_c": (0, 4),   "optimal_humidity_pct": (90, 95),  "shelf_life_days": 90,  "sensitivity": 1.0},
    "Banana":       {"optimal_temp_c": (13, 15), "optimal_humidity_pct": (90, 95),  "shelf_life_days": 14,  "sensitivity": 1.3},
    "Orange":       {"optimal_temp_c": (3, 9),   "optimal_humidity_pct": (85, 90),  "shelf_life_days": 60,  "sensitivity": 1.0},
    "Mango":        {"optimal_temp_c": (10, 13), "optimal_humidity_pct": (85, 90),  "shelf_life_days": 21,  "sensitivity": 1.2},
    "Grapes":       {"optimal_temp_c": (-1, 0),  "optimal_humidity_pct": (90, 95),  "shelf_life_days": 30,  "sensitivity": 1.1},
    "Tomato":       {"optimal_temp_c": (10, 13), "optimal_humidity_pct": (85, 90),  "shelf_life_days": 14,  "sensitivity": 1.3},
    "Cucumber":     {"optimal_temp_c": (10, 13), "optimal_humidity_pct": (90, 95),  "shelf_life_days": 14,  "sensitivity": 1.3},
    "Carrot":       {"optimal_temp_c": (0, 2),   "optimal_humidity_pct": (95, 100), "shelf_life_days": 120, "sensitivity": 0.8},
    "Spinach":      {"optimal_temp_c": (0, 2),   "optimal_humidity_pct": (95, 100), "shelf_life_days": 10,  "sensitivity": 1.6},
    "Potato":       {"optimal_temp_c": (4, 10),  "optimal_humidity_pct": (90, 95),  "shelf_life_days": 150, "sensitivity": 0.7},
    "Onion":        {"optimal_temp_c": (0, 4),   "optimal_humidity_pct": (65, 70),  "shelf_life_days": 180, "sensitivity": 0.6},
    "Bell Pepper":  {"optimal_temp_c": (7, 10),  "optimal_humidity_pct": (90, 95),  "shelf_life_days": 21,  "sensitivity": 1.1},
    "Broccoli":     {"optimal_temp_c": (0, 2),   "optimal_humidity_pct": (95, 100), "shelf_life_days": 14,  "sensitivity": 1.4},
    "Lettuce":      {"optimal_temp_c": (0, 2),   "optimal_humidity_pct": (95, 100), "shelf_life_days": 14,  "sensitivity": 1.5},
    "Cauliflower":  {"optimal_temp_c": (0, 2),   "optimal_humidity_pct": (95, 98),  "shelf_life_days": 21,  "sensitivity": 1.2},
}

# ---------------------------------------------------------------------------
# Feature / target schema
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = ["Produce"]

NUMERIC_FEATURES = [
    "Storage_Temperature_C",
    "Relative_Humidity_pct",
    "Storage_Duration_Days",
    "Firmness_Score",
    "Color_Score",
    "Smell_Score",
    "Visible_Mold_pct",
    "Surface_Damage_pct",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET_COLUMN = "Freshness_Status"

TARGET_CLASSES = ["Fresh", "Consume Soon", "Spoiled"]

# Spoilage-index thresholds used both by the generator (to assign labels)
# and by pages/3_Predict_Freshness.py (to explain a prediction)
SPOILAGE_INDEX_THRESHOLDS = {
    "fresh_max": 0.35,        # spoilage_index below this -> "Fresh"
    "consume_soon_max": 0.70, # spoilage_index below this -> "Consume Soon", above -> "Spoiled"
}

# ---------------------------------------------------------------------------
# Generation / reproducibility defaults
# ---------------------------------------------------------------------------

DEFAULT_NUM_SAMPLES = 12000
RANDOM_SEED = 42

DATASET_DIR = "dataset"
DATASET_FILENAME = "freshsense_dataset.csv"
MODEL_ARTIFACT_DIR = "dataset/artifacts"
MODEL_WEIGHTS_FILENAME = "freshsense_nn_weights.npz"
