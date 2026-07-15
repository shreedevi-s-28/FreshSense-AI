"""
pages/3_Predict_Freshness.py

Interactive form: enter storage conditions and observed quality
indicators for a produce item, and get a Fresh / Consume Soon / Spoiled
prediction from the trained neural network, plus the model's class
probabilities.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import (
    MODEL_ARTIFACT_DIR,
    MODEL_WEIGHTS_FILENAME,
    PRODUCE_CONFIG,
    PRODUCE_LIST,
    TARGET_CLASSES,
)
from utils.data_utils import produce_to_onehot
from utils.model_utils import FreshnessNeuralNetwork

st.set_page_config(page_title="Predict Freshness | FreshSense AI", page_icon="🔍", layout="wide")
st.title("🔍 Predict Freshness")

weights_path = os.path.join(MODEL_ARTIFACT_DIR, MODEL_WEIGHTS_FILENAME)
scaler_path = os.path.join(MODEL_ARTIFACT_DIR, "feature_scaler.npz")

if not (os.path.exists(weights_path) and os.path.exists(scaler_path)):
    st.warning(
        "No trained model found yet. Go to **Train Model** in the sidebar and train one first."
    )
    st.stop()

model = FreshnessNeuralNetwork.load(weights_path)
scaler = np.load(scaler_path)
numeric_mean, numeric_std = scaler["mean"], scaler["std"]

st.markdown("Enter the storage conditions and observable indicators for a produce item.")

col1, col2 = st.columns(2)

with col1:
    produce = st.selectbox("Produce", PRODUCE_LIST)
    cfg = PRODUCE_CONFIG[produce]
    st.caption(
        f"Recommended storage: {cfg['optimal_temp_c'][0]}–{cfg['optimal_temp_c'][1]} °C, "
        f"{cfg['optimal_humidity_pct'][0]}–{cfg['optimal_humidity_pct'][1]}% humidity, "
        f"~{cfg['shelf_life_days']} day typical shelf life at those conditions."
    )
    storage_temp = st.slider("Storage Temperature (°C)", -5.0, 40.0, float(sum(cfg["optimal_temp_c"]) / 2), 0.5)
    humidity = st.slider("Relative Humidity (%)", 20.0, 100.0, float(sum(cfg["optimal_humidity_pct"]) / 2), 1.0)
    duration = st.slider("Storage Duration (Days)", 0.0, float(cfg["shelf_life_days"]) * 2, float(cfg["shelf_life_days"]) * 0.3, 0.5)

with col2:
    firmness = st.slider("Firmness Score (0-10)", 0.0, 10.0, 7.0, 0.1)
    color = st.slider("Color Score (0-10)", 0.0, 10.0, 7.0, 0.1)
    smell = st.slider("Smell Score (0-10)", 0.0, 10.0, 7.0, 0.1)
    mold = st.slider("Visible Mold (%)", 0.0, 100.0, 0.0, 1.0)
    damage = st.slider("Surface Damage (%)", 0.0, 100.0, 5.0, 1.0)

if st.button("Predict freshness", type="primary"):
    row = pd.DataFrame(
        [
            {
                "Produce": produce,
                "Storage_Temperature_C": storage_temp,
                "Relative_Humidity_pct": humidity,
                "Storage_Duration_Days": duration,
                "Firmness_Score": firmness,
                "Color_Score": color,
                "Smell_Score": smell,
                "Visible_Mold_pct": mold,
                "Surface_Damage_pct": damage,
            }
        ]
    )

    onehot = produce_to_onehot(row["Produce"])
    numeric = row[
        [
            "Storage_Temperature_C",
            "Relative_Humidity_pct",
            "Storage_Duration_Days",
            "Firmness_Score",
            "Color_Score",
            "Smell_Score",
            "Visible_Mold_pct",
            "Surface_Damage_pct",
        ]
    ].to_numpy(dtype=np.float64)
    numeric_scaled = (numeric - numeric_mean) / numeric_std
    X = np.hstack([onehot, numeric_scaled])

    probs = model.predict_proba(X)[0]
    predicted_class = TARGET_CLASSES[int(np.argmax(probs))]

    st.divider()
    st.markdown("### Result")

    result_color = {"Fresh": "green", "Consume Soon": "orange", "Spoiled": "red"}[predicted_class]
    st.markdown(f"## :{result_color}[{predicted_class}]")

    prob_df = pd.DataFrame({"Class": TARGET_CLASSES, "Probability": probs}).set_index("Class")
    st.bar_chart(prob_df)

    st.caption(
        "Prediction from the project's from-scratch NumPy neural network, trained on the "
        "synthetic FreshSense dataset."
    )
