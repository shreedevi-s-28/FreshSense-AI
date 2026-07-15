"""
pages/2_Train_Model.py

Train the from-scratch NumPy neural network on the generated dataset,
show live loss/accuracy curves, and save the resulting weights so the
Predict Freshness page can load them.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import DATASET_DIR, DATASET_FILENAME, MODEL_ARTIFACT_DIR, MODEL_WEIGHTS_FILENAME
from utils.data_utils import (
    build_feature_matrix,
    labels_to_indices,
    load_dataset,
    train_test_split_arrays,
)
from utils.model_utils import FreshnessNeuralNetwork

st.set_page_config(page_title="Train Model | FreshSense AI", page_icon="🧠", layout="wide")
st.title("🧠 Train Model")

dataset_path = os.path.join(DATASET_DIR, DATASET_FILENAME)
if not os.path.exists(dataset_path):
    st.warning(
        "No dataset found. Generate one first by running:\n\n"
        "```\npython generate_dataset.py\n```\nfrom the project root, then reload this page."
    )
    st.stop()

st.markdown(
    "Trains the project's hand-implemented NumPy neural network "
    "(`utils/model_utils.py`) — a 2-hidden-layer MLP with ReLU activations "
    "and a softmax output, using mini-batch gradient descent with "
    "backpropagation written from scratch."
)

with st.sidebar:
    st.markdown("### Hyperparameters")
    epochs = st.slider("Epochs", 5, 200, 60, step=5)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128, 256], value=64)
    learning_rate = st.select_slider(
        "Learning rate", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.05
    )
    hidden_dim_1 = st.select_slider("Hidden layer 1 size", options=[8, 16, 32, 64], value=32)
    hidden_dim_2 = st.select_slider("Hidden layer 2 size", options=[8, 16, 32], value=16)
    test_fraction = st.slider("Validation split", 0.1, 0.4, 0.2, step=0.05)

if st.button("Train model", type="primary"):
    df = load_dataset()
    X, numeric_mean, numeric_std = build_feature_matrix(df)
    y = labels_to_indices(df["Freshness_Status"])
    X_train, X_val, y_train, y_val = train_test_split_arrays(X, y, test_fraction=test_fraction)

    model = FreshnessNeuralNetwork(
        input_dim=X.shape[1],
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        num_classes=3,
    )

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    chart_placeholder = st.empty()

    def on_epoch_end(epoch, total_epochs, history):
        progress_bar.progress(epoch / total_epochs)
        status_text.markdown(
            f"Epoch **{epoch}/{total_epochs}** — "
            f"train loss: `{history.train_loss[-1]:.4f}`, "
            f"train acc: `{history.train_accuracy[-1]:.3f}`, "
            f"val acc: `{history.val_accuracy[-1]:.3f}`"
        )
        chart_df = pd.DataFrame(
            {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
            }
        )
        chart_placeholder.line_chart(chart_df)

    history = model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        progress_callback=on_epoch_end,
    )

    os.makedirs(MODEL_ARTIFACT_DIR, exist_ok=True)
    weights_path = os.path.join(MODEL_ARTIFACT_DIR, MODEL_WEIGHTS_FILENAME)
    model.save(weights_path)

    scaler_path = os.path.join(MODEL_ARTIFACT_DIR, "feature_scaler.npz")
    np.savez(scaler_path, mean=numeric_mean, std=numeric_std)

    st.success(f"Training complete. Final validation accuracy: **{history.val_accuracy[-1]:.1%}**")

    st.markdown("#### Accuracy over training")
    st.line_chart(
        pd.DataFrame(
            {"train_accuracy": history.train_accuracy, "val_accuracy": history.val_accuracy}
        )
    )

    st.info("Weights saved — head to **Predict Freshness** in the sidebar to try the model.")
else:
    st.markdown("Adjust hyperparameters in the sidebar, then click **Train model**.")
