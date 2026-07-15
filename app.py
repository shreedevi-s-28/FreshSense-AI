"""
app.py

FreshSense AI - Streamlit entry point.

This is the landing page. Streamlit auto-discovers the numbered
scripts under pages/ and adds them to the sidebar navigation:

    1_Dataset_Overview.py
    2_Train_Model.py
    3_Predict_Freshness.py

Run with:
    streamlit run app.py
"""

import os

import streamlit as st

from constants import (
    PRODUCE_LIST,
    TARGET_CLASSES,
    NUMERIC_FEATURES,
    DATASET_DIR,
    DATASET_FILENAME,
    MODEL_ARTIFACT_DIR,
    MODEL_WEIGHTS_FILENAME,
)

st.set_page_config(page_title="FreshSense AI", page_icon="🥦", layout="wide")

st.title("🥦 FreshSense AI")
st.subheader("Predicting produce freshness from storage conditions and quality indicators")

st.markdown(
    """
FreshSense AI is a neural network project that predicts whether a fruit or
vegetable is **Fresh**, should be **Consumed Soon**, or is **Spoiled**,
based on how it has been stored and what can be observed about it.

The dataset is synthetically generated using storage recommendations and
spoilage trends adapted from postharvest food-science guidance — see
`research.md` for the full background and references.
"""
)

col1, col2, col3 = st.columns(3)
col1.metric("Supported produce types", len(PRODUCE_LIST))
col2.metric("Input features", len(NUMERIC_FEATURES) + 1)
col3.metric("Target classes", len(TARGET_CLASSES))

st.divider()

dataset_ready = os.path.exists(os.path.join(DATASET_DIR, DATASET_FILENAME))
model_ready = os.path.exists(os.path.join(MODEL_ARTIFACT_DIR, MODEL_WEIGHTS_FILENAME))

st.markdown("### Project status")
status_col1, status_col2 = st.columns(2)

with status_col1:
    if dataset_ready:
        st.success("Dataset generated — ready to explore on the **Dataset Overview** page.")
    else:
        st.warning("No dataset found yet. Run `python generate_dataset.py` in the terminal, "
                    "then reopen this app.")

with status_col2:
    if model_ready:
        st.success("Model trained — ready to try on the **Predict Freshness** page.")
    else:
        st.info("No trained model yet. Visit **Train Model** in the sidebar to train one.")

st.divider()

with st.expander("Supported produce"):
    st.write(", ".join(PRODUCE_LIST))

with st.expander("Features used by the model"):
    st.write("**Produce type** (categorical) plus:")
    st.write(", ".join(NUMERIC_FEATURES))

st.markdown("Use the sidebar to navigate between **Dataset Overview**, **Train Model**, "
            "and **Predict Freshness**.")
