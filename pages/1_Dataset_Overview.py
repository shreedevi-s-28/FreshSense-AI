"""
pages/1_Dataset_Overview.py

Explore the generated synthetic dataset: class balance, per-produce
counts, and feature distributions by freshness class.
"""

import os
import sys

import pandas as pd
import streamlit as st

# allow importing constants / utils from the project root when Streamlit
# runs this file directly as a page
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import DATASET_DIR, DATASET_FILENAME, NUMERIC_FEATURES, TARGET_COLUMN
from utils.data_utils import load_dataset

st.set_page_config(page_title="Dataset Overview | FreshSense AI", page_icon="📊", layout="wide")
st.title("📊 Dataset Overview")

dataset_path = os.path.join(DATASET_DIR, DATASET_FILENAME)

if not os.path.exists(dataset_path):
    st.warning(
        "No dataset found. Generate one first by running:\n\n"
        "```\npython generate_dataset.py\n```\nfrom the project root, then reload this page."
    )
    st.stop()

df = load_dataset()

st.markdown(f"**{len(df):,} rows** loaded from `{dataset_path}`.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Class distribution")
    st.bar_chart(df[TARGET_COLUMN].value_counts())

with col2:
    st.markdown("#### Rows per produce type")
    st.bar_chart(df["Produce"].value_counts())

st.divider()

st.markdown("#### Feature distribution by freshness class")
feature_choice = st.selectbox("Feature", NUMERIC_FEATURES, index=0)

pivot = df.pivot_table(index=pd.cut(df[feature_choice], bins=15), columns=TARGET_COLUMN, values="Produce", aggfunc="count", observed=False).fillna(0)
st.bar_chart(pivot)

st.divider()

st.markdown("#### Filter and inspect rows")
produce_filter = st.multiselect("Produce", sorted(df["Produce"].unique()))
class_filter = st.multiselect("Freshness status", sorted(df[TARGET_COLUMN].unique()))

filtered = df.copy()
if produce_filter:
    filtered = filtered[filtered["Produce"].isin(produce_filter)]
if class_filter:
    filtered = filtered[filtered[TARGET_COLUMN].isin(class_filter)]

st.dataframe(filtered.head(500), use_container_width=True)
st.caption(f"Showing up to 500 of {len(filtered):,} matching rows.")

st.divider()
st.markdown("#### Summary statistics")
st.dataframe(df[NUMERIC_FEATURES].describe().T, use_container_width=True)
