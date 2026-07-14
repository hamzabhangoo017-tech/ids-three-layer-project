"""
WEB DASHBOARD (Streamlit) FOR THE THREE-LAYERED IDS
-----------------------------------------------------
A lightweight interactive front-end over the existing 3-layer pipeline.
It does NOT duplicate any logic -- it simply calls the same functions
from simulate_data.py, detect.py, and evaluate.py, then displays the
results in a browser.

Run with:
    streamlit run app.py

This will open a local web page (usually http://localhost:8501) where
you (or your client) can:
  1. Generate the simulated attack dataset with one click
  2. Train the 3 ML models with one click
  3. View confusion matrices, metrics table, and comparison chart
     right in the browser -- no need to touch the terminal.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from simulate_data import generate_dataset
from detect import run_detection_engine
from evaluate import run_evaluation

BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

st.set_page_config(
    page_title="Three-Layered IDS Dashboard",
    layout="wide",
)

st.title("Three-Layered ML-Based Intrusion Detection System")
st.caption("Data Simulation -> ML Detection Engine -> Results & Evaluation")

# Keep track of pipeline progress across button clicks
if "dataset_ready" not in st.session_state:
    st.session_state.dataset_ready = os.path.exists(os.path.join(OUT_DIR, "simulated_cicids_dataset.csv"))
if "models_ready" not in st.session_state:
    st.session_state.models_ready = os.path.exists(os.path.join(OUT_DIR, "model_results.pkl"))

tab1, tab2, tab3 = st.tabs(["1. Data Simulation", "2. ML Detection", "3. Results & Evaluation"])

# ---------------- TAB 1: DATA SIMULATION ----------------
with tab1:
    st.header("Layer 1: Data Simulation")
    st.write(
        "Generates simulated network traffic following the CICIDS2017/2018 "
        "flow-based feature schema, across 5 classes: BENIGN, DoS, PortScan, "
        "BruteForce, and Botnet."
    )

    if st.button("Generate Simulated Dataset", type="primary"):
        with st.spinner("Simulating network traffic..."):
            df = generate_dataset()
            df.to_csv(os.path.join(OUT_DIR, "simulated_cicids_dataset.csv"), index=False)
            st.session_state.dataset_ready = True
        st.success(f"Generated {len(df)} rows across {df['Label'].nunique()} classes.")

    if st.session_state.dataset_ready:
        df = pd.read_csv(os.path.join(OUT_DIR, "simulated_cicids_dataset.csv"))
        st.subheader("Dataset preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Class distribution")
        st.bar_chart(df["Label"].value_counts())
    else:
        st.info("Click the button above to generate the dataset.")

# ---------------- TAB 2: ML DETECTION ----------------
with tab2:
    st.header("Layer 2: ML-Based Detection Engine")
    st.write(
        "Trains and compares three classifiers: Decision Tree, Random Forest, "
        "and a Multi-Layer Perceptron (small neural network)."
    )

    if not st.session_state.dataset_ready:
        st.warning("Generate the dataset in Tab 1 first.")
    else:
        if st.button("Train Models", type="primary"):
            with st.spinner("Training Decision Tree, Random Forest, and MLP... this can take a moment"):
                run_detection_engine()
                st.session_state.models_ready = True
            st.success("All 3 models trained successfully.")

        if st.session_state.models_ready:
            st.info("Models trained. Go to Tab 3 to view results.")

# ---------------- TAB 3: RESULTS & EVALUATION ----------------
with tab3:
    st.header("Layer 3: Results & Evaluation")

    if not st.session_state.models_ready:
        st.warning("Train the models in Tab 2 first.")
    else:
        if st.button("Generate Results", type="primary"):
            with st.spinner("Computing metrics, confusion matrices, and charts..."):
                summary_df = run_evaluation()
            st.session_state.summary_df = summary_df

        summary_path = os.path.join(OUT_DIR, "metrics_summary.csv")
        if os.path.exists(summary_path):
            st.subheader("Model comparison metrics")
            summary_df = pd.read_csv(summary_path)
            st.dataframe(summary_df, use_container_width=True)

            comparison_img = os.path.join(OUT_DIR, "model_comparison.png")
            if os.path.exists(comparison_img):
                st.subheader("Model comparison chart")
                st.image(comparison_img, use_container_width=True)

            st.subheader("Confusion matrices")
            col1, col2, col3 = st.columns(3)
            model_files = {
                "Decision Tree": "confusion_matrix_Decision_Tree.png",
                "Random Forest": "confusion_matrix_Random_Forest.png",
                "MLP (Neural Network)": "confusion_matrix_MLP_Neural_Network.png",
            }
            for col, (name, fname) in zip([col1, col2, col3], model_files.items()):
                path = os.path.join(OUT_DIR, fname)
                if os.path.exists(path):
                    with col:
                        st.caption(name)
                        st.image(path, use_container_width=True)
        else:
            st.info("Click 'Generate Results' above to compute evaluation metrics.")

st.divider()
st.caption(
    "Note: The dataset used is a simulated dataset modeled on the CICIDS2017/2018 "
    "feature schema, not the raw downloaded CICIDS2017/2018 dataset."
)
