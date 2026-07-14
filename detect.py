"""
LAYER 2: ML-BASED DETECTION ENGINE
-----------------------------------
Takes the simulated dataset from Layer 1 and trains machine learning
classifiers to detect and categorize network traffic into BENIGN vs
attack types. Three classifiers are trained and compared, which is
standard practice in IDS research papers (a comparison table is exactly
the kind of result reviewers/supervisors expect):

  - Decision Tree
  - Random Forest
  - Multi-Layer Perceptron (small neural network)

Output (used by Layer 3):
  - outputs/model_results.pkl  -> predictions + true labels per model
  - outputs/label_encoder.pkl  -> to decode class names later
"""

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "outputs", "simulated_cicids_dataset.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "model_results.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "outputs", "label_encoder.pkl")

RANDOM_STATE = 42


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Label"])
    y_raw = df["Label"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, encoder


def train_models(X_train, y_train):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=RANDOM_STATE, max_depth=15
        ),
        "MLP (Neural Network)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=RANDOM_STATE,
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def run_detection_engine():
    X_train, X_test, y_train, y_test, encoder = load_and_prepare_data()
    trained_models = train_models(X_train, y_train)

    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "y_true": y_test,
            "y_pred": y_pred,
        }

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    print(f"\nSaved model results to {RESULTS_PATH}")
    print(f"Saved label encoder to {ENCODER_PATH}")
    return results, encoder


if __name__ == "__main__":
    run_detection_engine()
