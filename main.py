"""
THREE-LAYERED IDS (ML/DATASET-BASED) - MAIN ENTRY POINT
---------------------------------------------------------
Layer 1 (simulate_data.py) -> generates simulated CICIDS-style dataset
Layer 2 (detect.py)        -> trains & runs ML classifiers
Layer 3 (evaluate.py)      -> confusion matrices, metrics, comparison chart

Run:
    python3 main.py

All outputs (dataset, models' results, plots, metrics table) are saved
in the outputs/ folder.
"""

from simulate_data import generate_dataset
from detect import run_detection_engine
from evaluate import run_evaluation
import os

BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "outputs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print(" LAYER 1: DATA SIMULATION")
    print("=" * 60)
    df = generate_dataset()
    df.to_csv(os.path.join(OUT_DIR, "simulated_cicids_dataset.csv"), index=False)
    print(f"Generated {len(df)} rows, {df['Label'].nunique()} classes.")

    print("\n" + "=" * 60)
    print(" LAYER 2: ML-BASED DETECTION ENGINE")
    print("=" * 60)
    run_detection_engine()

    print("\n" + "=" * 60)
    print(" LAYER 3: RESULTS & EVALUATION")
    print("=" * 60)
    summary_df = run_evaluation()

    print("\n" + "=" * 60)
    print(" DONE - all outputs saved in outputs/")
    print("=" * 60)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
