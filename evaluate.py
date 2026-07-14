"""
LAYER 3: RESULTS & EVALUATION LAYER
------------------------------------
Takes the predictions from Layer 2 and produces the analysis a research
paper needs:
  - Confusion matrix (per model)
  - Accuracy, Precision, Recall, F1-score (per model, per class)
  - A comparison bar chart across all 3 models
  - A metrics_summary.csv you can drop straight into a paper's results
    table

Outputs (all in outputs/):
  - confusion_matrix_<model>.png   (one heatmap per model)
  - model_comparison.png            (accuracy/precision/recall/F1 bar chart)
  - metrics_summary.csv             (numeric table for the paper)
  - classification_report_<model>.txt
"""

import pickle
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "model_results.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "outputs", "label_encoder.pkl")
OUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_results():
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return results, encoder


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_path = os.path.join(OUT_DIR, f"confusion_matrix_{safe_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def compute_summary_metrics(results, class_names):
    rows = []
    for model_name, data in results.items():
        y_true, y_pred = data["y_true"], data["y_pred"]
        rows.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1-score (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        })

        # Save full per-class classification report as text too
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        with open(os.path.join(OUT_DIR, f"classification_report_{safe_name}.txt"), "w") as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(report)

    return pd.DataFrame(rows)


def plot_model_comparison(summary_df):
    metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]
    df_melt = summary_df.melt(id_vars="Model", value_vars=metrics,
                               var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 6))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1.05)
    plt.title("Model Comparison Across Metrics")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def run_evaluation():
    results, encoder = load_results()
    class_names = list(encoder.classes_)

    for model_name, data in results.items():
        cm_path = plot_confusion_matrix(data["y_true"], data["y_pred"], class_names, model_name)
        print(f"Saved: {cm_path}")

    summary_df = compute_summary_metrics(results, class_names)
    summary_csv_path = os.path.join(OUT_DIR, "metrics_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved: {summary_csv_path}")
    print("\n" + summary_df.to_string(index=False))

    comparison_path = plot_model_comparison(summary_df)
    print(f"Saved: {comparison_path}")

    return summary_df


if __name__ == "__main__":
    run_evaluation()
