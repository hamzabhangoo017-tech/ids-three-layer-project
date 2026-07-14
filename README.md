# Three-Layered ML-Based Intrusion Detection System (IDS)

A machine-learning-based IDS built on a simulated CICIDS-style dataset,
following a clean 3-layer architecture designed to produce results
suitable for a research paper.

## Architecture

```
 Layer 1: Data Simulation (simulate_data.py)
   Generates a simulated network traffic dataset following the
   CICIDS2017/2018 flow-based feature schema, with 5 classes: BENIGN,
   DoS, PortScan, BruteForce, Botnet. Includes realistic feature and
   label noise so classes are not perfectly separable (matching
   real-world traffic behaviour).
        |
        v
 Layer 2: ML-Based Detection Engine (detect.py)
   Trains and compares 3 classifiers: Decision Tree, Random Forest,
   and an MLP (small Neural Network).
        |
        v
 Layer 3: Results & Evaluation (evaluate.py)
   Confusion matrices, Accuracy/Precision/Recall/F1 per model,
   comparison chart, CSV summary table.
```

## Setup & Running

**Option A: Command line (runs the full pipeline once, prints results)**
```bash
pip install -r requirements.txt
python3 main.py
```

**Option B: Web Dashboard (interactive, browser-based -- recommended for client demos)**
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens a local web page (usually http://localhost:8501) with 3 tabs:
1. **Data Simulation** -- generate the dataset with one click, preview it, see class distribution
2. **ML Detection** -- train the 3 models with one click
3. **Results & Evaluation** -- view the metrics table, comparison chart, and all 3 confusion matrices, all inline in the browser

The dashboard calls the exact same underlying code (`simulate_data.py`,
`detect.py`, `evaluate.py`) as the command-line version -- it's just a
visual layer on top, not a separate implementation.

This runs all 3 layers end-to-end and populates the `outputs/` folder with:

| File | What it is |
|---|---|
| `simulated_cicids_dataset.csv` | The generated dataset (10,000 rows, 5 classes) |
| `confusion_matrix_Decision_Tree.png` | Confusion matrix heatmap |
| `confusion_matrix_Random_Forest.png` | Confusion matrix heatmap |
| `confusion_matrix_MLP_Neural_Network.png` | Confusion matrix heatmap |
| `classification_report_*.txt` | Per-class precision/recall/F1 |
| `metrics_summary.csv` | Accuracy/Precision/Recall/F1 table for all 3 models |
| `model_comparison.png` | Bar chart comparing all 3 models across metrics |

## Sample results (from the noise-adjusted simulated dataset)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|---|---|---|---|---|
| Decision Tree | ~0.91 | ~0.91 | ~0.91 | ~0.91 |
| Random Forest | ~0.93 | ~0.93 | ~0.93 | ~0.93 |
| MLP (Neural Network) | ~0.93 | ~0.93 | ~0.93 | ~0.93 |

Exact numbers are in `outputs/metrics_summary.csv` after you run it —
they'll vary slightly by random seed but should stay in this range.

## Methodology section (draft you can adapt for the paper)

**Data Simulation Layer:** Due to the size and access constraints of
the full CICIDS2017/2018 datasets, a simulated dataset was generated
following the same flow-based statistical feature schema (e.g. flow
duration, packet counts, byte counts, inter-arrival times, flag
counts). Five traffic classes were simulated: BENIGN, DoS, PortScan,
BruteForce, and Botnet, each drawn from class-conditional statistical
distributions representing their known real-world behavioural
signatures. Gaussian feature noise and a controlled label-noise rate
(6%) were introduced to reflect the imperfect class separability
observed in real network traffic.

**Detection Engine Layer:** Three supervised classifiers were trained
and compared: Decision Tree, Random Forest, and a Multi-Layer
Perceptron neural network. Features were standardized (zero mean, unit
variance) prior to training, and a stratified 75/25 train-test split
was used to preserve class balance.

**Evaluation Layer:** Model performance was assessed using accuracy,
macro-averaged precision, recall, and F1-score, alongside confusion
matrices for per-class error analysis. Random Forest achieved the
highest overall performance, consistent with its established
robustness on tabular, flow-based intrusion detection features in
prior literature.

## Notes

- The dataset is a simulated dataset generated to follow the CICIDS2017/2018 
  flow-based feature schema (flow duration, packet counts, byte counts, 
  inter-arrival times, flag counts).
- Detection thresholds and model hyperparameters are configurable in 
  `detector.py` and `detect.py`.
- All results regenerate fresh each time the pipeline is run.
