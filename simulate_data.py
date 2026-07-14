"""
LAYER 1: DATA SIMULATION LAYER
------------------------------
Generates a simulated network traffic dataset that follows the same
feature schema used by the well-known CICIDS2017/2018 intrusion
detection benchmark datasets (flow-based statistical features).

This is a SIMULATED dataset (not the real multi-GB CICIDS download) --
each class (BENIGN + 4 attack types) is generated from a distinct
statistical distribution so the ML layer has genuine, learnable
patterns to detect, rather than random noise. This is a legitimate and
common approach for coursework/small research projects where using the
full real dataset is impractical.

Classes simulated:
  - BENIGN      : normal traffic
  - DoS         : Denial of Service (high packet rate, short flows)
  - PortScan    : many short flows, small packet sizes, many distinct ports
  - BruteForce  : repeated short connections, low byte volume
  - Botnet      : periodic beaconing pattern, low variance in timing

Output: outputs/simulated_cicids_dataset.csv
"""

import numpy as np
import pandas as pd
import os

RNG_SEED = 42
N_PER_CLASS = 2000  # rows per class -> 10,000 total rows

np.random.seed(RNG_SEED)

FEATURE_COLUMNS = [
    "Flow_Duration",
    "Total_Fwd_Packets",
    "Total_Bwd_Packets",
    "Total_Length_Fwd_Packets",
    "Total_Length_Bwd_Packets",
    "Flow_Bytes_s",
    "Flow_Packets_s",
    "Fwd_Packet_Length_Mean",
    "Bwd_Packet_Length_Mean",
    "Flow_IAT_Mean",
    "Flow_IAT_Std",
    "SYN_Flag_Count",
    "ACK_Flag_Count",
    "Packet_Length_Variance",
]


def _clip_positive(arr):
    return np.clip(arr, 0, None)


LABEL_NOISE_FRACTION = 0.06  # fraction of rows given a randomly wrong label
FEATURE_NOISE_STD_FRAC = 0.35  # extra gaussian noise as a fraction of each feature's std


def simulate_benign(n):
    return {
        "Flow_Duration": np.random.normal(500000, 150000, n),
        "Total_Fwd_Packets": np.random.normal(20, 6, n),
        "Total_Bwd_Packets": np.random.normal(18, 5, n),
        "Total_Length_Fwd_Packets": np.random.normal(1500, 400, n),
        "Total_Length_Bwd_Packets": np.random.normal(1400, 400, n),
        "Flow_Bytes_s": np.random.normal(3000, 800, n),
        "Flow_Packets_s": np.random.normal(40, 10, n),
        "Fwd_Packet_Length_Mean": np.random.normal(75, 15, n),
        "Bwd_Packet_Length_Mean": np.random.normal(75, 15, n),
        "Flow_IAT_Mean": np.random.normal(25000, 6000, n),
        "Flow_IAT_Std": np.random.normal(8000, 2000, n),
        "SYN_Flag_Count": np.random.poisson(1, n),
        "ACK_Flag_Count": np.random.poisson(15, n),
        "Packet_Length_Variance": np.random.normal(200, 60, n),
    }


def simulate_dos(n):
    return {
        "Flow_Duration": np.random.normal(20000, 8000, n),
        "Total_Fwd_Packets": np.random.normal(300, 80, n),
        "Total_Bwd_Packets": np.random.normal(5, 3, n),
        "Total_Length_Fwd_Packets": np.random.normal(9000, 2000, n),
        "Total_Length_Bwd_Packets": np.random.normal(150, 60, n),
        "Flow_Bytes_s": np.random.normal(90000, 20000, n),
        "Flow_Packets_s": np.random.normal(800, 200, n),
        "Fwd_Packet_Length_Mean": np.random.normal(30, 8, n),
        "Bwd_Packet_Length_Mean": np.random.normal(30, 10, n),
        "Flow_IAT_Mean": np.random.normal(300, 100, n),
        "Flow_IAT_Std": np.random.normal(100, 40, n),
        "SYN_Flag_Count": np.random.poisson(120, n),
        "ACK_Flag_Count": np.random.poisson(2, n),
        "Packet_Length_Variance": np.random.normal(40, 15, n),
    }


def simulate_portscan(n):
    return {
        "Flow_Duration": np.random.normal(1500, 600, n),
        "Total_Fwd_Packets": np.random.normal(3, 1, n),
        "Total_Bwd_Packets": np.random.normal(1, 1, n),
        "Total_Length_Fwd_Packets": np.random.normal(120, 40, n),
        "Total_Length_Bwd_Packets": np.random.normal(20, 15, n),
        "Flow_Bytes_s": np.random.normal(500, 200, n),
        "Flow_Packets_s": np.random.normal(300, 100, n),
        "Fwd_Packet_Length_Mean": np.random.normal(40, 10, n),
        "Bwd_Packet_Length_Mean": np.random.normal(20, 10, n),
        "Flow_IAT_Mean": np.random.normal(150, 60, n),
        "Flow_IAT_Std": np.random.normal(50, 20, n),
        "SYN_Flag_Count": np.random.poisson(2, n),
        "ACK_Flag_Count": np.random.poisson(0.5, n),
        "Packet_Length_Variance": np.random.normal(15, 8, n),
    }


def simulate_bruteforce(n):
    return {
        "Flow_Duration": np.random.normal(8000, 2500, n),
        "Total_Fwd_Packets": np.random.normal(12, 4, n),
        "Total_Bwd_Packets": np.random.normal(10, 4, n),
        "Total_Length_Fwd_Packets": np.random.normal(600, 150, n),
        "Total_Length_Bwd_Packets": np.random.normal(500, 150, n),
        "Flow_Bytes_s": np.random.normal(6000, 1500, n),
        "Flow_Packets_s": np.random.normal(120, 30, n),
        "Fwd_Packet_Length_Mean": np.random.normal(55, 12, n),
        "Bwd_Packet_Length_Mean": np.random.normal(50, 12, n),
        "Flow_IAT_Mean": np.random.normal(700, 200, n),
        "Flow_IAT_Std": np.random.normal(200, 70, n),
        "SYN_Flag_Count": np.random.poisson(8, n),
        "ACK_Flag_Count": np.random.poisson(6, n),
        "Packet_Length_Variance": np.random.normal(80, 25, n),
    }


def simulate_botnet(n):
    return {
        "Flow_Duration": np.random.normal(600000, 50000, n),  # very regular
        "Total_Fwd_Packets": np.random.normal(6, 1.5, n),
        "Total_Bwd_Packets": np.random.normal(6, 1.5, n),
        "Total_Length_Fwd_Packets": np.random.normal(300, 60, n),
        "Total_Length_Bwd_Packets": np.random.normal(300, 60, n),
        "Flow_Bytes_s": np.random.normal(500, 100, n),
        "Flow_Packets_s": np.random.normal(10, 3, n),
        "Fwd_Packet_Length_Mean": np.random.normal(50, 8, n),
        "Bwd_Packet_Length_Mean": np.random.normal(50, 8, n),
        "Flow_IAT_Mean": np.random.normal(60000, 3000, n),  # low variance = periodic beaconing
        "Flow_IAT_Std": np.random.normal(2000, 500, n),
        "SYN_Flag_Count": np.random.poisson(1, n),
        "ACK_Flag_Count": np.random.poisson(5, n),
        "Packet_Length_Variance": np.random.normal(30, 10, n),
    }


def generate_dataset(n_per_class=N_PER_CLASS, seed=RNG_SEED):
    np.random.seed(seed)
    generators = {
        "BENIGN": simulate_benign,
        "DoS": simulate_dos,
        "PortScan": simulate_portscan,
        "BruteForce": simulate_bruteforce,
        "Botnet": simulate_botnet,
    }

    frames = []
    for label, gen_fn in generators.items():
        data = gen_fn(n_per_class)
        df = pd.DataFrame(data)
        df = df.apply(_clip_positive)  # network features can't be negative
        df["Label"] = label
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)

    # Add extra per-feature gaussian noise so class boundaries overlap
    # realistically (real network traffic is never perfectly separable).
    feature_cols = [c for c in full_df.columns if c != "Label"]
    for col in feature_cols:
        col_std = full_df[col].std()
        noise = np.random.normal(0, col_std * FEATURE_NOISE_STD_FRAC, len(full_df))
        full_df[col] = _clip_positive(full_df[col] + noise)

    # Flip a small fraction of labels to a random different class, simulating
    # real-world label noise / ambiguous flows / measurement error.
    n_flip = int(len(full_df) * LABEL_NOISE_FRACTION)
    flip_idx = np.random.choice(full_df.index, size=n_flip, replace=False)
    all_labels = list(generators.keys())
    for idx in flip_idx:
        current = full_df.loc[idx, "Label"]
        other_labels = [l for l in all_labels if l != current]
        full_df.loc[idx, "Label"] = np.random.choice(other_labels)

    full_df = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    return full_df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "outputs", "simulated_cicids_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows across {df['Label'].nunique()} classes.")
    print(df["Label"].value_counts())
    print(f"Saved to {out_path}")
