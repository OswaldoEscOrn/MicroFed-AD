# Edge-Machine-Learning-Models/federated/scripts/summarize_results.py

import os
import json
import argparse
import pandas as pd


def safe_get(d, path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        if p not in cur:
            return default
        cur = cur[p]
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    rows = []

    for fname in os.listdir(args.result_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(args.result_dir, fname)

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skip broken json: {path}, error={e}")
            continue

        row = {
            "file": fname,
            "model": data.get("model"),
            "variant": data.get("variant"),
            "method": data.get("method"),
            "partition_type": data.get("partition_type", data.get("partition")),
            "num_clients": data.get("num_clients"),
            "alpha": data.get("alpha"),
            "rounds_requested": data.get("rounds_requested"),
            "rounds_completed": data.get("rounds_completed"),
            "local_epochs": data.get("local_epochs"),
            "batch_size": data.get("batch_size"),
            "lr": data.get("lr"),
            "mu": data.get("mu"),
            "score": data.get("score"),
            "seed": data.get("seed"),
            "total_params": data.get("total_params"),
            "best_val_loss": data.get("best_val_loss"),

            "f1": safe_get(data, ["evaluation", "f1"]),
            "precision": safe_get(data, ["evaluation", "precision"]),
            "recall": safe_get(data, ["evaluation", "recall"]),
            "accuracy": safe_get(data, ["evaluation", "accuracy"]),
            "threshold": safe_get(data, ["evaluation", "threshold"]),
            "mean_score": safe_get(data, ["evaluation", "mean_score"]),
            "std_score": safe_get(data, ["evaluation", "std_score"]),
            "predicted_anomaly_ratio": safe_get(data, ["evaluation", "predicted_anomaly_ratio"]),

            "pc_mean": safe_get(data, ["partition", "pc_mean"]),
            "pc_std": safe_get(data, ["partition", "pc_std"]),

            "global_anomaly_ratio": safe_get(data, ["data", "global_anomaly_ratio"]),
            "train_anomaly_ratio": safe_get(data, ["data", "train_anomaly_ratio"]),
            "val_anomaly_ratio": safe_get(data, ["data", "val_anomaly_ratio"]),
            "test_anomaly_ratio": safe_get(data, ["data", "test_anomaly_ratio"]),
            "raw_X_shape": safe_get(data, ["data", "raw_X_shape"]),
            "adapted_X_shape": safe_get(data, ["data", "adapted_X_shape"]),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("No results found.")
        return

    sort_cols = [
        "model",
        "variant",
        "partition_type",
        "num_clients",
        "method",
        "seed",
    ]
    sort_cols = [c for c in sort_cols if c in df.columns]

    df = df.sort_values(sort_cols)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Saved summary CSV to: {args.output_csv}")
    print(f"Rows: {len(df)}")

    print("\nTop rows:")
    print(df.head())

    # Also save compact table
    compact_cols = [
        "model",
        "variant",
        "partition_type",
        "num_clients",
        "method",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "pc_mean",
        "pc_std",
        "best_val_loss",
        "rounds_completed",
    ]
    compact_cols = [c for c in compact_cols if c in df.columns]

    compact_path = args.output_csv.replace(".csv", "_compact.csv")
    df[compact_cols].to_csv(compact_path, index=False)
    print(f"Saved compact CSV to: {compact_path}")

    # Pivot F1 table
    try:
        pivot = df.pivot_table(
            index=["model", "variant", "partition_type", "num_clients"],
            columns="method",
            values="f1",
            aggfunc="mean",
        ).reset_index()

        pivot_path = args.output_csv.replace(".csv", "_f1_pivot.csv")
        pivot.to_csv(pivot_path, index=False)
        print(f"Saved F1 pivot CSV to: {pivot_path}")
    except Exception as e:
        print(f"Could not create pivot: {e}")


if __name__ == "__main__":
    main()