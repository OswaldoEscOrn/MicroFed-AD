# Edge-Machine-Learning-Models/federated/make_labels_100k_from_original.py

import os
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_labels", type=str, required=True)
    parser.add_argument("--x100k", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    y_orig = np.load(args.original_labels)
    X100 = np.load(args.x100k)

    n_orig = len(y_orig)
    n100 = X100.shape[0]

    y100 = np.zeros(n100, dtype=np.int64)

    for i in range(n100):
        y100[i] = y_orig[i % n_orig]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, y100)

    info = {
        "method": "cyclic_inheritance_from_original_window_labels",
        "original_labels": args.original_labels,
        "x100k": args.x100k,
        "output_path": args.output_path,
        "n_original_labels": int(n_orig),
        "n_100k_windows": int(n100),
        "original_anomaly_ratio": float(y_orig.mean()),
        "real2_anomaly_ratio": float(y100.mean()),
    }

    info_path = args.output_path.replace(".npy", "_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print("Saved:", args.output_path)
    print("Info:", info_path)
    print("Original labels:", y_orig.shape, "ratio=", y_orig.mean())
    print("100k labels:", y100.shape, "ratio=", y100.mean())


if __name__ == "__main__":
    main()
