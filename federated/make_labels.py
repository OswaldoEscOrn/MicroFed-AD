# Edge-Machine-Learning-Models/federated/make_labels.py

import os
import json
import argparse
import numpy as np
import pandas as pd


def compute_hourly_anomaly_labels(df: pd.DataFrame):
    df = df.copy()

    # Air anomaly
    df["air_anomaly"] = df["avg_PM2.5"] > 75

    # Noise statistics
    duration_mean = df["total_noise_duration"].mean()
    duration_std = df["total_noise_duration"].std()
    event_mean = df["noise_event_count"].mean()
    event_std = df["noise_event_count"].std()

    high_event_threshold = event_mean + event_std
    high_salience_threshold = 1.5
    extreme_duration_threshold = duration_mean + 2 * duration_std
    medium_event_threshold = event_mean + 0.5 * event_std
    medium_duration_threshold = duration_mean + 0.5 * duration_std

    rule1 = (df["noise_event_count"] > high_event_threshold) & (df["avg_salience"] > high_salience_threshold)
    rule2 = df["total_noise_duration"] > extreme_duration_threshold
    rule3 = (df["noise_event_count"] > medium_event_threshold) & (df["total_noise_duration"] > medium_duration_threshold)
    rule4 = (df["avg_salience"] > high_salience_threshold) & (df["total_noise_duration"] > medium_duration_threshold)

    df["noise_anomaly"] = rule1 | rule2 | rule3 | rule4
    df["overall_anomaly"] = (df["air_anomaly"] | df["noise_anomaly"]).astype(int)

    config = {
        "pm25_threshold": 75,
        "high_salience_threshold": 1.5,
        "duration_mean": float(duration_mean),
        "duration_std": float(duration_std),
        "event_mean": float(event_mean),
        "event_std": float(event_std),
        "high_event_threshold": float(high_event_threshold),
        "extreme_duration_threshold": float(extreme_duration_threshold),
        "medium_event_threshold": float(medium_event_threshold),
        "medium_duration_threshold": float(medium_duration_threshold),
    }
    return df, config


def hourly_to_window_labels(hourly_labels, window_size=24, stride=1, horizon=1, min_anomaly_hours=3):
    labels = []
    for i in range(0, len(hourly_labels) - window_size - horizon + 1, stride):
        label_window = hourly_labels[i:i + window_size]
        labels.append(int(np.sum(label_window) >= min_anomaly_hours))
    return np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min_anomaly_hours", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    labeled_df, config = compute_hourly_anomaly_labels(df)

    hourly_labels = labeled_df["overall_anomaly"].astype(int).values
    window_labels = hourly_to_window_labels(
        hourly_labels,
        window_size=args.window_size,
        stride=args.stride,
        horizon=args.horizon,
        min_anomaly_hours=args.min_anomaly_hours,
    )

    np.save(os.path.join(args.output_dir, "hourly_anomaly_labels.npy"), hourly_labels)
    np.save(os.path.join(args.output_dir, "window_anomaly_labels.npy"), window_labels)
    labeled_df.to_csv(os.path.join(args.output_dir, "hourly_data_with_labels.csv"), index=False)

    info = {
        "window_size": args.window_size,
        "stride": args.stride,
        "horizon": args.horizon,
        "min_anomaly_hours": args.min_anomaly_hours,
        "hourly_anomaly_ratio": float(hourly_labels.mean()),
        "window_anomaly_ratio": float(window_labels.mean()),
        "n_hourly_samples": int(len(hourly_labels)),
        "n_window_samples": int(len(window_labels)),
        "rule_config": config,
    }

    with open(os.path.join(args.output_dir, "anomaly_label_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("Done.")
    print("hourly labels:", hourly_labels.shape, "ratio=", hourly_labels.mean())
    print("window labels:", window_labels.shape, "ratio=", window_labels.mean())


if __name__ == "__main__":
    main()