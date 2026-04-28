# Edge-Machine-Learning-Models/federated/metrics.py

import numpy as np
from sklearn.metrics import f1_score


def reconstruction_scores(model, X):
    recon = model.predict(X, verbose=0)
    scores = np.mean(np.abs(X - recon), axis=(1, 2))
    return scores


def choose_threshold_from_validation(model, X_val, percentile=95):
    scores = reconstruction_scores(model, X_val)
    return float(np.percentile(scores, percentile))


def evaluate_f1(model, X_test, y_test, threshold):
    scores = reconstruction_scores(model, X_test)
    y_pred = (scores > threshold).astype(int)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    return {
        "f1": float(f1),
        "threshold": float(threshold),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }


def partition_coefficient(y, client_indices):
    pcs = []
    classes = np.unique(y)

    for idx in client_indices:
        if len(idx) == 0:
            continue
        y_client = y[idx]
        probs = [np.mean(y_client == c) for c in classes]
        pc = np.sum(np.square(probs))
        pcs.append(pc)

    if len(pcs) == 0:
        return {"pc_mean": 0.0, "pc_std": 0.0, "pc_list": []}

    return {
        "pc_mean": float(np.mean(pcs)),
        "pc_std": float(np.std(pcs)),
        "pc_list": [float(x) for x in pcs],
    }