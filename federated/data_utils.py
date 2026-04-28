# Edge-Machine-Learning-Models/federated/data_utils.py

import os
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(variant: str, project_root: str):
    candidate_x = []
    candidate_y = []

    if variant == "real":
        candidate_x = [
            os.path.join(project_root, "Data-Preprocessing/preprocessed_data/X_windows.npy"),
            os.path.join(project_root, "Edge-Machine-Learning-Models/X_windows.npy"),
            os.path.join(project_root, "Edge-Machine-Learning-Models/preprocessed_data/X_windows.npy"),
            os.path.join(project_root, "preprocessed_data/X_windows.npy"),
        ]
        candidate_y = [
            os.path.join(project_root, "Data-Preprocessing/preprocessed_data/window_anomaly_labels.npy"),
            os.path.join(project_root, "Edge-Machine-Learning-Models/window_anomaly_labels.npy"),
            os.path.join(project_root, "Edge-Machine-Learning-Models/preprocessed_data/window_anomaly_labels.npy"),
            os.path.join(project_root, "preprocessed_data/window_anomaly_labels.npy"),
        ]

    elif variant == "real2":
        candidate_x = [
            os.path.join(project_root, "Data-Preprocessing/preprocessed_data/X_windows_100k.npy"),
            os.path.join(project_root, "Edge-Machine-Learning-Models/X_windows_100k.npy"),
            os.path.join(project_root, "Edge-Machine-Learning-Models/preprocessed_data/X_windows_100k.npy"),
            os.path.join(project_root, "preprocessed_data/X_windows_100k.npy"),
        ]
        candidate_y = [
            os.path.join(project_root, "Data-Preprocessing/preprocessed_data/window_anomaly_labels_100k.npy"),
            os.path.join(project_root, "Edge-Machine-Learning-Models/window_anomaly_labels_100k.npy"),
            os.path.join(project_root, "Edge-Machine-Learning-Models/preprocessed_data/window_anomaly_labels_100k.npy"),
            os.path.join(project_root, "preprocessed_data/window_anomaly_labels_100k.npy"),
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    x_path = next((p for p in candidate_x if os.path.exists(p)), None)
    y_path = next((p for p in candidate_y if os.path.exists(p)), None)

    if x_path is None:
        raise FileNotFoundError(f"Cannot find X file for variant={variant}. Tried: {candidate_x}")

    if y_path is None:
        raise FileNotFoundError(f"Cannot find y file for variant={variant}. Tried: {candidate_y}")

    X = np.load(x_path)
    y = np.load(y_path)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y size mismatch: X={X.shape}, y={y.shape}")

    return X.astype("float32"), y.astype("int64"), x_path, y_path


def adapt_x_for_model(X, model_name):
    model_name = model_name.lower()

    if model_name in ["deep_ae", "deepae", "deep-ae"]:
        if len(X.shape) == 3:
            n = X.shape[0]
            X = X.reshape(n, -1)
        return X.astype("float32")

    if model_name in [
        "conv_ae", "conv_vae", "lstm_ae",
        "convae", "convvae", "lstmae",
        "conv-ae", "conv-vae", "lstm-ae",
    ]:
        return X.astype("float32")

    raise ValueError(f"Unknown model_name: {model_name}")


def split_global_data(X, y, seed=2024):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.1,
        random_state=seed,
        stratify=y_trainval,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def dirichlet_partition(y, num_clients, alpha=0.5, seed=2024, min_size=10):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    while True:
        client_indices = [[] for _ in range(num_clients)]

        for c in classes:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions)[:-1] * len(idx_c)).astype(int)
            splits = np.split(idx_c, split_points)

            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        client_indices = [np.array(idx, dtype=np.int64) for idx in client_indices]
        sizes = [len(idx) for idx in client_indices]

        if min(sizes) >= min_size:
            for i in range(num_clients):
                rng.shuffle(client_indices[i])
            return client_indices


def uniform_partition(y, num_clients, seed=2024, shuffle=True):
    """
    Uniform IID partition.
    """
    rng = np.random.default_rng(seed)

    indices = np.arange(len(y))
    if shuffle:
        rng.shuffle(indices)

    client_indices = np.array_split(indices, num_clients)
    client_indices = [idx.astype(np.int64) for idx in client_indices]

    return client_indices


def make_partition(y, num_clients, partition="dirichlet", alpha=0.5, seed=2024):
    partition = partition.lower()

    if partition in ["dirichlet", "dir", "non-iid", "noniid"]:
        return dirichlet_partition(
            y,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
        )

    if partition in ["uniform", "iid"]:
        return uniform_partition(
            y,
            num_clients=num_clients,
            seed=seed,
            shuffle=True,
        )

    raise ValueError(f"Unknown partition: {partition}")


def make_client_datasets(X_train, y_train, client_indices):
    client_data = []
    for idx in client_indices:
        client_data.append((X_train[idx], y_train[idx]))
    return client_data