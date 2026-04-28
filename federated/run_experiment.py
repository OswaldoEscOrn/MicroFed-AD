# Edge-Machine-Learning-Models/federated/run_experiment.py

import os
import json
import argparse
import numpy as np
import tensorflow as tf

from models import build_model
from client import FederatedClient
from data_utils import (
    load_dataset,
    adapt_x_for_model,
    split_global_data,
    make_partition,
    make_client_datasets,
)
from metrics import choose_threshold_from_validation, evaluate_f1, partition_coefficient
from strategies import FedAvgStrategy, FedProxStrategy, FedNAGStrategy


def get_strategy(method):
    if method == "FedAvg":
        return FedAvgStrategy()
    elif method == "FedProx":
        return FedProxStrategy()
    elif method == "FedNAG":
        return FedNAGStrategy(server_lr=1.0, momentum=0.9, min_server_lr=0.1)
    else:
        raise ValueError(f"Unsupported method: {method}")


def get_eval_loss(eval_result):
    if isinstance(eval_result, list):
        return float(eval_result[0])
    return float(eval_result)


def default_score_for_model(model_name):
    model_name = model_name.lower()
    if model_name in ["deep_ae", "deepae", "deep-ae"]:
        return "mse"
    return "mae"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=["conv_ae", "conv_vae", "deep_ae", "lstm_ae"],
        default="conv_ae",
    )
    parser.add_argument("--variant", type=str, choices=["real", "real2"], required=True)
    parser.add_argument("--method", type=str, choices=["FedAvg", "FedProx", "FedNAG"], required=True)

    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["dirichlet", "uniform"],)

    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=0.01)

    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--kl_weight", type=float, default=0.001)
    parser.add_argument("--vae_output_activation", type=str, default="sigmoid", choices=["sigmoid", "linear"])

    parser.add_argument("--deep_encoding_dim", type=int, default=32)

    parser.add_argument("--lstm_units", type=int, default=128)
    parser.add_argument("--lstm_encoding_dim", type=int, default=32)

    parser.add_argument("--score", type=str, default=None, choices=["mae", "mse"])

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--result_dir", type=str, default="results")

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    os.makedirs(args.result_dir, exist_ok=True)

    # ===================== Load Data =====================
    X_raw, y, x_path, y_path = load_dataset(args.variant, args.project_root)
    X = adapt_x_for_model(X_raw, args.model)

    if args.score is None:
        args.score = default_score_for_model(args.model)

    print(f"Loaded X from: {x_path}")
    print(f"Loaded y from: {y_path}")
    print(f"Raw X shape={X_raw.shape}")
    print(f"Adapted X shape={X.shape}")
    print(f"y shape={y.shape}, anomaly_ratio={y.mean():.4f}")
    print(f"X min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}, std={X.std():.4f}")
    print(f"Reconstruction score: {args.score}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_global_data(
        X,
        y,
        seed=args.seed,
    )

    # ===================== Partition =====================
    client_indices = make_partition(
        y_train,
        num_clients=args.num_clients,
        partition=args.partition,
        alpha=args.alpha,
        seed=args.seed,
    )
    client_data = make_client_datasets(X_train, y_train, client_indices)

    pc_stats = partition_coefficient(y_train, client_indices)
    print("Partition Coefficient:", pc_stats["pc_mean"], pc_stats["pc_std"])

    # ===================== Global Model =====================
    global_model = build_model(
        model_name=args.model,
        input_shape=X.shape[1:],
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        vae_output_activation=args.vae_output_activation,
        deep_encoding_dim=args.deep_encoding_dim,
        lstm_units=args.lstm_units,
        lstm_encoding_dim=args.lstm_encoding_dim,
    )

    global_weights = global_model.get_weights()
    total_params = global_model.count_params()

    print(f"Model: {args.model}, total params: {total_params:,}")

    strategy = get_strategy(args.method)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    history = []

    # ===================== Federated Training =====================
    for round_idx in range(1, args.rounds + 1):
        client_updates = []

        for client_id, (Xc, yc) in enumerate(client_data):
            client = FederatedClient(
                client_id=client_id,
                X_train=Xc,
                y_train=yc,
                model_name=args.model,
                input_shape=X.shape[1:],
                lr=args.lr,
                latent_dim=args.latent_dim,
                kl_weight=args.kl_weight,
                vae_output_activation=args.vae_output_activation,
                deep_encoding_dim=args.deep_encoding_dim,
                lstm_units=args.lstm_units,
                lstm_encoding_dim=args.lstm_encoding_dim,
            )

            if args.method == "FedAvg":
                new_weights, num_examples = client.fit_fedavg(
                    global_weights,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                )
            elif args.method == "FedProx":
                new_weights, num_examples = client.fit_fedprox(
                    global_weights,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    mu=args.mu,
                )
            elif args.method == "FedNAG":
                new_weights, num_examples = client.fit_fedavg(
                    global_weights,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                )
            else:
                raise ValueError(args.method)

            client_updates.append(
                {
                    "weights": new_weights,
                    "num_examples": num_examples,
                }
            )

            tf.keras.backend.clear_session()

        global_weights = strategy.aggregate(
            global_weights=global_weights,
            client_updates=client_updates,
            round_idx=round_idx,
            total_rounds=args.rounds,
        )

        global_model = build_model(
            model_name=args.model,
            input_shape=X.shape[1:],
            learning_rate=args.lr,
            latent_dim=args.latent_dim,
            kl_weight=args.kl_weight,
            vae_output_activation=args.vae_output_activation,
            deep_encoding_dim=args.deep_encoding_dim,
            lstm_units=args.lstm_units,
            lstm_encoding_dim=args.lstm_encoding_dim,
        )
        global_model.set_weights(global_weights)

        eval_result = global_model.evaluate(
            X_val,
            X_val,
            batch_size=args.batch_size,
            verbose=0,
        )
        val_loss = get_eval_loss(eval_result)

        round_info = {
            "round": round_idx,
            "val_loss": float(val_loss),
        }
        history.append(round_info)

        print(f"[Round {round_idx:03d}] val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [w.copy() for w in global_weights]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at round {round_idx}")
            break

    if best_weights is not None:
        global_model = build_model(
            model_name=args.model,
            input_shape=X.shape[1:],
            learning_rate=args.lr,
            latent_dim=args.latent_dim,
            kl_weight=args.kl_weight,
            vae_output_activation=args.vae_output_activation,
            deep_encoding_dim=args.deep_encoding_dim,
            lstm_units=args.lstm_units,
            lstm_encoding_dim=args.lstm_encoding_dim,
        )
        global_model.set_weights(best_weights)

    # ===================== Evaluation =====================
    threshold = choose_threshold_from_validation(
        global_model,
        X_val,
        percentile=95,
        score=args.score,
        batch_size=args.batch_size,
    )

    f1_result = evaluate_f1(
        global_model,
        X_test,
        y_test,
        threshold=threshold,
        score=args.score,
        batch_size=args.batch_size,
    )

    summary = {
        "model": args.model,
        "variant": args.variant,
        "method": args.method,
        "partition_type": args.partition,
        "num_clients": args.num_clients,
        "alpha": args.alpha,
        "rounds_requested": args.rounds,
        "rounds_completed": len(history),
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mu": args.mu,
        "latent_dim": args.latent_dim,
        "kl_weight": args.kl_weight,
        "vae_output_activation": args.vae_output_activation,
        "deep_encoding_dim": args.deep_encoding_dim,
        "lstm_units": args.lstm_units,
        "lstm_encoding_dim": args.lstm_encoding_dim,
        "score": args.score,
        "seed": args.seed,
        "patience": args.patience,
        "total_params": int(total_params),
        "data": {
            "X_path": x_path,
            "y_path": y_path,
            "raw_X_shape": list(X_raw.shape),
            "adapted_X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "global_anomaly_ratio": float(y.mean()),
            "train_anomaly_ratio": float(y_train.mean()),
            "val_anomaly_ratio": float(y_val.mean()),
            "test_anomaly_ratio": float(y_test.mean()),
            "X_min": float(X.min()),
            "X_max": float(X.max()),
            "X_mean": float(X.mean()),
            "X_std": float(X.std()),
        },
        "partition": pc_stats,
        "evaluation": f1_result,
        "best_val_loss": float(best_val_loss),
        "history": history,
    }

    out_name = (
        f"{args.model}_{args.variant}_{args.method}_{args.partition}_"
        f"{args.num_clients}clients_seed{args.seed}.json"
    )
    out_path = os.path.join(args.result_dir, out_name)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    model_path = os.path.join(
        args.result_dir,
        f"{args.model}_{args.variant}_{args.method}_{args.partition}_"
        f"{args.num_clients}clients_seed{args.seed}.weights.h5",
    )
    global_model.save_weights(model_path)

    print(f"Saved result to: {out_path}")
    print(f"Saved model weights to: {model_path}")
    print("Final F1:", f1_result["f1"])
    print("Precision:", f1_result["precision"])
    print("Recall:", f1_result["recall"])
    print("Accuracy:", f1_result["accuracy"])
    print("PC mean/std:", pc_stats["pc_mean"], pc_stats["pc_std"])


if __name__ == "__main__":
    main()