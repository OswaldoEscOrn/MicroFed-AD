# Edge-Machine-Learning-Models/federated/scripts/make_job_list.py

import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    models = ["conv_ae", "conv_vae", "deep_ae", "lstm_ae"]
    variants = ["real", "real2"]
    methods = ["FedAvg", "FedProx", "FedNAG"]
    clients_list = [10, 50, 100]
    partitions = ["dirichlet", "uniform"]

    lines = []

    for model in models:
        for variant in variants:
            for method in methods:
                for num_clients in clients_list:
                    for partition in partitions:

                        # Batch size follows your original scripts
                        if model == "lstm_ae":
                            batch_size = 64
                        else:
                            batch_size = 128

                        # VAE default follows your original script
                        vae_output_activation = "sigmoid"

                        cmd = (
                            f"--project_root {args.project_root} "
                            f"--model {model} "
                            f"--variant {variant} "
                            f"--method {method} "
                            f"--partition {partition} "
                            f"--num_clients {num_clients} "
                            f"--alpha {args.alpha} "
                            f"--rounds {args.rounds} "
                            f"--local_epochs {args.local_epochs} "
                            f"--batch_size {batch_size} "
                            f"--patience {args.patience} "
                            f"--seed {args.seed} "
                            f"--vae_output_activation {vae_output_activation} "
                            f"--result_dir {args.result_dir}"
                        )

                        lines.append(cmd)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Saved job list to: {args.output}")
    print(f"Total jobs: {len(lines)}")


if __name__ == "__main__":
    main()