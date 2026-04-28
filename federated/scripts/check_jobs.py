# Edge-Machine-Learning-Models/federated/scripts/check_jobs.py

import os
import argparse


def parse_arg_value(tokens, key):
    if key not in tokens:
        return None
    idx = tokens.index(key)
    if idx + 1 >= len(tokens):
        return None
    return tokens[idx + 1]


def expected_json_name(cmd):
    tokens = cmd.strip().split()

    model = parse_arg_value(tokens, "--model")
    variant = parse_arg_value(tokens, "--variant")
    method = parse_arg_value(tokens, "--method")
    partition = parse_arg_value(tokens, "--partition")
    num_clients = parse_arg_value(tokens, "--num_clients")
    seed = parse_arg_value(tokens, "--seed")

    return f"{model}_{variant}_{method}_{partition}_{num_clients}clients_seed{seed}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_list", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--missing_out", type=str, default=None)
    args = parser.parse_args()

    with open(args.job_list, "r") as f:
        jobs = [line.strip() for line in f if line.strip()]

    missing = []
    completed = []

    for i, cmd in enumerate(jobs, start=1):
        name = expected_json_name(cmd)
        path = os.path.join(args.result_dir, name)

        if os.path.exists(path):
            completed.append((i, name))
        else:
            missing.append((i, cmd, name))

    print(f"Total jobs: {len(jobs)}")
    print(f"Completed: {len(completed)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print("\nMissing jobs:")
        for i, cmd, name in missing:
            print(f"  array_id={i}, expected={name}")

    if args.missing_out is not None:
        with open(args.missing_out, "w") as f:
            for _, cmd, _ in missing:
                f.write(cmd + "\n")
        print(f"\nSaved missing job list to: {args.missing_out}")


if __name__ == "__main__":
    main()