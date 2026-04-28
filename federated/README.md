# Federated Learning Experiments
This directory contains the federated learning implementation for environmental anomaly detection using autoencoder-based models. 
It supports multiple model architectures, datasets, federated aggregation methods, and client data partitioning strategies.

---

## File Structure
```text
federated/
├── run_experiment.py            # Main experiment entry
├── models.py                    # Model definitions
├── client.py                    # Federated client training logic
├── data_utils.py                # Data loading & partitioning
├── metrics.py                   # Evaluation metrics
├── strategies.py                # Server-side aggregation
├── make_labels_100k_from_original.py  # Label generation for 100k dataset
├── README.md
│
├── scripts/                     # Batch experiment scripts

```

---

## Core Files

### run_experiment.py
Main entry point for running a single federated experiment.
Handles argument parsing, dataset loading, model initialization, client partitioning, federated training, evaluation, and result saving.

### models.py
Implements four anomaly detection models:
`conv_ae`, `conv_vae`, `deep_ae`, `lstm_ae`

### client.py
Implements federated client logic:
receives global model → local training → returns updated weights
supports FedAvg and FedProx.

### data_utils.py
Data utilities:
dataset loading, train/val/test splitting, IID/non-IID client data partitioning.

### metrics.py
Computes evaluation metrics:
reconstruction error, detection threshold, F1-score, precision, recall, accuracy, and partition coefficient.

### strategies.py
Server-side federated aggregation strategies:
`FedAvg`, `FedProx`, `FedNAG` 

### make_labels_100k_from_original.py
Generates anomaly labels for the augmented 100k dataset from the original label file.

---

## Scripts

- **make_job_list.py**: Generates a list of experiment commands for batch execution
- **run_one_experiment.sbatch**: Slurm job script for single experiment
- **submit_all_federated.sh**: Submits all experiments as a Slurm array
- **check_jobs.py**: Checks for missing or failed experiment results
- **summarize_results.py**: Aggregates JSON results into CSV files

---

## Supported Options

- **Models**: `conv_ae`, `conv_vae`, `deep_ae`, `lstm_ae`
- **Datasets**: `real` (original), `real2` (augmented 100k)
- **Methods**: `FedAvg`, `FedProx`, `FedNAG`
- **Partitions**: `dirichlet` (non-IID), `uniform` (IID)

---

## Notes

- FedNAG is the renamed version of the original FedXJTLU.
- The `real2` dataset requires `window_anomaly_labels_100k.npy`.
- Use `check_jobs.py` to identify failed or missing cluster jobs.
