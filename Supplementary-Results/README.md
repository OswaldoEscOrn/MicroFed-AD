# Supplementary Experimental Results

This folder contains supplementary experimental results for the paper:

**MicroFed-AD: A Microcontroller-Based Federated Edge Intelligence Framework for Low-Latency Environmental Anomaly Detection**

The integrated table is provided as supplementary material because the complete comparison was too large to include in the camera-ready conference paper.

## Integrated Experimental Results Table

The table summarises the full experimental comparison across:

- Six experimental scenarios:
  - 100% real data
  - 70% real + 30% virtual data
  - 30% real + 70% virtual data
  - 100% virtual data
  - 70% outliers in 100% real data
  - 30% outliers in 100% real data

- Two dataset scales:
  - Original / 35k dataset
  - Extended / 100k dataset

- Four autoencoder models:
  - Deep-AE
  - LSTM-AE
  - Conv-AE
  - Conv-VAE

- Reported metrics:
  - Trainable parameters
  - Validation MAE loss
  - Anomaly threshold
  - Detected anomaly percentage
  - Training convergence epochs
  - Inference time, where measured

## Notes

A dash symbol (—) indicates that the value was not measured in that specific experiment.

These results should be interpreted together with the main paper. The paper reports the core MicroFed-AD framework, ESP32-S3 TinyML deployment, and simulation-based federated learning evaluation, while this table provides additional experimental detail for reproducibility and transparency.
