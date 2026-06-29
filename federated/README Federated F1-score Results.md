Supplementary Federated F1-score Results

This table provides the full federated learning F1-score results for the MicroFed-AD evaluation. It reports the mean and standard deviation of the F1-score across 10 independent runs for different aggregation methods, client populations, data distributions, and autoencoder backbones.

The table compares three federated aggregation methods:

FedAvg
FedProx
FedNAG

The experiments were evaluated under two client-partitioning settings:

Dirichlet partitioning with $\alpha = 0.5$, representing non-IID client data distributions.
Uniform partitioning, representing approximately IID client data distributions.

The table also includes the average F1-score for each aggregation method, computed across the Dirichlet and uniform settings.

Experimental Settings

The results are organised by:

Number of simulated clients: 10, 50, and 100
Autoencoder backbone:
Conv-AE
Conv-VAE
Deep-AE
LSTM-AE
Dataset variant:
real: baseline hybrid dataset constructed from public PM2.5 and acoustic data sources
real2: extended anomaly-enriched dataset used to evaluate robustness under broader abnormal conditions
Summary of Findings

The results show that FedProx and FedNAG generally outperform FedAvg across the evaluated settings. FedNAG achieves the strongest average performance for the 10-client setting, while FedProx remains highly competitive and slightly stronger in several 50-client and 100-client configurations. As the number of simulated clients increases, the average F1-score tends to decrease, reflecting the additional difficulty of federated training under larger and more heterogeneous client populations.

Overall, these results support the use of lightweight federated aggregation strategies for collaborative environmental anomaly detection, while reinforcing the paper’s main conclusion that the current federated evaluation is simulation-based and should be extended in future work to real multi-node ESP32-S3 deployments.

