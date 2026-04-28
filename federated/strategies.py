# Edge-Machine-Learning-Models/federated/strategies.py

import copy
import numpy as np


def weighted_average(weights_list, num_examples_list):
    total_examples = np.sum(num_examples_list)
    avg_weights = []

    for weights_tuple in zip(*weights_list):
        weighted = np.sum(
            [w * (n / total_examples) for w, n in zip(weights_tuple, num_examples_list)],
            axis=0
        )
        avg_weights.append(weighted)

    return avg_weights


class FedAvgStrategy:
    name = "FedAvg"

    def aggregate(self, global_weights, client_updates, round_idx, total_rounds):
        weights_list = [u["weights"] for u in client_updates]
        num_examples_list = [u["num_examples"] for u in client_updates]
        return weighted_average(weights_list, num_examples_list)


class FedProxStrategy:
    name = "FedProx"

    def aggregate(self, global_weights, client_updates, round_idx, total_rounds):
        weights_list = [u["weights"] for u in client_updates]
        num_examples_list = [u["num_examples"] for u in client_updates]
        return weighted_average(weights_list, num_examples_list)


class FedNAGStrategy:
    """
    Server-side momentum + cosine server lr schedule
    """
    name = "FedNAG"

    def __init__(self, server_lr=1.0, momentum=0.9, min_server_lr=0.1):
        self.server_lr = server_lr
        self.momentum = momentum
        self.min_server_lr = min_server_lr
        self.velocity = None

    def cosine_lr(self, round_idx, total_rounds):
        progress = round_idx / max(total_rounds, 1)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return self.min_server_lr + (self.server_lr - self.min_server_lr) * cosine_decay

    def aggregate(self, global_weights, client_updates, round_idx, total_rounds):
        weights_list = [u["weights"] for u in client_updates]
        num_examples_list = [u["num_examples"] for u in client_updates]

        avg_client_weights = weighted_average(weights_list, num_examples_list)

        delta = [avg_w - gw for avg_w, gw in zip(avg_client_weights, global_weights)]

        if self.velocity is None:
            self.velocity = [np.zeros_like(d) for d in delta]

        current_lr = self.cosine_lr(round_idx, total_rounds)

        new_velocity = []
        new_weights = []

        for v, d, gw in zip(self.velocity, delta, global_weights):
            v_new = self.momentum * v + current_lr * d
            w_new = gw + v_new
            new_velocity.append(v_new)
            new_weights.append(w_new)

        self.velocity = new_velocity
        return new_weights