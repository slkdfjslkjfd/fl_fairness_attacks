from functools import reduce
import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class MalStrategy(fl.server.strategy.FedAvg):  # IMPORTANT: the attack is on the client not the strategy
    def __init__(self, name="", attack_round=80, *args, **kwargs):
        self.debug = False
        self.name = name
        self.attack_round = attack_round
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, server_round, results, failures):

        num_clients = len(results) - 1

        results = sorted(results, key=lambda x : x[0].cid)

        if server_round >= self.attack_round:

            target_parameters = parameters_to_ndarrays(results[0][1].parameters)

            if self.debug:
                weights_results = [
                    parameters_to_ndarrays(i[1].parameters) for i in results
                ][2:]
                predicted_parameters = [
                    reduce(np.add, layer) / (num_clients - 1) for layer in zip(*weights_results)
                ]
            else:
                predicted_parameters = parameters_to_ndarrays(results[1][1].parameters)

            # num_clients clients: (num_clients-1) clean + 1 malicious
            malicious_parameters = [(t * num_clients - p * (num_clients - 1)) / 1 for p,t in zip(predicted_parameters, target_parameters)]
            results[1][1].parameters = ndarrays_to_parameters(malicious_parameters)

        results = results[1:]

        np.save(f"outputs/updates_round_{server_round}_{self.name}.npy", np.array([i[1] for i in results], dtype=object), allow_pickle=True)

        return super().aggregate_fit(server_round, results, failures)