from functools import reduce
import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class MalStrategy(fl.server.strategy.FedAvg):  # IMPORTANT: the attack is on the client not the strategy
    def __init__(self, *args, **kwargs):
        self.debug = False
        self.attack_round = 10
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, server_round, results, failures):

        results = sorted(results, key=lambda x : x[0].cid)

        if server_round >= self.attack_round:

            target_parameters = parameters_to_ndarrays(results[0][1].parameters)

            if self.debug:
                weights_results = [
                    parameters_to_ndarrays(i[1].parameters) for i in results
                ][2:]
                predicted_parameters = [
                    reduce(np.add, layer) / 9 for layer in zip(*weights_results)
                ]
            else:
                predicted_parameters = parameters_to_ndarrays(results[1][1].parameters)

            # 10 clients - 9 clean + 1 malicious
            malicious_parameters = [(t * 10 - p * 9) / 1 for p,t in zip(predicted_parameters, target_parameters)]
            results[1][1].parameters = ndarrays_to_parameters(malicious_parameters)

        results = results[1:]

        np.save(f"outputs/updates_round_{server_round}.npy", np.array([i[1] for i in results], dtype=object), allow_pickle=True)

        return super().aggregate_fit(server_round, results, failures)