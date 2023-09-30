#import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import flwr as fl

from client import get_client_fn
from evaluate import get_evaluate_fn
from models import ResNet50
from datasets import get_cifar10, ClassSubsetDataset
from attack import MalStrategy

def main(num_clients):

    SEED = 0
    #random.seed(SEED)
    #np.random.seed(SEED)
    torch.manual_seed(SEED)

    train, test = get_cifar10()

    trains = random_split(train, [1 / num_clients] * num_clients)
    tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(10)]

    # for 2 gpus
    train_loaders = [DataLoader(t, batch_size=32, shuffle=True, num_workers=12) for t in trains]
    test_loaders = [(s, DataLoader(c, batch_size=32, num_workers=12)) for s, c in tests]

    strategy = fl.server.strategy.FedAvg(
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for n, val in ResNet50().state_dict().items() if 'num_batches_tracked' not in n
        ]),
        evaluate_fn=get_evaluate_fn(ResNet50, test_loaders, file_name=str(num_clients)),
        fraction_fit=1,
        on_fit_config_fn=lambda x : {"round": x}
    )

    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(ResNet50, train_loaders),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
        client_resources={"num_cpus": 4, "num_gpus": 0.5}
    )

if __name__ == "__main__":

    for num_clients in [3, 10, 30]:
        main(num_clients)