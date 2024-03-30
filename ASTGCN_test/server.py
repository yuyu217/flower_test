from typing import List, Tuple

import flwr as fl
import flwr.common.parameter
from flwr.common import Metrics
import torch

import argparse

from pathlib import Path

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


parser = argparse.ArgumentParser()
parser.add_argument("--num_rounds", type=int, default=3)
args = parser.parse_args()
num_rounds = args.num_rounds

# load pre-trained params
model_param_file = "experiments/PEMS08/astgcn_r_h1d2w0_channel1_1.000000e-03/epoch_10.params"
model_param = torch.load(model_param_file)

initial_parameters = fl.common.Parameters([flwr.common.parameter.ndarray_to_bytes(val.cpu().numpy()) for _, val in model_param.items()], "numpy.ndarray")

# Define strategy
strategy = fl.server.strategy.FedAdam(evaluate_metrics_aggregation_fn=weighted_average, initial_parameters=initial_parameters)

flower_server = fl.server.Server(strategy=strategy, client_manager=fl.server.SimpleClientManager())

# Start Flower server
history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    server=flower_server,
    certificates=(
        Path("cert/ca.crt").read_bytes(),
        Path("cert/server.pem").read_bytes(),
        Path("cert/server.key").read_bytes()
    )
)

trained_parameters = [fl.common.parameter.bytes_to_ndarray(x) for x in flower_server.parameters.tensors]
