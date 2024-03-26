import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,  # fraction of clients used for training
    min_fit_clients=2,  # minimum number of clients used for training
    min_available_clients=3,  # minimum number of clients available for training
)

# Start server
fl.server.start_server(
    server_address="[::]:8080",  
    config=fl.server.ServerConfig(num_rounds=10),  
    strategy=strategy  
)
