import flwr as fl

strategy = fl.server.strategy.FedProx(
    fraction_fit=0.1,  # fraction of clients used for training
    min_fit_clients=2,  # minimum number of clients used for training
    min_available_clients=2,  # minimum number of clients available for training
    proximal_mu=0.1  # proximal mu value
)

# Start server
fl.server.start_server(
    server_address="[::]:8080",  
    config=fl.server.ServerConfig(num_rounds=15),  
    strategy=strategy  
)
