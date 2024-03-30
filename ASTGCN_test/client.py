import argparse
import warnings
from collections import OrderedDict

import flwr as fl
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import os

import configparser
from model.ASTGCN_r import make_model
from lib.utils import load_graphdata_channel1, get_adjacency_matrix

import numpy as np

from pathlib import Path

# From FlowerFL Quickstart PyTorch
warnings.filterwarnings("ignore", category=UserWarning)

# To define what cuda device id will be accepted when parsing arguments.
available_cuda_device_id = [id for id in range(torch.cuda.device_count())]

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS08_astgcn.conf', type=str,
                     help="The path of configuration file.")
parser.add_argument("--partition-id", default=1, type=int,
                     help="The ID of dataset partition.")
parser.add_argument("--partition-size", default=3, type=int,
                     help="The number of partitions.")
parser.add_argument("--force-cpu", action='store_true',
                     help="Force use CPU for training and testing even if there are cuda device(s) available.")
parser.add_argument("--cuda-device-id", default=0, type=int, choices=available_cuda_device_id,
                     help="Select what cuda-device you want to use for training.")

args = parser.parse_args()

# Use cuda if available
cuda_device_name = f"cuda:{args.cuda_device_id}"

device_name = "cpu"
if not args.force_cpu:
    if torch.cuda.is_available():
        device_name = cuda_device_name
    elif torch.backends.mps.is_available():
        device_name = "mps"


DEVICE = torch.device(device_name)
print("device_name: ", device_name)

# From train_ASTGCN_r. Loading configurations.
config = configparser.ConfigParser()
config.read(args.config, encoding='utf-8')
data_config = config['Data']
training_config = config['Training']
partition_id = args.partition_id
partition_size = args.partition_size

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])


def train(net, train_loader, epochs):
    criterion = nn.L1Loss().to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters())
    for batch in tqdm(train_loader, "Training"):
        encoder_inputs, labels = batch
        optimizer.zero_grad()
        outputs = net(encoder_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def test(net, test_loader, mean):
    criterion = nn.L1Loss().to(DEVICE)
    loss = 0.0
    accuracy = torch.tensor(0.0).to(DEVICE)
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, "Testing"):
            encoder_inputs, labels = batch
            outputs = net(encoder_inputs)
            loss += criterion(outputs, labels).item()
            accuracy += ((1 - torch.clamp(torch.abs(outputs - labels) / torch.tensor(_mean).to(DEVICE), 0, 1)) * 100).mean()
            batch_count += 1
    accuracy /= batch_count
    accuracy = accuracy.cpu().item()
    loss /= test_loader.batch_size
    return loss, accuracy


def load_data(partition_id, partition_count, shuffle=True):
    '''
    from utils.py.
    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) +'_astcgn'

    rat = 1.0 / partition_count
    rel_range = np.array([partition_id, partition_id + 1]) * rat

    print('load file:', filename)

    file_data = np.load(filename + '.npz')
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x_timestamps = [int(x) for x in np.round(rel_range * (train_x.shape[0] - 1))]
    train_x = train_x[train_x_timestamps[0]:train_x_timestamps[1], :, 0:1, :]
    train_target = file_data['train_target'][train_x_timestamps[0]:train_x_timestamps[1], :, :]  # (10181, 307, 12)

    val_x = file_data['val_x']
    val_x_timestamps = [int(x) for x in np.round(rel_range * (val_x.shape[0] - 1))]
    val_x = val_x[val_x_timestamps[0]:val_x_timestamps[1], :, 0:1, :]
    val_target = file_data['val_target'][val_x_timestamps[0]:val_x_timestamps[1], :, :]

    test_x = file_data['test_x']
    test_x_timestamps = [int(x) for x in np.round(rel_range * (test_x.shape[0] - 1))]
    test_x = test_x[test_x_timestamps[0]:test_x_timestamps[1], :, 0:1, :]
    test_target = file_data['test_target'][test_x_timestamps[0]:test_x_timestamps[1], :, :]

    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std


# Define Flower client
class ASTGCNClient(fl.client.NumPyClient):
    def __init__(self, mean):
        super(ASTGCNClient, self).__init__()
        self.mean = mean

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_loader, epochs=1)
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, test_loader, self.mean)
        return loss, len(test_loader.dataset), {"accuracy": accuracy}


# Loading train data, value data and test data.
train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_data(partition_id, partition_size)

adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, len_input, num_of_vertices).to(DEVICE)

flower_client = ASTGCNClient(_mean)

# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=flower_client.to_client(),
    root_certificates=Path("cert/ca.crt").read_bytes()
)
