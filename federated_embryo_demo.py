#!/usr/bin/env python3
"""
EmbryoML Federated Learning Demo

This script implements a federated learning system for embryo classification.
It can be run in two modes:
- Server: python federated_embryo_demo.py server --port=8080
- Client: python federated_embryo_demo.py client --server_address=<server_ip>:<port> --client_id=<id>

Example usage:
- On server computer: python federated_embryo_demo.py server --port=8080
- On client computer 1: python federated_embryo_demo.py client --server_address=192.168.1.100:8080 --client_id=1
- On client computer 2: python federated_embryo_demo.py client --server_address=192.168.1.100:8080 --client_id=2
"""

import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import NDArrays, Metrics
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import utilities to avoid code duplication
from embryo_fl_utils import (
    get_local_ip, find_image_files, EmbryoDataset, Net, 
    DEVICE, DATA_FOLDER, BATCH_SIZE,
    train, test, get_parameters, set_parameters
)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_datasets(partition_id=None):
    """Load and preprocess the dataset."""
    try:
        # Find all image files
        image_files = find_image_files()
        
        # Extract labels from filenames (assuming format: embryo_XXXX_label_Y.png)
        labels = []
        for img_path in image_files:
            filename = os.path.basename(img_path)
            if "_label_1" in filename:
                labels.append(1)  # Viable
            else:
                labels.append(0)  # Non-viable
        
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Create dataset
        dataset = EmbryoDataset(image_files, labels, transform=transform)
        
        if partition_id is not None:
            # For client mode: Partition the dataset
            num_clients = 5  # Default number of partitions
            partition_size = len(dataset) // num_clients
            start_idx = int(partition_id) * partition_size
            end_idx = start_idx + partition_size
            
            indices = list(range(start_idx, min(end_idx, len(dataset))))
            train_indices = indices[:int(0.8 * len(indices))]
            val_indices = indices[int(0.8 * len(indices)):]
            
            trainloader = DataLoader(
                torch.utils.data.Subset(dataset, train_indices),
                batch_size=BATCH_SIZE, 
                shuffle=True
            )
            valloader = DataLoader(
                torch.utils.data.Subset(dataset, val_indices),
                batch_size=BATCH_SIZE
            )
            
            return trainloader, valloader
        else:
            # For server mode or general use: Return the whole dataset
            return dataset
    
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return None

class EmbryoClient(fl.client.NumPyClient):
    """Flower client implementing embryo classification."""
    
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
    
    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)
    
    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        
        # Train the model
        epochs = config.get("epochs", 1)
        loss, accuracy = train(self.net, self.trainloader, epochs)
        
        # Return updated parameters and training metrics
        return get_parameters(self.net), len(self.trainloader.dataset), {"loss": float(loss), "accuracy": float(accuracy)}
    
    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate")
        set_parameters(self.net, parameters)
        
        # Evaluate the model
        loss, accuracy = test(self.net, self.valloader)
        
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

def start_client(server_address, client_id):
    """Start a federated learning client."""
    print(f"Starting client {client_id}, connecting to server at {server_address}")
    
    # Load data
    trainloader, valloader = load_datasets(client_id)
    
    # Create model
    net = Net().to(DEVICE)
    print(f"Model created on {DEVICE}")
    
    # Start Flower client
    client = EmbryoClient(client_id, net, trainloader, valloader)
    
    # Run client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )
    
    print("Client finished")

def weighted_average(metrics):
    """Compute weighted average of metrics."""
    # Weighted accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}

def start_server(port):
    """Start a federated learning server."""
    print(f"Starting server on port {port}")
    
    # Load the dataset for evaluation
    dataset = load_datasets()
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=2,  # Minimum number of clients to train in each round
        min_evaluate_clients=2,  # Minimum number of clients to evaluate in each round
        min_available_clients=2,  # Minimum number of clients that need to be connected
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate evaluation metrics
        initial_parameters=None,  # Set to None to wait for client to provide model
    )
    
    # Start Flower server
    server_ip = get_local_ip()
    print(f"Server IP: {server_ip}")
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

def main():
    parser = argparse.ArgumentParser(description="EmbryoML Federated Learning Demo")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start a federated learning server")
    server_parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    
    # Client command
    client_parser = subparsers.add_parser("client", help="Start a federated learning client")
    client_parser.add_argument("--server_address", type=str, required=True, 
                              help="Server address in format IP:PORT")
    client_parser.add_argument("--client_id", type=str, required=True,
                              help="Unique identifier for this client")
    
    args = parser.parse_args()
    
    if args.command == "server":
        start_server(args.port)
    elif args.command == "client":
        start_client(args.server_address, args.client_id)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  Server: python federated_embryo_demo.py server --port=8080")
        print("  Client: python federated_embryo_demo.py client --server_address=192.168.1.100:8080 --client_id=1")

if __name__ == "__main__":
    main() 