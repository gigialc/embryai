#!/usr/bin/env python3
"""
EmbryoML Federated Learning Demo

This script demonstrates how to run a real federated learning system with multiple computers.
One computer acts as the server, and other computers join as clients.

Usage:
  - Server: python federated_embryo_demo.py server
  - Client: python federated_embryo_demo.py client --server_address=<server_ip>:<port> --client_id=<id>

Example:
  - On server computer: python federated_embryo_demo.py server --port=8080
  - On client computer 1: python federated_embryo_demo.py client --server_address=192.168.1.100:8080 --client_id=1
  - On client computer 2: python federated_embryo_demo.py client --server_address=192.168.1.100:8080 --client_id=2
"""

import os
import sys
import argparse
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import flwr as fl
from flwr.common import Metrics, Parameters
import socket

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FOLDER = "embryo_data"
BATCH_SIZE = 32

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Connect to a public server to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # Fallback to localhost

class EmbryoDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels if labels is not None else torch.ones(len(image_paths), dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx].long()

        if self.transform:
            image = self.transform(image)

        return image, label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def find_image_files():
    """Find all image files in the data directory."""
    all_images = []
    search_patterns = [
        os.path.join(DATA_FOLDER, "*.png"),
        os.path.join(DATA_FOLDER, "*.jpg"),
        os.path.join(DATA_FOLDER, "*.jpeg")
    ]

    for pattern in search_patterns:
        all_images.extend(glob.glob(pattern))

    if not all_images:
        raise ValueError("No image files found in the directory structure")

    print(f"Found {len(all_images)} image files")
    return all_images

def load_datasets():
    """Load and preprocess the dataset."""
    try:
        # Find all image files
        image_files = find_image_files()
        if not image_files:
            print(f"No image files found in {DATA_FOLDER}")
            return None, None

        print(f"Found {len(image_files)} images in {DATA_FOLDER}")

        # Extract labels from filenames (assuming filenames contain "label_X")
        labels = []
        for file_path in image_files:
            file_name = os.path.basename(file_path)
            if "label_1" in file_name:
                labels.append(1)
            else:
                labels.append(0)
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Create dataset
        dataset = EmbryoDataset(image_files, labels_tensor, transform=transform)

        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        return trainloader, valloader

    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return None, None

def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader)
    accuracy = correct / total
    return loss, accuracy

def get_parameters(net):
    """Get model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class EmbryoClient(fl.client.NumPyClient):
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
        
        # Get epochs from config or use default
        epochs = config.get("epochs", 1)
        
        # Train the model
        train(self.net, self.trainloader, epochs=epochs)
        
        # Return updated parameters and number of training examples
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

def start_client(server_address, client_id):
    """Start a federated learning client."""
    print(f"Starting client {client_id}, connecting to server at {server_address}")
    
    # Check if data directory exists
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER, exist_ok=True)
        print(f"Created data directory {DATA_FOLDER}")
        print(f"Please add your embryo images to {DATA_FOLDER} before continuing.")
        print("Embryo images should be named with the pattern *label_0*.png or *label_1*.png")
        print("where 1 represents survival and 0 represents non-survival.")
        return
    
    # Load data
    trainloader, valloader = load_datasets()
    if trainloader is None or valloader is None:
        print("Failed to load datasets. Please check your data directory.")
        return
    
    # Create model
    net = Net().to(DEVICE)
    print(f"Model created on {DEVICE}")
    
    # Create client
    client = EmbryoClient(client_id, net, trainloader, valloader)
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )

def weighted_average(metrics):
    """Compute weighted average of metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def start_server(port):
    """Start a federated learning server."""
    local_ip = get_local_ip()
    print(f"Starting server on {local_ip}:{port}")
    print(f"Clients should connect to: {local_ip}:{port}")
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,  # Wait for at least 2 clients
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=None,  # Set to None to wait for client to provide model
    )
    
    # Start server
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