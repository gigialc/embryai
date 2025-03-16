"""EmbryoML binary classification and Federated Learning

A Python script for binary classification on embryo survival.
Implementing a federated learning framework where all information remain local.
By: Maria, Gigi, and Age.
"""

import os
import sys
import subprocess
import random
from collections import OrderedDict
from typing import List, Tuple, Optional
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import gdown
import requests
from bs4 import BeautifulSoup
import re
import zipfile
import shutil

import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import (
    Metrics,
    Context,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

# Global constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 5
BATCH_SIZE = 32
DATA_FOLDER = "embryo_data"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1NmDSVTyvLU-izHOGoe98kCGaivDAgOJg"

print(f"Training on {DEVICE}")
print(f"Flower {fl.__version__} / PyTorch {torch.__version__}")

def download_public_drive_folder(folder_url, output_folder):
    """Downloads files from a public Google Drive folder without authentication."""
    os.makedirs(output_folder, exist_ok=True)
    folder_id = folder_url.split('/')[-1]
    response = requests.get(f"https://drive.google.com/drive/folders/{folder_id}")
    
    if response.status_code != 200:
        print("Error: Could not access the folder. Make sure it's public.")
        return False

    file_ids = re.findall(r'"(https://drive.google.com/file/d/([^/]*)/view[^"]*)"', response.text)
    if not file_ids:
        print("No files found or folder might require authentication.")
        return False

    print(f"Found {len(file_ids)} files to download")
    for i, (_, file_id) in enumerate(file_ids):
        print(f"Downloading file {i+1}/{len(file_ids)} (ID: {file_id})")
        output = f"{output_folder}/{file_id}"
        download_url = f"https://drive.google.com/uc?id={file_id}"
        success = gdown.download(download_url, output, quiet=False)
        if not success:
            print(f"Failed to download file {file_id}")
            return False

    print("\nAll files downloaded successfully!")
    return True

def create_synthetic_dataset():
    """Create a synthetic dataset for testing."""
    print("Creating synthetic dataset...")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Create synthetic images (random noise)
    num_images = 1000
    image_size = (224, 224)
    
    for i in range(num_images):
        # Create random noise image
        img = np.random.randint(0, 255, (image_size[0], image_size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img)
        
        # Save image
        label = 1 if i < 500 else 0  # First 500 are positive, rest negative
        filename = f"{DATA_FOLDER}/embryo_{i:04d}_label_{label}.png"
        img.save(filename)
        
        if i % 100 == 0:
            print(f"Created {i}/{num_images} images")
    
    print("Synthetic dataset created successfully!")
    return True

def setup_data():
    """Set up the data directory and download if needed."""
    if not os.path.exists(DATA_FOLDER):
        print("Creating synthetic dataset for testing...")
        success = create_synthetic_dataset()
        if not success:
            raise RuntimeError("Failed to create the synthetic dataset.")
    
    # Verify data exists
    try:
        all_images = find_image_files()
        print(f"Found {len(all_images)} images in {DATA_FOLDER}")
        return True
    except ValueError as e:
        print(f"Error: {str(e)}")
        return False

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
        label = self.labels[idx].long()  # Convert label to long

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
    """Find all image files in the extracted directory structure"""
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
        if not image_files:
            print("No image files found in the directory structure")
            return None

        print(f"Found {len(image_files)} images in {DATA_FOLDER}")

        # Create labels (assuming first half is positive, second half is negative)
        labels = torch.cat([
            torch.ones(len(image_files) // 2, dtype=torch.long),
            torch.zeros(len(image_files) // 2, dtype=torch.long)
        ])

        # Create dataset
        dataset = EmbryoDataset(image_files, labels, transform=transform)

        if partition_id is not None:
            # Partition the dataset for federated learning
            partition_size = len(dataset) // NUM_CLIENTS
            start_idx = partition_id * partition_size
            end_idx = start_idx + partition_size

            indices = list(range(start_idx, min(end_idx, len(dataset))))
            train_indices = indices[:int(0.8 * len(indices))]
            val_indices = indices[int(0.8 * len(indices)):]

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)

            trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            valloader = DataLoader(val_dataset, batch_size=32)
            testloader = None  # Not used in this implementation

            return trainloader, valloader, testloader
        else:
            # Return the full dataset
            return dataset

    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return None

def train(net, trainloader, epochs: int, verbose=False):
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
        if verbose:
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

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate weighted average of metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

def get_parameters(net):
    """Get model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def client_fn(node_id: int) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    trainloader, valloader, _ = load_datasets(partition_id=node_id)
    net = Net().to(DEVICE)
    return FlowerClient(node_id, net, trainloader, valloader)

def server_fn(context: Context) -> ServerAppComponents:
    """Create server instance."""
    # Create strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=2,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Return ServerAppComponents
    return ServerAppComponents(
        config=ServerConfig(num_rounds=5),
        strategy=strategy
    )

def create_virtual_clients(dataset, num_clients):
    """Create virtual clients for federated learning."""
    clients = []
    partition_size = len(dataset) // num_clients

    for i in range(num_clients):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size

        indices = list(range(start_idx, min(end_idx, len(dataset))))
        train_indices = indices[:int(0.8 * len(indices))]
        val_indices = indices[int(0.8 * len(indices)):]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=32)

        net = Net().to(DEVICE)
        client = FlowerClient(i, net, trainloader, valloader)
        clients.append(client)
        print(f"Client {i+1}/{num_clients} created")

    return clients

def main():
    # Set up data
    if not setup_data():
        print("Failed to set up data")
        return

    # Load and preprocess data
    all_data = load_datasets()
    if all_data is None:
        print("Failed to load datasets")
        return

    # Create model
    net = Net()
    
    # Create clients
    num_clients = 5
    clients = create_virtual_clients(all_data, num_clients)
    
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    # Start federated learning
    print("\nStarting federated learning...")
    
    try:
        # Configure logging
        fl.common.logger.configure(identifier="Embryo Classification", filename="flower.log")
        
        # Initialize global model parameters
        global_parameters = get_parameters(net)
        
        # Training rounds
        for round_num in range(3):
            print(f"\nRound {round_num + 1}/3")
            
            # Client training
            client_results = []
            for client in clients:
                # Send global parameters to client
                client.net.load_state_dict(
                    {
                        key: torch.tensor(global_parameters[i], dtype=torch.float32)
                        for i, key in enumerate(client.net.state_dict().keys())
                    }
                )
                
                # Train on client's data
                train(client.net, client.trainloader, epochs=1)
                
                # Get updated parameters
                updated_parameters = get_parameters(client.net)
                client_results.append((len(client.trainloader.dataset), updated_parameters))
            
            # Aggregate parameters (FedAvg)
            total_size = sum(size for size, _ in client_results)
            
            # Initialize a list to hold the aggregated parameters
            aggregated_parameters = []
            
            # For each layer/parameter in the model
            for param_idx in range(len(global_parameters)):
                # Get weighted parameters from each client for this specific layer
                weighted_params = [
                    size * params[param_idx] 
                    for size, params in client_results
                ]
                
                # Sum the weighted parameters and divide by total size
                aggregated_param = np.sum(weighted_params, axis=0) / total_size
                aggregated_parameters.append(aggregated_param.astype(np.float32))
            
            # Update global parameters
            global_parameters = aggregated_parameters
            
            # Evaluate global model
            net.load_state_dict(
                {
                    key: torch.tensor(global_parameters[i], dtype=torch.float32)
                    for i, key in enumerate(net.state_dict().keys())
                }
            )
            
            # Test on each client's validation set
            total_loss = 0
            total_accuracy = 0
            total_examples = 0
            
            for client in clients:
                loss, accuracy = test(net, client.valloader)
                n_examples = len(client.valloader.dataset)
                total_loss += loss * n_examples
                total_accuracy += accuracy * n_examples
                total_examples += n_examples
            
            avg_loss = total_loss / total_examples
            avg_accuracy = total_accuracy / total_examples
            print(f"Round {round_num + 1} results:")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Average accuracy: {avg_accuracy:.4f}")
        
        print("\nFederated learning completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"Error during federated learning: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
