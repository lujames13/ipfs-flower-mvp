"""
Flower client with IPFS integration for federated learning.
"""

import os
import argparse
import time
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

# Local imports
from models.model import create_model
from utils.data_loader import load_mnist, load_cifar10
from ipfs_connector import ModelIPFSConnector
import config


class FlowerIPFSClient(fl.client.NumPyClient):
    """Flower client that uses IPFS for model exchange."""

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        ipfs_connector: ModelIPFSConnector,
        client_id: str,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.ipfs = ipfs_connector
        self.client_id = client_id
        self.device = device
        self.model_history = []  # Track CIDs of client models
        
        print(f"Client {client_id} initialized with device: {device}")
        self.model.to(self.device)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on the locally held dataset."""
        
        # Check if IPFS CID is provided in config
        model_cid = config.get("model_cid", None)
        current_round = config.get("current_round", 0)
        
        # If CID is provided, download parameters from IPFS
        if model_cid:
            print(f"Downloading model from IPFS with CID: {model_cid}")
            try:
                parameters = self.ipfs.download_model(model_cid)
                print(f"Successfully downloaded model from IPFS")
            except Exception as e:
                print(f"Error downloading model from IPFS: {str(e)}")
                print("Using parameters directly from server message instead")
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Get training config
        epochs = int(config.get("epochs", config.EPOCHS_PER_ROUND))
        lr = float(config.get("lr", config.LEARNING_RATE))
        batch_size = int(config.get("batch_size", config.BATCH_SIZE))
        
        # Train the model
        train_loss = train(
            self.model,
            self.trainloader,
            epochs=epochs,
            learning_rate=lr,
            device=self.device,
        )
        
        # Upload trained model to IPFS
        model_params = self.get_parameters(config)
        model_id = f"client_{self.client_id}_round_{current_round}"
        
        try:
            result = self.ipfs.upload_model(model_params, model_id=model_id)
            model_cid = result["Hash"]
            self.model_history.append(model_cid)
            print(f"Uploaded model to IPFS with CID: {model_cid}")
            
            # Return the CID instead of the model parameters
            return model_params, len(self.trainloader.dataset), {"model_cid": model_cid}
        except Exception as e:
            print(f"Error uploading model to IPFS: {str(e)}")
            # Fall back to standard Flower behavior
            return model_params, len(self.trainloader.dataset), {"ipfs_error": str(e)}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on the locally held dataset."""
        
        # Check if IPFS CID is provided in config
        model_cid = config.get("model_cid", None)
        
        # If CID is provided, download parameters from IPFS
        if model_cid:
            print(f"Downloading model from IPFS with CID: {model_cid}")
            try:
                parameters = self.ipfs.download_model(model_cid)
                print(f"Successfully downloaded model from IPFS")
            except Exception as e:
                print(f"Error downloading model from IPFS: {str(e)}")
                print("Using parameters directly from server message instead")
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate the model
        loss, accuracy = test(self.model, self.testloader, self.device)
        
        # Return metrics
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> float:
    """Train the model on the training set."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    epoch_loss = 0.0
    for epoch in range(epochs):
        batch_loss = 0.0
        for batch_idx, (data, target) in enumerate(trainloader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Get loss
            loss = criterion(outputs, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            batch_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx}/{len(trainloader)}]: Loss: {loss.item():.6f}")
        
        # Print epoch statistics
        epoch_loss = batch_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{epochs} complete. Average loss: {epoch_loss:.6f}")
    
    return epoch_loss


def test(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on the test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    
    with torch.no_grad():
        for data, target in testloader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Compute loss
            batch_loss = criterion(outputs, target)
            loss += batch_loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update metrics
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # Calculate metrics
    accuracy = correct / total
    average_loss = loss / len(testloader)
    
    print(f"Test set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2%})")
    
    return average_loss, accuracy


def main(
    client_id: str,
    server_address: str = config.SERVER_ADDRESS,
    ipfs_api_url: str = config.IPFS_API_URL,
    model_type: str = config.MODEL_TYPE,
    dataset: str = "mnist",
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    output_size: int = config.OUTPUT_SIZE,
    data_dir: str = "./data",
    batch_size: int = config.BATCH_SIZE,
) -> None:
    """Create and start a Flower client with IPFS integration."""
    
    # Initialize IPFS connector
    print(f"Connecting to IPFS at {ipfs_api_url}")
    ipfs = ModelIPFSConnector(api_url=ipfs_api_url)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data based on dataset parameter
    if dataset.lower() == "mnist":
        print(f"Loading MNIST dataset into client {client_id}")
        trainloaders, testloader = load_mnist(
            data_dir=data_dir, 
            train_batch_size=batch_size, 
            test_batch_size=batch_size,
            num_clients=5,  # Assume 5 clients for partitioning
            iid=True        # IID data partitioning
        )
        # Select this client's dataloader based on client_id
        client_idx = int(client_id) % len(trainloaders)
        trainloader = trainloaders[client_idx]
        
    elif dataset.lower() == "cifar10":
        print(f"Loading CIFAR-10 dataset into client {client_id}")
        input_shape = (3, 32, 32)  # Update input shape for CIFAR-10
        output_size = 10
        trainloaders, testloader = load_cifar10(
            data_dir=data_dir, 
            train_batch_size=batch_size, 
            test_batch_size=batch_size,
            num_clients=5,  # Assume 5 clients for partitioning
            iid=True        # IID data partitioning  
        )
        # Select this client's dataloader based on client_id
        client_idx = int(client_id) % len(trainloaders)
        trainloader = trainloaders[client_idx]
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Print data distribution
    print(f"Client {client_id} has {len(trainloader.dataset)} training examples")
    
    # Create model
    model = create_model(model_type=model_type, input_shape=input_shape, output_size=output_size)
    model.to(device)
    
    # Create and start client
    client = FlowerIPFSClient(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        ipfs_connector=ipfs,
        client_id=client_id,
        device=device,
    )
    
    # Start Flower client
    print(f"Starting Flower client (ID: {client_id}) connecting to server {server_address}")
    fl.client.start_client(server_address=server_address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client with IPFS")
    parser.add_argument(
        "--client-id",
        type=str,
        required=True,
        help="Client ID (required)",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=config.SERVER_ADDRESS,
        help=f"Server address (default: {config.SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--ipfs-api",
        type=str,
        default=config.IPFS_API_URL,
        help=f"IPFS API URL (default: {config.IPFS_API_URL})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to use (default: mnist)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_TYPE,
        choices=["cnn", "mlp", "custom_cnn"],
        help=f"Model type (default: {config.MODEL_TYPE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory (default: ./data)",
    )
    
    args = parser.parse_args()
    
    main(
        client_id=args.client_id,
        server_address=args.server_address,
        ipfs_api_url=args.ipfs_api,
        model_type=args.model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )