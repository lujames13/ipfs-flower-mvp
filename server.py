"""
Flower server with IPFS integration for federated learning.
"""

import os
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import Scalar
from flwr.common.logger import log
from flwr.server.strategy import Strategy
from logging import INFO

# Local imports
from models.model import create_model
from strategies.fedavg_ipfs import FedAvgIPFS
from ipfs_connector import ModelIPFSConnector
import config


# Configure logger
fl.common.logger.configure(identifier="fedavg-ipfs", level=INFO)


def get_eval_fn(model: nn.Module):
    """Return an evaluation function for server-side evaluation."""

    # Load data here to avoid the overhead of loading it in `evaluate` itself
    _, testloader = torch.utils.data.random_split(
        list(range(10)), [8, 2]  # Dummy data, replace with real test data loading
    )

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # Placeholder for actual evaluation logic
        # Replace with actual model evaluation on test dataset
        loss = 0.5  # Dummy loss value
        accuracy = 0.85  # Dummy accuracy value
        
        return loss, {"accuracy": accuracy}

    return evaluate


def main(
    server_address: str = config.SERVER_ADDRESS,
    num_rounds: int = config.NUM_ROUNDS,
    ipfs_api_url: str = config.IPFS_API_URL,
    fraction_fit: float = config.FRACTION_FIT,
    min_fit_clients: int = config.MIN_CLIENTS,
    min_available_clients: int = config.MIN_AVAILABLE_CLIENTS,
    model_type: str = config.MODEL_TYPE,
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    output_size: int = config.OUTPUT_SIZE,
    save_model_path: str = config.MODEL_SAVE_PATH if config.SAVE_MODELS else None,
) -> None:
    """Start the Flower server with IPFS-enabled federated learning."""
    
    # Initialize IPFS connector
    print(f"Connecting to IPFS at {ipfs_api_url}")
    ipfs = ModelIPFSConnector(api_url=ipfs_api_url)
    
    # Create model for server-side evaluation
    model = create_model(model_type=model_type, input_shape=input_shape, output_size=output_size)
    
    # Get initial model parameters
    initial_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    # Upload initial model to IPFS and get CID
    try:
        upload_result = ipfs.upload_model(initial_parameters, model_id="initial_model")
        initial_model_cid = upload_result["Hash"]
        ipfs.pin_model(initial_model_cid)
        print(f"Initial model uploaded to IPFS with CID: {initial_model_cid}")
    except Exception as e:
        print(f"Error uploading initial model to IPFS: {str(e)}")
        print("Starting without IPFS integration for initial model")
        initial_model_cid = None
    
    # Define strategy
    strategy = FedAvgIPFS(
        ipfs_connector=ipfs,
        initial_model_cid=initial_model_cid,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=lambda round: {
            "lr": config.LEARNING_RATE,
            "epochs": config.EPOCHS_PER_ROUND,
            "batch_size": config.BATCH_SIZE,
        },
        save_model_path=save_model_path
    )
    
    # Create server directory if saving models
    if save_model_path:
        os.makedirs(save_model_path, exist_ok=True)
        print(f"Models will be saved to: {save_model_path}")
    
    # Start Flower server
    print(f"Starting Flower server with IPFS integration, running for {num_rounds} rounds")
    print(f"Server address: {server_address}")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Server with IPFS")
    parser.add_argument(
        "--server-address",
        type=str,
        default=config.SERVER_ADDRESS,
        help=f"Server address (default: {config.SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=config.NUM_ROUNDS,
        help=f"Number of rounds (default: {config.NUM_ROUNDS})",
    )
    parser.add_argument(
        "--ipfs-api",
        type=str,
        default=config.IPFS_API_URL,
        help=f"IPFS API URL (default: {config.IPFS_API_URL})",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=config.MIN_CLIENTS,
        help=f"Minimum number of clients (default: {config.MIN_CLIENTS})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_TYPE,
        choices=["cnn", "mlp", "custom_cnn"],
        help=f"Model type (default: {config.MODEL_TYPE})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=config.MODEL_SAVE_PATH if config.SAVE_MODELS else None,
        help="Path to save models (empty for no saving)",
    )
    
    args = parser.parse_args()
    
    main(
        server_address=args.server_address,
        num_rounds=args.rounds,
        ipfs_api_url=args.ipfs_api,
        min_fit_clients=args.min_clients,
        model_type=args.model,
        save_model_path=args.save_path,
    )