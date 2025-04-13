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
import logging

# Local imports
from models.model import create_model
from strategies.fedavg_ipfs import FedAvgIPFS
from ipfs_connector import ModelIPFSConnector
import user_config


# Configure logger
# Updated to match Flower 1.16.0 API
fl.common.logger.configure(identifier="fedavg-ipfs")
log(logging.INFO, "Starting Flower server with IPFS integration")


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
    server_address: str = None,
    num_rounds: int = None,
    ipfs_api_url: str = None,
    fraction_fit: float = None,
    min_fit_clients: int = None,
    min_available_clients: int = None,
    model_type: str = None,
    input_shape: Tuple[int, int, int] = None,
    output_size: int = None,
    save_model_path: str = None,
) -> None:
    """Start the Flower server with IPFS-enabled federated learning."""
    
    # 導入用戶配置
    from user_config import user_config
    
    # 優先使用傳入的參數，如果未提供則使用用戶配置
    server_address = server_address or user_config.SERVER_ADDRESS
    num_rounds = num_rounds or user_config.NUM_ROUNDS
    ipfs_api_url = ipfs_api_url or user_config.IPFS_API_URL
    fraction_fit = fraction_fit or user_config.FRACTION_FIT
    min_fit_clients = min_fit_clients or user_config.MIN_CLIENTS
    min_available_clients = min_available_clients or user_config.MIN_AVAILABLE_CLIENTS
    model_type = model_type or user_config.MODEL_TYPE
    input_shape = input_shape or user_config.INPUT_SHAPE
    output_size = output_size or user_config.OUTPUT_SIZE
    
    # 處理 save_model_path 的特殊邏輯
    if save_model_path is None and user_config.SAVE_MODELS:
        save_model_path = user_config.MODEL_SAVE_PATH
    
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
            "lr": user_config.LEARNING_RATE,
            "epochs": user_config.EPOCHS_PER_ROUND,
            "batch_size": user_config.BATCH_SIZE,
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
        default=user_config.SERVER_ADDRESS,
        help=f"Server address (default: {user_config.SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=user_config.NUM_ROUNDS,
        help=f"Number of rounds (default: {user_config.NUM_ROUNDS})",
    )
    parser.add_argument(
        "--ipfs-api",
        type=str,
        default=user_config.IPFS_API_URL,
        help=f"IPFS API URL (default: {user_config.IPFS_API_URL})",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=user_config.MIN_CLIENTS,
        help=f"Minimum number of clients (default: {user_config.MIN_CLIENTS})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=user_config.MODEL_TYPE,
        choices=["cnn", "mlp", "custom_cnn"],
        help=f"Model type (default: {user_config.MODEL_TYPE})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=user_config.MODEL_SAVE_PATH if user_config.SAVE_MODELS else None,
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