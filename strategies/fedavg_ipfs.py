"""
Simplified FedAvg strategy with IPFS integration for model exchange.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.server.strategy import FedAvg

# Import IPFS connector
from ipfs_connector import ModelIPFSConnector


class FedAvgIPFS(FedAvg):
    """Simplified Federated Averaging with IPFS integration for model exchange."""

    def __init__(
        self,
        ipfs_connector: ModelIPFSConnector,
        initial_model_cid: Optional[str] = None,
        initial_parameters: Optional[Parameters] = None,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        save_model_path: Optional[str] = None,
    ) -> None:
        """Initialize simplified FedAvg with IPFS strategy.

        Args:
            ipfs_connector: IPFS connector for model exchange
            initial_model_cid: CID of initial model (if available)
            initial_parameters: Initial global model parameters
            fraction_fit: Fraction of clients used during training
            min_fit_clients: Minimum number of clients for training
            min_available_clients: Minimum number of total clients in the system
            evaluate_fn: Optional function to evaluate the global model
            on_fit_config_fn: Function to configure training
            save_model_path: Path to save models to (optional)
        """
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=initial_parameters,
        )

        self.ipfs = ipfs_connector
        self.global_model_cid = initial_model_cid
        self.current_round = 0
        self.save_model_path = save_model_path
        
        # If initial global model CID is provided, download it
        if self.global_model_cid and initial_parameters is None:
            try:
                parameters_list = self.ipfs.download_model(self.global_model_cid)
                self.initial_parameters = ndarrays_to_parameters(parameters_list)
                print(f"Initial model loaded from IPFS CID: {self.global_model_cid}")
            except Exception as e:
                print(f"Error loading initial model from IPFS: {str(e)}")
                self.initial_parameters = initial_parameters

    def num_fit_clients(self, num_available_clients: int) -> int:
        """Return the number of clients to be used for fit."""
        return max(
            int(num_available_clients * self.fraction_fit),
            self.min_fit_clients,
        )
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training.

        Args:
            server_round: Current round of federated learning
            parameters: Global model parameters
            client_manager: Client manager containing available clients

        Returns:
            List of tuples of (client, fit_instructions)
        """
        self.current_round = server_round
        
        # Convert parameters to NumPy arrays
        parameter_arrays = parameters_to_ndarrays(parameters)
        
        # Upload current model to IPFS
        try:
            upload_result = self.ipfs.upload_model(
                parameter_arrays, 
                model_id=f"global_round_{server_round-1}"
            )
            self.global_model_cid = upload_result["Hash"]
            print(f"Global model (round {server_round-1}) uploaded to IPFS: {self.global_model_cid}")
            
            # Save model locally if path is specified
            if self.save_model_path:
                save_path = os.path.join(
                    self.save_model_path, 
                    f"global_model_round_{server_round-1}.npz"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(save_path, *parameter_arrays)
                print(f"Global model saved locally to: {save_path}")
        except Exception as e:
            print(f"Error uploading global model to IPFS: {str(e)}")
            self.global_model_cid = None
        
        # Get clients for this round
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom function for fit config
            config = self.on_fit_config_fn(server_round)
            
        # Add IPFS model CID to config if available
        if self.global_model_cid:
            config["model_cid"] = self.global_model_cid
            
        # Add current round to config
        config["current_round"] = server_round
        
        # Sample clients
        num_clients = self.num_fit_clients(client_manager.num_available())
        if isinstance(num_clients, tuple):
            num_clients = num_clients[0]  # 取元組的第一個元素
        clients = client_manager.sample(
            num_clients=num_clients,
            min_num_clients=self.min_fit_clients,
        )
        
        # Create fit instructions
        fit_ins = fl.common.FitIns(parameters, config)
            
        # Return client/instruction pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients using IPFS.

        Args:
            server_round: Current round of federated learning
            results: List of tuples of (client, fit_result)
            failures: List of failures

        Returns:
            Tuple of (parameters, metrics)
        """
        if not results:
            return None, {}
        
        # Check if clients used IPFS for model exchange
        if results[0][1].metrics and "model_cid" in results[0][1].metrics:
            # IPFS-based aggregation
            weights_results = []
            
            # Collect weights from all clients via IPFS
            for _, fit_res in results:
                client_cid = fit_res.metrics.get("model_cid")
                if client_cid:
                    try:
                        # Download client model from IPFS
                        weights = self.ipfs.download_model(client_cid)
                        client_samples = fit_res.num_examples
                        weights_results.append((weights, client_samples))
                        print(f"Downloaded client model from IPFS CID: {client_cid}")
                    except Exception as e:
                        print(f"Error downloading client model from IPFS: {str(e)}")
            
            # Perform weighted aggregation
            if not weights_results:
                return None, {}
                
            # Federated averaging
            fedavg_result = aggregate(weights_results)
            parameters_aggregated = ndarrays_to_parameters(fedavg_result)
            
            return parameters_aggregated, {}
            
        else:
            # Standard FedAvg aggregation (fallback)
            print("Using standard FedAvg aggregation (clients did not use IPFS)")
            return super().aggregate_fit(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the global model on the server (if evaluate_fn is provided).

        Args:
            server_round: Current round of federated learning
            parameters: Global model parameters

        Returns:
            Tuple of (loss, metrics) or None
        """
        if self.evaluate_fn is None:
            return None
        
        parameter_arrays = parameters_to_ndarrays(parameters)
        
        # Based on Flower 1.16.0 version of evaluate_fn
        eval_result = self.evaluate_fn(server_round, parameter_arrays, {})
        
        if eval_result is not None:
            loss, metrics = eval_result
            if metrics is None:
                metrics = {}
                
            # Add IPFS CID to metrics if available
            if self.global_model_cid:
                metrics["global_model_cid"] = self.global_model_cid
                
            return loss, metrics
        return None