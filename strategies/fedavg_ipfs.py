"""
FedAvg strategy with IPFS integration for model exchange.
"""

import os
import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, cast

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
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
    """Federated Averaging with IPFS integration for model exchange."""

    def __init__(
        self,
        ipfs_connector: ModelIPFSConnector,
        initial_model_cid: Optional[str] = None,
        initial_parameters: Optional[Parameters] = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            callable[[NDArrays], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_metrics: Optional[Dict[str, Scalar]] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        save_model_path: Optional[str] = None,
    ) -> None:
        """Initialize FedAvg with IPFS strategy.

        Args:
            ipfs_connector: IPFS connector for model exchange
            initial_model_cid: CID of initial model (if available)
            initial_parameters: Initial global model parameters
            fraction_fit: Fraction of clients used during training
            fraction_evaluate: Fraction of clients used during validation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for validation
            min_available_clients: Minimum number of total clients in the system
            evaluate_fn: Optional function to evaluate the global model
            on_fit_config_fn: Function to configure training
            on_evaluate_config_fn: Function to configure validation
            accept_failures: Whether to accept client failures
            initial_metrics: Initial metrics, default empty dict
            fit_metrics_aggregation_fn: Metrics aggregation function for training
            evaluate_metrics_aggregation_fn: Metrics aggregation function for evaluation
            save_model_path: Path to save models to (optional)
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.ipfs = ipfs_connector
        self.global_model_cid = initial_model_cid
        self.current_round = 0
        self.save_model_path = save_model_path
        self.model_history = []  # Track CIDs of global models
        
        # If initial global model CID is provided, download it
        if self.global_model_cid and initial_parameters is None:
            try:
                parameters_list = self.ipfs.download_model(self.global_model_cid)
                self.initial_parameters = ndarrays_to_parameters(parameters_list)
                print(f"Initial model loaded from IPFS CID: {self.global_model_cid}")
            except Exception as e:
                print(f"Error loading initial model from IPFS: {str(e)}")
                self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
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
        
        # If this is not the first round, upload current model to IPFS
        if server_round > 1 or self.global_model_cid is None:
            try:
                upload_result = self.ipfs.upload_model(
                    parameter_arrays, 
                    model_id=f"global_round_{server_round-1}"
                )
                self.global_model_cid = upload_result["Hash"]
                self.ipfs.pin_model(self.global_model_cid)
                self.model_history.append(self.global_model_cid)
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
                # Fall back to standard FedAvg if IPFS upload fails
                print("Falling back to standard parameter exchange...")
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
        clients = client_manager.sample(
            num_clients=self.num_fit_clients(client_manager.num_available()),
            min_num_clients=self.min_fit_clients,
        )
        
        # Create fit instructions with either IPFS CID or parameters
        if self.global_model_cid:
            # If we have a CID, send minimal parameters and the CID
            # Create a small dummy parameter set (just to satisfy Flower's API requirements)
            dummy_parameters = ndarrays_to_parameters([np.array([0.0])])
            fit_ins = FitIns(dummy_parameters, config)
        else:
            # Otherwise send the full parameters
            fit_ins = FitIns(parameters, config)
            
        # Return client/instruction pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
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
        if "model_cid" in results[0][1].metrics:
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
                        if not self.accept_failures:
                            raise
            
            # Perform weighted aggregation
            if not weights_results:
                return None, {}
                
            # Federated averaging
            fedavg_result = aggregate(weights_results)
            parameters_aggregated = ndarrays_to_parameters(fedavg_result)
            
            # Calculate metrics across clients
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            
            return parameters_aggregated, metrics_aggregated
            
        else:
            # Standard FedAvg aggregation (fallback)
            print("Using standard FedAvg aggregation (clients did not use IPFS)")
            return super().aggregate_fit(server_round, results, failures)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        Args:
            server_round: Current round of federated learning
            parameters: Global model parameters
            client_manager: Client manager containing available clients

        Returns:
            List of tuples of (client, eval_instructions)
        """
        # Convert parameters to NumPy arrays
        parameter_arrays = parameters_to_ndarrays(parameters)
        
        # Get evaluation configuration
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom function for evaluation config
            config = self.on_evaluate_config_fn(server_round)
            
        # Add IPFS model CID to config if available
        if self.global_model_cid:
            config["model_cid"] = self.global_model_cid
            config["eval_round"] = server_round
            
            # Create a small dummy parameter set (just to satisfy Flower's API requirements)
            dummy_parameters = ndarrays_to_parameters([np.array([0.0])])
            evaluate_ins = EvaluateIns(dummy_parameters, config)
        else:
            # Otherwise send the full parameters
            evaluate_ins = EvaluateIns(parameters, config)
            
        # Sample clients
        clients = client_manager.sample(
            num_clients=self.num_evaluate_clients(client_manager.num_available()),
            min_num_clients=self.min_evaluate_clients,
        )
            
        # Return client/instruction pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results.

        Args:
            server_round: Current round of federated learning
            results: List of tuples of (client, eval_result)
            failures: List of failures

        Returns:
            Tuple of (optional loss, metrics)
        """
        if not results:
            return None, {}
            
        # Standard FedAvg evaluation aggregation
        return super().aggregate_evaluate(server_round, results, failures)
    
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
        
        # Log CID of the global model that's being evaluated
        parameter_arrays = parameters_to_ndarrays(parameters)
        eval_result = self.evaluate_fn(parameter_arrays)
        
        if eval_result is not None:
            loss, metrics = eval_result
            if metrics is None:
                metrics = {}
                
            # Add IPFS CID to metrics if available
            if self.global_model_cid:
                metrics["global_model_cid"] = self.global_model_cid
                
            return loss, metrics
        return None