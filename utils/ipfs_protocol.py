"""
IPFS Protocol utilities for Flower federated learning.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union

import flwr as fl
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from ipfs_connector import ModelIPFSConnector


class IPFSModelExchangeProtocol:
    """
    Protocol for model exchange using IPFS in Flower federated learning.
    This class provides utilities for handling the protocol-level aspects
    of using IPFS for model exchange.
    """
    
    def __init__(self, ipfs_connector: ModelIPFSConnector):
        """
        Initialize the IPFS model exchange protocol.
        
        Args:
            ipfs_connector: IPFS connector instance
        """
        self.ipfs = ipfs_connector
        self.protocol_version = "1.0.0"
    
    def create_metadata(
        self, 
        model_id: str, 
        round_number: int, 
        client_id: Optional[str] = None,
        is_global: bool = False,
    ) -> Dict[str, Any]:
        """
        Create metadata for a model.
        
        Args:
            model_id: Identifier for the model
            round_number: Federated learning round number
            client_id: Optional client identifier (for client models)
            is_global: Whether this is a global model
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "model_id": model_id,
            "round_number": round_number,
            "timestamp": time.time(),
            "protocol_version": self.protocol_version,
            "is_global": is_global,
        }
        
        if client_id:
            metadata["client_id"] = client_id
            
        return metadata
    
    def upload_model(
        self, 
        parameters: List[Any], 
        model_id: str,
        round_number: int,
        client_id: Optional[str] = None,
        is_global: bool = False,
    ) -> Dict[str, Any]:
        """
        Upload model parameters to IPFS with protocol metadata.
        
        Args:
            parameters: Model parameters
            model_id: Identifier for the model
            round_number: Federated learning round number
            client_id: Optional client identifier (for client models)
            is_global: Whether this is a global model
            
        Returns:
            Dictionary containing CID and metadata
        """
        # Create metadata
        metadata = self.create_metadata(
            model_id=model_id,
            round_number=round_number,
            client_id=client_id,
            is_global=is_global,
        )
        
        # Upload model with metadata
        result = self.ipfs.upload_model(parameters, model_id=model_id)
        
        # Add protocol metadata to result
        result.update(metadata)
        
        # If global model, pin it to ensure availability
        if is_global:
            self.ipfs.pin_model(result["Hash"])
            
        return result
    
    def download_model(
        self, 
        cid: str,
        as_parameters: bool = False,
    ) -> Union[List[Any], Parameters]:
        """
        Download model parameters from IPFS.
        
        Args:
            cid: Content identifier
            as_parameters: If True, returns as Flower Parameters object
            
        Returns:
            Model parameters as list or Parameters object
        """
        # Download model from IPFS
        parameters = self.ipfs.download_model(cid)
        
        # Convert to Flower Parameters if requested
        if as_parameters:
            return ndarrays_to_parameters(parameters)
        
        return parameters
    
    def extract_cid_from_fit_res(self, fit_res: fl.common.FitRes) -> Optional[str]:
        """
        Extract CID from FitRes metrics.
        
        Args:
            fit_res: Flower FitRes object
            
        Returns:
            CID string or None if not found
        """
        if fit_res and fit_res.metrics and "model_cid" in fit_res.metrics:
            return fit_res.metrics["model_cid"]
        return None
    
    def extract_cid_from_evaluate_res(self, evaluate_res: fl.common.EvaluateRes) -> Optional[str]:
        """
        Extract CID from EvaluateRes metrics.
        
        Args:
            evaluate_res: Flower EvaluateRes object
            
        Returns:
            CID string or None if not found
        """
        if evaluate_res and evaluate_res.metrics and "model_cid" in evaluate_res.metrics:
            return evaluate_res.metrics["model_cid"]
        return None
    
    def add_cid_to_config(self, config: Dict[str, Any], cid: str) -> Dict[str, Any]:
        """
        Add model CID to configuration dictionary.
        
        Args:
            config: Configuration dictionary
            cid: Content identifier
            
        Returns:
            Updated configuration dictionary
        """
        config["model_cid"] = cid
        return config