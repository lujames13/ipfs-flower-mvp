"""
Utilities for model serialization/deserialization and parameter handling.
"""

import os
import json
import torch
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Any, Tuple, Union


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as a list of NumPy arrays.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of NumPy arrays representing model parameters
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Set model parameters from a list of NumPy arrays.
    
    Args:
        model: PyTorch model
        parameters: List of NumPy arrays containing model parameters
        
    Returns:
        None
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def save_model(model: torch.nn.Module, path: str, metadata: Dict = None) -> None:
    """
    Save model to disk.
    
    Args:
        model: PyTorch model
        path: Path to save the model
        metadata: Optional metadata to save with the model
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model with metadata
    if metadata:
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, path)
    else:
        torch.save(model.state_dict(), path)
    
    print(f"Model saved to: {path}")


def load_model(model: torch.nn.Module, path: str) -> Tuple[torch.nn.Module, Union[Dict, None]]:
    """
    Load model from disk.
    
    Args:
        model: PyTorch model instance to load parameters into
        path: Path to the saved model
        
    Returns:
        Tuple of (model, metadata) where metadata might be None
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    checkpoint = torch.load(path)
    
    # Check if the saved file contains a state dict or a dictionary with state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = checkpoint.get('metadata', None)
    else:
        model.load_state_dict(checkpoint)
        metadata = None
    
    print(f"Model loaded from: {path}")
    return model, metadata


def serialize_parameters(parameters: List[np.ndarray], path: str) -> None:
    """
    Serialize model parameters to JSON format.
    
    Args:
        parameters: List of NumPy arrays
        path: Path to save the serialized parameters
        
    Returns:
        None
    """
    # Convert NumPy arrays to lists for JSON serialization
    serializable_params = [p.tolist() for p in parameters]
    
    with open(path, 'w') as f:
        json.dump(serializable_params, f)
    
    print(f"Parameters serialized to: {path}")


def deserialize_parameters(path: str) -> List[np.ndarray]:
    """
    Deserialize model parameters from JSON format.
    
    Args:
        path: Path to the serialized parameters
        
    Returns:
        List of NumPy arrays
    """
    with open(path, 'r') as f:
        serializable_params = json.load(f)
    
    # Convert lists back to NumPy arrays
    parameters = [np.array(p) for p in serializable_params]
    
    print(f"Parameters deserialized from: {path}")
    return parameters


def compare_models(model1: torch.nn.Module, model2: torch.nn.Module) -> Tuple[float, Dict[str, float]]:
    """
    Compare two models by calculating the average and per-layer parameter differences.
    
    Args:
        model1: First PyTorch model
        model2: Second PyTorch model
        
    Returns:
        Tuple of (average_difference, layer_differences)
    """
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    if state_dict1.keys() != state_dict2.keys():
        raise ValueError("Models have different architecture (different layer names)")
    
    total_diff = 0.0
    total_params = 0
    layer_diffs = {}
    
    for key in state_dict1:
        layer1 = state_dict1[key].cpu().numpy()
        layer2 = state_dict2[key].cpu().numpy()
        
        if layer1.shape != layer2.shape:
            raise ValueError(f"Layer shapes don't match for {key}: {layer1.shape} vs {layer2.shape}")
        
        # Calculate absolute differences
        diff = np.mean(np.abs(layer1 - layer2))
        layer_diffs[key] = float(diff)
        
        # Update totals
        num_params = np.prod(layer1.shape)
        total_diff += diff * num_params
        total_params += num_params
    
    avg_diff = total_diff / total_params if total_params > 0 else 0
    
    return avg_diff, layer_diffs