"""
IPFS connector specialized for machine learning model parameter storage and retrieval.
Based on the SimpleIPFSConnector from ipfs_mvp.py but adapted for ML model exchange.
"""

import os
import json
import time
import tempfile
import requests
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union

class ModelIPFSConnector:
    """
    IPFS connector specialized for machine learning model parameter storage and retrieval.
    Extends functionality from the SimpleIPFSConnector example.
    """
    
    def __init__(self, api_url: str = "http://localhost:5001/api/v0"):
        """
        Initialize IPFS connector.
        
        Args:
            api_url: IPFS API endpoint, defaults to local node
        """
        self.api_url = api_url
        # Test connection
        try:
            response = requests.post(f"{self.api_url}/id")
            if response.status_code == 200:
                node_id = response.json()["ID"]
                print(f"成功連接到IPFS節點: {node_id}")
                print(f"節點地址: {response.json()['Addresses']}")
            else:
                print(f"無法連接到IPFS API: {response.status_code}")
        except Exception as e:
            print(f"連接IPFS時出錯: {str(e)}")
    
    def upload_model(self, model_weights: Union[List[np.ndarray], Dict, torch.nn.Module], 
                    model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Serialize model weights and upload to IPFS.
        
        Args:
            model_weights: Model weights as list of NumPy arrays, state dict, or PyTorch model
            model_id: Optional identifier for the model
            
        Returns:
            Dictionary containing CID and metadata
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
            temp_path = temp_file.name
        
        try:
            # Serialize model weights based on type
            if isinstance(model_weights, torch.nn.Module):
                # If full PyTorch model is provided
                torch.save(model_weights.state_dict(), temp_path)
            elif isinstance(model_weights, dict):
                # If state dict is provided
                torch.save(model_weights, temp_path)
            else:
                # If list of NumPy arrays is provided (Flower format)
                state_dict = {}
                if model_id:
                    # Store metadata
                    state_dict['__metadata__'] = {
                        'model_id': model_id,
                        'timestamp': time.time()
                    }
                state_dict['__weights__'] = [arr.tolist() for arr in model_weights]
                with open(temp_path, 'w') as f:
                    json.dump(state_dict, f)
            
            print(f"Serialized model to: {temp_path}")
            
            # Upload to IPFS
            with open(temp_path, 'rb') as f:
                files = {'file': (os.path.basename(temp_path), f)}
                response = requests.post(f"{self.api_url}/add", files=files)
                
            if response.status_code != 200:
                raise Exception(f"IPFS add request failed: {response.text}")
                
            result = response.json()
            print(f"Model uploaded to IPFS with CID: {result['Hash']}")
            
            # Add metadata to result
            if model_id:
                result['model_id'] = model_id
            result['timestamp'] = time.time()
            
            return result
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def download_model(self, cid: str, output_path: Optional[str] = None, 
                      as_numpy: bool = True) -> Union[List[np.ndarray], Dict]:
        """
        Download model weights from IPFS using CID.
        
        Args:
            cid: Content identifier for the model
            output_path: Optional path to save the model
            as_numpy: If True, returns list of NumPy arrays (Flower format), 
                     otherwise returns state dict
            
        Returns:
            Model weights
        """
        print(f"Downloading model from IPFS with CID: {cid}")
        response = requests.post(f"{self.api_url}/cat?arg={cid}")
        
        if response.status_code != 200:
            raise Exception(f"IPFS cat request failed: {response.text}")
            
        content = response.content
        
        # Create temporary file for downloaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Save to output path if specified
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(content)
                print(f"Saved model to: {output_path}")
            
            # Deserialize model weights
            try:
                # Try loading as PyTorch model
                state_dict = torch.load(temp_path)
                if as_numpy:
                    # Convert to NumPy arrays (Flower format)
                    return [val.cpu().numpy() for key, val in state_dict.items() 
                            if not key.startswith('__')]
                else:
                    return state_dict
            except:
                # Try loading as JSON
                try:
                    with open(temp_path, 'r') as f:
                        state_dict = json.load(f)
                    
                    if '__weights__' in state_dict:
                        weights_list = state_dict['__weights__']
                        if as_numpy:
                            return [np.array(arr) for arr in weights_list]
                        else:
                            return state_dict
                    else:
                        raise ValueError("Invalid model format: missing weights")
                except:
                    raise ValueError("Failed to deserialize model: invalid format")
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def pin_model(self, cid: str) -> Dict[str, Any]:
        """
        Pin a model to ensure it remains in IPFS storage.
        
        Args:
            cid: Content identifier to pin
            
        Returns:
            API response
        """
        print(f"正在釘住CID: {cid}")
        response = requests.post(f"{self.api_url}/pin/add?arg={cid}")
        
        if response.status_code != 200:
            raise Exception(f"IPFS pin add請求失敗: {response.text}")
            
        result = response.json()
        print(f"CID已釘住: {result}")
        return result
    
    def unpin_model(self, cid: str) -> Dict[str, Any]:
        """
        Unpin a model to allow it to be garbage collected.
        
        Args:
            cid: Content identifier to unpin
            
        Returns:
            API response
        """
        print(f"正在移除釘住的CID: {cid}")
        response = requests.post(f"{self.api_url}/pin/rm?arg={cid}")
        
        if response.status_code != 200:
            raise Exception(f"IPFS pin rm請求失敗: {response.text}")
            
        result = response.json()
        print(f"釘住已移除: {result}")
        return result
    
    def list_pinned_models(self) -> Dict[str, Any]:
        """
        List all pinned models.
        
        Returns:
            Dictionary of pinned CIDs
        """
        response = requests.post(f"{self.api_url}/pin/ls")
        
        if response.status_code != 200:
            raise Exception(f"IPFS pin ls請求失敗: {response.text}")
            
        result = response.json()
        return result

    def get_node_info(self) -> Dict[str, Any]:
        """
        Get IPFS node information.
        
        Returns:
            Node information dictionary
        """
        response = requests.post(f"{self.api_url}/id")
        
        if response.status_code != 200:
            raise Exception(f"獲取節點信息失敗: {response.text}")
            
        return response.json()