"""
Utility to verify IPFS installation and setup.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ipfs_connector import ModelIPFSConnector
from user_config import user_config


def check_ipfs_daemon(api_url=user_config.IPFS_API_URL, timeout=10):
    """
    Check if IPFS daemon is running.
    
    Args:
        api_url: IPFS API URL
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with status and details
    """
    try:
        # Try to connect to IPFS
        connector = ModelIPFSConnector(api_url=api_url)
        
        # Get node info
        node_info = connector.get_node_info()
        
        # Simple test of adding and retrieving from IPFS
        test_start = time.time()
        test_content = {"test": "Hello IPFS!", "timestamp": time.time()}
        
        # Convert to list of arrays for the upload_model function
        import numpy as np
        test_data = [np.array([1, 2, 3]), np.array([[4, 5], [6, 7]])]
        
        # Upload test data
        upload_result = connector.upload_model(test_data, model_id="test_model")
        test_cid = upload_result["Hash"]
        
        # Wait a moment for IPFS to process
        time.sleep(1)
        
        # Download test data
        downloaded_data = connector.download_model(test_cid)
        
        test_duration = time.time() - test_start
        
        # Verify data integrity
        import numpy as np
        data_matches = all(
            np.array_equal(orig, downloaded) 
            for orig, downloaded in zip(test_data, downloaded_data)
        )
        
        return {
            "status": "ok",
            "node_id": node_info["ID"],
            "api_url": api_url,
            "addresses": node_info["Addresses"],
            "test_cid": test_cid,
            "data_integrity": "Passed" if data_matches else "Failed",
            "test_duration": f"{test_duration:.2f} seconds"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "api_url": api_url
        }


def attempt_start_ipfs_daemon():
    """
    Attempt to start the IPFS daemon if not running.
    
    Returns:
        Boolean indicating success
    """
    try:
        # Check if daemon is already running
        result = check_ipfs_daemon()
        if result["status"] == "ok":
            print("IPFS daemon is already running.")
            return True
            
        print("IPFS daemon not detected. Attempting to start...")
        
        # Try to start the IPFS daemon
        process = subprocess.Popen(
            ["ipfs", "daemon"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the daemon to start
        time.sleep(5)
        
        # Check if daemon started successfully
        for i in range(3):  # Try a few times
            result = check_ipfs_daemon()
            if result["status"] == "ok":
                print("Successfully started IPFS daemon.")
                return True
            time.sleep(2)
            
        print("Failed to start IPFS daemon.")
        return False
        
    except Exception as e:
        print(f"Error attempting to start IPFS daemon: {str(e)}")
        return False


def verify_ipfs_installation():
    """
    Verify if IPFS is installed and accessible.
    
    Returns:
        Boolean indicating if IPFS is properly installed
    """
    try:
        # Check if ipfs command is available
        process = subprocess.run(
            ["ipfs", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if process.returncode == 0:
            version = process.stdout.decode().strip()
            print(f"IPFS is installed: {version}")
            return True
        else:
            print("IPFS command not found or returned an error.")
            return False
            
    except Exception as e:
        print(f"Error checking IPFS installation: {str(e)}")
        print("IPFS may not be installed or not in PATH.")
        return False


def main(api_url=user_config.IPFS_API_URL, start_daemon=False):
    """
    Main function to verify IPFS setup.
    
    Args:
        api_url: IPFS API URL
        start_daemon: Whether to attempt to start the IPFS daemon if not running
    """
    print("=== IPFS Verification Utility ===")
    
    # Verify IPFS installation
    ipfs_installed = verify_ipfs_installation()
    if not ipfs_installed:
        print("\nIPFS does not appear to be installed or is not in PATH.")
        print("Please install IPFS following instructions at https://docs.ipfs.tech/install/")
        return
        
    # Check IPFS daemon
    daemon_result = check_ipfs_daemon(api_url=api_url)
    
    if daemon_result["status"] == "ok":
        print("\nIPFS daemon is running:")
        print(f"  Node ID: {daemon_result['node_id']}")
        print(f"  API URL: {daemon_result['api_url']}")
        print(f"  Test CID: {daemon_result['test_cid']}")
        print(f"  Data Integrity Test: {daemon_result['data_integrity']}")
        print(f"  Test Duration: {daemon_result['test_duration']}")
        print("\nYour IPFS setup appears to be working correctly!")
        
    else:
        print("\nIPFS daemon is not running or not accessible.")
        print(f"Error: {daemon_result.get('message', 'Unknown error')}")
        
        if start_daemon:
            print("\nAttempting to start IPFS daemon...")
            success = attempt_start_ipfs_daemon()
            
            if success:
                print("\nIPFS daemon started successfully!")
                # Re-verify
                daemon_result = check_ipfs_daemon(api_url=api_url)
                print(f"  Node ID: {daemon_result['node_id']}")
                print(f"  API URL: {daemon_result['api_url']}")
                print(f"  Test CID: {daemon_result['test_cid']}")
            else:
                print("\nFailed to start IPFS daemon. Please start it manually with 'ipfs daemon'")
                
        else:
            print("\nPlease start the IPFS daemon with 'ipfs daemon' before running the federated learning system.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify IPFS Installation and Setup")
    parser.add_argument(
        "--api", 
        type=str, 
        default=user_config.IPFS_API_URL, 
        help=f"IPFS API URL (default: {user_config.IPFS_API_URL})"
    )
    parser.add_argument(
        "--start-daemon", 
        action="store_true", 
        help="Attempt to start IPFS daemon if not running"
    )
    
    args = parser.parse_args()
    
    main(api_url=args.api, start_daemon=args.start_daemon)