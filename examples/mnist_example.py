"""
Example script for IPFS-enhanced federated learning with MNIST dataset.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config


def start_server(num_rounds: int, min_clients: int, save_dir: str = "./saved_models"):
    """
    Start the Flower server with IPFS integration.
    
    Args:
        num_rounds: Number of federated learning rounds
        min_clients: Minimum number of clients for aggregation
        save_dir: Directory to save models
    
    Returns:
        Subprocess object
    """
    cmd = [
        "python", 
        "server.py",
        "--rounds", str(num_rounds),
        "--min-clients", str(min_clients),
        "--model", "cnn",
        "--save-path", save_dir,
    ]
    
    print(f"Starting server: {' '.join(cmd)}")
    server_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # Wait a moment for server to start
    time.sleep(3)
    
    return server_process


def start_client(client_id: int, dataset: str = "mnist"):
    """
    Start a Flower client with IPFS integration.
    
    Args:
        client_id: Client identifier
        dataset: Dataset to use
    
    Returns:
        Subprocess object
    """
    cmd = [
        "python",
        "client.py",
        "--client-id", str(client_id),
        "--dataset", dataset,
        "--model", "cnn",
        "--batch-size", str(config.BATCH_SIZE),
    ]
    
    print(f"Starting client {client_id}: {' '.join(cmd)}")
    client_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    return client_process


def run_example(num_clients: int = 3, num_rounds: int = 5):
    """
    Run a complete federated learning example with MNIST.
    
    Args:
        num_clients: Number of clients to start
        num_rounds: Number of federated learning rounds
    """
    # Create directory for saved models
    save_dir = "./saved_models/mnist_example"
    os.makedirs(save_dir, exist_ok=True)
    
    # Start server
    server_process = start_server(
        num_rounds=num_rounds,
        min_clients=num_clients,
        save_dir=save_dir,
    )
    
    # Start clients
    client_processes = []
    for i in range(num_clients):
        client_process = start_client(client_id=i, dataset="mnist")
        client_processes.append(client_process)
        # Stagger client starts to avoid overwhelming the server
        time.sleep(1)
    
    # Wait for server to complete
    try:
        server_stdout, server_stderr = server_process.communicate()
        print("\n===== SERVER OUTPUT =====")
        print(server_stdout.decode())
        if server_stderr:
            print("\n===== SERVER ERRORS =====")
            print(server_stderr.decode())
    except KeyboardInterrupt:
        print("Stopping processes...")
    finally:
        # Terminate all processes
        for i, proc in enumerate(client_processes):
            print(f"Terminating client {i}...")
            proc.terminate()
        
        server_process.terminate()
        
        # Wait for processes to terminate
        for proc in client_processes:
            proc.wait()
        server_process.wait()
    
    print("\nExample completed!")
    print(f"Saved models can be found in: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MNIST Example with IPFS-enhanced Federated Learning")
    parser.add_argument(
        "--clients", 
        type=int, 
        default=3, 
        help="Number of clients (default: 3)"
    )
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=5, 
        help="Number of rounds (default: 5)"
    )
    
    args = parser.parse_args()
    
    run_example(num_clients=args.clients, num_rounds=args.rounds)