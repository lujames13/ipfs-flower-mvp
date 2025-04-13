"""
Script to run the IPFS-enhanced federated learning system.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


def verify_ipfs():
    """
    Verify IPFS installation and daemon.
    
    Returns:
        Boolean indicating if IPFS is properly set up
    """
    print("Verifying IPFS setup...")
    result = subprocess.run(
        [sys.executable, "utils/ipfs_verify.py", "--start-daemon"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    # Check if verification was successful
    return "IPFS setup appears to be working correctly" in result.stdout


def create_project_structure():
    """
    Create project directory structure.
    
    Returns:
        Boolean indicating success
    """
    print("Creating project structure...")
    result = subprocess.run(
        [sys.executable, "create_structure.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


def run_server(rounds=5, min_clients=2, model="cnn", save_path="./saved_models", detached=False):
    """
    Run the Flower server.
    
    Args:
        rounds: Number of training rounds
        min_clients: Minimum number of clients
        model: Model type to use
        save_path: Path to save models
        detached: Whether to run in detached mode (background)
        
    Returns:
        Subprocess object or None if detached
    """
    cmd = [
        sys.executable,
        "server.py",
        "--rounds", str(rounds),
        "--min-clients", str(min_clients),
        "--model", model,
        "--save-path", save_path
    ]
    
    print(f"Starting server: {' '.join(cmd)}")
    
    if detached:
        # Start server in background
        if os.name == 'nt':  # Windows
            subprocess.Popen(
                cmd, 
                creationflags=subprocess.DETACHED_PROCESS,
                stdout=open("server_output.log", "a"),
                stderr=open("server_error.log", "a")
            )
            return None
        else:  # Unix/Linux
            process = subprocess.Popen(
                cmd,
                stdout=open("server_output.log", "a"),
                stderr=open("server_error.log", "a"),
                start_new_session=True
            )
            return None
    else:
        # Start server in foreground
        process = subprocess.Popen(cmd)
        return process


def run_client(client_id, dataset="mnist", model="cnn", batch_size=32, detached=False):
    """
    Run a Flower client.
    
    Args:
        client_id: Client identifier
        dataset: Dataset to use
        model: Model type to use
        batch_size: Batch size for training
        detached: Whether to run in detached mode (background)
        
    Returns:
        Subprocess object or None if detached
    """
    cmd = [
        sys.executable,
        "client.py",
        "--client-id", str(client_id),
        "--dataset", dataset,
        "--model", model,
        "--batch-size", str(batch_size)
    ]
    
    print(f"Starting client {client_id}: {' '.join(cmd)}")
    
    if detached:
        # Start client in background
        log_file = f"client_{client_id}_output.log"
        err_file = f"client_{client_id}_error.log"
        
        if os.name == 'nt':  # Windows
            subprocess.Popen(
                cmd, 
                creationflags=subprocess.DETACHED_PROCESS,
                stdout=open(log_file, "a"),
                stderr=open(err_file, "a")
            )
            return None
        else:  # Unix/Linux
            process = subprocess.Popen(
                cmd,
                stdout=open(log_file, "a"),
                stderr=open(err_file, "a"),
                start_new_session=True
            )
            return None
    else:
        # Start client in foreground
        process = subprocess.Popen(cmd)
        return process


def run_example(dataset="mnist", num_clients=3, rounds=5, model="cnn", batch_size=32):
    """
    Run a complete example with server and multiple clients.
    
    Args:
        dataset: Dataset to use
        num_clients: Number of clients to start
        rounds: Number of training rounds
        model: Model type to use
        batch_size: Batch size for training
    """
    # Create save directory
    save_dir = f"./saved_models/{dataset}_{model}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Start server
    server_process = run_server(
        rounds=rounds,
        min_clients=num_clients,
        model=model,
        save_path=save_dir
    )
    
    # Wait for server to initialize
    time.sleep(3)
    
    # Start clients
    client_processes = []
    for i in range(num_clients):
        client_process = run_client(
            client_id=i,
            dataset=dataset,
            model=model,
            batch_size=batch_size
        )
        client_processes.append(client_process)
        # Stagger client starts
        time.sleep(1)
    
    print(f"\nSystem running with {num_clients} clients, {rounds} rounds...")
    print(f"Models will be saved to: {save_dir}")
    print("Press Ctrl+C to stop the system")
    
    try:
        # Wait for server to complete
        if server_process:
            server_process.wait()
        
        # Wait for all clients to complete
        for i, proc in enumerate(client_processes):
            if proc:
                print(f"Waiting for client {i} to complete...")
                proc.wait()
        
        print("\nTraining completed successfully!")
        print(f"Saved models can be found in: {save_dir}")
        
    except KeyboardInterrupt:
        print("\nStopping the system...")
        
        # Terminate clients
        for i, proc in enumerate(client_processes):
            if proc:
                print(f"Terminating client {i}...")
                proc.terminate()
        
        # Terminate server
        if server_process:
            print("Terminating server...")
            server_process.terminate()
        
        print("System stopped.")


def main():
    """
    Main function to run the IPFS-enhanced federated learning system.
    """
    parser = argparse.ArgumentParser(description="Run IPFS-Enhanced Federated Learning System")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--verify", action="store_true", help="Verify IPFS setup")
    mode_group.add_argument("--server", action="store_true", help="Run server only")
    mode_group.add_argument("--client", action="store_true", help="Run client only")
    mode_group.add_argument("--example", action="store_true", help="Run complete example")
    
    # Server options
    parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds (default: 5)")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum number of clients (default: 2)")
    
    # Client options
    parser.add_argument("--client-id", type=str, help="Client ID (required for client mode)")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"], 
                        help="Dataset to use (default: mnist)")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp", "custom_cnn"],
                        help="Model type (default: cnn)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    
    # Example options
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients for example (default: 3)")
    
    # Other options
    parser.add_argument("--detached", action="store_true", help="Run in detached mode (background)")
    
    args = parser.parse_args()
    
    # Create project structure if needed
    if not os.path.exists("models") or not os.path.exists("strategies"):
        create_project_structure()
    
    # Run the selected mode
    if args.verify:
        verify_ipfs()
        
    elif args.server:
        run_server(
            rounds=args.rounds,
            min_clients=args.min_clients,
            model=args.model,
            detached=args.detached
        )
        print("Server started.")
        
    elif args.client:
        if not args.client_id:
            parser.error("--client-id is required for client mode")
        
        run_client(
            client_id=args.client_id,
            dataset=args.dataset,
            model=args.model,
            batch_size=args.batch_size,
            detached=args.detached
        )
        print(f"Client {args.client_id} started.")
        
    elif args.example:
        run_example(
            dataset=args.dataset,
            num_clients=args.num_clients,
            rounds=args.rounds,
            model=args.model,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()