# IPFS-Enhanced Federated Learning with Flower

This project implements a federated learning system using the Flower framework with IPFS (InterPlanetary File System) as the storage mechanism for model parameters. The system uses FedAvg (Federated Averaging) as its aggregation strategy.

## Overview

Federated Learning enables training machine learning models across multiple devices or servers without exchanging the actual data. This project enhances traditional federated learning by:

1. Using IPFS for decentralized storage of model parameters
2. Providing resilience through IPFS's content-addressing and distributed nature
3. Reducing central server storage requirements
4. Enabling more flexible client-server communication patterns

## Architecture

The system consists of:

- **Flower Server**: Coordinates the federated learning process, aggregates models using FedAvg
- **Flower Clients**: Train local models on their private data
- **IPFS Network**: Stores and retrieves model parameters using content addressing
- **Model Exchange Protocol**: Uses IPFS CIDs (Content Identifiers) to reference models

## Project Structure

```
ipfs-flower-fedavg/
├── client.py                 # Flower client implementation with IPFS integration
├── config.py                 # Configuration parameters
├── create_structure.py       # Script to create the project directory structure
├── ipfs_connector.py         # IPFS interaction utilities
├── models/                   # Model definitions
│   ├── __init__.py
│   └── model.py              # Model architectures (CNN, MLP)
├── README.md                 # Project documentation
├── requirements.txt          # Project dependencies
├── run_system.py             # Script to run the entire system
├── server.py                 # Flower server implementation with IPFS
├── strategies/               # Federated learning strategies
│   ├── __init__.py
│   └── fedavg_ipfs.py        # FedAvg strategy with IPFS support
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading and partitioning
│   ├── ipfs_protocol.py      # IPFS model exchange protocol
│   ├── ipfs_verify.py        # IPFS setup verification
│   └── model_utils.py        # Model parameter handling utilities
└── examples/                 # Example implementations
    ├── __init__.py
    └── mnist_example.py      # MNIST dataset example
```

## Prerequisites

- Python 3.8+
- Running IPFS node (local or remote)
- PyTorch for the ML components
- Network connectivity between clients, server, and IPFS nodes

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ipfs-flower-fedavg.git
cd ipfs-flower-fedavg

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create the project structure
python create_structure.py
```

## IPFS Setup

Ensure you have a running IPFS node. If not, install and initialize:

```bash
# Install IPFS (this varies by OS, see https://docs.ipfs.tech/install/)
# Example for Linux:
wget https://dist.ipfs.tech/kubo/v0.17.0/kubo_v0.17.0_linux-amd64.tar.gz
tar -xvzf kubo_v0.17.0_linux-amd64.tar.gz
cd kubo
sudo bash install.sh

# Initialize IPFS
ipfs init

# Start the IPFS daemon
ipfs daemon
```

Verify your IPFS setup using the provided utility:

```bash
python utils/ipfs_verify.py --start-daemon
```

## Usage

### Running the Complete System

For a quick start with the default configuration (MNIST dataset, 3 clients, 5 rounds):

```bash
python run_system.py --example
```

With custom settings:

```bash
python run_system.py --example --dataset cifar10 --num-clients 5 --rounds 10 --model custom_cnn
```

### Running Server and Clients Separately

1. **Start the server**:

```bash
python run_system.py --server --rounds 10 --min-clients 3
```

2. **Start multiple clients** (run each in a separate terminal):

```bash
python run_system.py --client --client-id 1 --dataset mnist
python run_system.py --client --client-id 2 --dataset mnist
python run_system.py --client --client-id 3 --dataset mnist
```

### Advanced Usage

You can also run server and clients with specific configurations:

```bash
# Start server
python server.py --rounds 20 --min-clients 2 --model cnn --save-path ./saved_models/my_experiment

# Start clients
python client.py --client-id 1 --dataset mnist --model cnn --batch-size 64
python client.py --client-id 2 --dataset mnist --model cnn --batch-size 64
```

## Customizing Models

To use your own model:

1. Define your model in `models/model.py`
2. Add it to the `create_model` factory function
3. Update the imports in `models/__init__.py`

Example for a custom CNN model:

```python
# models/model.py
class MyCustomCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(MyCustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Define forward pass
        ...
        return x
        
# Update create_model function
def create_model(model_type="cnn", input_shape=(1, 28, 28), output_size=10):
    # Existing code...
    
    elif model_type.lower() == "my_custom_cnn":
        return MyCustomCNN(input_channels=input_shape[0], num_classes=output_size)
    
    # Existing code...
```

## FedAvg Strategy with IPFS

The project implements the Federated Averaging (FedAvg) strategy with IPFS integration for model exchange. The main steps in the protocol are:

1. The server initializes a global model
2. The server uploads the model to IPFS and shares the CID with clients
3. Clients download the model from IPFS using the CID
4. Clients train locally and upload their updated models to IPFS
5. Clients send their model CIDs back to the server
6. The server downloads all client models from IPFS and aggregates them
7. The process repeats for multiple rounds

The implementation in `strategies/fedavg_ipfs.py` extends Flower's built-in FedAvg strategy with IPFS integration for model exchange.

## IPFS Integration

The `ipfs_connector.py` module handles all IPFS interactions:

- Uploading models to IPFS
- Retrieving models from IPFS using CIDs
- Pinning important models to ensure they remain available
- Managing IPFS connections and error handling

The IPFS protocol provides:
- Content verification through cryptographic hashing
- Resilient storage through content addressing
- Reduced direct data transfer between server and clients

## Security Considerations

- IPFS data is public by default - consider encryption for sensitive models
- Secure your IPFS nodes appropriately
- Validate client identities to prevent malicious participation
- Consider model poisoning attacks in your implementation

## Troubleshooting

Common issues and solutions:

1. **IPFS Connection Issues**
   - Ensure your IPFS daemon is running: `ipfs daemon`
   - Check IPFS API URL in config.py (default: http://localhost:5001/api/v0)
   - Verify network connectivity between nodes

2. **Model Serialization Errors**
   - Check model_utils.py for proper serialization/deserialization
   - Ensure consistent model structures

3. **Client Timeout**
   - Increase CLIENT_TIMEOUT in config.py for larger models or slower connections
   - Check network connectivity

4. **IPFS Verification**
   - Run `python utils/ipfs_verify.py` to check your IPFS setup
   - Use `--start-daemon` flag to attempt automatic daemon startup

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Flower Federated Learning Framework](https://flower.dev/)
- [IPFS](https://ipfs.tech/)
- [PySyft](https://github.com/OpenMined/PySyft) for inspiration on secure federated learning