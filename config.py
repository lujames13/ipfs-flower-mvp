"""
Configuration file for IPFS-Enhanced Federated Learning with Flower
"""

# IPFS Configuration
IPFS_API_URL = "http://localhost:5001/api/v0"  # IPFS API endpoint
IPFS_GATEWAY_URL = "http://localhost:8080/ipfs/"  # IPFS Gateway URL for viewing content

# Federated Learning Configuration
NUM_ROUNDS = 10                   # Number of federated learning rounds
MIN_CLIENTS = 2                   # Minimum number of clients for aggregation
CLIENT_TIMEOUT = 120              # Client timeout in seconds
SERVER_ADDRESS = "0.0.0.0:8080"   # Server address for Flower
MIN_AVAILABLE_CLIENTS = 2         # Minimum number of available clients to start round
FRACTION_FIT = 1.0                # Fraction of clients used for training
FRACTION_EVALUATE = 0.5           # Fraction of clients used for evaluation

# Model Configuration
MODEL_TYPE = "cnn"                # Model type: "cnn", "mlp", etc.
INPUT_SHAPE = (1, 28, 28)         # Input shape for the model
OUTPUT_SIZE = 10                  # Output size (e.g., number of classes)
LEARNING_RATE = 0.01              # Learning rate for client training
BATCH_SIZE = 32                   # Batch size for training
EPOCHS_PER_ROUND = 5              # Local epochs per FL round

# Logging Configuration
LOG_LEVEL = "INFO"                # Logging level
SAVE_MODELS = True                # Whether to save models locally
MODEL_SAVE_PATH = "./saved_models"  # Path to save models