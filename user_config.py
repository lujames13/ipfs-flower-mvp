"""
用戶自定義配置模組，用於分離應用程式配置和內建配置。
從原始 config.py 遷移的配置參數。
"""

class UserConfig:
    """用戶配置類，將配置參數封裝在類實例中以避免名稱衝突"""
    
    # IPFS Configuration
    IPFS_API_URL = "http://localhost:5001/api/v0"  # IPFS API endpoint
    IPFS_GATEWAY_URL = "http://localhost:8081/ipfs/"  # IPFS Gateway URL for viewing content

    # Federated Learning Configuration
    NUM_ROUNDS = 10                   # Number of federated learning rounds
    MIN_CLIENTS = 2                   # Minimum number of clients for aggregation
    CLIENT_TIMEOUT = 120              # Client timeout in seconds
    SERVER_ADDRESS = "[::]:8081"      # Updated server address for Flower 1.16.0
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

# 創建一個全局實例以便導入
user_config = UserConfig()