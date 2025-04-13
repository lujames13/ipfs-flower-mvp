# Utils module initialization
from .data_loader import load_mnist, load_cifar10, create_non_iid_partitions
from .model_utils import get_model_parameters, set_model_parameters, save_model, load_model
from .ipfs_protocol import IPFSModelExchangeProtocol

__all__ = [
    "load_mnist",
    "load_cifar10",
    "create_non_iid_partitions",
    "get_model_parameters",
    "set_model_parameters",
    "save_model",
    "load_model",
    "IPFSModelExchangeProtocol",
]