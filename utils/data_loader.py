"""
Data loading utilities for federated learning.
"""

import os
import torch
import random
import numpy as np
from typing import Tuple, Dict, List, Optional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST


def load_mnist(data_dir: str = "./data", 
              train_batch_size: int = 32, 
              test_batch_size: int = 32,
              num_clients: int = 5,
              iid: bool = True) -> Tuple[List[DataLoader], DataLoader]:
    """
    Load and partition MNIST dataset for federated learning.
    
    Args:
        data_dir: Directory to store the dataset
        train_batch_size: Batch size for training
        test_batch_size: Batch size for testing
        num_clients: Number of clients for partitioning
        iid: Whether to partition data in an IID manner
        
    Returns:
        Tuple of (client_loaders, test_loader)
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load training and test datasets
    train_dataset = MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    # Partition training dataset for federated learning
    if iid:
        # IID partitioning: random split
        partition_size = len(train_dataset) // num_clients
        lengths = [partition_size] * num_clients
        if sum(lengths) < len(train_dataset):
            lengths[-1] += len(train_dataset) - sum(lengths)
        
        datasets = random_split(train_dataset, lengths)
    else:
        # Non-IID partitioning: sort by label and distribute
        datasets = create_non_iid_partitions(train_dataset, num_clients)
    
    # Create client data loaders
    client_loaders = [DataLoader(ds, batch_size=train_batch_size, shuffle=True) for ds in datasets]
    
    print(f"Dataset loaded and partitioned for {num_clients} clients (IID={iid})")
    return client_loaders, test_loader


def load_cifar10(data_dir: str = "./data", 
                train_batch_size: int = 32, 
                test_batch_size: int = 32,
                num_clients: int = 5,
                iid: bool = True) -> Tuple[List[DataLoader], DataLoader]:
    """
    Load and partition CIFAR-10 dataset for federated learning.
    
    Args:
        data_dir: Directory to store the dataset
        train_batch_size: Batch size for training
        test_batch_size: Batch size for testing
        num_clients: Number of clients for partitioning
        iid: Whether to partition data in an IID manner
        
    Returns:
        Tuple of (client_loaders, test_loader)
    """
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load training and test datasets
    train_dataset = CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    # Partition training dataset for federated learning
    if iid:
        # IID partitioning: random split
        partition_size = len(train_dataset) // num_clients
        lengths = [partition_size] * num_clients
        if sum(lengths) < len(train_dataset):
            lengths[-1] += len(train_dataset) - sum(lengths)
        
        datasets = random_split(train_dataset, lengths)
    else:
        # Non-IID partitioning: sort by label and distribute
        datasets = create_non_iid_partitions(train_dataset, num_clients)
    
    # Create client data loaders
    client_loaders = [DataLoader(ds, batch_size=train_batch_size, shuffle=True) for ds in datasets]
    
    print(f"Dataset loaded and partitioned for {num_clients} clients (IID={iid})")
    return client_loaders, test_loader


def create_non_iid_partitions(dataset: Dataset, num_clients: int) -> List[Subset]:
    """
    Create non-IID partitions of the dataset for federated learning.
    This function creates partitions where each client has samples from a limited set of classes.
    
    Args:
        dataset: PyTorch dataset to partition
        num_clients: Number of clients for partitioning
        
    Returns:
        List of dataset partitions
    """
    # Get labels for all samples
    if isinstance(dataset, MNIST) or isinstance(dataset, FashionMNIST) or isinstance(dataset, CIFAR10):
        labels = dataset.targets
        if torch.is_tensor(labels):
            labels = labels.numpy()
    else:
        # Try to extract labels for unknown dataset
        try:
            labels = np.array([y for _, y in dataset])
        except:
            raise ValueError("Unable to extract labels from dataset for non-IID partitioning")
    
    # Group sample indices by label
    label_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)
    
    # Determine number of classes per client
    num_classes = len(label_indices)
    classes_per_client = max(1, num_classes // 2)  # Each client gets half the classes by default
    
    # Assign classes to clients
    client_class_assignments = []
    for i in range(num_clients):
        # Assign classes_per_client random classes to this client
        if i == num_clients - 1:  # Last client gets all remaining classes
            assigned_classes = list(set(range(num_classes)) - set().union(*client_class_assignments))
        else:
            # Randomly select classes for this client
            available_classes = list(set(range(num_classes)) - set().union(*client_class_assignments) if client_class_assignments else set())
            if len(available_classes) < classes_per_client:  # If not enough unique classes available
                available_classes = list(range(num_classes))  # Allow reuse of classes
            assigned_classes = random.sample(available_classes, min(classes_per_client, len(available_classes)))
        
        client_class_assignments.append(assigned_classes)
    
    # Create client datasets
    client_datasets = []
    for client_classes in client_class_assignments:
        # Collect indices for this client's classes
        client_indices = []
        for cls in client_classes:
            # Get a portion of samples for this class
            cls_indices = label_indices[cls]
            # Each client gets a fraction of the samples for each assigned class
            samples_per_client = max(1, len(cls_indices) // sum(1 for c in client_class_assignments if cls in c))
            client_indices.extend(cls_indices[:samples_per_client])
            # Remove used indices
            label_indices[cls] = cls_indices[samples_per_client:]
        
        # Create a subset of the dataset for this client
        client_datasets.append(Subset(dataset, client_indices))
    
    # Check if any clients have very few samples and redistribute if necessary
    min_samples = min(len(ds) for ds in client_datasets)
    if min_samples < 10:  # Arbitrary threshold
        print(f"Warning: Some clients have very few samples (min={min_samples}). Redistributing...")
        return create_balanced_non_iid_partitions(dataset, num_clients)
    
    return client_datasets


def create_balanced_non_iid_partitions(dataset: Dataset, num_clients: int) -> List[Subset]:
    """
    Create balanced non-IID partitions ensuring each client has a reasonable number of samples.
    This is a fallback for create_non_iid_partitions when the distribution is too imbalanced.
    
    Args:
        dataset: PyTorch dataset to partition
        num_clients: Number of clients for partitioning
        
    Returns:
        List of dataset partitions
    """
    # Get labels for all samples
    if isinstance(dataset, MNIST) or isinstance(dataset, FashionMNIST) or isinstance(dataset, CIFAR10):
        labels = dataset.targets
        if torch.is_tensor(labels):
            labels = labels.numpy()
    else:
        try:
            labels = np.array([y for _, y in dataset])
        except:
            raise ValueError("Unable to extract labels from dataset for non-IID partitioning")
    
    # Group sample indices by label
    label_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)
    
    # Determine samples per client (approximately equal total samples)
    target_samples_per_client = len(dataset) // num_clients
    
    # Create partitions using Dirichlet distribution for label distribution
    client_datasets = []
    for _ in range(num_clients):
        client_indices = []
        remaining_samples = target_samples_per_client
        
        # Sample from each class according to a skewed distribution
        proportions = np.random.dirichlet(np.ones(len(label_indices)) * 0.5)  # Alpha=0.5 gives skewed distribution
        
        for label, p in enumerate(proportions):
            if remaining_samples <= 0:
                break
                
            # Calculate number of samples to take from this class
            class_samples = min(int(p * target_samples_per_client), len(label_indices.get(label, [])), remaining_samples)
            if class_samples <= 0 or label not in label_indices:
                continue
                
            # Take samples from this class
            sampled_indices = np.random.choice(label_indices[label], class_samples, replace=False)
            client_indices.extend(sampled_indices)
            
            # Remove sampled indices from available pool
            label_indices[label] = list(set(label_indices[label]) - set(sampled_indices))
            remaining_samples -= class_samples
        
        client_datasets.append(Subset(dataset, client_indices))
    
    # Check if the distribution is reasonably balanced
    sizes = [len(ds) for ds in client_datasets]
    print(f"Client dataset sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes)}")
    
    return client_datasets


class CustomDataset(Dataset):
    """
    Custom dataset for handling arbitrary data in federated learning.
    """
    def __init__(self, features, labels, transform=None):
        """
        Initialize custom dataset.
        
        Args:
            features: Feature data (numpy array or tensor)
            labels: Label data (numpy array or tensor)
            transform: Optional transform to apply to features
        """
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length")
            
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform and not isinstance(x, torch.Tensor):
            x = self.transform(x)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
            
        return x, y


def load_custom_dataset(features: np.ndarray, 
                        labels: np.ndarray,
                        train_ratio: float = 0.8,
                        batch_size: int = 32,
                        num_clients: int = 5,
                        iid: bool = True,
                        transform = None) -> Tuple[List[DataLoader], DataLoader]:
    """
    Load and partition custom dataset for federated learning.
    
    Args:
        features: Feature data as numpy array
        labels: Label data as numpy array
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for both training and testing
        num_clients: Number of clients for partitioning
        iid: Whether to partition data in an IID manner
        transform: Optional transform to apply to features
        
    Returns:
        Tuple of (client_loaders, test_loader)
    """
    # Create full dataset
    full_dataset = CustomDataset(features, labels, transform)
    
    # Split into train and test
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Partition training dataset for federated learning
    if iid:
        # IID partitioning: random split
        partition_size = len(train_dataset) // num_clients
        lengths = [partition_size] * num_clients
        if sum(lengths) < len(train_dataset):
            lengths[-1] += len(train_dataset) - sum(lengths)
        
        datasets = random_split(train_dataset, lengths)
    else:
        # Extract indices from train_dataset
        train_indices = train_dataset.indices
        train_features = [full_dataset.features[i] for i in train_indices]
        train_labels = [full_dataset.labels[i] for i in train_indices]
        
        # Create a temporary dataset for non-IID partitioning
        temp_dataset = CustomDataset(train_features, train_labels, transform)
        
        # Non-IID partitioning
        datasets = create_balanced_non_iid_partitions(temp_dataset, num_clients)
    
    # Create client data loaders
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in datasets]
    
    print(f"Custom dataset loaded and partitioned for {num_clients} clients (IID={iid})")
    return client_loaders, test_loader