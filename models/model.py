"""
Model definitions for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistCNN(nn.Module):
    """
    Simple CNN model for MNIST classification.
    This is used as the default model for the federated learning system.
    """
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SimpleMLP(nn.Module):
    """
    Simple MLP model for tabular data or simple classification tasks.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CustomCNN(nn.Module):
    """
    Customizable CNN model for image classification.
    """
    def __init__(self, input_channels=3, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def create_model(model_type="cnn", input_shape=(1, 28, 28), output_size=10):
    """
    Factory function to create a model based on configuration.
    
    Args:
        model_type: Type of model to create ("cnn", "mlp", or "custom_cnn")
        input_shape: Shape of input data (channels, height, width)
        output_size: Number of output classes
        
    Returns:
        PyTorch model instance
    """
    if model_type.lower() == "cnn":
        # For MNIST-like data
        if input_shape[0] == 1:  # Single channel (grayscale)
            return MnistCNN()
        else:
            return CustomCNN(input_channels=input_shape[0], num_classes=output_size)
    
    elif model_type.lower() == "mlp":
        # Calculate input dimension from input shape
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        return SimpleMLP(input_dim=input_dim, output_dim=output_size)
    
    elif model_type.lower() == "custom_cnn":
        return CustomCNN(input_channels=input_shape[0], num_classes=output_size)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")