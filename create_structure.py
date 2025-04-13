"""
Script to create the project folder structure.
"""

import os
import sys
from pathlib import Path


def create_directory_structure():
    """
    Create the directory structure for the IPFS-enhanced Federated Learning project.
    """
    # Get the current directory
    current_dir = Path().resolve()
    
    # Directories to create
    directories = [
        "models",
        "strategies",
        "utils",
        "examples",
        "data",
        "saved_models",
        "tests"
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(os.path.join(current_dir, directory), exist_ok=True)
        # Create __init__.py in each directory
        init_file = os.path.join(current_dir, directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write(f"# {directory} module initialization\n")
    
    # Create root __init__.py
    root_init = os.path.join(current_dir, "__init__.py")
    if not os.path.exists(root_init):
        with open(root_init, "w") as f:
            f.write("# IPFS-Enhanced Federated Learning\n")
    
    print("Directory structure created successfully!")
    
    # Print structure
    print("\nProject structure:")
    for directory in [""] + directories:
        path = os.path.join(current_dir, directory)
        print(f"{path}/")
        if directory:  # Skip listing files in the root directory
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    print(f"|-- {file}")


if __name__ == "__main__":
    create_directory_structure()