"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: Deep Residual Neural Networks with Self-Attention for Landslide Susceptibility Mapping
Description: This script handles the training of the ResNet152+Att model on geospatial data.
License: MIT License
"""

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from models import ResNet152WithAttention

class CustomDataset(Dataset):
    def __init__(self, data_path):
        # Initialize dataset
        self.data_path = data_path
        # Load and preprocess your data here

    def __len__(self):
        # Return the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Return a sample from the dataset
        sample = self.data[idx]
        return sample

def train_model(config):
    # Load configuration
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset and dataloader
    dataset = CustomDataset(cfg['data_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = ResNet152WithAttention(num_classes=cfg['num_classes']).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    
    # Training loop
    model.train()
    for epoch in range(cfg['num_epochs']):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{cfg["num_epochs"]}], Loss: {running_loss/len(dataloader)}')

if __name__ == "__main__":
    train_model('config.yaml')