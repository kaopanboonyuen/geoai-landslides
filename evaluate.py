"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: Deep Residual Neural Networks with Self-Attention for Landslide Susceptibility Mapping
Description: This script evaluates the performance of the trained ResNet152+Att model.
License: MIT License
"""

import torch
from torch.utils.data import DataLoader
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

def evaluate_model(config):
    # Load configuration
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset and dataloader
    dataset = CustomDataset(cfg['data_path'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
    
    # Initialize model
    model = ResNet152WithAttention(num_classes=cfg['num_classes']).to(device)
    model.load_state_dict(torch.load(cfg['model_path']))
    model.eval()
    
    # Evaluation loop
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy}%')

if __name__ == "__main__":
    evaluate_model('config.yaml')