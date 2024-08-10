"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: Deep Residual Neural Networks with Self-Attention for Landslide Susceptibility Mapping
Description: This script performs inference using the trained ResNet152+Att model to generate susceptibility maps.
License: MIT License
"""

import torch
from PIL import Image
from torchvision import transforms
from models import ResNet152WithAttention

def infer(model_path, image_path):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet152WithAttention(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()
    
    print(f'Predicted Susceptibility: {prediction}')

if __name__ == "__main__":
    infer('path_to_model.pth', 'path_to_image.jpg')