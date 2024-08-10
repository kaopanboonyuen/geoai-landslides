"""
Author: Teerapong Panboonyuen (Kao Panboonyuen)
Project: Deep Residual Neural Networks with Self-Attention for Landslide Susceptibility Mapping
Description: This module defines the enhanced ResNet152 model with self-attention mechanisms for landslide susceptibility mapping.
License: MIT License
"""

import torch
import torch.nn as nn
import torchvision.models as models

class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fc_out = nn.Conv2d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C)
        key = self.key_conv(x).view(B, -1, H * W)  # (B, C, HW)
        value = self.value_conv(x).view(B, -1, H * W)  # (B, C, HW)

        attention = torch.bmm(query, key)  # (B, HW, HW)
        attention = torch.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)  # (B, C, H, W)
        out = self.fc_out(out)
        return out

class ResNet152WithAttention(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet152WithAttention, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the original classification head
        self.attention = SelfAttention(in_channels=2048, out_channels=512)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.resnet(x)
        attention_out = self.attention(features)
        pooled_features = torch.mean(attention_out, dim=[2, 3])  # Global average pooling
        out = self.fc(pooled_features)
        return out