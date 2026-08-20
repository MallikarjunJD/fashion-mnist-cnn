"""
CNN architecture for Fashion-MNIST classification.

This module is imported by BOTH src/train.py and every deploy/*.py app, so the
exact architecture that produced the saved weights is guaranteed to be the
exact architecture running in the deployed demo — no risk of drift.
"""

import torch.nn as nn
import torch.nn.functional as F


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Block 1: 1x28x28 -> 32x28x28 -> pooled to 32x14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 32x14x14 -> 64x14x14 -> pooled to 64x7x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # -> (B, 32, 14, 14)
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> (B, 64, 7, 7)
        x = self.dropout_conv(x)

        x = x.flatten(1)                                  # -> (B, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)                                   # raw logits, no softmax here
        return x


if __name__ == "__main__":
    import torch

    model = FashionCNN()
    dummy = torch.randn(4, 1, 28, 28)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect (4, 10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
