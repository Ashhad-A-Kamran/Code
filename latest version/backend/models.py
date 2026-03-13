import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class DeepNeuralNetwork(nn.Module):
    def __init__(self, n_features):
        super(DeepNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

import torchvision.models as models

class ResNet18Custom(nn.Module):
    def __init__(self):
        super(ResNet18Custom, self).__init__()
        # Load a pretrained or unpretrained ResNet18 (unpretrained for faster local non-download, or pretrained if requested)
        # Using pretrained=False to avoid massive downloads for synthetic tests
        self.model = models.resnet18(pretrained=False)
        # Modify the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
        
    def forward(self, x):
        # return a sigmoid prob
        return torch.sigmoid(self.model(x))
