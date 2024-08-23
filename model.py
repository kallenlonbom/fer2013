import torch.nn as nn
import matplotlib.pyplot as plt

class FERecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # convolutional 1
            # 1,48,48
            nn.Conv2d(1, 16, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            # pooling 1
            # 16,48,48
            nn.MaxPool2d((2,2)),
            # convolutional 2
            # 16,24,24
            nn.Conv2d(16, 32, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            # pooling 2
            # 32,24,24
            nn.MaxPool2d((2,2)),
            # convolutional 3
            # 32,12,12
            nn.Conv2d(32, 64, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # pooling 3
            # 64,12,12
            nn.MaxPool2d((2,2)),
            # 64,6,6
            nn.Flatten(),
            # fully connected layers
            # 2304
            nn.Linear(2304, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            # output
            nn.Linear(512,7)
        )
    
    def forward(self, x):
        return self.layers(x)