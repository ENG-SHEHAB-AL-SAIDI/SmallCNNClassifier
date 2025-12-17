import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN with BN and DropOut 


class CNNClassifier2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.p1 = nn.MaxPool2d(2, 2)

        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.p2 = nn.MaxPool2d(2, 2)

        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.p3 = nn.MaxPool2d(2, 2)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64*4*4, 128)
        self.drop = nn.Dropout(0.5)  # dropout 50%
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.c1(x)))
        x = self.p1(x)
        x = F.relu(self.bn2(self.c2(x)))
        x = self.p2(x)
        x = F.relu(self.bn3(self.c3(x)))
        x = self.p3(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
