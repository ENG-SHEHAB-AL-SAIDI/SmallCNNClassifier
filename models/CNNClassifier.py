import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # for input 32x32
        self.fc2 = nn.Linear(128, 10)          # 10 classes

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.p1(x)
        x = F.relu(self.c2(x))
        x = self.p2(x)
        x = F.relu(self.c3(x))
        x = self.p3(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
