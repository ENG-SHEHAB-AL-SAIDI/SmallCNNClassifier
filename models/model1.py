import torch as nn


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.c1 = nn.Conv2d(10, 5, kernel_size=3)

    def forward(self, x):
        return self.layer(x)