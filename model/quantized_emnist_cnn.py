# ONLY FOR INFERENCE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import Linear, Conv2d
from torch.ao.nn.quantized import ReLU
from torch.nn import MaxPool2d

class QuantizedEMNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = ReLU(inplace=True)

        self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = ReLU(inplace=True)

        self.pool = MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = Linear(32 * 7 * 7, 128)
        self.relu3 = ReLU(inplace=True)
        self.fc2 = Linear(128, 47)

    # same as training forward method
    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x))) # Conv1 + ReLU + Pool
        x = self.pool(self.relu2(self.conv2(x))) # Conv2 + ReLU + Pool
        x = x.view(-1, 32 * 7 * 7) # flatten
        x = self.relu3(self.fc1(x))
        return self.fc2(x)