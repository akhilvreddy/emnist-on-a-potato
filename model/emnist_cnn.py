import torch
import torch.nn as nn
import torch.nn.functional as F

class EMNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # 28x28 → 28x28
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 28x28 → 28x28
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 → 14x14 → 7x7

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 47)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x))) # Conv1 + ReLU + Pool
        x = self.pool(self.relu2(self.conv2(x))) # Conv2 + ReLU + Pool
        x = x.view(-1, 32 * 7 * 7) # flatten
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

    def fuse_model(self):
        """
        Fuses convolution + relu and linear + relu layers in-place for quantization.
        Must be called before quantization steps.
        """
        torch.quantization.fuse_modules(
            self,
            [['conv1', 'relu1'], ['conv2', 'relu2'], ['fc1', 'relu3']],
            inplace=True
        )