import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model.emnist_cnn import EMNIST_CNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

model = torch.load("weights/emnist_cnn_dynamic_quantized.pt", map_location="cpu")
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

correct = 0
total = 0
total_time = 0

with torch.no_grad():
    for x, y in test_loader:
        start = time.time()
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        batch_time = time.time() - start
        total_time += batch_time
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Simulated Kindle Accuracy: {100 * correct / total:.2f}%")