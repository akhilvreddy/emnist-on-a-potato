import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.emnist_cnn import EMNIST_CNN
from utils.label_maps import get_label_map
from utils.eval_helpers import evaluate_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
test_data  = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=1000, shuffle=False)

model = EMNIST_CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "emnist_cnn.pth")

label_map = get_label_map("emnist_balanced")
acc = evaluate_model(model, test_loader, device=device, label_map=label_map)
print(f"EMNIST Test Accuracy: {acc:.2f}%")