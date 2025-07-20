import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model.emnist_cnn import EMNIST_CNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.label_maps import get_label_map
from utils.eval_helpers import evaluate_model

model_fp32 = EMNIST_CNN()
model_fp32.load_state_dict(torch.load("weights/emnist_cnn.pth"))
model_fp32.to("cpu")
model_fp32.eval()
torch.backends.quantized.engine = 'qnnpack'

model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

torch.save(model_int8, "weights/emnist_cnn_dynamic_quantized.pt")
print("Dynamic quantized model saved to weights/emnist_cnn_dynamic_quantized.pt")

print("Evaluating quantized model...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_data = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

label_map = get_label_map("emnist_balanced")
model_int8.eval()
acc = evaluate_model(model_int8, test_loader, device="cpu", label_map=label_map)

print(f"Dynamic quantized model accuracy: {acc:.2f}%")