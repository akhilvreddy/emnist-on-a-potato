import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from model.mnist_mlp import MNISTNet
from utils.label_maps import get_label_map
from scipy.ndimage import center_of_mass, shift

label_map = get_label_map("mnist")

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

model = MNISTNet().to(device)
model.load_state_dict(torch.load("weights/mnist_model.pth", map_location=device))
model.eval()

st.title("MNIST Digit Classifier")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, 0]
    img = cv2.resize(img, (28, 28))
    img = 255 - img

    com = center_of_mass(img)
    shifted = shift(img, shift=np.array([14, 14]) - com)  # shift to center

    img = img / 255.0
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    st.markdown(f"### Prediction: **{label_map[pred]}**")
    st.bar_chart(probs.cpu().numpy()[0])
else:
    st.info("Draw a digit to get a prediction.")