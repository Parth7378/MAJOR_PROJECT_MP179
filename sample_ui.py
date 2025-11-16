import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import gradio as gr
from torchvision import models
import torch.nn as nn

MODEL_PATH = r"C:\Users\SAKSHI\Desktop\MAJOR_PROJECT_GEU\model1_pose\model1_best.pth"
NUM_KP = 17

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# BUILD EXACT SAME MODEL USED DURING TRAINING
# ------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 34)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ------------------------------------------------
# Preprocess (same as training)
# ------------------------------------------------
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

def draw_kp(image, preds):
    h, w = image.shape[:2]
    out = image.copy()

    for i in range(NUM_KP):
        x = preds[2*i]
        y = preds[2*i + 1]

        if x == 0 and y == 0:
            continue

        px = int(x * w)
        py = int(y * h)
        cv2.circle(out, (px, py), 4, (0, 255, 0), -1)

    return out

def predict(image):
    pil = Image.fromarray(image).convert("RGB")

    inp = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp).cpu().numpy().flatten()

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out = draw_kp(img_bgr, pred)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    return out_rgb

ui = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Thermal Image"),
    outputs=gr.Image(type="numpy", label="Output with Keypoints"),
    title="Model-1 Thermal Pose Estimation"
)

ui.launch()
