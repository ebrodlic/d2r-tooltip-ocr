# examples/infer.py
import os
import sys
import torch
import argparse
from PIL import Image
from torchvision import transforms

# --- Make src visible ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.dataset import LineOCRDataset
from src.model import CRNN

# --- Command line arguments ---
parser = argparse.ArgumentParser(description="Run OCR on a single line image")
parser.add_argument("--image", required=True, help="Path to line image")
parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
args = parser.parse_args()

# --- Load dataset char list ---
# This assumes you have the same char_list as training
char_list = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:'-+/()% "))
char_to_idx = {char: idx + 1 for idx, char in enumerate(char_list)}  # 0 reserved for CTC blank
idx_to_char = {idx + 1: char for idx, char in enumerate(char_list)}

# --- Load model ---
num_classes = len(char_list) + 1
model = CRNN(img_height=28, num_channels=3, num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

checkpoint = args.checkpoint
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()
print(f"Loaded model from {checkpoint}")

# --- Preprocess image ---
transform_infer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

image_path = args.image
img = Image.open(image_path).convert("RGB")
img_tensor = transform_infer(img).unsqueeze(0).to(device)  # [1, C, H, W]

# --- Forward pass ---
with torch.no_grad():
    logits = model(img_tensor)             # [T, B, num_classes]
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    pred_indices = log_probs.argmax(dim=2)  # [T, B]

# --- CTC decoding ---
pred_indices = pred_indices[:, 0].cpu().numpy()  # take batch=0
pred_text = ""
prev_idx = 0
for idx in pred_indices:
    if idx != 0 and idx != prev_idx:  # skip blank and duplicates
        pred_text += idx_to_char[idx]
    prev_idx = idx

print(f"Predicted text: {pred_text}")