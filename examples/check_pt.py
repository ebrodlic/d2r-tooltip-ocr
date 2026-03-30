import os
import sys
import torch

# add project root to sys.path dynamically (minimal and clean)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model import CRNN  # your model class

char_list = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:'-+/()% "))
num_classes = len(char_list) + 1  # +1 for CTC blank

model = CRNN(img_height=28, num_channels=3, num_classes=num_classes)

model.load_state_dict(torch.load("checkpoints/crnn_best.pt", map_location="cpu"))
model.eval()

for w in [100, 150, 200, 250, 300]:
    x = torch.randn(1, 3, 28, w)
    out = model(x)
    print(f"Input width: {w}, Output shape: {out.shape}")

