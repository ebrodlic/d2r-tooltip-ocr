import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

# add project root to sys.path dynamically (minimal and clean)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# -------------------------
# CONFIG
# -------------------------
IMAGE_PATH = "data/infer/lines/2026-03-28_19-22-48_06.png"
CHECKPOINT = "checkpoints/crnn_best.pt"
ONNX_PATH = "models/d2r_tooltip_crnn_best.onnx"

# -------------------------
# CHARSET (same as yours)
# -------------------------
char_list = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:'-+/()% "))
idx_to_char = {idx + 1: char for idx, char in enumerate(char_list)}

# -------------------------
# PREPROCESS (EXACT COPY)
# -------------------------
def pad_to_width(img_tensor, target_width=512):
    _, _, h, w = img_tensor.shape
    if w >= target_width:
        return img_tensor[:, :, :, :target_width]

    pad_width = target_width - w
    pad = torch.full((1, 3, h, pad_width), -1.0)  # -1.0 because we normalized to [-1, 1], so this is effectively black padding
    return torch.cat([img_tensor, pad], dim=3)


transform_infer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform_infer(img).unsqueeze(0)  # [1, C, H, W]
input_tensor = pad_to_width(input_tensor, target_width=512)

print("Input shape:", input_tensor.shape)
print("Input range:", input_tensor.min().item(), input_tensor.max().item())

np.save("debug/debug_input.npy", input_tensor.numpy())
np_input = input_tensor.cpu().numpy().astype(np.float32)
np_input.tofile("debug/debug_input.bin")

# -------------------------
# LOAD PYTORCH MODEL
# -------------------------
from src.model import CRNN

num_classes = len(char_list) + 1
model = CRNN(img_height=28, num_channels=3, num_classes=num_classes)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

with torch.no_grad():
    torch_logits = model(input_tensor)  # [T, B, C]
    torch_log_probs = torch.nn.functional.log_softmax(torch_logits, dim=2)

torch_np = torch_log_probs.cpu().numpy()

print("\nPyTorch output shape:", torch_np.shape)

# -------------------------
# LOAD ONNX MODEL
# -------------------------
sess = ort.InferenceSession(ONNX_PATH)

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

onnx_logits = sess.run([output_name], {input_name: input_tensor.numpy()})[0]

np.save("debug/debug_onnx_output.npy", onnx_logits)
np_output = onnx_logits.astype(np.float32)
np_output.tofile("debug/debug_onnx_output.bin")  # new format

# APPLY SAME log_softmax
onnx_log_probs = torch.nn.functional.log_softmax(
    torch.from_numpy(onnx_logits), dim=2
).numpy()

print("ONNX output shape:", onnx_log_probs.shape)

# -------------------------
# COMPARE
# -------------------------
print("\n=== COMPARISON ===")

print("Max abs diff:", np.max(np.abs(torch_np - onnx_log_probs)))
print("Mean abs diff:", np.mean(np.abs(torch_np - onnx_log_probs)))
print("Allclose:", np.allclose(torch_np, onnx_log_probs, atol=1e-4))

# -------------------------
# ARGMAX CHECK
# -------------------------
torch_argmax = torch_np.argmax(axis=2)
onnx_argmax = onnx_log_probs.argmax(axis=2)

print("Argmax equal:", np.array_equal(torch_argmax, onnx_argmax))

# -------------------------
# CTC DECODE FUNCTION
# -------------------------
def decode(pred_indices):
    pred_text = ""
    prev_idx = 0
    for idx in pred_indices:
        if idx != 0 and idx != prev_idx:
            pred_text += idx_to_char.get(idx, "?")
        prev_idx = idx
    return pred_text

# Decode both
torch_seq = torch_argmax[:, 0]
onnx_seq = onnx_argmax[:, 0]

print("\nTorch decoded:", decode(torch_seq))
print("ONNX decoded:", decode(onnx_seq))