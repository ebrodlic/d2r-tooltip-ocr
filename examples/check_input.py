import os
import sys

import torch
import onnxruntime as ort
import numpy as np

# add project root to sys.path dynamically (minimal and clean)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# -----------------------------
# 1. Parameters
# -----------------------------
batch_size = 1
channels = 3
height = 28
widths_to_test = [64, 128, 256, 512]  # test multiple widths


# -----------------------------
# 2. Load PyTorch model
# -----------------------------
from src.model import CRNN  # your model class

crnn = CRNN(img_height=28, num_channels=3, num_classes=len("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:'-+/()% ")+1)
crnn.eval()

pt_model_path = "checkpoints/crnn_best.pt"
crnn.load_state_dict(torch.load(pt_model_path, map_location="cpu"))

# -----------------------------
# 3. Load ONNX model
# -----------------------------
onnx_model_path = "models/crnn_best.onnx"
onnx_sess = ort.InferenceSession(onnx_model_path)

# ONNX expects float32 NumPy array
#onnx_input = dummy_input.numpy().astype(np.float32)
#onnx_outputs = onnx_sess.run(None, {"input": onnx_input})
#onnx_output_np = onnx_outputs[0]

# -----------------------------
# Test multiple widths
# -----------------------------
for w in widths_to_test:
    print(f"\nTesting width: {w}")
    
    # Create constant input
    dummy_input = torch.ones(batch_size, channels, height, w, dtype=torch.float32)
    
    # PyTorch output
    with torch.no_grad():
        pytorch_out = crnn(dummy_input).cpu().numpy()
    
    # ONNX output
    onnx_input = dummy_input.numpy().astype(np.float32)
    onnx_out = onnx_sess.run(None, {"input": onnx_input})[0]
    
    # Compare
    max_diff = np.max(np.abs(pytorch_out - onnx_out))
    mean_diff = np.mean(np.abs(pytorch_out - onnx_out))
    
    print(f"PyTorch output shape: {pytorch_out.shape}")
    print(f"ONNX output shape: {onnx_out.shape}")
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    print("First 5 values (PyTorch):", pytorch_out.flatten()[:5])
    print("First 5 values (ONNX):   ", onnx_out.flatten()[:5])

# -----------------------------
# 4. Compare outputs
# -----------------------------
# print("PyTorch output shape:", pytorch_output_np.shape)
# print("ONNX output shape:", onnx_output_np.shape)

# max_diff = np.max(np.abs(pytorch_output_np - onnx_output_np))
# mean_diff = np.mean(np.abs(pytorch_output_np - onnx_output_np))
# print("Max absolute difference:", max_diff)
# print("Mean absolute difference:", mean_diff)

# # Optional: print first 10 elements of flattened outputs for quick inspection
# print("\nPyTorch first 10 values:", pytorch_output_np.flatten()[:10])
# print("ONNX first 10 values:", onnx_output_np.flatten()[:10])