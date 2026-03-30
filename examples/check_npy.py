import os
import sys
import numpy as np

# add project root to sys.path dynamically (minimal and clean)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)



# Load saved tensors
input_np = np.load("debug/debug_input.npy")
output_np = np.load("debug/debug_onnx_output.npy")

print("=== INPUT ===")
print("Shape:", input_np.shape)
print("Dtype:", input_np.dtype)
print("Min/Max:", input_np.min(), input_np.max())
print("First 10 values:", input_np.flatten()[:10])

print("\n=== OUTPUT (logits) ===")
print("Shape:", output_np.shape)
print("Dtype:", output_np.dtype)
print("Min/Max:", output_np.min(), output_np.max())
print("First 10 values:", output_np.flatten()[:10])