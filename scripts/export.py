import os
import sys
import torch

# --- Make src visible ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model import CRNN

char_list = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:'-+/()% "))
num_classes = len(char_list) + 1  # +1 for CTC blank

# Paths
checkpoint_path = "checkpoints/crnn_best.pt"

# Initialize model
model = CRNN(img_height=28, num_channels=3, num_classes=num_classes)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()  # important for inference

# Dummy input: 1 image, 3 channels, height 28, width 200 (example width)
dummy_input = torch.randn(1, 3, 28, 200)  # adjust width as needed

onnx_path = "models/d2r_tooltip_crnn_best.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,        # store weights in ONNX file
    opset_version=17,          # newer ONNX version is usually safer
    do_constant_folding=True,  # optimize constants
    input_names=['input'],     # optional: name your input
    output_names=['output'],   # optional: name your output
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # optional: batch size dynamic
)
print(f"Model exported to {onnx_path}")


# Validate the ONNX model
import onnx

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Optional: Run inference with ONNX Runtime to verify it works
import onnxruntime as ort

ort_session = ort.InferenceSession(onnx_path)
dummy_input_np = dummy_input.numpy()
outputs = ort_session.run(None, {'input': dummy_input_np})
print(outputs[0].shape)