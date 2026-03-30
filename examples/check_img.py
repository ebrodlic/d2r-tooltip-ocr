from PIL import Image
import numpy as np
import onnxruntime as ort

import os
import sys

# add project root to sys.path dynamically (minimal and clean)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- Step 0: Setup --- #
image_path = "data/infer/lines/2026-03-28_19-22-48_06.png"
onnx_path = "models/d2r_tooltip_crnn_best.onnx"
target_width = 512


img = Image.open(image_path).convert("RGB")  # 3 channels, RGB

# Convert to float32 numpy array, shape H x W x C
np_img = np.array(img, dtype=np.float32)
H, W, C = np_img.shape
print("Step 1 - raw float32 array, first 10 flattened values:")
print(np_img.flatten()[:10])

# --- Step 2: Scale 0-1 ---
np_img /= 255.0
print("Step 2 - scaled 0-1, first 10 flattened values:")
print(np_img.flatten()[:10])

# --- Step 3: Normalize (mean=0.5, std=0.5) ---
np_img = (np_img - 0.5) / 0.5
print("Step 3 - normalized, first 10 flattened values:")
print(np_img.flatten()[:10])

# --- Step 4: HWC -> CHW ---
np_img_chw = np_img.transpose(2, 0, 1)
print("Step 4 - CHW shape:", np_img_chw.shape)
print("Step 4 - flattened first 10 values:")
print(np_img_chw.flatten()[:10])

# --- Step 5: Pad width to 512 --- #
pad_width = target_width - W
if pad_width > 0:
    # pad along width (axis=2 in CHW)
    np_img_chw = np.pad(np_img_chw, ((0,0), (0,0), (0,pad_width)), 'constant', constant_values=-1.0)
print("Step 5 - padded CHW shape:", np_img_chw.shape)
print("Step 5 - padded flattened first 10 values:")
print(np_img_chw.flatten()[:10])

# --- Step 6: Add batch dimension --- #
input_tensor = np.expand_dims(np_img_chw, axis=0)  # shape [1, 3, H, 512]

# --- Step 7: Run ONNX inference --- #
session = ort.InferenceSession(onnx_path)
inputs = {session.get_inputs()[0].name: input_tensor.astype(np.float32)}
onnx_out = session.run(None, inputs)[0]  # logits
print("Step 7 - ONNX output shape:", onnx_out.shape)
print("Step 7 - first 10 logits:", onnx_out.flatten()[:10])

# --- Step 8: CTC decoding --- #
# simple greedy decoding
def ctc_greedy_decode(logits, blank=0):
    # logits: [T, 1, num_classes] -> remove batch dim
    logits = logits[:,0,:]
    pred_labels = np.argmax(logits, axis=1)
    prev = None
    output = []
    for l in pred_labels:
        if l != blank and l != prev:
            output.append(l)
        prev = l
    return output

decoded_indices = ctc_greedy_decode(onnx_out)
print("Step 8 - decoded indices:", decoded_indices)

# --- Step 9: Map indices to characters correctly ---
char_list = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:'-+/()% "))
idx_to_char = {idx + 1: char for idx, char in enumerate(char_list)}

decoded_text = ""
prev_idx = 0
for idx in decoded_indices:
    if idx != 0 and idx != prev_idx:
        decoded_text += idx_to_char.get(idx, "?")
    prev_idx = idx

print("Predicted text:", decoded_text)