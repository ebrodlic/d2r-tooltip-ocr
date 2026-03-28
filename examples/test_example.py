# examples/test_example.py

import os
import sys
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# --- Make src visible ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.dataset import LineOCRDataset

# --- Dataset paths ---
images_dir = os.path.join(project_root, "data", "train", "lines")
labels_csv = os.path.join(project_root, "data", "train", "labels.csv")

# --- Load dataset ---
dataset = LineOCRDataset(images_dir=images_dir, labels_csv=labels_csv)
print(f"Dataset loaded: {len(dataset)} line images")

# --- Validation ---
errors = 0
valid_indices = []

for idx, (img_tensor, label_tensor) in enumerate(dataset):
    # --- Decode label tensor to string ---
    label_str = dataset.decode_text(label_tensor.tolist())
    if label_str == "":
        print(f"Warning: Empty label at index {idx}")
        errors += 1
        continue  # skip this sample
    
    # --- Validate image tensor ---
    try:
        img_arr = img_tensor.permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
        if img_arr.size == 0 or img_arr.shape[0] < 2 or img_arr.shape[1] < 2:
            print(f"Skipping invalid image at index {idx}, shape={img_arr.shape}")
            errors += 1
            continue
        valid_indices.append(idx)
    except Exception as e:
        print(f"Error: Invalid image at index {idx} - {e}")
        errors += 1

if errors == 0:
    print("All images and labels look good!")
else:
    print(f"Found {errors} issues in dataset, skipped those images.")

# --- Display first 5 valid images ---
to_pil = ToPILImage()
for i in valid_indices[:5]:
    img_tensor, label_tensor = dataset[i]
    label_str = dataset.decode_text(label_tensor.tolist())
    
    # Convert tensor to HxWxC for display
    img_arr = img_tensor.permute(1, 2, 0).numpy()
    if img_arr.max() <= 1:  # normalize for display
        img_arr = img_arr * 255
    img_arr = img_arr.astype('uint8')
    
    plt.imshow(img_arr)
    plt.title(f"Label: {label_str}")
    plt.axis('off')
    plt.show()