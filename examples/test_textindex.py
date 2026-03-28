import os
import sys

# add project root to sys.path dynamically (minimal and clean)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.dataset import LineOCRDataset

char_list = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:'-+/()% "))

dataset = LineOCRDataset(images_dir="data/train/lines", labels_csv="data/train/labels.csv", char_list=char_list)

print("Vocabulary (char list):")
print(dataset.chars)
print(f"Number of characters (vocab size): {len(dataset.chars)}")

print("\nCharacter to index mapping (0 is CTC blank):")
for char, idx in dataset.char_to_idx.items():
    print(f"'{char}': {idx}")