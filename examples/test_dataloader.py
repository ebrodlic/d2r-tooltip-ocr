import os
import sys

# add project root to sys.path dynamically (minimal and clean)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.dataloader import get_dataloader
from src.dataset import LineOCRDataset

loaded_dataset = LineOCRDataset(images_dir="data/train/lines", labels_csv="data/train/labels.csv")
loaded_data = get_dataloader(loaded_dataset, batch_size=4)

for images, labels, label_lengths in loaded_data:
    print(images.shape)        # [B, 3, 28, max_width]
    print(labels.shape)        # [total_chars]
    print(label_lengths.shape) # [B]
    break