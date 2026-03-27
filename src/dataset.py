# src/dataset.py

import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LineOCRDataset(Dataset):
    def __init__(self, images_dir, labels_csv, char_list=None, img_height=28, transform=None):
        """
        images_dir: path to line images
        labels_csv: path to CSV, format: filename,text
        char_list: list of all possible characters (your vocab)
        img_height: fixed height for images
        transform: optional torchvision transforms
        """
        self.images_dir = images_dir
        self.img_height = img_height
        self.transform = transform

        # Load CSV
        self.samples = []
        with open(labels_csv, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 2:
                    continue
                filename, text = row
                self.samples.append((filename, text))

        # Build character dictionary
        if char_list is None:
            self.chars = self._extract_chars()
        else:
            self.chars = char_list

        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}  # 0 reserved for CTC blank
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}

    def _extract_chars(self):
        """Extract unique characters from labels"""
        chars = set()
        for _, text in self.samples:
            chars.update(list(text))
        return sorted(list(chars))

    def encode_text(self, text):
        """Convert text to list of integer indices"""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode_text(self, indices):
        """Convert list of indices back to text"""
        return ''.join([self.idx_to_char[i] for i in indices if i in self.idx_to_char])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        img_path = os.path.join(self.images_dir, filename)
        image = Image.open(img_path).convert('RGB')  # keep color

        # Resize height, maintain aspect ratio
        w, h = image.size
        new_h = self.img_height
        new_w = int(w * new_h / h)
        image = image.resize((new_w, new_h), Image.BICUBIC)

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize [0,1]
            transform_default = transforms.ToTensor()
            image = transform_default(image)

        # Encode text
        label = torch.tensor(self.encode_text(text), dtype=torch.long)

        return image, label