import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def collate_fn(batch):
    images, labels = zip(*batch)

    # --- pad images to same width ---
    widths = [img.shape[2] for img in images]
    max_width = max(widths)

    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_w = max_width - w
        padded = F.pad(img, (0, pad_w))  # pad right
        padded_images.append(padded)

    images_tensor = torch.stack(padded_images)  # [B, C, H, W]

    # --- labels ---
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_tensor = torch.cat(labels) # CTC expects 1D concatenated labels

    return images_tensor, labels_tensor, label_lengths


def get_dataloader(dataset, batch_size=8):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )