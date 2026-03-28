import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# add project root to sys.path dynamically (minimal and clean)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.dataset import LineOCRDataset
from src.dataloader import get_dataloader

# --- Dataset & DataLoader ---
images_dir = os.path.join(project_root, "data", "train", "lines")
labels_csv = os.path.join(project_root, "data", "train", "labels.csv")

dataset = LineOCRDataset(images_dir=images_dir, labels_csv=labels_csv)
print(f"Dataset loaded: {len(dataset)} samples")

dataloader = get_dataloader(dataset, batch_size=4)
print(f"DataLoader created with batch size 4, total batches: {len(dataloader)}")

# --- Minimal CRNN ---
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, rnn_hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        cnn_out_channels = 128
        cnn_out_height = img_height // 4
        self.rnn_input_size = cnn_out_channels * cnn_out_height
        self.rnn = nn.LSTM(self.rnn_input_size, rnn_hidden, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden*2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(3,0,1,2)  # [W, B, C, H]
        x = x.contiguous().view(w, b, c*h)  # [T, B, feat]
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

num_classes = len(dataset.chars) + 1  # +1 for CTC blank
model = CRNN(img_height=28, num_channels=3, num_classes=num_classes)

# --- Loss ---
ctc_loss = nn.CTCLoss(blank=0)

# --- One batch test ---
images, labels, label_lengths = next(iter(dataloader))
logits = model(images)  # [T, B, num_classes]
log_probs = nn.functional.log_softmax(logits, dim=2)
input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long)

loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
print("CTC Loss on one batch:", loss.item())