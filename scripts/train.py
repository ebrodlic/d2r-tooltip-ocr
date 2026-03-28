import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# --- Make src visible ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.dataset import LineOCRDataset
from src.model import CRNN
from src.dataloader import get_dataloader

# --- Dataset & DataLoader ---
images_dir = os.path.join(project_root, "data", "train", "lines")
labels_csv = os.path.join(project_root, "data", "train", "labels.csv")

char_list = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:'-+/()% "))
transform = transforms.Compose([
    transforms.ToTensor(),                     # convert to [C,H,W]
    transforms.ColorJitter(brightness=0.1, contrast=0.1),        # simulate lighting changes    
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)              # normalize
])

dataset = LineOCRDataset(images_dir=images_dir, labels_csv=labels_csv, char_list=char_list, transform=transform)
print(f"Dataset loaded: {len(dataset)} samples")

dataloader = get_dataloader(dataset, batch_size=8)
print(f"DataLoader created with batch size 8, total batches: {len(dataloader)}")

# --- Model ---
num_classes = len(dataset.chars) + 1  # +1 for CTC blank
model = CRNN(img_height=28, num_channels=3, num_classes=num_classes)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Optimizer & Loss ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
ctc_loss = nn.CTCLoss(blank=0)

# --- Checkpoints folder ---
checkpoint_dir = os.path.join(project_root, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
best_loss = float("inf")

# --- Training loop ---
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()
        logits = model(images)  # [T, B, num_classes]
        log_probs = nn.functional.log_softmax(logits, dim=2)
        input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long, device=device)

        loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if (batch_idx+1) % 50 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(dataloader)} Loss: {loss.item():.4f}")

        
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished, average loss: {avg_loss:.4f}")

    # --- Save checkpoint ---
    checkpoint_path = os.path.join(checkpoint_dir, f"crnn_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

      # --- Save best model separately ---
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = os.path.join(checkpoint_dir, "crnn_best.pt")
        torch.save(model.state_dict(), best_path)
        print(f"Saved new best model: {best_path}")