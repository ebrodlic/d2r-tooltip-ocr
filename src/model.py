# src/model.py
import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, rnn_hidden=128):
        """
        Minimal CRNN for line OCR
        img_height: height of input images (pixels)
        num_channels: usually 3 (RGB)
        num_classes: number of characters + 1 for CTC blank
        rnn_hidden: hidden size for LSTM
        """
        super().__init__()
        # --- CNN backbone ---
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )

        cnn_out_channels = 128
        cnn_out_height = img_height // 4  # after two maxpools
        self.rnn_input_size = cnn_out_channels * cnn_out_height

        # --- BiLSTM ---
        self.rnn = nn.LSTM(self.rnn_input_size, rnn_hidden, num_layers=2, bidirectional=True)

        # --- Final linear layer ---
        self.fc = nn.Linear(rnn_hidden*2, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.cnn(x)
        b, c, h, w = x.size()

        # Prepare for RNN: [T, B, feat]
        x = x.permute(3,0,1,2)          # [W, B, C, H]
        x = x.contiguous().view(w, b, c*h)

        x, _ = self.rnn(x)
        x = self.fc(x)                  # [T, B, num_classes]
        return x