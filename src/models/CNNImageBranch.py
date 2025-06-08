import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNImageBranch(nn.Module):
    """
    CNN for single‐frame feature extraction.
    Takes a 3‐channel image [B, 3, H, W] and outputs a 64‐dim feature [B, 64],
    with batch normalization, and dropout for regularization.
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim

        # Convolutional backbone with BatchNorm and Dropout
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # → [B, 32, H/2, W/2]
            nn.Dropout2d(0.2),

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # → [B, 64, H/4, W/4]
            nn.Dropout2d(0.2),

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # → [B, 128, H/8, W/8]
            nn.Dropout2d(0.2),

            # Block 4: 128 → 256 (no pooling)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            # Block 5: 256 → 128 (no pooling)
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            # Block 6: 128 → feature_dim (64)
            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),

            # Global average‐pool to 1×1 spatial
            nn.AdaptiveAvgPool2d((1, 1)),  # → [B, feature_dim, 1, 1]
        )

        # A small fully‐connected head for extra nonlinearity and dropout
        self.fc_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image tensor of shape [B, 3, H, W]

        Returns:
            feat: tensor of shape [B, feature_dim] (e.g., [B, 64])
        """
        x = self.features(x)                 # → [B, feature_dim, 1, 1]
        feat = x.view(x.size(0), -1)         # → [B, feature_dim]
        feat = self.fc_head(feat)            # → [B, feature_dim]
        return feat


class CNNImageBranch_Lite(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # idx 0
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            # idx 1
            nn.ReLU(inplace=True),
            # idx 2 (placeholder)
            nn.Identity(),
            # idx 3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # idx 4
            nn.ReLU(inplace=True),
            # idx 5 (placeholder)
            nn.Identity(),
            # idx 6
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # idx 7
            nn.ReLU(inplace=True),
            # idx 8 (placeholder)
            nn.Identity(),
            # idx 9
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # idx 10
            nn.ReLU(inplace=True),
            # idx 11
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
        )

        # After features[11], spatial is still 224×224; collapse to 1×1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Now the output dimension is 64
        self.out_dim = 64

    def forward(self, x):
        x = self.features(x)      # [B, 128, 224, 224]
        x = self.pool(x)          # [B, 128,   1,   1]
        x = x.view(x.size(0), -1) # [B, 128]
        return x

