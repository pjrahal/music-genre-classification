import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.GELU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class GenreCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, groups=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, groups=4),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.se = SEBlock(128)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(1)
        x = self.features(x)
        x = self.se(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
