# -*- coding: utf-8 -*-
"""
U-Net Model dla segmentacji zdjęć satelitarnych
Klasy:
    0 - Inne (tło)
    1 - Tereny zurbanizowane (Urban)
    2 - Wody (Water)
    3 - Tereny zielone (Vegetation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Podwójna konwolucja: (Conv2D -> BatchNorm -> ReLU) x 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling: UpConv/Bilinear -> Concat -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Dopasowanie rozmiaru
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Warstwa wyjściowa: Conv 1x1"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net dla segmentacji satelitarnej
    
    Args:
        n_channels: liczba kanałów wejściowych (3 dla RGB)
        n_classes: liczba klas segmentacji (4: inne, urban, water, vegetation)
        bilinear: użyj bilinear upsampling zamiast transposed conv
    """
    
    def __init__(self, n_channels=3, n_classes=4, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (ścieżka kurcząca)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (ścieżka rozszerzająca)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder z skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits
    
    def predict(self, x):
        """Predykcja z softmax"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


# Słownik klas i kolorów do wizualizacji
CLASS_NAMES = {
    0: "Inne",
    1: "Tereny zurbanizowane",
    2: "Wody",
    3: "Tereny zielone"
}

CLASS_COLORS = {
    0: (128, 128, 128),    # Szary - Inne
    1: (255, 0, 0),        # Czerwony - Urban
    2: (0, 0, 255),        # Niebieski - Water
    3: (0, 255, 0)         # Zielony - Vegetation
}


def get_model(n_channels=3, n_classes=4, pretrained_path=None):
    """
    Funkcja pomocnicza do tworzenia modelu
    
    Args:
        n_channels: liczba kanałów wejściowych
        n_classes: liczba klas
        pretrained_path: ścieżka do wag modelu (opcjonalnie)
    """
    model = UNet(n_channels=n_channels, n_classes=n_classes)
    
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Załadowano wagi z: {pretrained_path}")
    
    return model


if __name__ == "__main__":
    # Test modelu
    model = UNet(n_channels=3, n_classes=4)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Liczba parametrów: {sum(p.numel() for p in model.parameters()):,}")

