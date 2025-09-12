import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks=4):
        super(Generator, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        residual_blocks = [ResidualBlock(64, 64) for _ in range(num_residual_blocks)]
        self.residual_layers = nn.Sequential(*residual_blocks)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.residual_layers(out)
        out = self.final_conv(out)
        return out
