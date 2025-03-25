# cnn_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """A simple residual block for demonstration."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class DeepCNNBackbone(nn.Module):
    """A deeper CNN backbone with multiple residual blocks."""
    def __init__(self, input_c=3, base_ch=32):
        super(DeepCNNBackbone, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(input_c, base_ch, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(base_ch)
        self.relu  = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = BasicBlock(base_ch, base_ch, stride=1)
        self.layer2 = BasicBlock(base_ch, base_ch*2, stride=2)
        self.layer3 = BasicBlock(base_ch*2, base_ch*2, stride=1)
        self.layer4 = BasicBlock(base_ch*2, base_ch*4, stride=2)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.relu(self.bn1(self.conv1(x)))  # -> [B, base_ch, H/2, W/2]
        x = self.layer1(x)
        x = self.layer2(x)                      # -> [B, 2*base_ch, H/4, W/4]
        x = self.layer3(x)
        x = self.layer4(x)                      # -> [B, 4*base_ch, H/16, W/16]
        x = self.global_pool(x)                 # -> [B, 4*base_ch, 1,1]
        x = x.view(x.size(0), -1)               # -> [B, 4*base_ch]
        return x

if __name__ == "__main__":
    # Quick test for the CNN backbone
    model = DeepCNNBackbone()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)
