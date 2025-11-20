import torch.nn as nn

class dH_to_dT_conv(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # local receptive field
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # outputs 2 channels
        )

    def forward(self, x):
        return self.net(x)