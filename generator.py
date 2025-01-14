import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, feature_maps):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # State: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # Output: img_channels x 32 x 32
        )

    def forward(self, x):
        return self.model(x)
