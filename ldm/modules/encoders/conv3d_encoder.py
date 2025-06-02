import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DEncoder(nn.Module):
    def __init__(self, in_channels=1, dim_z=128, base_channels=64, dropout=0.0):
        super().__init__()
        self.dim_z = dim_z
        self.encoder = nn.Sequential(
            # 32³ → 16³
            nn.Conv3d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels), nn.SiLU(),
            nn.Dropout3d(p=dropout),

            # 16³ → 8³
            nn.Conv3d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*2), nn.SiLU(),
            nn.Dropout3d(p=dropout),

            # 8³ → 4³
            nn.Conv3d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*4), nn.SiLU(),
            nn.Dropout3d(p=dropout),

            # 4³ → 4³ (no stride here)
            nn.Conv3d(base_channels*4, 2*dim_z, 3, stride=1, padding=1),
            nn.GroupNorm(8, 2*dim_z), nn.SiLU(),
            nn.Dropout3d(p=dropout)
        )

        # Output head: maps to μ and logσ²
        self.output_head = nn.Conv2d(2 * dim_z, 2 * dim_z, kernel_size=1)

    def forward(self, x):  # x: (B, 1, 32, 32, 32)
        x = self.encoder(x)        # → (B, dim_z, 4, 4, 4)
        x = x.mean(dim=2)          # collapse D → (B, dim_z, 4, 4)
        x = self.output_head(x)    # → (B, 2 * dim_z, 4, 4)
        return x