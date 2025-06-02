import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import reparam, deactivate
from ldm.modules.decoders.transformer import Attention, LocalAttention
import einops

class LatentTokenizer(nn.Module):
    
    def __init__(self, latent_dim, latent_size, patch_size, dim, n_head, head_dim, padding=0, dropout=0.):
        super().__init__()
        if isinstance(latent_size, int):
            latent_size = (latent_size, latent_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding
    
        # Patch embedding: Converts image patches into latent tokens
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * latent_dim, dim)

        n_patches = ((latent_size[0] + padding[0] * 2) // patch_size[0]) * ((latent_size[1] + padding[1] * 2)  // patch_size[1])

        # Positional embeddings for patch tokens
        self.posemb = nn.Parameter(torch.randn(1, n_patches, dim))  # (1, N, D)
        
        # Local + Global Attention Layers
        self.local_attn = LocalAttention(dim, window_size=patch_size[0], n_head=n_head, head_dim=head_dim)
        self.global_attn = Attention(dim, n_head=n_head, head_dim=head_dim)

    def forward(self, x, *args, **kwargs):
        
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding)  # (B, C * p * p, L)
        x = x.permute(0, 2, 1).contiguous()  # (B, N, D)

        x = self.prefc(x)  # Apply token embedding

        # Add position embeddings
        x = x + self.posemb

        # Apply Local Attention before Global Attention
        x = self.local_attn(x)
        x = self.global_attn(x)

        return x  # (B, N, D) → Tokenized representation