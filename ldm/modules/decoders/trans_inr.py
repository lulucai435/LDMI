import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from ldm.util import instantiate_from_config
from utils import make_coord_grid   

def normalize_weights(w, x):
    return F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)

def scale_weights(w, x):
    return w * (1 + x.repeat(1, 1, w.shape[2] // x.shape[2]))  # Scale instead of normalize

def identity_weights(w, x):
    """Weight update strategy: Identity (No modification)"""
    return w  # Keeps weights unchanged

# Define the update strategy mapping
update_strategies = {
    "normalize": normalize_weights,
    "scale": scale_weights,
    "identity": identity_weights
}


class TransInr(nn.Module):
    def __init__(self, tokenizer, inr, n_groups, data_shape, transformer, update_strategy="normalize", *args, **kwargs):
        super().__init__()
        self.inr = instantiate_from_config(inr)
        self.register_buffer('shared_coord', make_coord_grid(data_shape, (-1, 1)), persistent=False)

    def forward(self, data=None, nsamples=None, coord=None, **kwargs):
        # Get batch size from input data if available, otherwise default to 1
        B = data.shape[0] if data is not None else 1

        if coord is None:
            coord = self.shared_coord

        # Handle coordinate batching
        if len(coord.shape) == 3:  # 2D case
            coord = einops.repeat(coord, 'h w d -> b h w d', b=B)
        elif len(coord.shape) == 4:  # 3D case
            coord = einops.repeat(coord, 'x y z d -> b x y z d', b=B)

        # Generate prediction
        pred = self.inr(coord)

        # Rearrange output dimensions
        if len(pred.shape) == 4:  # 2D case
            pred = pred.permute(0, -1, 1, 2).contiguous()
        elif len(pred.shape) == 5:  # 3D case
            pred = pred.permute(0, -1, 1, 2, 3).contiguous()

        return pred

    def get_last_layer(self):
        return self.inr.get_last_layer()
