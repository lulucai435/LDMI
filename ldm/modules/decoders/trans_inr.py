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
        dim = transformer['params']['dim']

        # We will use a special tokenizer for the latents (see latent_tokenizer.py)
        # The tokenizer will need to load the encoder and the prior
        self.tokenizer = instantiate_from_config(tokenizer, extra_args={'dim': dim})
        self.inr = instantiate_from_config(inr)
        self.transformer = instantiate_from_config(transformer)

        self.register_buffer('shared_coord', make_coord_grid(data_shape, (-1, 1)), persistent=False)
        
        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        for name, shape in self.inr.param_shapes.items():
            self.base_params[name] = nn.Parameter(self.inr.init_wb(shape, name=name))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, shape[0] - 1),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))
        self.update_strategy = update_strategies[update_strategy]

        nparams = sum(p.numel() for p in self.transformer.parameters())
        nparams += sum(p.numel() for p in self.tokenizer.parameters())
        nparams += sum(p.numel() for p in self.base_params.values())
        nparams += self.wtokens.numel()
        
        print(f"Hypernetwork Parameters: {nparams / 1e6:.2f}M")

    def forward(self, data=None, nsamples=None, coord=None, **kwargs):
        
        # Inside the latent_tokenizer, we will use the encoder to map x -> z, then tokenize z
        dtokens = self.tokenizer(data, nsamples, **kwargs)
        B = dtokens.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        # If self.transformer is a transformer encoder we concatenate and feed
        if self.transformer.__class__.__name__ == 'TransformerEncoder':
            trans_out = self.transformer(torch.cat([dtokens, wtokens], dim=1))
        elif self.transformer.__class__.__name__ == 'Transformer':
            trans_out = self.transformer(src=dtokens, tgt=wtokens)
        #trans_out = trans_out[:, -len(self.wtokens):, :]

        params = dict()
        for name, shape in self.inr.param_shapes.items():
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng[name]
            x = self.wtoken_postfc[name](trans_out[:, l: r, :])
            x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
            #w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)
            w = self.update_strategy(w, x)

            wb = torch.cat([w, b], dim=1)
            params[name] = wb

        self.inr.set_params(params)
        
        if coord is None:
            coord = self.shared_coord
        
        if len(coord.shape) == 3:
            coord = einops.repeat(coord, 'h w d -> b h w d', b=B)
        elif len(coord.shape) == 4:
            coord = einops.repeat(coord, 'x y z d -> b x y z d', b=B)
        pred = self.inr(coord) 

        if len(pred.shape) == 4:
            pred = pred.permute(0, -1, 1, 2).contiguous()
        elif len(pred.shape) == 5:
            pred = pred.permute(0, -1, 1, 2, 3).contiguous() #bs x c x h x w x d
            
        return pred

    def sample(self, nsamples=10):
        return self.forward(data=None, nsamples=nsamples)
    
    def get_last_layer(self):
        return self.inr.get_last_layer()