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

class HyperMLP(nn.Module):
    def __init__(self, tokenizer, inr, mlp, n_groups, data_shape, update_strategy="normalize", *args, **kwargs):
        super().__init__()
        
        # Extract parameters from tokenizer config even though we don't use the tokenizer
        self.latent_dim = tokenizer.get('params', {}).get('latent_dim', 3)
        latent_size = tokenizer.get('params', {}).get('latent_size', [8, 8])
        if isinstance(latent_size, int):
            latent_size = [latent_size, latent_size]
        self.latent_size = latent_size
        
        # Use dim from tokenizer config or a default value
        self.dim = tokenizer.get('params', {}).get('dim', 128)
        if not self.dim and 'head_dim' in tokenizer.get('params', {}) and 'n_head' in tokenizer.get('params', {}):
            # If dim isn't specified but head_dim and n_head are, calculate dim
            self.dim = tokenizer.get('params', {}).get('head_dim', 32) * tokenizer.get('params', {}).get('n_head', 4)
        
        # Direct INR instantiation
        self.inr = instantiate_from_config(inr)
        
        # Make coordinate grid for sampling
        self.register_buffer('shared_coord', make_coord_grid(data_shape, (-1, 1)), persistent=False)
        
        # Base parameters of the INR
        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        
        # Configure weight parameters for each group
        for name, shape in self.inr.param_shapes.items():
            self.base_params[name] = nn.Parameter(self.inr.init_wb(shape, name=name))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, shape[0] - 1),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        
        # Learnable weight tokens (initial values)
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, self.dim))
        
        # Calculate input and output dimensions for MLP
        mlp_input_dim = self.latent_dim * self.latent_size[0] * self.latent_size[1]
        mlp_output_dim = n_wtokens * self.dim
        
        # Update MLP config with calculated dimensions
        mlp['params'] = mlp.get('params', {})
        mlp['params']['input_dim'] = mlp_input_dim
        mlp['params']['output_dim'] = mlp_output_dim
        
        # Instantiate the MLP
        self.mlp = instantiate_from_config(mlp)
        
        # Update strategy selection
        self.update_strategy = update_strategies[update_strategy]


        nparams = sum(p.numel() for p in self.mlp.parameters())
        nparams += sum(p.numel() for p in self.base_params.values())
        nparams += self.wtokens.numel()
        
        print(f"Hypernetwork Parameters: {nparams / 1e6:.2f}M")
        
    def forward(self, z=None, coord=None, **kwargs):
        """
        Forward pass taking latent z directly
        Args:
            z: Latent input tensor (B, C, H, W)
            coord: Optional coordinate grid for sampling
        """
        if z is None:
            # If no input is provided, use a batch size of 1
            B = kwargs.get('nsamples', 1)
            # Generate default weight token modifiers (zeros)
            weight_modifiers = torch.zeros(B, len(self.wtokens), self.dim).to(self.wtokens.device)
        else:
            # Get batch size from input
            B = z.shape[0]
            
            # Flatten the latent input
            z_flat = z.flatten(start_dim=1)  # (B, C*H*W)
            
            # Process through the MLP to get weight token modifiers
            weight_modifiers = self.mlp(z_flat)
            
            # Reshape to match the weight tokens
            weight_modifiers = weight_modifiers.view(B, len(self.wtokens), self.dim)
        
        # Process parameters for each group
        params = dict()
        for name, shape in self.inr.param_shapes.items():
            # Get base parameters and expand to batch dimensions
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]
            
            # Get the relevant weight tokens for this parameter group
            l, r = self.wtoken_rng[name]
            
            # Apply modifications from the MLP output
            modified_tokens = self.wtokens[l:r].unsqueeze(0) + weight_modifiers[:, l:r]
            
            # Process through the postprocessing network
            x = self.wtoken_postfc[name](modified_tokens)
            x = x.transpose(-1, -2)  # (B, shape[0] - 1, g)
            
            # Apply the update strategy
            w = self.update_strategy(w, x)
            
            # Combine weights and biases
            wb = torch.cat([w, b], dim=1)
            params[name] = wb
        
        # Set the parameters in the INR
        self.inr.set_params(params)
        
        # If no coordinates are provided, use the shared coordinate grid
        if coord is None:
            coord = self.shared_coord
        
        # Expand coordinates to match batch size
        if len(coord.shape) == 3:
            coord = einops.repeat(coord, 'h w d -> b h w d', b=B)
        elif len(coord.shape) == 4:
            coord = einops.repeat(coord, 'x y z d -> b x y z d', b=B)
        
        # Run the INR with the modified parameters
        pred = self.inr(coord)
        
        # Rearrange the prediction to the right format
        if len(pred.shape) == 4:
            pred = pred.permute(0, -1, 1, 2).contiguous()
        elif len(pred.shape) == 5:
            pred = pred.permute(0, -1, 1, 2, 3).contiguous()  # bs x c x h x w x d
        
        return pred
    
    def sample(self, nsamples=1):
        """Generate samples without input"""
        return self.forward(z=None, nsamples=nsamples)
    
    def get_last_layer(self):
        """Get the last layer of the INR"""
        return self.inr.get_last_layer()