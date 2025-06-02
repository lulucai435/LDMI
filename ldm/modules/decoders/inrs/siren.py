import torch
import torch.nn as nn
import numpy as np
import math
from .layers import batched_linear_mm



class SIREN(nn.Module):

    def __init__(self, depth, in_dim, out_dim, hidden_dim, out_bias=0, omega=30.0):
        super().__init__()
        self.omega = omega
        self.depth = depth
        self.param_shapes = dict()

        last_dim = in_dim

        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f'wb{i}'] = (last_dim + 1, cur_dim)
            last_dim = cur_dim
        self.params = None
        self.out_bias = out_bias

    def siren_activation(self, x):
        """SIREN non-linearity with frequency scaling."""
        return torch.sin(self.omega * x)
    
    def init_wb(self, shape, name):
        """Initialize weights and biases for a given layer."""
        if name == 'wb0':
            num_input = shape[0] - 1
            bound = 1 / num_input  # First layer should not use omega scaling
            weight = torch.empty(shape[1], shape[0] - 1)
            nn.init.uniform_(weight, -bound, bound)
            bias = torch.zeros(shape[1], 1)
            return torch.cat([weight, bias], dim=1).t().detach()
        else:
            weight = torch.empty(shape[1], shape[0] - 1)
            num_input = shape[0] - 1
            bound = np.sqrt(6 / num_input) / self.omega
            nn.init.uniform_(weight, -bound, bound)

            bias = torch.zeros(shape[1], 1)  # Zero bias is standard in SIREN
            return torch.cat([weight, bias], dim=1).t().detach()


    def siren_init(self, weight, omega=None):
        """Apply SIREN-specific initialization scaling."""
        if omega is None:
            omega = self.omega  # Default to the SIREN model's omega
        num_input = weight.shape[0]
        return weight * (np.sqrt(6 / num_input) / omega)  # Scale weights properly

    def siren_init_first_layer(self, weight):
        """First layer initialization (no omega scaling)."""
        num_input = weight.shape[0]
        return weight * (1 / num_input)  # First layer has different scaling

    def set_params(self, params):
        """Receives weights from the hypernetwork and rescales them."""
        self.params = params

    def forward(self, x):
        B, query_shape = x.shape[0], x.shape[1: -1]
        x = x.view(B, -1, x.shape[-1])

        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f'wb{i}'])  # External weight application
            if i < self.depth - 1:
                x = self.siren_activation(x)  # SIREN non-linearity
            else:
                x = x + self.out_bias  # Final bias adjustment
                
        x = x.view(B, *query_shape, -1)
        return x

    def get_last_layer(self):
        return self.params[f'wb{self.depth - 1}']
    
