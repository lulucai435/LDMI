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

        # Initialize layers
        last_dim = in_dim
        self.linears = nn.ModuleList()

        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            linear = nn.Linear(last_dim, cur_dim)
            self.linears.append(linear)
            last_dim = cur_dim

        self.init_siren_linears()

    def siren_activation(self, x, is_first=True):
        """SIREN non-linearity with proper frequency scaling."""
        omega = self.omega if is_first else self.omega
        return torch.sin(omega * x)

    def init_siren_linears(self):
        """Initialize weights following SIREN paper."""
        for i, linear in enumerate(self.linears):
            if i == 0:
                # First layer
                bound = 1 / linear.weight.shape[1]
            else:
                # Hidden layers
                bound = np.sqrt(6 / linear.weight.shape[1]) / self.omega

            nn.init.uniform_(linear.weight, -bound, bound)
            nn.init.zeros_(linear.bias)

    def forward(self, x):
        B, query_shape = x.shape[0], x.shape[1: -1]
        x = x.view(B, -1, x.shape[-1])

        for i in range(self.depth):
            x = self.linears[i](x)
            if i < self.depth - 1:
                x = self.siren_activation(x, is_first=(i == 0))
            else:
                # Final layer - scale to [-1, 1] range
                # x = torch.tanh(x)
                x = x

        x = x.view(B, *query_shape, -1)
        return x

    def get_last_layer(self):
        return self.linears[-1].weight
