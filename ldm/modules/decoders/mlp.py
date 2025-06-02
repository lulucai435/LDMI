import torch.nn as nn
import torch.nn.functional as F

# Dictionary of activation functions from PyTorch
ACTIVATION_CLASSES = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "swish": nn.SiLU(),  # Swish is the same as SiLU
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "celu": nn.CELU(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "softsign": nn.Softsign(),
    "hardswish": nn.Hardswish(),
    "mish": nn.Mish(),
}


class MLP(nn.Module):
    """Simple MLP with configurable hidden dimensions"""
    def __init__(self, input_dim, output_dim, hidden_dims=None, activation='gelu', dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim * 2, input_dim * 2]
       
            
        # Build the MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(ACTIVATION_CLASSES[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)