import torch
import torch.nn as nn
import math


class MLP_Base(nn.Module):
    def __init__(self, layers, residual=False):
        super().__init__()
        if residual:
            self.layers = nn.ModuleList(layers)
        else:
            self.network = nn.Sequential(*layers)
        self.residual = residual
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., in_dim).

        Returns:
            Output tensor of shape (..., out_dim).
        """
        if self.residual:
            for i, layer in enumerate(self.layers):
                out = layer(x)
                if self.residual and out.shape[-1] == x.shape[-1]:
                    x = out + x
                else:
                    x = out
            return x            
        return self.network(x)


class MLP(MLP_Base):
    """
    Standard Multi-Layer Perceptron.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers, activation=nn.SiLU, residual=False):
        """
        Args:
            in_dim: Input feature size.
            out_dim: Output feature size.
            hidden_dim: Number of neurons per hidden layer.
            n_layers: Number of hidden layers.
            activation: PyTorch activation function class.
        """
        layers = []
        prev_dim = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        
        super().__init__(layers, residual)


class SineLayer(nn.Module):
    """
    A single linear layer with a Sine activation function.
    """
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        """
        Initialization as described in the SIREN paper (Sitzmann et al., 2020).
        """
        with torch.no_grad():
            if self.is_first:
                # First layer initialization: U(-1/in_features, 1/in_features)
                bound = 1 / self.linear.weight.size(1)
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layers initialization: U(-sqrt(6/in)/omega_0, sqrt(6/in)/omega_0)
                bound = math.sqrt(6 / self.linear.weight.size(1)) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
            # Bias is typically initialized to 0 or uniformly
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(MLP_Base):
    """
    Standard Multi-Layer Perceptron.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers, residual=False):
        
        # omega_0 = 30.0 / 30 # lower values are more stable than the default 30.0 used for image reconstruction
        first_omega_0 = 1.0#30.0 
        hidden_omega_0 = 1.0/10.0
        
        # -- Build the SIREN MLP
        
        layers = []
        prev_dim = in_dim
        for i in range(n_layers):
            is_first = (i == 0)
            omega = first_omega_0 if is_first else hidden_omega_0            
            layer = SineLayer(prev_dim, hidden_dim, is_first=is_first, omega_0=omega)
            layers.append(layer)
            prev_dim = hidden_dim

        final_layer = nn.Linear(prev_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6 / prev_dim) # Xavier/Glorot init
            final_layer.weight.uniform_(-bound * 0.01, bound * 0.01)
            # final_layer.bias.fill_(0.1) 
        layers.append(final_layer)
        
        super().__init__(layers, residual)
    