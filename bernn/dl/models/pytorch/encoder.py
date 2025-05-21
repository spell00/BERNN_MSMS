import torch
import torch.nn as nn


class Encoder3(nn.Module):
    """A flexible encoder network with batch normalization and dropout.
    
    This encoder consists of multiple linear layers with batch normalization and dropout
    between layers. All layers except the last use LeakyReLU activation.
    
    Args:
        in_shape (int): Input dimension
        layers (dict): Dictionary of layer configurations. Each key should be a layer name,
                      and the value should be a dict with 'size' and optional 'dropout' keys.
                      Example: {
                          'layer1': {'size': 256, 'dropout': 0.2},
                          'layer2': {'size': 128, 'dropout': 0.2},
                          'output': {'size': 64}
                      }
        dropout (float): Default dropout probability if not specified in layer config
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
    """
    
    def __init__(self, in_shape: int, layers: dict, dropout: float, device: str = 'cuda'):
        super(Encoder3, self).__init__()
        
        # Build the network layers
        self.layers = nn.ModuleDict()
        prev_size = in_shape
        
        # Add all layers except the last one
        for layer_name, layer_config in list(layers.items())[:-1]:
            layer_size = layer_config['size']
            layer_dropout = layer_config.get('dropout', dropout)
            
            self.layers[layer_name] = nn.Sequential(
                nn.Linear(prev_size, layer_size),
                nn.BatchNorm1d(layer_size),
                nn.Dropout(layer_dropout),
                nn.LeakyReLU(),
            )
            prev_size = layer_size
            
        # Add the final layer without activation
        final_layer_name = list(layers.keys())[-1]
        final_layer_size = layers[final_layer_name]['size']
        self.layers[final_layer_name] = nn.Sequential(
            nn.Linear(prev_size, final_layer_size),
            nn.BatchNorm1d(final_layer_size),
        )
        
        self.random_init()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_shape)
            
        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, layers[-1]['size'])
        """
        for layer in self.layers.values():
            x = layer(x)
        return x
    
    def random_init(self, init_func=nn.init.kaiming_uniform_):
        """Initialize the network weights using Kaiming initialization.
        
        Args:
            init_func: Initialization function to use. Defaults to kaiming_uniform_.
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_() 