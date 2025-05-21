import torch
import torch.nn as nn


class Decoder3(nn.Module):
    """A flexible decoder network with batch normalization and dropout.
    
    This decoder consists of multiple linear layers with batch normalization and dropout
    between layers. All layers except the last use ReLU activation. The decoder can optionally
    incorporate batch information at each layer.
    
    Args:
        in_shape (int): Input dimension
        layers (list[int]): List of layer sizes. The last value will be the output dimension.
        n_batches (int): Number of batch categories. If > 0, batch information will be concatenated
                         at each layer.
        dropout (float): Dropout probability
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
    """
    
    def __init__(self, in_shape: int, layers: list[int], n_batches: int, 
                 dropout: float, device: str = 'cuda'):
        super(Decoder3, self).__init__()
        
        # Build the network layers
        self.layers = nn.ModuleList()
        prev_size = in_shape
        
        # Add all layers except the last one
        for layer_size in layers[:-1]:
            # If we have batch information, add it to the input size
            if n_batches > 0:
                self.layers.append(nn.Sequential(
                    nn.Linear(prev_size + n_batches, layer_size),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(prev_size, layer_size),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                ))
            prev_size = layer_size
            
        # Add the final layer without activation
        if n_batches > 0:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_size + n_batches, layers[-1]),
                nn.BatchNorm1d(layers[-1]),
            ))
        else:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_size, layers[-1]),
                nn.BatchNorm1d(layers[-1]),
            ))
        
        self.n_batches = n_batches
        self.random_init()
        
    def forward(self, x: torch.Tensor, batches: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_shape)
            batches (torch.Tensor, optional): Batch information tensor of shape (batch_size, n_batches)
            
        Returns:
            torch.Tensor: Decoded representation of shape (batch_size, layers[-1])
        """
        for layer in self.layers:
            if self.n_batches > 0 and batches is not None:
                x = torch.cat((x, batches), dim=1)
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