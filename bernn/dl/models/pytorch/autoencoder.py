import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .encoder import Encoder3
from .decoder import Decoder3
from .utils.stochastic import GaussianSample
from .utils.distributions import log_normal_standard, log_normal_diag, log_gaussian
from .utils.utils import to_categorical
from .utils.activations import MeanAct, DispAct
from .classifier import Classifier, Classifier2


class Autoencoder3(nn.Module):
    """A flexible autoencoder network with batch normalization and dropout.
    
    This autoencoder consists of an encoder and decoder, each with multiple linear layers
    with batch normalization and dropout between layers. The encoder uses LeakyReLU activation
    while the decoder uses ReLU activation.
    
    Args:
        in_shape (int): Input dimension
        encoder_layers (dict): Dictionary of encoder layer configurations
        decoder_layers (dict): Dictionary of decoder layer configurations
        n_batches (int): Number of batch categories for the decoder
        dropout (float): Default dropout probability if not specified in layer config
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
    """
    
    def __init__(self, in_shape: int, encoder_layers: Dict[str, Dict[str, Any]], 
                 decoder_layers: Dict[str, Dict[str, Any]], n_batches: int, 
                 dropout: float, device: str = 'cuda') -> None:
        super(Autoencoder3, self).__init__()
        
        # Create encoder and decoder
        self.encoder = Encoder3(
            in_shape=in_shape,
            layers=encoder_layers,
            dropout=dropout,
            device=device
        )
        
        self.decoder = Decoder3(
            in_shape=in_shape,
            layers=decoder_layers,
            n_batches=n_batches,
            dropout=dropout,
            device=device
        )
        
    def forward(self, x: torch.Tensor, batches: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_shape)
            batches (torch.Tensor, optional): Batch information tensor of shape (batch_size, n_batches)
            
        Returns:
            torch.Tensor: Reconstructed input of shape (batch_size, in_shape)
        """
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded, batches)
        
        return decoded 

class SHAPAutoencoder3(nn.Module):
    """A flexible SHAP autoencoder network with batch normalization and dropout.

    This autoencoder consists of multiple linear layers with batch normalization and dropout
    between layers. It includes SHAP-specific functionality like variational encoding,
    batch effect removal, and classification capabilities.

    Args:
        in_shape (int): Input dimension
        n_batches (int): Number of batch categories
        nb_classes (int): Number of output classes
        n_emb (int): Number of embedding dimensions
        n_meta (int): Number of metadata dimensions
        mapper (bool): Whether to use batch effect mapping
        variational (bool): Whether to use variational encoding
        layers (dict): Dictionary of layer configurations. Each key should be a layer name,
                      and the value should be a dict with 'size' and optional 'dropout' keys.
        dropout (float): Default dropout probability if not specified in layer config
        n_layers (int): Number of layers in the classifier
        zinb (bool): Whether to use ZINB loss
        conditional (bool): Whether to use conditional decoding
        add_noise (bool): Whether to add noise during training
        tied_weights (int): Whether to tie encoder/decoder weights
        use_gnn (bool): Whether to use graph neural network
        device (str): Device to use ('cuda' or 'cpu')
    """

    def __init__(self, in_shape: int, n_batches: int, nb_classes: int, n_emb: int, 
                 n_meta: int, mapper: bool, variational: bool, 
                 layers: Dict[str, Dict[str, Any]], dropout: float, n_layers: int,
                 zinb: bool = False, conditional: bool = False, add_noise: bool = False,
                 tied_weights: int = 0, use_gnn: bool = False, device: str = 'cuda') -> None:
        super(SHAPAutoencoder3, self).__init__()
        self.n_emb = n_emb
        self.add_noise = add_noise
        self.n_meta = n_meta
        self.device = device
        self.use_gnn = use_gnn
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'

        # Create encoder and decoder
        self.enc = Encoder3(in_shape + n_meta, layers, dropout, device)
        if conditional:
            self.dec = Decoder3(in_shape + n_meta, n_batches, layers, dropout, device)
        else:
            self.dec = Decoder3(in_shape + n_meta, 0, layers, dropout, device)

        # Create mapper for batch effect removal
        self.mapper = Classifier(n_batches + 1, layers[-1]['size'])

        # Create variational sampling if needed
        if variational:
            self.gaussian_sampling = GaussianSample(layers[-1]['size'], layers[-1]['size'], device)
        else:
            self.gaussian_sampling = None

        # Create discriminator and classifier
        self.dann_discriminator = Classifier2(layers[-1]['size'], 64, n_batches)
        self.classifier = Classifier(layers[-1]['size'] + n_emb, nb_classes, n_layers=n_layers)

        # Create ZINB-specific layers if needed
        if zinb:
            self._dec_mean = nn.Sequential(
                nn.Linear(layers[-2]['size'], in_shape + n_meta), MeanAct())
            self._dec_disp = nn.Sequential(
                nn.Linear(layers[-2]['size'], in_shape + n_meta), DispAct())
            self._dec_pi = nn.Sequential(
                nn.Linear(layers[-2]['size'], in_shape + n_meta), nn.Sigmoid())

        self.random_init(nn.init.xavier_uniform_)

    def forward(self, x: torch.Tensor, batches: Optional[torch.Tensor] = None, 
                sampling: bool = False, beta: float = 1.0) -> torch.Tensor:
        """Forward pass through the SHAP autoencoder.

        Args:
            x (torch.Tensor): Input tensor
            batches (torch.Tensor, optional): Batch information tensor
            sampling (bool): Whether to sample from the variational distribution
            beta (float): KL divergence weight

        Returns:
            torch.Tensor: Output tensor
        """
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values).to(self.device)

        # Handle metadata if present
        if self.n_emb > 0:
            meta_values = x[:, -2:]
            x = x[:, :-2]

        # Add noise if specified
        if self.add_noise:
            noise = Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1
            x = x * noise.type_as(x)

        # Encode
        enc = self.enc(x)

        # Apply variational sampling if enabled
        if self.gaussian_sampling is not None:
            if sampling:
                enc, mu, log_var = self.gaussian_sampling(enc, train=True, beta=beta)
            else:
                enc, _, _ = self.gaussian_sampling(enc, train=False)

        # Apply batch effect mapping if enabled
        if self.use_mapper and batches is not None:
            bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            enc = enc + self.mapper(bs).squeeze()

        # Classify
        if self.n_emb > 0:
            out = self.classifier(torch.cat((enc, meta_values), 1))
        else:
            out = self.classifier(enc)

        return out

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        """Initialize the network weights.

        Args:
            init_func: Initialization function to use
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Get probability predictions.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            np.ndarray: Probability predictions
        """
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Get class predictions.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            np.ndarray: Class predictions
        """
        return self.classifier(x).argmax(1).detach().cpu().numpy()

    def _kld(self, z: torch.Tensor, q_param: Tuple[torch.Tensor, torch.Tensor], 
             h_last: Optional[torch.Tensor] = None, 
             p_param: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """Calculate KL divergence.

        Args:
            z (torch.Tensor): Latent representation
            q_param (tuple): Parameters of the approximate posterior
            h_last (torch.Tensor, optional): Last hidden state
            p_param (tuple, optional): Parameters of the prior

        Returns:
            torch.Tensor: KL divergence
        """
        if len(z.shape) == 1:
            z = z.view(1, -1)

        if (self.flow_type == "nf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif (self.flow_type == "iaf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z, h_last)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif (self.flow_type in ['hf', 'ccliniaf']) and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z = self.flow(z, h_last)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        elif self.flow_type in ["o-sylvester", "h-sylvester", "t-sylvester"] and self.n_flows > 0:
            mu, log_var, r1, r2, q_ortho, b = q_param
            f_z = self.flow(z, r1, r2, q_ortho, b)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        else:
            (mu, log_var) = q_param
            qz = log_normal_diag(z, mu, log_var)

        if p_param is None:
            pz = log_normal_standard(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = -(pz - qz)
        return kl

    def zinb_loss(self, x: torch.Tensor, mean: torch.Tensor, disp: torch.Tensor, 
                  pi: torch.Tensor, scale_factor: float = 1.0, 
                  ridge_lambda: float = 0.0) -> torch.Tensor:
        """Calculate ZINB loss.

        Args:
            x (torch.Tensor): Input tensor
            mean (torch.Tensor): Mean parameter
            disp (torch.Tensor): Dispersion parameter
            pi (torch.Tensor): Dropout parameter
            scale_factor (float): Scale factor
            ridge_lambda (float): Ridge regularization parameter

        Returns:
            torch.Tensor: ZINB loss
        """
        eps = 1e-10
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result 