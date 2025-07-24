import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Tuple, Any, List
from bernn.dl.models.pytorch.utils.stochastic import GaussianSample
from bernn.dl.models.pytorch.utils.distributions import log_normal_standard, log_normal_diag, log_gaussian
from bernn.dl.models.pytorch.utils.utils import to_categorical
import pandas as pd
import numpy as np


# https://github.com/DHUDBlab/scDSC/blob/1247a63aac17bdfb9cd833e3dbe175c4c92c26be/layers.py#L43
class MeanAct(nn.Module):
    def __init__(self) -> None:
        super(MeanAct, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


# https://github.com/DHUDBlab/scDSC/blob/1247a63aac17bdfb9cd833e3dbe175c4c92c26be/layers.py#L43
class DispAct(nn.Module):
    def __init__(self) -> None:
        super(DispAct, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        output = grad_output.neg() * ctx.alpha

        return output, None


def grad_reverse(x: torch.Tensor) -> torch.Tensor:
    return ReverseLayerF()(x)


class Classifier(nn.Module):
    def __init__(self, in_shape: int = 64, out_shape: int = 9, n_layers: int = 2, use_softmax: bool = True) -> None:
        super(Classifier, self).__init__()
        self.use_softmax = use_softmax
        if n_layers == 2:
            self.linear2 = nn.Sequential(
                nn.Linear(in_shape, in_shape),
            )
            self.linear3 = nn.Sequential(
                nn.Linear(in_shape, out_shape),
            )
        if n_layers == 1:
            self.linear2 = nn.Sequential(
                nn.Linear(in_shape, out_shape),
            )

        self.random_init()
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(x)
        if self.n_layers == 2:
            x = self.linear3(x)
        if self.use_softmax:
            x = F.softmax(x, dim=1)
        return x

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        return self.linear2(x).detach().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.linear2(x).argmax(1).detach().cpu().numpy()


class Classifier2(nn.Module):
    def __init__(self, in_shape: int = 64, hidden: int = 64, out_shape: int = 9, use_softmax: bool = True) -> None:
        super(Classifier2, self).__init__()
        self.use_softmax = use_softmax
        self.linear1 = nn.Sequential(
            nn.Linear(in_shape, hidden),
            nn.BatchNorm1d(hidden),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden, out_shape),
        )
        self.random_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        if self.use_softmax:
            x = F.softmax(x, dim=1)
        return x

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        return self.linear2(x).detach().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.linear2(x).argmax(1).detach().cpu().numpy()


class Encoder(nn.Module):
    def __init__(self, in_shape: int, layer1: int, dropout: float) -> None:
        super(Encoder, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(in_shape, layer1),
            nn.BatchNorm1d(layer1),
            # nn.LeakyReLU(),
        )
        self.random_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        # x = self.linear2(x)
        return x

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class Encoder2(nn.Module):
    def __init__(self, in_shape: int, layer1: int, layer2: int, dropout: float) -> None:
        super(Encoder2, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_shape, layer1),
            nn.BatchNorm1d(layer1),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(layer1, layer2),
            nn.BatchNorm1d(layer2),
            # nn.Dropout(dropout),
            # nn.Sigmoid(),
            # nn.ReLU(),

        )

        self.random_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class Encoder3(nn.Module):
    """A flexible encoder network with batch normalization and dropout.

    This encoder consists of multiple linear layers with batch normalization and dropout
    between layers. All layers except the last use LeakyReLU activation.

    Args:
        in_shape (int): Input dimension
        layers (list[int]): List of layer sizes. The last value will be the output dimension.
        dropout (float): Dropout probability
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
    """

    def __init__(self, in_shape: int, layers: dict, dropout: float, device: str = 'cuda'):
        super(Encoder3, self).__init__()

        # Build the network layers
        self.layers = nn.ModuleList()
        prev_size = in_shape

        # Add all layers except the last one
        for layer_size in list(layers.values())[:-1]:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_size, layer_size),
                nn.BatchNorm1d(layer_size),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
            ))
            prev_size = layer_size

        # Add the final layer without activation
        self.layers.append(nn.Sequential(
            nn.Linear(prev_size, layers[list(layers.keys())[-1]]),
            nn.BatchNorm1d(layers[list(layers.keys())[-1]]),
        ))

        self.random_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_shape)

        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, layers[-1])
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        """Initialize the network weights using Kaiming initialization.

        Args:
            init_func: Initialization function to use. Defaults to kaiming_uniform_.
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class Decoder2(nn.Module):
    def __init__(self, in_shape: int, n_batches: int, layer1: int, layer2: int, dropout: float) -> None:
        super(Decoder2, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(layer1 + n_batches, layer2),
            nn.BatchNorm1d(layer2),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(layer2, in_shape),
            # nn.BatchNorm1d(in_shape),
            # nn.Sigmoid(),
        )
        self.n_batches = n_batches
        self.random_init()

    def forward(self, x: torch.Tensor, batches: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        if batches is not None and self.n_batches > 0:
            x = torch.cat((x, batches), 1)
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        # x2 = torch.sigmoid(x2)
        return [x1, x2]

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class Decoder(nn.Module):
    def __init__(self, in_shape: int, n_batches: int, layer1: int, dropout: float) -> None:
        super(Decoder, self).__init__()
        self.linear2 = nn.Sequential(
            nn.Linear(layer1 + n_batches, in_shape),
        )
        self.n_batches = n_batches
        self.random_init()

    def forward(self, x: torch.Tensor, batches: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batches is not None and self.n_batches > 0:
            x = torch.cat((x, batches), 1)
        x1 = self.linear1(x)
        return x1

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class Decoder3(nn.Module):
    """A flexible decoder network with batch normalization and dropout.

    This decoder consists of multiple linear layers with batch normalization and dropout
    between layers. All layers except the last use ReLU activation. The decoder can optionally
    incorporate batch information at each layer.

    Args:
        in_shape (int): Input dimension
        layers (dict): Dictionary of layer configurations. Each key should be a layer name,
                      and the value should be a dict with 'size' and optional 'dropout' keys.
                      Example: {
                          'layer1': {'size': 256, 'dropout': 0.2},
                          'layer2': {'size': 512, 'dropout': 0.2},
                          'output': {'size': 784}
                      }
        n_batches (int): Number of batch categories. If > 0, batch information will be concatenated
                         at each layer.
        dropout (float): Default dropout probability if not specified in layer config
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
    """

    def __init__(self, in_shape: int, n_batches: int, layers: dict,
                 dropout: float, device: str = 'cuda'):
        super(Decoder3, self).__init__()

        # Build the network layers
        self.layers = nn.ModuleDict()
        prev_size = list(layers.values())[-1]

        # Add all layers except the last one
        for layer_name, layer_size in list(layers.items())[::-1]:
            if layer_name == list(layers.keys())[-1]:
                continue
            # If we have batch information, add it to the input size
            if n_batches > 0:
                self.layers[layer_name] = nn.Sequential(
                    nn.Linear(prev_size + n_batches, layer_size),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                )
            else:
                self.layers[layer_name] = nn.Sequential(
                    nn.Linear(prev_size, layer_size),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                )
            prev_size = layer_size

        # Add the final layer without activation
        self.layers[list(layers.keys())[-1]] = nn.Sequential(
            nn.Linear(prev_size, in_shape),
            nn.BatchNorm1d(in_shape),
        )

        # Create a new ModuleDict with reversed order
        # reversed_layers = nn.ModuleDict()
        # for key in reversed(list(self.layers.keys())):
        #     reversed_layers[key] = self.layers[key]
        # self.layers = reversed_layers

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
        for layer in self.layers.values():
            if self.n_batches > 0 and batches is not None:
                x = torch.cat((x, batches), dim=1)
            x = layer(x)
        return x

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        """Initialize the network weights using Kaiming initialization.

        Args:
            init_func: Initialization function to use. Defaults to kaiming_uniform_.
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class SHAPAutoEncoder2(nn.Module):
    def __init__(self, in_shape: int, n_batches: int, nb_classes: int, n_emb: int,
                 n_meta: int, mapper: bool, variational: bool, layer1: int, layer2: int,
                 dropout: float, n_layers: int, zinb: bool = False, conditional: bool = True,
                 add_noise: bool = False, tied_weights: int = 0, use_gnn: bool = False,
                 device: str = 'cuda') -> None:
        super(SHAPAutoEncoder2, self).__init__()
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
        # self.gnn1 = GCNConv(in_shape, in_shape)
        self.enc = Encoder2(in_shape + n_meta, layer1, layer2, dropout)
        if conditional:
            self.dec = Decoder2(in_shape + n_meta, n_batches, layer2, layer1, dropout)
        else:
            self.dec = Decoder2(in_shape + n_meta, 0, layer2, layer1, dropout)
        self.mapper = Classifier(n_batches + 1, layer2)

        if variational:
            self.gaussian_sampling = GaussianSample(layer2, layer2, device)
        else:
            self.gaussian_sampling = None
        self.dann_discriminator = Classifier2(layer2, 64, n_batches)
        self.classifier = Classifier(layer2 + n_emb, nb_classes, n_layers=n_layers)
        self._dec_mean = nn.Sequential(nn.Linear(layer1, in_shape + n_meta), nn.Sigmoid())
        self._dec_disp = nn.Sequential(nn.Linear(layer1, in_shape + n_meta), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(layer1, in_shape + n_meta), nn.Sigmoid())
        self.random_init(nn.init.xavier_uniform_)

    def forward(self, x: torch.Tensor, batches: Optional[torch.Tensor] = None,
                sampling: bool = False, beta: float = 1.0) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values).to(self.device)
        if self.n_emb > 0:
            meta_values = x[:, -2:]
            x = x[:, :-2]
        # if self.n_meta > 0:
        #     x = x[:, :-2]
        # rec = {}
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        # if self.use_gnn:
        #     x = self.gnn1(x)
        enc = self.enc(x)
        if self.gaussian_sampling is not None:
            if sampling:
                enc, mu, log_var = self.gaussian_sampling(enc, train=True, beta=beta)
            else:
                enc, _, _ = self.gaussian_sampling(enc, train=False)

        if self.n_emb > 0:
            out = self.classifier(torch.cat((enc, meta_values), 1))
        else:
            out = self.classifier(enc)

        return out

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.classifier(x).argmax(1).detach().cpu().numpy()

    def _kld(self, z: torch.Tensor, q_param: Tuple[torch.Tensor, torch.Tensor],
             h_last: Optional[torch.Tensor] = None,
             p_param: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
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
        # vanilla
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

    # # based on https://github.com/DHUDBlab/scDSC/blob/master/layers.py
    def zinb_loss(self, x: torch.Tensor, mean: torch.Tensor, disp: torch.Tensor,
                  pi: torch.Tensor, scale_factor: float = 1.0,
                  ridge_lambda: float = 0.0) -> torch.Tensor:
        eps = 1e-10
        # scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result


class SHAPAutoEncoder3(nn.Module):
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

    def __init__(self, in_shape: int, n_batches: int, nb_classes: int, n_emb: int, n_meta: int,
                 mapper: bool, variational: bool, layers: dict, dropout: float, n_layers: int,
                 zinb: bool = False, conditional: bool = True, add_noise: bool = False,
                 tied_weights: int = 0, use_gnn: bool = False, device: str = 'cuda'):
        super(SHAPAutoEncoder3, self).__init__()
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
        self.mapper = Classifier(n_batches + 1, layers[list(layers.keys())[-1]])

        # Create variational sampling if needed
        if variational:
            self.gaussian_sampling = GaussianSample(
                layers[list(layers.keys())[-1]],
                layers[list(layers.keys())[-1]],
                device,
            )
        else:
            self.gaussian_sampling = None

        # Create discriminator and classifier
        self.dann_discriminator = Classifier2(layers[list(layers.keys())[-1]], 64, n_batches)
        self.classifier = Classifier(layers[list(layers.keys())[-1]] + n_emb, nb_classes, n_layers=n_layers)

        # Create ZINB-specific layers if needed
        if zinb:
            self._dec_mean = nn.Sequential(nn.Linear(layers[list(layers.keys())[-2]], in_shape + n_meta), MeanAct())
            self._dec_disp = nn.Sequential(nn.Linear(layers[list(layers.keys())[-2]], in_shape + n_meta), DispAct())
            self._dec_pi = nn.Sequential(nn.Linear(layers[list(layers.keys())[-2]], in_shape + n_meta), nn.Sigmoid())

        self.random_init(nn.init.xavier_uniform_)

    def forward(self, x: torch.Tensor, batches: torch.Tensor = None,
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
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)

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


class AutoEncoder2(nn.Module):
    def __init__(self, in_shape: int, n_batches: int, nb_classes: int, n_meta: int,
                 n_emb: int, mapper: bool, variational: bool, layer1: int, layer2: int,
                 dropout: float, n_layers: int, prune_threshold: float, zinb: bool = False,
                 conditional: bool = True, add_noise: bool = False, tied_weights: int = 0,
                 update_grid: bool = False, use_gnn: bool = False, device: str = 'cuda') -> None:
        """
        TODO MAKE DESCRIPTION
        """
        super(AutoEncoder2, self).__init__()
        self.add_noise = add_noise
        self.device = device
        self.use_gnn = use_gnn
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'
        # self.gnn1 = GCNConv(in_shape, in_shape)
        self.enc = Encoder2(in_shape + n_meta, layer1, layer2, dropout)
        if conditional:
            self.dec = Decoder2(in_shape + n_meta, n_batches, layer2, layer1, dropout)
        else:
            self.dec = Decoder2(in_shape + n_meta, 0, layer2, layer1, dropout)
        self.mapper = Classifier(n_batches + 1, layer2)

        if variational:
            self.gaussian_sampling = GaussianSample(layer2, layer2, device)
        else:
            self.gaussian_sampling = None
        self.dann_discriminator = Classifier2(layer2, 64, n_batches)
        self.classifier = Classifier(layer2 + n_emb, nb_classes, n_layers=n_layers)
        self._dec_mean = nn.Sequential(nn.Linear(layer1, in_shape + n_meta), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(layer1, in_shape + n_meta), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(layer1, in_shape + n_meta), nn.Sigmoid())
        self.random_init(nn.init.kaiming_uniform_)

    def forward(self, x: torch.Tensor, to_rec: torch.Tensor,
                batches: Optional[torch.Tensor] = None, sampling: bool = False,
                beta: float = 1.0, mapping: bool = True) -> List[Any]:
        rec = {}
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        enc = self.enc(x)
        if self.gaussian_sampling is not None:
            if sampling:
                enc, mu, log_var = self.gaussian_sampling(enc, train=True, beta=beta)
                # Kullback-Leibler Divergence
                # kl = self._kld(enc, (mu, log_var))
                # mean_sq = mu * mu
                # std = log_var.exp().sqrt()
                # stddev_sq = std * std
                # kl = 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
                # https://arxiv.org/pdf/1312.6114.pdf equation 10, first part and
                # https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
                kl = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), axis=1)
            else:
                enc, _, _ = self.gaussian_sampling(enc, train=False)
                kl = torch.Tensor([0])
        else:
            kl = torch.Tensor([0])
        if self.use_mapper and mapping:
            bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            enc_be = enc + self.mapper(bs).squeeze()
        else:
            enc_be = enc
        if not self.tied_weights:
            try:
                bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            except Exception as e:
                print(e)
                bs = to_categorical(batches.long(), self.n_batches + 1).to(self.device).float()

            rec = {"mean": self.dec(enc_be, bs)}
        elif not self.zinb:
            rec = [F.relu(F.linear(enc, self.enc.linear2[0].weight.t()))]
            rec += [F.relu(F.linear(rec[0], self.enc.linear1[0].weight.t()))]
            rec = {"mean": rec}  # TODO rec does not need to be a dict no more
        elif self.zinb:
            rec = {"mean": [F.relu(F.linear(enc, self.enc.linear2[0].weight.t()))]}

        if self.zinb:
            _mean = self._dec_mean(rec['mean'][0])
            _disp = self._dec_disp(rec['mean'][0])
            _pi = self._dec_pi(rec['mean'][0])
            zinb_loss = self.zinb_loss(to_rec, _mean, _disp, _pi)
            # if not sampling:
            rec = {'mean': _mean, 'rec': to_rec}
        else:
            zinb_loss = torch.Tensor([0])

        # reverse = ReverseLayerF.apply(enc, alpha)
        # b_preds = self.classifier(reverse)
        # rec[-1] = torch.clamp(rec[-1], min=0, max=1)
        return [enc, rec, zinb_loss, kl]

    def prune_model_paperwise(self, is_classification: bool, is_dann: bool,
                              weight_threshold: float = 0) -> int:
        print("Pruning not available for this model")
        # TODO implement pruning or count the number of neurons
        return 0

    def count_n_neurons(self) -> int:
        return 0

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.classifier(x).argmax(1).detach().cpu().numpy()

    def _kld(self, z: torch.Tensor, q_param: Tuple[torch.Tensor, torch.Tensor],
             h_last: Optional[torch.Tensor] = None,
             p_param: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
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
        # vanilla
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

    # based on https://github.com/DHUDBlab/scDSC/blob/master/layers.py
    def zinb_loss(self, x: torch.Tensor, mean: torch.Tensor, disp: torch.Tensor,
                  pi: torch.Tensor, scale_factor: float = 1.0,
                  ridge_lambda: float = 0.0) -> torch.Tensor:
        eps = 1e-10
        # scale_factor = scale_factor[:, None]
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


class AutoEncoder3(nn.Module):
    def __init__(self, in_shape: int, n_batches: int, nb_classes: int, n_meta: int,
                 n_emb: int, mapper: bool, variational: bool, layers: dict,
                 dropout: float, n_layers: int, prune_threshold: float, zinb: bool = False,
                 conditional: bool = True, add_noise: bool = False, tied_weights: int = 0,
                 update_grid: bool = False, use_gnn: bool = False, device: str = 'cuda') -> None:
        """
        TODO MAKE DESCRIPTION
        """
        super(AutoEncoder3, self).__init__()
        self.add_noise = add_noise
        self.device = device
        self.use_gnn = use_gnn
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'
        # self.gnn1 = GCNConv(in_shape, in_shape)
        self.enc = Encoder3(in_shape + n_meta, layers, dropout, device)
        if conditional:
            self.dec = Decoder3(in_shape + n_meta, n_batches, layers, dropout, device)
        else:
            self.dec = Decoder3(in_shape + n_meta, 0, layers, dropout, device)
        self.mapper = Classifier(n_batches + 1, layers[list(layers.keys())[-1]])

        if variational:
            self.gaussian_sampling = GaussianSample(
                layers[list(layers.keys())[-1]],
                layers[list(layers.keys())[-1]],
                device,
            )
        else:
            self.gaussian_sampling = None
        self.dann_discriminator = Classifier2(
            layers[list(layers.keys())[-1]],
            64, n_batches)
        self.classifier = Classifier(layers[list(layers.keys())[-1]] + n_emb, nb_classes, n_layers=n_layers)
        if self.zinb:
            self._dec_mean = nn.Sequential(nn.Linear(layers[list(layers.keys())[-2]], in_shape + n_meta), MeanAct())
            self._dec_disp = nn.Sequential(nn.Linear(layers[list(layers.keys())[-2]], in_shape + n_meta), DispAct())
            self._dec_pi = nn.Sequential(nn.Linear(layers[list(layers.keys())[-2]], in_shape + n_meta), nn.Sigmoid())
        self.random_init(nn.init.kaiming_uniform_)

    def forward(self, x: torch.Tensor, to_rec: torch.Tensor,
                batches: Optional[torch.Tensor] = None, sampling: bool = False,
                beta: float = 1.0, mapping: bool = True) -> List[Any]:
        rec = {}
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        enc = self.enc(x)
        if self.gaussian_sampling is not None:
            if sampling:
                enc, mu, log_var = self.gaussian_sampling(enc, train=True, beta=beta)
                # Kullback-Leibler Divergence
                # kl = self._kld(enc, (mu, log_var))
                # mean_sq = mu * mu
                # std = log_var.exp().sqrt()
                # stddev_sq = std * std
                # kl = 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
                # https://arxiv.org/pdf/1312.6114.pdf equation 10, first part and
                # https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
                kl = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), axis=1)
            else:
                enc, _, _ = self.gaussian_sampling(enc, train=False)
                kl = torch.Tensor([0])
        else:
            kl = torch.Tensor([0])
        if self.use_mapper and mapping:
            bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            enc_be = enc + self.mapper(bs).squeeze()
        else:
            enc_be = enc
        if not self.tied_weights:
            try:
                bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            except Exception as e:
                print(e)
                bs = to_categorical(batches.long(), self.n_batches + 1).to(self.device).float()

            rec = {"mean": self.dec(enc_be, bs)}
        elif not self.zinb:
            rec = [F.relu(F.linear(enc, self.enc.linear2[0].weight.t()))]
            rec += [F.relu(F.linear(rec[0], self.enc.linear1[0].weight.t()))]
            rec = {"mean": rec}  # TODO rec does not need to be a dict no more
        elif self.zinb:
            rec = {"mean": [F.relu(F.linear(enc, self.enc.linear2[0].weight.t()))]}

        if self.zinb:
            _mean = self._dec_mean(rec['mean'][0])
            _disp = self._dec_disp(rec['mean'][0])
            _pi = self._dec_pi(rec['mean'][0])
            zinb_loss = self.zinb_loss(to_rec, _mean, _disp, _pi)
            # if not sampling:
            rec = {'mean': _mean, 'rec': to_rec}
        else:
            zinb_loss = torch.Tensor([0])

        # reverse = ReverseLayerF.apply(enc, alpha)
        # b_preds = self.classifier(reverse)
        # rec[-1] = torch.clamp(rec[-1], min=0, max=1)
        return [enc, rec, zinb_loss, kl]

    def prune_model_paperwise(self, is_classification: bool, is_dann: bool,
                              weight_threshold: float = 0) -> int:
        print("Pruning not available for this model")
        # TODO implement pruning or count the number of neurons
        return 0

    def count_n_neurons(self) -> int:
        return 0

    def random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, inputs: torch.Tensor) -> np.ndarray:
        x = self.enc(inputs)
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        x = self.enc(inputs)
        return self.classifier(x).argmax(1).detach().cpu().numpy()

    def _kld(self, z: torch.Tensor, q_param: Tuple[torch.Tensor, torch.Tensor],
             h_last: Optional[torch.Tensor] = None,
             p_param: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
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
        # vanilla
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

    # based on https://github.com/DHUDBlab/scDSC/blob/master/layers.py
    def zinb_loss(self, x: torch.Tensor, mean: torch.Tensor, disp: torch.Tensor,
                  pi: torch.Tensor, scale_factor: float = 1.0,
                  ridge_lambda: float = 0.0) -> torch.Tensor:
        eps = 1e-10
        # scale_factor = scale_factor[:, None]
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


def log_zinb_positive(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor,
                      pi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    log likelihood (scalar) of a minibatch according to a zinb model.
    Notes:
    We parametrize the bernouilli using the logits, hence the softplus functions appearing

    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    pi: logit of the dropout parameter (real support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    case_zero = F.softplus(- pi + theta * torch.log(theta + eps) - theta * torch.log(theta + mu + eps)) \
        - F.softplus(-pi)
    case_non_zero = - pi - F.softplus(- pi) \
        + theta * torch.log(theta + eps) - theta * torch.log(theta + mu + eps) \
        + x * torch.log(mu + eps) - x * torch.log(theta + mu + eps) \
        + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)

    # mask = tf.cast(torch.less(x, eps), torch.float32)
    mask = torch.less(x, eps).float()
    res = torch.multiply(mask, case_zero) + torch.multiply(1 - mask, case_non_zero)
    res = torch.nan_to_num(res, 0)
    return torch.sum(res, axis=-1)
