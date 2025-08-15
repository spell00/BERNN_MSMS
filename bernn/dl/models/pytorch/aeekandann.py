# Rewritten to mirror aedann.py but using KANLinear instead of nn.Linear.
from typing import Any, Optional, Tuple, List

# try:
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function
# TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False

# def _check_torch_available():
#     if not TORCH_AVAILABLE:
#         raise ImportError(
#             "PyTorch is not installed. Install with: pip install bernn[deep-learning] or pip install torch"
#         )

from .utils.stochastic import GaussianSample
from .utils.distributions import log_normal_standard, log_normal_diag, log_gaussian
from .utils.utils import to_categorical
from .ekan import KANLinear
import pandas as pd
import numpy as np

# -------- KAN grid update mixin -------- #
class KANGridMixin:
    """
    Mixin giving unified KAN grid maintenance utilities.

    Methods:
      iter_kan_layers() -> iterator over all KANLinear layers
      update_grids(*args, **kwargs) -> calls each KANLinear.update_grid(...)
      maybe_update_grids(step, every=100, *args, **kwargs) -> conditional call
         (call inside training loop if you want periodic updates)

    Typical usage in a training loop:
        if args.update_grid:
            model.update_grids()                     # once per epoch
        # or periodic:
        model.maybe_update_grids(global_step, every=args.update_grid_every)

    If your KANLinear.update_grid signature expects specific kwargs (e.g. data, percentile),
    pass them through:
        model.update_grids(data=batch_x, percentile=0.95)
    """
    def iter_kan_layers(self):
        for m in self.modules():
            if isinstance(m, KANLinear):
                yield m

    def update_grids(self, *args, **kwargs) -> int:
        """
        Returns number of KANLinear layers whose grid was updated.
        """
        updated = 0
        for layer in self.iter_kan_layers():
            if hasattr(layer, "update_grid"):
                try:
                    layer.update_grid(*args, **kwargs)
                except TypeError:
                    # Fallback to no-arg call if signature mismatch
                    layer.update_grid()
                updated += 1
        return updated

    def maybe_update_grids(self, step: int, every: int = 100, *args, **kwargs) -> int:
        """
        Conditionally update grids every `every` steps. Returns number updated or 0.
        Pass every=0 or None to disable.
        """
        if every and every > 0 and step % every == 0:
            return self.update_grids(*args, **kwargs)
        return 0


# -------------------- Utility activations -------------------- #
class MeanAct(nn.Module):
    def __init__(self) -> None:
        # _check_torch_available()
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self) -> None:
        # _check_torch_available()
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


# -------------------- Gradient Reversal -------------------- #
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor) -> torch.Tensor:
    # _check_torch_available()
    return ReverseLayerF.apply(x, 1.0)


# -------------------- Classifiers (KAN) -------------------- #
class Classifier(KANGridMixin, nn.Module):
    def __init__(
        self,
        in_shape: int = 64,
        out_shape: int = 9,
        n_layers: int = 2,
        hidden_sizes: Optional[List[int]] = None,
        use_softmax: bool = True,
        activation: Any = nn.ReLU,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_softmax = use_softmax
        if hidden_sizes is None:
            hidden_sizes = [in_shape // 2 ** i for i in range(1, n_layers)] if n_layers > 1 else []
        layers: List[nn.Module] = []
        prev = in_shape
        for h in hidden_sizes:
            layers += [
                KANLinear(prev, h),
                # nn.BatchNorm1d(h),
                nn.Dropout(dropout),
                # activation(),
            ]
            prev = h
        layers.append(KANLinear(prev, out_shape))
        self.net = nn.Sequential(*layers)
        self._random_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, (KANLinear, nn.Conv2d, nn.ConvTranspose2d)):
                # KANLinear may use 'W' instead of 'weight'
                if hasattr(m, "weight") and m.weight is not None:
                    init_func(m.weight)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif hasattr(m, "W") and m.W is not None:
                    init_func(m.W)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        out = self.net(x)
        if self.use_softmax:
            out = F.softmax(out, dim=1)
        return out.detach().float().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.net(x).argmax(1).detach().float().cpu().numpy()


class Classifier2(KANGridMixin, nn.Module):
    def __init__(
        self,
        in_shape: int = 64,
        hidden: int = 64,
        out_shape: int = 9,
        use_softmax: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_softmax = use_softmax
        self.linear1 = nn.Sequential(
            KANLinear(in_shape, hidden),
            # nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            # nn.ReLU(),
        )
        self.linear2 = KANLinear(hidden, out_shape)
        self._random_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))

    def _random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, (KANLinear, nn.Conv2d, nn.ConvTranspose2d)):
                # KANLinear may use 'W' instead of 'weight'
                if hasattr(m, "weight") and m.weight is not None:
                    init_func(m.weight)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif hasattr(m, "W") and m.W is not None:
                    init_func(m.W)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        out = self.forward(x)
        if self.use_softmax:
            out = F.softmax(out, dim=1)
        return out.detach().float().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.forward(x).argmax(1).detach().float().cpu().numpy()


# -------------------- Encoders / Decoders (KAN) -------------------- #
class Encoder2(KANGridMixin, nn.Module):
    def __init__(self, in_shape: int, layer1: int, layer2: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Sequential(
            KANLinear(in_shape, layer1),
            nn.BatchNorm1d(layer1),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.linear2 = nn.Sequential(
            KANLinear(layer1, layer2),
            nn.BatchNorm1d(layer2),
        )
        self._random_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))

    def _random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, (KANLinear, nn.Conv2d, nn.ConvTranspose2d)):
                # KANLinear may use 'W' instead of 'weight'
                if hasattr(m, "weight") and m.weight is not None:
                    init_func(m.weight)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif hasattr(m, "W") and m.W is not None:
                    init_func(m.W)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)


class Encoder3(KANGridMixin, nn.Module):
    def __init__(self, in_shape: int, layers: dict, dropout: float, device: str = 'cuda'):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev = in_shape
        sizes = list(layers.values())
        for size in sizes[:-1]:
            self.blocks.append(nn.Sequential(
                KANLinear(prev, size),
                # nn.BatchNorm1d(size),
                nn.Dropout(dropout),
                #nn.LeakyReLU(),
            ))
            prev = size
        self.blocks.append(nn.Sequential(KANLinear(prev, sizes[-1])))
        self._random_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

    def _random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, (KANLinear, nn.Conv2d, nn.ConvTranspose2d)):
                # KANLinear may use 'W' instead of 'weight'
                if hasattr(m, "weight") and m.weight is not None:
                    init_func(m.weight)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif hasattr(m, "W") and m.W is not None:
                    init_func(m.W)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)


class Decoder2(KANGridMixin, nn.Module):
    def __init__(self, in_shape: int, n_batches: int, layer1: int, layer2: int, dropout: float) -> None:
        super().__init__()
        self.n_batches = n_batches
        self.linear1 = nn.Sequential(
            KANLinear(layer1 + n_batches, layer2),
            nn.BatchNorm1d(layer2),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.linear2 = KANLinear(layer2, in_shape)
        self._random_init()

    def forward(self, x: torch.Tensor, batches: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        if batches is not None and self.n_batches > 0:
            x = torch.cat((x, batches), 1)
        h = self.linear1(x)
        out = self.linear2(h)
        return [h, out]

    def _random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, (KANLinear, nn.Conv2d, nn.ConvTranspose2d)):
                # KANLinear may use 'W' instead of 'weight'
                if hasattr(m, "weight") and m.weight is not None:
                    init_func(m.weight)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif hasattr(m, "W") and m.W is not None:
                    init_func(m.W)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)


class Decoder3(KANGridMixin, nn.Module):
    def __init__(self, in_shape: int, n_batches: int, layers: dict, dropout: float, device: str = 'cuda'):
        super().__init__()
        self.n_batches = n_batches
        self.blocks = nn.ModuleList()
        rev_sizes = list(layers.values())[::-1]  # largest -> smallest
        prev = rev_sizes[0]
        for size in rev_sizes[1:]:
            self.blocks.append(nn.Sequential(
                KANLinear(prev + (n_batches if n_batches > 0 else 0), size),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout),
                nn.ReLU(),
            ))
            prev = size
        self.out = KANLinear(prev + (n_batches if n_batches > 0 else 0), in_shape)
        self._random_init()

    def forward(self, x: torch.Tensor, batches: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.blocks:
            if self.n_batches > 0 and batches is not None:
                x = torch.cat((x, batches), dim=1)
            x = blk(x)
        if self.n_batches > 0 and batches is not None:
            x = torch.cat((x, batches), dim=1)
        return self.out(x)

    def _random_init(self, init_func: Any = nn.init.kaiming_uniform_) -> None:
        for m in self.modules():
            if isinstance(m, (KANLinear, nn.Conv2d, nn.ConvTranspose2d)):
                # KANLinear may use 'W' instead of 'weight'
                if hasattr(m, "weight") and m.weight is not None:
                    init_func(m.weight)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif hasattr(m, "W") and m.W is not None:
                    init_func(m.W)
                    if getattr(m, "bias", None) is not None and m.bias is not None:
                        nn.init.zeros_(m.bias)


# -------------------- SHAP + AutoEncoders (KAN) -------------------- #
class SHAPKANAutoEncoder2(KANGridMixin, nn.Module):
    def __init__(
        self,
        in_shape: int,
        n_batches: int,
        nb_classes: int,
        n_emb: int,
        n_meta: int,
        mapper: bool,
        variational: bool,
        layer1: int,
        layer2: int,
        dropout: float,
        n_layers: int,
        zinb: bool = False,
        conditional: bool = True,
        add_noise: bool = False,
        tied_weights: int = 0,
        device: str = 'cuda',
        is_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        self.n_emb = n_emb
        self.add_noise = add_noise
        self.n_meta = n_meta
        self.device = device
        self.is_sigmoid = is_sigmoid
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'

        self.enc = Encoder2(in_shape + n_meta, layer1, layer2, dropout)
        if conditional:
            self.dec = Decoder2(in_shape + n_meta, n_batches, layer2, layer1, dropout)
        else:
            self.dec = Decoder2(in_shape + n_meta, 0, layer2, layer1, dropout)
        self.mapper = Classifier(n_batches + 1, layer2, n_layers=1, hidden_sizes=[], dropout=dropout)

        if variational:
            self.gaussian_sampling = GaussianSample(layer2, layer2, device)
        else:
            self.gaussian_sampling = None

        self.dann_discriminator = Classifier2(layer2, 64, n_batches)
        self.classifier = Classifier(layer2 + n_emb, nb_classes, n_layers=n_layers)
        self._dec_mean = nn.Sequential(KANLinear(layer1, in_shape + n_meta), nn.Sigmoid())
        self._dec_disp = nn.Sequential(KANLinear(layer1, in_shape + n_meta), DispAct())
        self._dec_pi = nn.Sequential(KANLinear(layer1, in_shape + n_meta), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        batches: Optional[torch.Tensor] = None,
        sampling: bool = False,
        beta: float = 1.0
    ) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values).to(self.device)
        if self.n_emb > 0:
            meta_values = x[:, -2:]
            x = x[:, :-2]
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -0.1).type_as(x)

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

    # Probability / prediction helpers
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        return self.classifier(x).detach().float().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.classifier(x).argmax(1).detach().float().cpu().numpy()


class SHAPKANAutoEncoder3(KANGridMixin, nn.Module):
    def __init__(
        self,
        in_shape: int,
        n_batches: int,
        nb_classes: int,
        n_emb: int,
        n_meta: int,
        mapper: bool,
        variational: bool,
        layers: dict,
        dropout: float,
        n_layers: int,
        zinb: bool = False,
        conditional: bool = True,
        add_noise: bool = False,
        tied_weights: int = 0,
        device: str = 'cuda',
        is_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        self.n_emb = n_emb
        self.add_noise = add_noise
        self.n_meta = n_meta
        self.device = device
        self.is_sigmoid = is_sigmoid
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'

        self.enc = Encoder3(in_shape + n_meta, layers, dropout, device)
        if conditional:
            self.dec = Decoder3(in_shape + n_meta, n_batches, layers, dropout, device)
        else:
            self.dec = Decoder3(in_shape + n_meta, 0, layers, dropout, device)

        last_dim = list(layers.values())[-1]
        self.mapper = Classifier(n_batches + 1, last_dim, n_layers=1, hidden_sizes=[], dropout=dropout)

        if variational:
            self.gaussian_sampling = GaussianSample(last_dim, last_dim, device)
        else:
            self.gaussian_sampling = None

        self.dann_discriminator = Classifier2(last_dim, 64, n_batches)
        self.classifier = Classifier(last_dim + n_emb, nb_classes, n_layers=n_layers)

        if zinb and len(layers) > 1:
            penultimate = list(layers.values())[-2]
            self._dec_mean = nn.Sequential(KANLinear(penultimate, in_shape + n_meta), MeanAct())
            self._dec_disp = nn.Sequential(KANLinear(penultimate, in_shape + n_meta), DispAct())
            self._dec_pi = nn.Sequential(KANLinear(penultimate, in_shape + n_meta), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        batches: Optional[torch.Tensor] = None,
        sampling: bool = False,
        beta: float = 1.0
    ) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values).to(self.device)
        if self.n_emb > 0:
            meta_values = x[:, -2:]
            if self.n_meta == 0:
                x = x[:, :-2]
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -0.1).type_as(x)
        enc = self.enc(x)

        if self.gaussian_sampling is not None:
            if sampling:
                enc, mu, log_var = self.gaussian_sampling(enc, train=True, beta=beta)
            else:
                enc, _, _ = self.gaussian_sampling(enc, train=False)

        if self.use_mapper and batches is not None:
            bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            enc = enc + self.mapper(bs).squeeze()

        if self.n_emb > 0:
            out = self.classifier(torch.cat((enc, meta_values), 1))
        else:
            out = self.classifier(enc)
        return out

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        return self.classifier(x).detach().float().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.classifier(x).argmax(1).detach().float().cpu().numpy()


# -------------------- Variational / Loss helpers -------------------- #
def log_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    case_zero = F.softplus(-pi + theta * torch.log(theta + eps) - theta * torch.log(theta + mu + eps)) - F.softplus(-pi)
    case_non_zero = (
        -pi
        - F.softplus(-pi)
        + theta * torch.log(theta + eps)
        - theta * torch.log(theta + mu + eps)
        + x * torch.log(mu + eps)
        - x * torch.log(theta + mu + eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mask = torch.less(x, eps).float()
    res = mask * case_zero + (1.0 - mask) * case_non_zero
    res = torch.nan_to_num(res, 0)
    return torch.sum(res, dim=-1)


class KANAutoEncoder3(KANGridMixin, nn.Module):
    """KAN-based AutoEncoder3 analogue.

    Mirrors AutoEncoder3 (in aedann.py) but swaps nn.Linear for KANLinear and
    uses the already defined KAN Encoder3 / Decoder3. Provides reconstruction,
    optional variational sampling (GaussianSample), optional ZINB outputs, and
    batch-effect mapping. Forward returns [enc, rec, zinb_loss, kl].
    """
    def __init__(
        self,
        in_shape: int,
        n_batches: int,
        nb_classes: int,
        n_meta: int,
        n_emb: int,
        mapper: bool,
        variational: bool,
        layers: dict,
        dropout: float,
        n_layers: int,
        prune_threshold: float,
        zinb: bool = False,
        conditional: bool = True,
        add_noise: bool = False,
        tied_weights: int = 0,
        update_grid: bool = False,
        device: str = 'cuda',
        is_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        self.add_noise = add_noise
        self.is_sigmoid = is_sigmoid
        self.device = device
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'
        self.n_emb = n_emb
        self.n_meta = n_meta
        self.prune_threshold = prune_threshold

        # Encoder / Decoder
        self.enc = Encoder3(in_shape + n_meta, layers, dropout, device)
        if conditional:
            self.dec = Decoder3(in_shape + n_meta, n_batches, layers, dropout, device)
        else:
            self.dec = Decoder3(in_shape + n_meta, 0, layers, dropout, device)

        last_dim = list(layers.values())[-1]
        self.mapper = Classifier(n_batches + 1, last_dim, n_layers=1, hidden_sizes=[], dropout=dropout)

        # Variational sampling
        if variational:
            self.gaussian_sampling = GaussianSample(last_dim, last_dim, device)
        else:
            self.gaussian_sampling = None

        self.dann_discriminator = Classifier2(last_dim, 64, n_batches)
        self.classifier = Classifier(last_dim + n_emb, nb_classes, n_layers=n_layers, hidden_sizes=None, dropout=dropout)

        if self.zinb and len(layers) > 1:
            penultimate = list(layers.values())[-2]
            self._dec_mean = nn.Sequential(KANLinear(penultimate, in_shape + n_meta), MeanAct())
            self._dec_disp = nn.Sequential(KANLinear(penultimate, in_shape + n_meta), DispAct())
            self._dec_pi = nn.Sequential(KANLinear(penultimate, in_shape + n_meta), nn.Sigmoid())
        elif self.zinb:
            # Fallback if only one layer defined
            self._dec_mean = nn.Sequential(KANLinear(last_dim, in_shape + n_meta), MeanAct())
            self._dec_disp = nn.Sequential(KANLinear(last_dim, in_shape + n_meta), DispAct())
            self._dec_pi = nn.Sequential(KANLinear(last_dim, in_shape + n_meta), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        to_rec: torch.Tensor,
        batches: Optional[torch.Tensor] = None,
        sampling: bool = False,
        beta: float = 1.0,
        mapping: bool = True,
    ) -> List[Any]:
        rec: dict = {}
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values).to(self.device)
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -0.1).type_as(x)

        enc = self.enc(x)

        # Variational sampling
        if self.gaussian_sampling is not None:
            if sampling:
                enc, mu, log_var = self.gaussian_sampling(enc, train=True, beta=beta)
                kl = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
            else:
                enc, _, _ = self.gaussian_sampling(enc, train=False)
                kl = torch.zeros(enc.size(0), device=enc.device)
        else:
            kl = torch.zeros(enc.size(0), device=enc.device)

        # Batch-effect mapping
        if self.use_mapper and mapping and batches is not None:
            bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            enc_be = enc + self.mapper(bs).squeeze()
        else:
            enc_be = enc

        # Reconstruction (no tied-weights support for multi-KAN yet)
        if self.tied_weights:
            raise NotImplementedError("tied_weights not implemented for KANAutoEncoder3")
        else:
            if batches is not None and self.n_batches > 0:
                rec_mean = self.dec(enc_be, to_categorical(batches, self.n_batches + 1).to(self.device).float())
            else:
                rec_mean = self.dec(enc_be, None)
            rec = {"mean": rec_mean}

        # ZINB optional
        if self.zinb:
            penult_act = enc_be  # simplification; could capture penultimate hidden if needed
            _mean = self._dec_mean(penult_act)
            _disp = self._dec_disp(penult_act)
            _pi = self._dec_pi(penult_act)
            zinb_loss = self.zinb_loss(to_rec, _mean, _disp, _pi)
            rec = {"mean": _mean, "rec": to_rec}
        else:
            zinb_loss = torch.zeros(1, device=enc.device)

        if self.is_sigmoid:
            rec['mean'] = torch.sigmoid(rec['mean'])

        return [enc, rec, zinb_loss, kl]

    # Utilities reused from aedann AutoEncoder3
    def predict_proba(self, inputs: torch.Tensor) -> np.ndarray:
        x = self.enc(inputs)
        return self.classifier(x).detach().float().cpu().float().numpy()

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        x = self.enc(inputs)
        return self.classifier(x).argmax(1).detach().float().cpu().float().numpy()

    def zinb_loss(self, x: torch.Tensor, mean: torch.Tensor, disp: torch.Tensor, pi: torch.Tensor, scale_factor: float = 1.0, ridge_lambda: float = 0.0) -> torch.Tensor:
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
        return torch.mean(result)

    def prune_model_paperwise(self, is_classification: bool, is_dann: bool, weight_threshold: float = 0) -> int:
        print("Pruning not available for this model")
        return 0

    def count_n_neurons(self) -> int:
        total = 0
        layer_counts = {}
        for name, module in self.named_modules():
            if isinstance(module, KANLinear):
                layer_counts[name] = module.out_features
                total += module.out_features
        return {"total": total, "layers": layer_counts}


class KANAutoEncoder2(KANGridMixin, nn.Module):
    """KAN-based analogue of AutoEncoder2.

    Uses Encoder2/Decoder2 with KANLinear layers, optional variational sampling,
    optional ZINB reconstruction, and batch-effect mapping. Forward returns
    [enc, rec, zinb_loss, kl] where rec is a dict containing at least key 'mean'.
    """
    def __init__(
        self,
        in_shape: int,
        n_batches: int,
        nb_classes: int,
        n_meta: int,
        n_emb: int,
        mapper: bool,
        variational: bool,
        layer1: int,
        layer2: int,
        dropout: float,
        n_layers: int,
        prune_threshold: float,
        zinb: bool = False,
        conditional: bool = True,
        add_noise: bool = False,
        tied_weights: int = 0,
        update_grid: bool = False,
        device: str = 'cuda',
        is_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        self.add_noise = add_noise
        self.device = device
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'
        self.n_emb = n_emb
        self.n_meta = n_meta
        self.prune_threshold = prune_threshold
        self.is_sigmoid = is_sigmoid

        # Encoder / Decoder
        self.enc = Encoder2(in_shape + n_meta, layer1, layer2, dropout)
        if conditional:
            self.dec = Decoder2(in_shape + n_meta, n_batches, layer2, layer1, dropout)
        else:
            self.dec = Decoder2(in_shape + n_meta, 0, layer2, layer1, dropout)
        self.mapper = Classifier(n_batches + 1, layer2, n_layers=1, hidden_sizes=[], dropout=dropout)

        # Variational sampling
        if variational:
            self.gaussian_sampling = GaussianSample(layer2, layer2, device)
        else:
            self.gaussian_sampling = None

        self.dann_discriminator = Classifier2(layer2, 64, n_batches)
        self.classifier = Classifier(layer2 + n_emb, nb_classes, n_layers=n_layers, hidden_sizes=None, dropout=dropout)

        # ZINB projection heads (operate on first decoder hidden output)
        if self.zinb:
            self._dec_mean = nn.Sequential(KANLinear(layer1, in_shape + n_meta), MeanAct())
            self._dec_disp = nn.Sequential(KANLinear(layer1, in_shape + n_meta), DispAct())
            self._dec_pi = nn.Sequential(KANLinear(layer1, in_shape + n_meta), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        to_rec: torch.Tensor,
        batches: Optional[torch.Tensor] = None,
        sampling: bool = False,
        beta: float = 1.0,
        mapping: bool = True,
    ) -> List[Any]:
        rec: dict = {}
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values).to(self.device)
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -0.1).type_as(x)

        enc = self.enc(x)

        # Variational sampling
        if self.gaussian_sampling is not None:
            if sampling:
                enc, mu, log_var = self.gaussian_sampling(enc, train=True, beta=beta)
                kl = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
            else:
                enc, _, _ = self.gaussian_sampling(enc, train=False)
                kl = torch.zeros(enc.size(0), device=enc.device)
        else:
            kl = torch.zeros(enc.size(0), device=enc.device)

        # Batch-effect mapping
        if self.use_mapper and mapping and batches is not None:
            bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            enc_be = enc + self.mapper(bs).squeeze()
        else:
            enc_be = enc

        # Reconstruction (Decoder2 returns [hidden, out])
        if self.tied_weights:
            raise NotImplementedError("tied_weights not implemented for KANAutoEncoder2")
        else:
            if batches is not None and self.n_batches > 0:
                try:
                    bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
                except Exception:
                    bs = to_categorical(batches.long(), self.n_batches + 1).to(self.device).float()
            else:
                bs = None
            dec_out = self.dec(enc_be, bs)
            rec = {"mean": dec_out}

        # ZINB optional (operate on first hidden from decoder)
        if self.zinb:
            hidden_first = rec['mean'][0]  # first element from decoder list
            _mean = self._dec_mean(hidden_first)
            _disp = self._dec_disp(hidden_first)
            _pi = self._dec_pi(hidden_first)
            zinb_loss = self.zinb_loss(to_rec, _mean, _disp, _pi)
            rec = {"mean": _mean, "rec": to_rec}
        else:
            zinb_loss = torch.zeros(1, device=enc.device)

        if self.is_sigmoid:
            if isinstance(rec['mean'], torch.Tensor):
                rec['mean'] = torch.sigmoid(rec['mean'])
            # else leave list structure for compatibility if not zinb

        return [enc, rec, zinb_loss, kl]

    def predict_proba(self, inputs: torch.Tensor) -> np.ndarray:
        x = self.enc(inputs)
        return self.classifier(x).detach().float().cpu().float().numpy()

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        x = self.enc(inputs)
        return self.classifier(x).argmax(1).detach().float().cpu().float().numpy()

    def zinb_loss(self, x: torch.Tensor, mean: torch.Tensor, disp: torch.Tensor, pi: torch.Tensor, scale_factor: float = 1.0, ridge_lambda: float = 0.0) -> torch.Tensor:
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
        return torch.mean(result)

    def prune_model_paperwise(self, is_classification: bool, is_dann: bool, weight_threshold: float = 0) -> int:
        print("Pruning not available for this model")
        return 0

    def count_n_neurons(self) -> int:
        total = 0
        layer_counts = {}
        for name, module in self.named_modules():
            if isinstance(module, KANLinear):
                layer_counts[name] = module.out_features
                total += module.out_features
        return {"total": total, "layers": layer_counts}

# Preserve legacy alias if earlier code expects KANAutoEncoder3 name
