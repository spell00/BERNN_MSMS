# Rewritten to mirror aedann.py but using KANLinear instead of nn.Linear.
from typing import Any, Optional, Tuple, List

try:
    import torch
    from torch import nn
    from torch.autograd import Variable
    import torch.nn.functional as F
    from torch.autodiff import forward_ad  # optional, safe to ignore if unused
    from torch.autograd import Function
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def _check_torch_available():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Install with: pip install bernn[deep-learning] or pip install torch"
        )

from .utils.stochastic import GaussianSample
from .utils.distributions import log_normal_standard, log_normal_diag, log_gaussian
from .utils.utils import to_categorical
from ...train.pytorch.ekan import KANLinear
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
        _check_torch_available()
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self) -> None:
        _check_torch_available()
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
    _check_torch_available()
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
                init_func(m.weight)
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        out = self.net(x)
        if self.use_softmax:
            out = F.softmax(out, dim=1)
        return out.detach().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.net(x).argmax(1).detach().cpu().numpy()


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
                init_func(m.weight)
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        out = self.forward(x)
        if self.use_softmax:
            out = F.softmax(out, dim=1)
        return out.detach().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.forward(x).argmax(1).detach().cpu().numpy()


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
                init_func(m.weight)
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
                init_func(m.weight)
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
                init_func(m.weight)
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
                init_func(m.weight)
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
        device: str = 'cuda'
    ):
        super().__init__()
        self.n_emb = n_emb
        self.add_noise = add_noise
        self.n_meta = n_meta
        self.device = device
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
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.classifier(x).argmax(1).detach().cpu().numpy()


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
        device: str = 'cuda'
    ):
        super().__init__()
        self.n_emb = n_emb
        self.add_noise = add_noise
        self.n_meta = n_meta
        self.device = device
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
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.classifier(x).argmax(1).detach().cpu().numpy()


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


# -------------------- Backward compatibility (aliases) -------------------- #
# Legacy class names (if older code imports them)
#KANAutoencoder2 = SHAPKANAutoEncoder2
#KANAutoencoder3 = SHAPKANAutoEncoder3
#SHAPKANAutoencoder2 = SHAPKANAutoEncoder2
#SHAPKANAutoencoder3 = SHAPKANAutoEncoder3
