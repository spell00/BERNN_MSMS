import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function
from .utils.stochastic import GaussianSample
from .utils.distributions import log_normal_standard, log_normal_diag, log_gaussian
from .utils.utils import to_categorical
import pandas as pd
from .ekan.src.efficient_kan.kan import KANLinear
import copy
from .utils.utils import to_categorical
import numpy as np

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if args.cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


# https://github.com/DHUDBlab/scDSC/blob/1247a63aac17bdfb9cd833e3dbe175c4c92c26be/layers.py#L43
class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


# https://github.com/DHUDBlab/scDSC/blob/1247a63aac17bdfb9cd833e3dbe175c4c92c26be/layers.py#L43
class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def grad_reverse(x):
    return ReverseLayerF()(x)


class Classifier(nn.Module):
    def __init__(self, in_shape=64, out_shape=9, n_layers=2, prune_threshold=0, update_grid=False, name=None):
        super().__init__()
        self.name = name
        self.update_grid = update_grid
        self.n_layers = n_layers

        self.layers = nn.ModuleDict()

        if n_layers == 2:
            self.layers["layer1"] = nn.Sequential(
                KANLinear(in_shape, in_shape, name=f'{name}_classifier1', prune_threshold=prune_threshold),
            )
            self.layers["layer2"] = nn.Sequential(
                KANLinear(in_shape, out_shape, name='classifier2', prune_threshold=0.),
            )
        elif n_layers == 1:
            self.layers["layer1"] = nn.Sequential(
                KANLinear(in_shape, out_shape, name='classifier0', prune_threshold=0.),
            )

        self.random_init()

    def forward(self, x):
        if self.update_grid and self.training:
            try:
                self.layers["layer1"][0].update_grid(x.contiguous(), 1e-4)
            except Exception:
                pass

        x = self.layers["layer1"](x)

        if self.n_layers == 2:
            if self.update_grid and self.training:
                try:
                    self.layers["layer2"][0].update_grid(x.contiguous(), 1e-4)
                except Exception:
                    pass
            x = self.layers["layer2"](x)

        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x):
        if self.n_layers == 2:
            return self.layers["layer2"](self.layers["layer1"](x)).detach().cpu().numpy()
        return self.layers["layer1"](x).detach().cpu().numpy()

    def predict(self, x):
        if self.n_layers == 2:
            return self.layers["layer2"](self.layers["layer1"](x)).argmax(1).detach().cpu().numpy()
        return self.layers["layer1"](x).argmax(1).detach().cpu().numpy()


class Classifier2(nn.Module):
    def __init__(self, in_shape=64, hidden=64, out_shape=9, prune_threshold=1e-4, update_grid=False, name=None):
        super().__init__()
        self.name = name
        self.update_grid = update_grid

        self.layers = nn.ModuleDict({
            "layer1": nn.Sequential(
                KANLinear(in_shape, hidden, name=f'{name}_classifier1', prune_threshold=prune_threshold),
                nn.BatchNorm1d(hidden),
                nn.Dropout(),
            ),
            "layer2": nn.Sequential(
                KANLinear(hidden, out_shape, name=f'{name}_classifier2', prune_threshold=0.),
            )
        })

        self.random_init()

    def forward(self, x):
        if self.update_grid and self.training:
            try:
                self.layers["layer1"][0].update_grid(x.contiguous(), 1e-4)
            except Exception:
                pass

        x = self.layers["layer1"](x)

        if self.update_grid and self.training:
            try:
                self.layers["layer2"][0].update_grid(x.contiguous(), 1e-4)
            except Exception:
                pass

        x = self.layers["layer2"](x)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def predict_proba(self, x):
        return self.layers["layer2"](self.layers["layer1"](x)).detach().cpu().numpy()

    def predict(self, x):
        return self.layers["layer2"](self.layers["layer1"](x)).argmax(1).detach().cpu().numpy()


class Classifier3(nn.Module):
    def __init__(self, in_shape=64, hidden=64, out_shape=9, name=None):
        super(Classifier3, self).__init__()
        self.name = name
        self.linear1 = nn.Sequential(
            KANLinear(in_shape, hidden, name=f'{name}_classifier1'),
            nn.BatchNorm1d(hidden),
            nn.Dropout(),
            # nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            KANLinear(hidden, hidden, name=f'{name}_classifier2'),
            nn.BatchNorm1d(hidden),
            nn.Dropout(),
            # nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            KANLinear(hidden, out_shape, prune_threshold=0., name=f'{name}_classifier3'),
        )
        self.random_init()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x):
        return self.linear2(x).detach().cpu().numpy()

    def predict(self, x):
        return self.linear2(x).argmax(1).detach().cpu().numpy()


class Encoder(nn.Module):
    def __init__(self, in_shape, layer1, dropout, name=None):
        super(Encoder, self).__init__()

        self.linear1 = nn.Sequential(
            KANLinear(in_shape, layer1, name=f'{name}_encoder1'),
            nn.BatchNorm1d(layer1),
            # nn.LeakyReLU(),
        )
        self.random_init()

    def forward(self, x):
        x = self.linear1(x)
        # x = self.linear2(x)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, KANLinear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class Encoder2(nn.Module):
    def __init__(self, in_shape, layer1, layer2, dropout, prune_threshold, update_grid=False, name=None):
        super().__init__()
        self.update_grid = update_grid
        self.name = name

        self.layers = nn.ModuleDict({
            "layer1": nn.Sequential(
                KANLinear(in_shape, layer1, name=f'{name}_encoder1', prune_threshold=prune_threshold),
                nn.BatchNorm1d(layer1),
                nn.Dropout(dropout),
            ),
            "layer2": nn.Sequential(
                KANLinear(layer1, layer2, name=f'{name}_encoder2', prune_threshold=prune_threshold),
                nn.BatchNorm1d(layer2),
            )
        })

        self.random_init()

    def forward(self, x, batches=None):
        if self.update_grid and self.training:
            try:
                self.layers["layer1"][0].update_grid(x.contiguous(), 1e-4)
            except Exception:
                pass

        x = self.layers["layer1"](x)

        if self.update_grid and self.training:
            try:
                self.layers["layer2"][0].update_grid(x.contiguous(), 1e-4)
            except Exception:
                pass

        x = self.layers["layer2"](x)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class Decoder2(nn.Module):
    def __init__(self, in_shape, n_batches, layer1, layer2, dropout, prune_threshold, update_grid=False, name=None):
        super().__init__()
        self.update_grid = update_grid
        self.name = name
        self.n_batches = n_batches
        
        self.layers = nn.ModuleDict({
            "layer1": nn.Sequential(
                KANLinear(layer1 + n_batches, layer2, name=f'{name}_decoder1', prune_threshold=prune_threshold),
                nn.BatchNorm1d(layer2),
                nn.Dropout(dropout),
            ),
            "layer2": nn.Sequential(
                KANLinear(layer2, in_shape, name=f'{name}_decoder2', prune_threshold=0.),
            )
        })
        
        self.random_init()

    def forward(self, x, batches=None):
        if batches is not None and self.n_batches > 0:
            x = torch.cat((x, batches), dim=1)
            
        if self.update_grid and self.training:
            try:
                self.layers["layer1"][0].update_grid(x.contiguous(), 1e-4)
            except Exception:
                pass

        x1 = self.layers["layer1"](x)

        if self.update_grid and self.training:
            try:
                self.layers["layer2"][0].update_grid(x1.contiguous(), 1e-4)
            except Exception:
                pass

        x2 = self.layers["layer2"](x1)
        return [x1, x2]

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class Decoder(nn.Module):
    def __init__(self, in_shape, n_batches, layer1, dropout, name):
        super(Decoder, self).__init__()
        self.name = name
        self.linear2 = nn.Sequential(
            KANLinear(layer1 + n_batches, in_shape, name=f'{name}_decoder2'),
        )
        self.n_batches = n_batches
        self.random_init()

    def forward(self, x, batches=None):
        if batches is not None and self.n_batches > 0:
            x = torch.cat((x, batches), 1)
        x1 = self.linear1(x)
        return x1

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class SHAPKANAutoencoder2(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_emb, n_meta, mapper, variational,
                 layer1, layer2, dropout, n_layers, zinb=False, conditional=False,
                 add_noise=False, tied_weights=0, use_gnn=False, device='cuda'):
        super(SHAPKANAutoencoder2, self).__init__()
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

        self.enc = Encoder2(in_shape + n_meta, layer1, layer2, dropout, prune_threshold=0, name='encoder2')
        if conditional:
            self.dec = Decoder2(in_shape + n_meta, n_batches, layer2, layer1, dropout, prune_threshold=0, name='decoder2')
        else:
            self.dec = Decoder2(in_shape + n_meta, 0, layer2, layer1, dropout, prune_threshold=0, name='decoder2')

        self.mapper = Classifier(n_batches + 1, layer2, n_layers=2, prune_threshold=0, name='mapper')
        if variational:
            self.gaussian_sampling = GaussianSample(layer2, layer2, device)
        else:
            self.gaussian_sampling = None

        self.dann_discriminator = Classifier2(layer2, 64, n_batches, prune_threshold=0, name='dann_discriminator')
        self.classifier = Classifier(layer2 + n_emb, nb_classes, n_layers=n_layers, prune_threshold=0, name='classifier')

        self._dec_mean = nn.Sequential(KANLinear(layer1, in_shape + n_meta), nn.Sigmoid())
        self._dec_disp = nn.Sequential(KANLinear(layer1, in_shape + n_meta), DispAct())
        self._dec_pi = nn.Sequential(KANLinear(layer1, in_shape + n_meta), nn.Sigmoid())

        self.random_init(nn.init.kaiming_uniform_)

    def forward(self, x, batches=None, sampling=False, beta=1.0):
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values).to(self.device)
        if self.n_emb > 0:
            meta_values = x[:, -2:]
            x = x[:, :-2]
        # if self.n_meta > 0:
        #     x = x[:, :-self.n_meta]
        # if self.n_meta > 0:
        #     x = x[:, :-2]
        # rec = {}
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)

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

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x):
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, x):
        return self.classifier(x).argmax(1).detach().cpu().numpy()
    def _kld(self, z, q_param, h_last=None, p_param=None):
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
    def zinb_loss(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
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


class KANAutoencoder2(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_meta, n_emb, mapper, 
                 variational, layer1, layer2, dropout, n_layers, prune_threshold, zinb=False,
                 conditional=False, add_noise=False, tied_weights=0, 
                 use_gnn=False, update_grid=False, device='cuda'):
        super(KANAutoencoder2, self).__init__()
        self.prune_threshold = prune_threshold
        self.add_noise = add_noise
        self.device = device
        self.use_gnn = use_gnn
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'
        self.n_meta = n_meta
        self.n_emb = n_emb
        # self.gnn1 = GCNConv(in_shape, in_shape)
        self.enc = Encoder2(in_shape + n_meta, layer1, layer2, dropout, prune_threshold=prune_threshold, update_grid=0, name='encoder2')  # TODO update_grid causes an error, but no idea why
        if conditional:
            self.dec = Decoder2(in_shape + n_meta, n_batches, layer2, layer1, dropout, prune_threshold=prune_threshold, update_grid=update_grid, name='decoder2')
        else:
            self.dec = Decoder2(in_shape + n_meta, 0, layer2, layer1, dropout, prune_threshold=prune_threshold, update_grid=update_grid, name='decoder2')
        self.mapper = Classifier(n_batches + 1, layer2, update_grid=update_grid, prune_threshold=prune_threshold, name='mapper')

        if variational:
            self.gaussian_sampling = GaussianSample(layer2, layer2, device) 
        else:
            self.gaussian_sampling = None
        # TODO dann_disc needs to be at prune_threshold=0, otherwise it will prune away the whole model
        # TODO update_grid causes an error, but no idea why
        self.dann_discriminator = Classifier2(layer2, 64, n_batches, update_grid=0, name='dann_discriminator', prune_threshold=0)  
        self.classifier = Classifier(layer2 + n_emb, nb_classes, n_layers=n_layers, update_grid=update_grid, prune_threshold=prune_threshold, name='classifier')
        self._dec_mean = nn.Sequential(KANLinear(layer1, in_shape + n_meta), MeanAct())
        self._dec_disp = nn.Sequential(KANLinear(layer1, in_shape + n_meta), DispAct())
        self._dec_pi = nn.Sequential(KANLinear(layer1, in_shape + n_meta), nn.Sigmoid())
        # self.random_init(nn.init.kaiming_uniform_)
        

    def forward(self, x, to_rec, batches=None, sampling=False, beta=1.0, mapping=True):
        rec = {}
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        try:
            enc = self.enc(x)
        except:
            pass
        if torch.isnan(enc).any():
            print('nan in enc')
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
            except:
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

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x):
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, x):
        return self.classifier(x).argmax(1).detach().cpu().numpy()

    def _kld(self, z, q_param, h_last=None, p_param=None):
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
    def zinb_loss(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
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

    # def prune_model(self, weight_threshold):
    #     total = 0
    #     layers = {}
    #     for idx, layer in enumerate(self.children()):
    #         try:
    #             layers[layer.name] = {}
    #         except Exception as e:
    #             continue
    #         for idx2, layer2 in enumerate(layer.children()):
    #             if isinstance(layer2, KANLinear):
    #                 layer2.prune_neurons(weight_threshold)
    #                 n = layer2.mask.sum().item()
    #                 total += n
    #                 layers[layer2.name] = n
    #             # If subscriptable
    #             for layer3 in layer2.keys():
    #                 if isinstance(layer2[layer3][0], KANLinear):
    #                     if not (layer.name == 'decoder2' and layer3 == 'layer2') \
    #                         and not (layer.name == 'classifier' and layer3 == 'layer2') \
    #                             and not (layer.name == 'dann_discriminator' and layer3 == 'layer2'):
    #                         layer2[layer3][0].prune_neurons(weight_threshold)
    #                         n = layer2[layer3][0].mask.sum().item()
    #                         total += n
    #                         layers[layer.name][layer3] = n
    #     layers['total'] = total
    #     return layers

    def prune_model(self, weight_threshold):
        """
        Iterates through all KANLinear layers and prunes neurons using node-level pruning
        based on incoming and outgoing spline weights (Eq. 2.21 of the KAN paper).
        """
        total = 0
        layers = {}

        for module in self.children().layers:
            if not hasattr(module, "name"):
                continue

            layers[module.name] = {}
            prev_kan = None
            prev_name = None

            # Flatten submodules
            if isinstance(module, nn.ModuleDict):
                sublayers = list(module.values())
            elif isinstance(module, nn.Sequential):
                sublayers = [module]
            else:
                sublayers = list(module.children())

            for name, sub in enumerate(sublayers):
                if isinstance(sub, nn.Sequential):
                    sub = sub[0]  # grab KANLinear from Sequential

                if not isinstance(sub, KANLinear):
                    continue

                if prev_kan is not None:
                    prev_kan.prune_neurons(weight_threshold, next_layer_weights=sub.scaled_spline_weight)
                    n = prev_kan.mask.sum().item()
                    layers[module.name][prev_name] = n
                    total += n

                prev_kan = sub
                prev_name = f"layer{name}"

            # Prune the last one
            if prev_kan is not None:
                prev_kan.prune_neurons(weight_threshold, next_layer_weights=None)
                n = prev_kan.mask.sum().item()
                layers[module.name][prev_name] = n
                total += n

        layers["total"] = total
        return layers

    @torch.no_grad()
    def prune_model_paperwise(self, is_classification, is_dann, weight_threshold: float = 1e-2) -> dict:
        summary = {}
        total_remaining = 0

        # Collect all KANLinear layers across parts
        all_layers = {}
        for part_name in ['enc', 'dec', 'classifier', 'mapper', 'dann_discriminator']:
            part = getattr(self, part_name, None)
            if part is None or not hasattr(part, 'layers'):
                continue
            for lname, seq in part.layers.items():
                layer = seq[0]
                if isinstance(layer, KANLinear):
                    full_name = f"{part_name}.{lname}"
                    all_layers[full_name] = layer

        for full_name, layer in all_layers.items():
            part_name, layer_name = full_name.split(".")

            outgoing_total = []

            # Determine target next layers
            if full_name == 'enc.layer2':
                next_targets = ['dec.layer1', 'classifier.layer1']
            elif full_name == 'mapper.layer2':
                next_targets = ['dec.layer1']
            else:
                # Try to find the next layer in the same part
                part = getattr(self, part_name)
                layer_names = list(part.layers.keys())
                idx = layer_names.index(layer_name)
                if idx + 1 < len(layer_names):
                    next_layer_name = layer_names[idx + 1]
                    next_targets = [f"{part_name}.{next_layer_name}"]
                else:
                    next_targets = []

            # Accumulate outgoing weights from the next target layer(s)
            for next_name in next_targets:
                next_layer = all_layers.get(next_name)
                if next_layer is None:
                    continue
                if next_layer.in_features == layer.out_features:
                    out = next_layer.base_weight.abs().mean(dim=0)
                    if hasattr(next_layer, "scaled_spline_weight"):
                        out += next_layer.scaled_spline_weight.abs().mean(dim=(0, 2))
                    # print(f"{full_name} → {next_name}: {out.shape}")
                    outgoing_total += [out]
            if len(outgoing_total) > 0 and not (part_name == 'classifier' and not is_classification):
                outgoing_total = torch.max(torch.stack(outgoing_total, axis=0), axis=0).values
                mask = (outgoing_total >= weight_threshold)
            else:
                mask = torch.ones_like(layer.mask, dtype=torch.bool)

            mask = mask & layer.mask

            if mask.sum() == 0:
                pass


            kept = int(mask.sum().item())

            summary[full_name] = kept
            total_remaining += kept

            if (part_name == 'classifier' and not is_classification) or (part_name == 'dann_discriminator' and is_dann):
                continue
            if part_name == 'classifier' and is_classification:
                is_classification = True
            else:
                is_classification = is_classification
            if hasattr(layer, "mask"):
                layer.mask.copy_(mask.to(layer.mask.device))
                # part.layers[layer_name] = nn.Sequential(layer) 
                if mask.sum() == 0:
                    pass
                #     topk = importance.topk(1).indices
                #     mask[topk] = True

        summary["total_remaining"] = total_remaining
        return summary


    def prune_model_pathwise(self, weight_threshold):
        total = 0
        layers = {}

        # Run enc first
        # Init at 1 to avoid unbalanced importance
        weights = {
            'decoder2': {
                'layer2': np.ones_like(self.dec.layers['layer2'][0].counts),
                'layer1': np.zeros_like(self.dec.layers['layer1'][0].counts),
            },
            'encoder2': {
                'layer2': np.zeros_like(self.enc.layers['layer2'][0].counts),
                'layer1': np.zeros_like(self.enc.layers['layer1'][0].counts),    
            },
            'classifier': {
                'layer2': np.ones_like(self.classifier.layers['layer2'][0].counts),
                'layer1': np.zeros_like(self.classifier.layers['layer1'][0].counts),
            },
            'mapper': {
                'layer2': np.ones_like(self.mapper.layers['layer2'][0].counts),
                'layer1': np.zeros_like(self.mapper.layers['layer1'][0].counts),
            },
            # TODO make this part for when normAE or DANN only
            # 'dann_discriminator': {
            #     'layer2': np.zeros_like(self.dann_discriminator.layers['layer2'][0].counts),
            #     'layer1': np.zeros_like(self.dann_discriminator.layers['layer1'][0].counts),
            #},
        }
        accumulated_n = {
            'decoder2': 1,
            'encoder2': 1,
            'classifier': 1,
            'mapper': 1,
            # TODO make this part for when normAE or DANN only
            # 'dann_discriminator': {
            #     'layer2': np.zeros_like(self.dann_discriminator.layers['layer2'][0].counts),
            #     'layer1': np.zeros_like(self.dann_discriminator.layers['layer1'][0].counts),
            #},
        }
        prev_layer = ''
        layers['decoder2'] = {}
        for idx, layer in enumerate(reversed(list(self.dec.layers.keys()))):
            if isinstance(self.dec.layers[layer][0], KANLinear):
                if not layer == 'layer2':
                    base = self.dec.layers[prev_layer][0].base_weight.detach().abs().cpu()
                    spline = self.dec.layers[prev_layer][0].scaled_spline_weight.detach().abs().mean(-1).cpu()
                    W = (base + spline).numpy()
                    # n = len(self.dec.layers[layer][0].mask) * self.dec.layers[layer][0].n
                    n = len(self.dec.layers[layer][0].mask)
                    # self.dec.layers[layer][0].counts = (weights['decoder2'][prev_layer] @ W * self.dec.layers[layer][0].counts) / self.dec.layers[layer][0].n
                    self.dec.layers[layer][0].counts = ((W * self.dec.layers[layer][0].counts)).mean(0)
                    # weights['decoder2'][layer] = self.dec.layers[layer][0].counts
                    # self.dec.layers[layer][0].counts = (weights['decoder2'][prev_layer] @ W * self.dec.layers[layer][0].counts) / n
                    total += self.dec.layers[layer][0].mask.sum().item()
                    self.dec.layers[layer][0].prune_neurons(weight_threshold)
                    layers['decoder2'][layer] = self.dec.layers[layer][0].mask.sum().item()
                else:
                    # n = len(self.dec.layers[layer][0].mask) * self.dec.layers[layer][0].n
                    n = self.dec.layers[layer][0].n
                accumulated_n['decoder2'] *= n
                prev_layer = layer

        prev_layer = ''
        layers['classifier'] = {}
        for idx, layer in enumerate(reversed(list(self.classifier.layers.keys()))):
            if isinstance(self.classifier.layers[layer][0], KANLinear):
                if not layer == 'layer2':
                    base = self.classifier.layers[prev_layer][0].base_weight.detach().abs().cpu()
                    spline = self.classifier.layers[prev_layer][0].scaled_spline_weight.detach().abs().mean(-1).cpu()
                    W = (base + spline).numpy()
                    # n = len(self.classifier.layers[layer][0].mask) * self.classifier.layers[layer][0].n
                    n = len(self.classifier.layers[layer][0].mask)
                    # weights['classifier'][layer] = self.classifier.layers[layer][0].counts
                    # self.classifier.layers[layer][0].counts = (weights['classifier'][prev_layer] @ W * self.classifier.layers[layer][0].counts) / self.classifier.layers[layer][0].n
                    self.classifier.layers[layer][0].counts = ((W * self.classifier.layers[layer][0].counts)).mean(0)
                    # self.classifier.layers[layer][0].counts = (weights['classifier'][prev_layer] @ W * self.classifier.layers[layer][0].counts) / n
                    total += self.classifier.layers[layer][0].mask.sum().item()
                    self.classifier.layers[layer][0].prune_neurons(weight_threshold)
                    layers['classifier'][layer] = self.classifier.layers[layer][0].mask.sum().item()
                else:
                    # n = len(self.classifier.layers[layer][0].mask) * self.classifier.layers[layer][0].n
                    n = len(self.classifier.layers[layer][0].mask)
                accumulated_n['classifier'] *= n
                prev_layer = layer
        
        # prev_layer = ''
        # layers['mapper'] = {}
        # for idx, layer in enumerate(reversed(list(self.mapper.layers.keys()))):
        #     if isinstance(self.mapper.layers[layer][0], KANLinear):
        #         if not layer == 'layer2':
        #             base = self.mapper.layers[prev_layer][0].base_weight.detach().abs().cpu()
        #             spline = self.mapper.layers[prev_layer][0].scaled_spline_weight.detach().abs().mean(-1).cpu()
        #             W = (base + spline).numpy()
        #             # n = len(self.mapper.layers[layer][0].mask) * self.mapper.layers[layer][0].n
        #             n = self.mapper.layers[layer][0].n
        #             self.mapper.layers[layer][0].counts = (weights['mapper'][prev_layer] @ W * self.mapper.layers[layer][0].counts) / (n * accumulated_n['mapper'])
        #             weights['mapper'][layer] = self.mapper.layers[layer][0].counts
        #             total += n
        #             self.mapper.layers[layer][0].prune_neurons(weight_threshold)
        #             layers['mapper'][layer] = n
        #         else:
        #             # n = len(self.mapper.layers[layer][0].mask) * self.mapper.layers[layer][0].n
        #             n = self.mapper.layers[layer][0].n
        #             weights['mapper'][layer] = self.mapper.layers[layer][0].counts / n
        #             total += n
        #             self.mapper.layers[layer][0].prune_neurons(weight_threshold)
        #             layers['mapper'][layer] = n
        #         accumulated_n['mapper'] += n
        #         prev_layer = layer
        
        prev_layer = ''
        layers['encoder2'] = {}
        for idx, layer in enumerate(reversed(list(self.enc.layers.keys()))):
            if isinstance(self.enc.layers[layer][0], KANLinear):
                if not layer == 'layer2':
                    base = self.enc.layers[prev_layer][0].base_weight.detach().abs().cpu()
                    spline = self.enc.layers[prev_layer][0].scaled_spline_weight.detach().abs().mean(-1).cpu()
                    W = (base + spline).numpy()
                    # n = len(self.enc.layers[layer][0].mask) * self.enc.layers[layer][0].n
                    n = len(self.enc.layers[layer][0].mask)
                    # self.enc.layers[layer][0].counts = (weights['encoder2'][prev_layer] @ W * self.enc.layers[layer][0].counts) / self.enc.layers[layer][0].n
                    self.enc.layers[layer][0].counts = ((W * self.enc.layers[layer][0].counts)).mean(0)
                    # self.enc.layers[layer][0].counts = (weights['encoder2'][prev_layer] @ W * self.enc.layers[layer][0].counts) / n
                    # weights['encoder2'][layer] = self.enc.layers[layer][0].counts
                    total += self.enc.layers[layer][0].mask.sum().item()
                    self.enc.layers[layer][0].prune_neurons(weight_threshold)
                    layers['encoder2'][layer] = self.enc.layers[layer][0].mask.sum().item()
                else:
                    # base_mapper = self.mapper.layers['layer1'][0].base_weight.detach().abs().cpu()
                    # spline_mapper = self.mapper.layers['layer1'][0].scaled_spline_weight.detach().abs().mean(-1).cpu()
                    # W_mapper = (base_mapper + spline_mapper).numpy()

                    base_dec = self.dec.layers['layer1'][0].base_weight.detach().abs().cpu()
                    spline_dec = self.dec.layers['layer1'][0].scaled_spline_weight.detach().abs().mean(-1).cpu()
                    W_dec = (base_dec + spline_dec).numpy()

                    if self.n_emb > 0:
                        base_classifier = self.classifier.layers['layer1'][0].base_weight.detach().abs().cpu()[:-self.n_emb, :-self.n_emb]
                        spline_classifier = self.classifier.layers['layer1'][0].scaled_spline_weight.detach().abs().mean(-1).cpu()[:-self.n_emb, :-self.n_emb]
                        weights_classif_layer1 = weights['classifier']['layer1'][:-self.n_emb]
                    else:
                        base_classifier = self.classifier.layers['layer1'][0].base_weight.detach().abs().cpu()
                        spline_classifier = self.classifier.layers['layer1'][0].scaled_spline_weight.detach().abs().mean(-1).cpu()
                        weights_classif_layer1 = weights['classifier']['layer1']

                    W_classifier = (base_classifier + spline_classifier).numpy()
                    # classif_val = ((weights_classif_layer1 @ W_classifier * self.enc.layers[layer][0].counts) / (n * accumulated_n['classifier']))
                    classif_val = ((W_classifier * self.enc.layers[layer][0].counts)).mean(0)
                    if np.isnan(classif_val.sum()):
                        classif_val = 0
                    n = len(self.enc.layers[layer][0].mask)
                    # n = len(self.enc.layers[layer][0].mask) * self.enc.layers[layer][0].n
                    # weights['encoder2'][layer] = self.enc.layers[layer][0].counts
                    self.enc.layers[layer][0].counts = \
                        classif_val + ((W_dec * self.enc.layers[layer][0].counts)).mean(0)
                    # self.enc.layers[layer][0].counts = \
                    #     classif_val + ((weights['decoder2']['layer1'] @ W_dec * self.enc.layers[layer][0].counts) / (accumulated_n['decoder2'] * self.enc.layers[layer][0].n))
                    total += self.enc.layers[layer][0].mask.sum().item()
                    self.enc.layers[layer][0].prune_neurons(weight_threshold)
                    layers['encoder2'][layer] = self.enc.layers[layer][0].mask.sum().item()

                accumulated_n['encoder2'] *= n
                prev_layer = layer
        
        return layers

    def prune(self, threshold=1e-4, mode="auto", active_neurons_id=None):
        '''
        pruning KAN on the node level. If a node has small incoming or outgoing connection, it will be pruned away.
        
        Args:
        -----
            threshold : float
                the threshold used to determine whether a node is small enough
            mode : str
                "auto" or "manual". If "auto", the thresold will be used to automatically prune away nodes. If "manual", active_neuron_id is needed to specify which neurons are kept (others are thrown away).
            active_neuron_id : list of id lists
                For example, [[0,1],[0,2,3]] means keeping the 0/1 neuron in the 1st hidden layer and the 0/2/3 neuron in the 2nd hidden layer. Pruning input and output neurons is not supported yet.
            
        Returns:
        --------
            model2 : KAN
                pruned model
         
        Example
        -------
        >>> # for more interactive examples, please see demos
        >>> from utils import create_dataset
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
        >>> model.prune()
        >>> model.plot(mask=True)
        '''
        mask = [torch.ones(self.width[0], )]
        active_neurons = [list(range(self.width[0]))]
        for i in range(len(self.acts_scale) - 1):
            if mode == "auto":
                in_important = torch.max(self.acts_scale[i], dim=1)[0] > threshold
                out_important = torch.max(self.acts_scale[i + 1], dim=0)[0] > threshold
                overall_important = in_important * out_important
            elif mode == "manual":
                overall_important = torch.zeros(self.width[i + 1], dtype=torch.bool)
                overall_important[active_neurons_id[i + 1]] = True
            mask.append(overall_important.float())
            active_neurons.append(torch.where(overall_important == True)[0])
        active_neurons.append(list(range(self.width[-1])))
        mask.append(torch.ones(self.width[-1], ))

        self.mask = mask  # this is neuron mask for the whole model

        # update act_fun[l].mask
        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width[l + 1]):
                if i not in active_neurons[l + 1]:
                    self.remove_node(l + 1, i)

        model2 = KANAutoencoder2(copy.deepcopy(self.width), self.grid, self.k, base_fun=self.base_fun, device=self.device)
        model2.load_state_dict(self.state_dict())
        for i in range(len(self.acts_scale)):
            if i < len(self.acts_scale) - 1:
                model2.biases[i].weight.data = model2.biases[i].weight.data[:, active_neurons[i + 1]]

            model2.act_fun[i] = model2.act_fun[i].get_subset(active_neurons[i], active_neurons[i + 1])
            model2.width[i] = len(active_neurons[i])
            model2.symbolic_fun[i] = self.symbolic_fun[i].get_subset(active_neurons[i], active_neurons[i + 1])

        return model2

    def remove_edge(self, l, i, j):
        '''
        remove activtion phi(l,i,j) (set its mask to zero)
        
        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
        
        Returns:
        --------
            None
        '''
        self.act_fun[l].mask[j * self.width[l] + i] = 0.

    def remove_node(self, l, i):
        '''
        remove neuron (l,i) (set the masks of all incoming and outgoing activation functions to zero)
        
        Args:
        -----
            l : int
                layer index
            i : int
                neuron index
        
        Returns:
        --------
            None
        '''
        self.act_fun[l - 1].mask[i * self.width[l - 1] + torch.arange(self.width[l - 1])] = 0.
        self.act_fun[l].mask[torch.arange(self.width[l + 1]) * self.width[l] + i] = 0.
        self.symbolic_fun[l - 1].mask[i, :] *= 0.
        self.symbolic_fun[l].mask[:, i] *= 0.

    def increase_pruning_threshold(self):
        '''
        increase the pruning threshold
        
        Args:
        -----
            threshold : float
                the amount of increase
        
        Returns:
        --------
            None
        '''
        if self.prune_threshold == 0:
            self.prune_threshold = 1e-8
        else:
            self.prune_threshold *= 10

    def count_n_neurons(self):
        '''
        count the number of neurons in the model
        
        Args:
        -----
            None
        
        Returns:
        --------
            n_neurons : int
                number of neurons
        '''
        total = 0
        layers = {}
        for idx, layer in enumerate(self.children()):
            try:
                layers[layer.name] = {}
            except Exception as e:
                continue
            for idx2, layer2 in enumerate(layer.children()):
                if isinstance(layer2, KANLinear):
                    n = layer2.mask.sum().item()
                    total += n
                    layers[layer2.name] = n
                # If subscriptable
                for layer3 in layer2.keys():
                    if isinstance(layer2[layer3][0], KANLinear):
                        if not (layer.name == 'decoder2' and layer3 == 'layer2') \
                            and not (layer.name == 'classifier' and layer3 == 'layer2') \
                                and not (layer.name == 'dann_discriminator' and layer3 == 'layer2'):
                            n = layer2[layer3][0].mask.sum().item()
                            total += n
                            layers[layer.name][layer3] = n

        layers['total'] = total
        return layers

class Encoder3(nn.Module):
    def __init__(self, in_shape, layer1, layer2, layer3, dropout, prune_threshold, name=None):
        super(Encoder3, self).__init__()
        self.name = name
        self.linear1 = nn.Sequential(
            KANLinear(in_shape, layer1, prune_threshold=prune_threshold, name='encoder1'),
            nn.BatchNorm1d(layer1),
            nn.Dropout(dropout),
            # nn.LeakyReLU(),
        )
        self.linear2 = nn.Sequential(
            KANLinear(layer1, layer2, prune_threshold=prune_threshold, name='encoder2'),
            nn.BatchNorm1d(layer2),
            nn.Dropout(dropout),
            # nn.LeakyReLU(),
        )
        self.linear3 = nn.Sequential(
            KANLinear(layer2, layer3, prune_threshold=prune_threshold, name='encoder3'),
            nn.BatchNorm1d(layer3),
            # nn.Dropout(dropout),
            # nn.Sigmoid(),
        )

        self.random_init()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class Decoder3(nn.Module):
    def __init__(self, in_shape, n_batches, layer1, layer2, layer3, dropout):
        super(Decoder3, self).__init__()
        self.linear1 = nn.Sequential(
            KANLinear(layer1 + n_batches, layer2, name='decoder1'),
            nn.BatchNorm1d(layer2),
            nn.Dropout(dropout),
            # nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            KANLinear(layer2 + n_batches, layer3, name='decoder2'),
            nn.BatchNorm1d(layer3),
            nn.Dropout(dropout),
            # nn.ReLU(),
        )

        self.linear3 = nn.Sequential(
            KANLinear(layer3, in_shape, name='decoder3'),
            # nn.BatchNorm1d(in_shape),
            # nn.Sigmoid(),
        )
        self.n_batches = n_batches
        self.random_init()

    def forward(self, x, batches=None):
        if batches is not None and self.n_batches > 0:
            x = torch.cat((x, batches), 1)
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        x3 = self.linear3(x2)
        x2 = torch.sigmoid(x3)
        return [x1, x2, x3]

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class SHAPKANAutoencoder3(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_emb, n_meta, mapper, variational, layer1, layer2, layer3, dropout, zinb=False,
                 conditional=False, add_noise=False, tied_weights=0, use_gnn=False, device='cuda'):
        super(SHAPKANAutoencoder3, self).__init__()
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
        self.enc = Encoder3(in_shape + n_meta, layer1, layer2, layer3, dropout)
        if conditional:
            self.dec = Decoder3(in_shape + n_meta, n_batches, layer3, layer2, layer1, dropout)
        else:
            self.dec = Decoder3(in_shape + n_meta, 0, layer3, layer2, layer1, dropout)
        self.mapper = Classifier(n_batches + 1, layer3)

        if variational:
            self.gaussian_sampling = GaussianSample(layer3, layer3, device)
        else:
            self.gaussian_sampling = None
        self.dann_discriminator = Classifier2(layer3, 64, n_batches)
        self.classifier = Classifier(layer3 + n_emb, nb_classes)
        self._dec_mean = nn.Sequential(KANLinear(layer2, in_shape + n_meta), nn.Sigmoid())
        self._dec_disp = nn.Sequential(KANLinear(layer2, in_shape + n_meta), DispAct())
        self._dec_pi = nn.Sequential(KANLinear(layer2, in_shape + n_meta), nn.Sigmoid())
        self.random_init(nn.init.xavier_uniform_)

    def forward(self, x, batches=None, sampling=False, beta=1.0):
        if self.n_emb > 0:
            meta_values = x[:, -2:]
        if self.n_meta == 0:
            x = x[:, :-2]
        # rec = {}
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        # if self.use_gnn:
        #     x = self.gnn1(x)
        try:
            enc = self.enc(x).squeeze()
        except:
            pass
        if self.gaussian_sampling is not None:
            if sampling:
                enc, mu, log_var = self.gaussian_sampling(enc, train=True, beta=beta)
                # Kullback-Leibler Divergence
                # kl = self._kld(enc, (mu, log_var))
                mean_sq = mu * mu
                std = log_var.exp().sqrt()
                stddev_sq = std * std
                # kl = 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
                # https://arxiv.org/pdf/1312.6114.pdf equation 10, first part and
                # https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
                kl = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), axis=1)
            else:
                enc, _, _ = self.gaussian_sampling(enc, train=False)
                kl = torch.Tensor([0])
        else:
            kl = torch.Tensor([0])
        if self.use_mapper:
            bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            # needs the batch
            enc += self.mapper(bs).squeeze()
        if not self.tied_weights:
            try:
                bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            except:
                bs = to_categorical(batches.long(), self.n_batches + 1).to(self.device).float()

            rec = {"mean": self.dec(enc, bs)}
        elif not self.zinb:
            rec = [F.relu(F.linear(enc, self.enc.linear2[0].weight.t()))]
            rec += [F.relu(F.linear(rec[0], self.enc.linear1[0].weight.t()))]
            rec = {"mean": rec}  # TODO rec does not need to be a dict no more
        elif self.zinb:
            rec = {"mean": [F.relu(F.linear(enc, self.enc.linear3[0].weight.t()))]}

        if self.n_emb > 0:
            out = self.classifier(torch.cat((enc, meta_values), 1))
        else:
            out = self.classifier(enc)

        return out

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)
            if isinstance(m, KANLinear):
                m.reset_parameters()

    def predict_proba(self, x):
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, x):
        return self.classifier(x).argmax(1).detach().cpu().numpy()

    def _kld(self, z, q_param, h_last=None, p_param=None):
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
    def zinb_loss(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        # scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
                    x * (torch.log(disp + eps) - torch.log(mean + eps)))
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


class KANAutoencoder3(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_meta, n_emb, mapper, variational, layer1, layer2, layer3, dropout, zinb=False,
                 conditional=False, add_noise=False, tied_weights=0, use_gnn=False, device='cuda'):
        super(KANAutoencoder3, self).__init__()
        self.add_noise = add_noise
        self.device = device
        self.use_gnn = use_gnn
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'
        # self.gnn1 = GCNConv(in_shape, in_shape)
        self.enc = Encoder3(in_shape + n_meta, layer1, layer2, layer3, dropout)
        if conditional:
            self.dec = Decoder3(in_shape + n_meta, n_batches, layer3, layer2, layer1, dropout)
        else:
            self.dec = Decoder3(in_shape + n_meta, 0, layer3, layer2, layer1, dropout)
        self.mapper = Classifier(n_batches + 1, layer3)

        if variational:
            self.gaussian_sampling = GaussianSample(layer3, layer3, device)
        else:
            self.gaussian_sampling = None
        self.dann_discriminator = Classifier2(layer3, 64, n_batches)
        self.classifier = Classifier(layer3 + n_emb, nb_classes)
        self._dec_mean = nn.Sequential(KANLinear(layer2, in_shape + n_meta), MeanAct())
        self._dec_disp = nn.Sequential(KANLinear(layer2, in_shape + n_meta), DispAct())
        self._dec_pi = nn.Sequential(KANLinear(layer2, in_shape + n_meta), nn.Sigmoid())
        # self.random_init(nn.init.kaiming_uniform_)

    def forward(self, x, to_rec, batches=None, sampling=False, beta=1.0):
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
        if self.use_mapper:
            bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            enc += self.mapper(bs)
        if not self.tied_weights:
            try:
                bs = to_categorical(batches, self.n_batches + 1).to(self.device).float()
            except:
                bs = to_categorical(batches.long(), self.n_batches + 1).to(self.device).float()

            rec = {"mean": self.dec(enc, bs)}
        elif not self.zinb:
            rec = [F.relu(F.linear(enc, self.enc.linear2[0].weight.t()))]
            rec += [F.relu(F.linear(rec[0], self.enc.linear1[0].weight.t()))]
            rec = {"mean": rec}  # TODO rec does not need to be a dict no more
        elif self.zinb:
            rec = {"mean": [F.relu(F.linear(enc, self.enc.linear3[0].weight.t()))]}

        if self.zinb:
            _mean = self._dec_mean(rec['mean'][0])
            _disp = self._dec_disp(rec['mean'][0])
            _pi = self._dec_pi(rec['mean'][0])
            zinb_loss = self.zinb_loss(to_rec, _mean, _disp, _pi, scale_factor=1)
            # if not sampling:
            rec = {'mean': _mean, 'rec': None}
        else:
            zinb_loss = torch.Tensor([0])

        # reverse = ReverseLayerF.apply(enc, alpha)
        # b_preds = self.classifier(reverse)
        # rec[-1] = torch.clamp(rec[-1], min=0, max=1)
        return [enc, rec, zinb_loss, kl]

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x):
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, x):
        return self.classifier(x).argmax(1).detach().cpu().numpy()

    def _kld(self, z, q_param, h_last=None, p_param=None):
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
    def zinb_loss(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
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


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
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
                                - F.softplus( - pi)
    case_non_zero = - pi - F.softplus(- pi) \
                                + theta * torch.log(theta + eps) - theta * torch.log(theta + mu + eps) \
                                + x * torch.log(mu + eps) - x * torch.log(theta + mu + eps) \
                                + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)

    # mask = tf.cast(torch.less(x, eps), torch.float32)
    mask = torch.less(x, eps).float()
    res = torch.multiply(mask, case_zero) + torch.multiply(1 - mask, case_non_zero)
    res = torch.nan_to_num(res, 0)
    return torch.sum(res, axis=-1)
