import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function
from .utils.stochastic import GaussianSample
from .utils.distributions import log_normal_standard, log_normal_diag, log_gaussian
from .utils.utils import to_categorical
import pandas as pd
from ...models.pytorch.ekan.src.efficient_kan.kan import KANLinear
import copy
import numpy as np


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
            return self.layers["layer2"](self.layers["layer1"](x)).detach().float().cpu().numpy()
        return self.layers["layer1"](x).detach().float().cpu().numpy()

    def predict(self, x):
        if self.n_layers == 2:
            return self.layers["layer2"](self.layers["layer1"](x)).argmax(1).detach().float().cpu().numpy()
        return self.layers["layer1"](x).argmax(1).detach().float().cpu().numpy()


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
        return self.layers["layer2"](self.layers["layer1"](x)).detach().float().cpu().numpy()

    def predict(self, x):
        return self.layers["layer2"](self.layers["layer1"](x)).argmax(1).detach().float().cpu().numpy()


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
        return self.linear2(x).detach().float().cpu().numpy()

    def predict(self, x):
        return self.linear2(x).argmax(1).detach().float().cpu().numpy()


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


class SHAPKANAutoEncoder2(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_emb, n_meta, mapper, variational,
                 layer1, layer2, dropout, n_layers, zinb=False, conditional=False,
                 add_noise=False, tied_weights=0, device='cuda'):
        super(SHAPKANAutoEncoder2, self).__init__()
        self.n_emb = n_emb
        self.add_noise = add_noise
        self.n_meta = n_meta
        self.device = device
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'

        self.enc = Encoder2(in_shape + n_meta, layer1, layer2, dropout, prune_threshold=0, name='encoder2')
        if conditional:
            self.dec = Decoder2(in_shape + n_meta, n_batches, layer2, layer1, dropout, prune_threshold=0,
                                name='decoder2')
        else:
            self.dec = Decoder2(in_shape + n_meta, 0, layer2, layer1, dropout, prune_threshold=0, name='decoder2')

        self.mapper = Classifier(n_batches + 1, layer2, n_layers=2, prune_threshold=0, name='mapper')
        if variational:
            self.gaussian_sampling = GaussianSample(layer2, layer2, device)
        else:
            self.gaussian_sampling = None

        self.dann_discriminator = Classifier2(layer2, 64, n_batches, prune_threshold=0, name='dann_discriminator')
        self.classifier = Classifier(layer2 + n_emb, nb_classes, n_layers=n_layers, prune_threshold=0,
                                     name='classifier')

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
        return self.classifier(x).detach().float().cpu().numpy()

    def predict(self, x):
        return self.classifier(x).argmax(1).detach().float().cpu().numpy()

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


class KANAutoEncoder2(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_meta, n_emb, mapper,
                 variational, layer1, layer2, dropout, n_layers, prune_threshold, zinb=False,
                 conditional=False, add_noise=False, tied_weights=0,
                 update_grid=False, device='cuda'):
        super(KANAutoEncoder2, self).__init__()
        self.prune_threshold = prune_threshold
        self.add_noise = add_noise
        self.device = device
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'
        self.n_meta = n_meta
        self.n_emb = n_emb
        # self.gnn1 = GCNConv(in_shape, in_shape)
        self.enc = Encoder2(in_shape + n_meta, layer1, layer2, dropout, prune_threshold=prune_threshold, update_grid=0,
                            name='encoder2')  # TODO update_grid causes an error, but no idea why
        if conditional:
            self.dec = Decoder2(in_shape + n_meta, n_batches, layer2, layer1, dropout, prune_threshold=prune_threshold,
                                update_grid=update_grid, name='decoder2')
        else:
            self.dec = Decoder2(in_shape + n_meta, 0, layer2, layer1, dropout, prune_threshold=prune_threshold,
                                update_grid=update_grid, name='decoder2')
        self.mapper = Classifier(n_batches + 1, layer2, update_grid=update_grid, prune_threshold=prune_threshold,
                                 name='mapper')

        if variational:
            self.gaussian_sampling = GaussianSample(layer2, layer2, device)
        else:
            self.gaussian_sampling = None
        # TODO dann_disc needs to be at prune_threshold=0, otherwise it will prune away the whole model
        # TODO update_grid causes an error, but no idea why
        self.dann_discriminator = Classifier2(layer2, 64, n_batches, update_grid=0,
                                              name='dann_discriminator', prune_threshold=0)
        self.classifier = Classifier(layer2 + n_emb, nb_classes, n_layers=n_layers,
                                     update_grid=update_grid, prune_threshold=prune_threshold, name='classifier')
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
        except Exception as e:
            print(f'{e}')
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
            except Exception as e:
                print(f'{e}')
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
        return self.classifier(x).detach().float().cpu().numpy()

    def predict(self, x):
        return self.classifier(x).argmax(1).detach().float().cpu().numpy()

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


class KANAutoEncoder3(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_meta, n_emb, mapper, variational, layers: dict,
                 dropout, n_layers, prune_threshold, zinb=False, conditional=True, add_noise=False,
                 tied_weights=0, update_grid=False, device='cuda', is_sigmoid=False):
        super(KANAutoEncoder3, self).__init__()
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
        self.update_grid = update_grid
        self.is_sigmoid = is_sigmoid  # Store the parameter for potential use

        # Encoder and Decoder using KAN layers
        self.enc = Encoder3(in_shape + n_meta, layers, dropout, device)
        if conditional:
            self.dec = Decoder3(in_shape + n_meta, n_batches, layers, dropout, device)
        else:
            self.dec = Decoder3(in_shape + n_meta, 0, layers, dropout, device)

        # Mapper for batch effect removal
        self.mapper = Classifier(n_batches + 1, layers[list(layers.keys())[-1]], device=device)

        # Variational sampling
        if variational:
            self.gaussian_sampling = GaussianSample(layers[list(layers.keys())[-1]], layers[list(layers.keys())[-1]], device)
        else:
            self.gaussian_sampling = None

        # Discriminator and classifier
        self.dann_discriminator = Classifier2(layers[list(layers.keys())[-1]], 128, n_batches, device=device)
        self.classifier = Classifier2(layers[list(layers.keys())[-1]] + n_emb, 128, nb_classes, device=device)

        # ZINB layers if needed
        if zinb:
            self._dec_mean = nn.Sequential(
                KANLinear(layers[list(layers.keys())[-2]], in_shape + n_meta, device=device), MeanAct())
            self._dec_disp = nn.Sequential(
                KANLinear(layers[list(layers.keys())[-2]], in_shape + n_meta, device=device), DispAct())
            self._dec_pi = nn.Sequential(
                KANLinear(layers[list(layers.keys())[-2]], in_shape + n_meta, device=device), nn.Sigmoid())

        self.random_init(nn.init.kaiming_uniform_)

    def forward(self, x, to_rec, batches=None, sampling=False, beta=1.0, mapping=True):
        rec = {}
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        enc = self.enc(x)
        if self.gaussian_sampling is not None:
            if sampling:
                enc, mu, log_var = self.gaussian_sampling(enc, train=True, beta=beta)
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
                print(f'{e}')
                bs = to_categorical(batches.long(), self.n_batches + 1).to(self.device).float()
            rec = {"mean": self.dec(enc_be, bs)}
        elif not self.zinb:
            rec = [F.relu(F.linear(enc, self.enc.linear2[0].weight.t()))]
            rec += [F.relu(F.linear(rec[0], self.enc.linear1[0].weight.t()))]
            rec = {"mean": rec}
        elif self.zinb:
            rec = {"mean": [F.relu(F.linear(enc, self.enc.linear2[0].weight.t()))]}

        if self.zinb:
            _mean = self._dec_mean(rec['mean'][0])
            _disp = self._dec_disp(rec['mean'][0])
            _pi = self._dec_pi(rec['mean'][0])
            zinb_loss = self.zinb_loss(to_rec, _mean, _disp, _pi)
            rec = {'mean': _mean, 'rec': to_rec}
        else:
            zinb_loss = torch.Tensor([0])

        return [enc, rec, zinb_loss, kl]

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

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, KANLinear) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def predict_proba(self, x):
        enc = self.enc(x)
        return self.classifier(enc).detach().float().cpu().numpy()

    def predict(self, x):
        enc = self.enc(x)
        return self.classifier(enc).argmax(1).detach().float().cpu().numpy()

    def _kld(self, z, q_param, h_last=None, p_param=None):
        if len(z.shape) == 1:
            z = z.view(1, -1)
        (mu, log_var) = q_param
        qz = log_normal_diag(z, mu, log_var)
        if p_param is None:
            pz = log_normal_standard(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)
        kl = -(pz - qz)
        return kl

    def zinb_loss(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
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


class Encoder3(nn.Module):
    def __init__(self, in_shape, layers: dict, dropout, device='cuda'):
        super().__init__()
        self.device = device
        self.layer_names = list(layers.keys())
        self.n_layers = len(self.layer_names)
        self.kan_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        prev_shape = in_shape
        for i, name in enumerate(self.layer_names):
            out_shape = layers[name]
            self.kan_layers.append(KANLinear(prev_shape, out_shape, name=f'encoder3_{name}'))
            self.dropouts.append(nn.Dropout(dropout))
            prev_shape = out_shape
        self.random_init()

    def forward(self, x):
        for kan, drop in zip(self.kan_layers, self.dropouts):
            x = kan(x)
            x = drop(x)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, KANLinear):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class Decoder3(nn.Module):
    def __init__(self, in_shape, n_batches, layers: dict, dropout, device='cuda'):
        super().__init__()
        self.device = device
        self.n_batches = n_batches
        self.layer_names = list(layers.keys())
        self.n_layers = len(self.layer_names)
        self.kan_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        prev_shape = layers[self.layer_names[-1]] + n_batches if n_batches > 0 else layers[self.layer_names[-1]]
        for name in reversed(self.layer_names):
            out_shape = layers[name]
            self.kan_layers.append(KANLinear(prev_shape, out_shape, name=f'decoder3_{name}'))
            self.dropouts.append(nn.Dropout(dropout))
            prev_shape = out_shape
        self.kan_layers.append(KANLinear(prev_shape, in_shape, name='decoder3_out'))
        self.random_init()

    def forward(self, x, batches=None):
        if batches is not None and self.n_batches > 0:
            x = torch.cat((x, batches), dim=1)
        for kan, drop in zip(self.kan_layers, self.dropouts):
            x = kan(x)
            x = drop(x)
        x = self.kan_layers[-1](x)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, KANLinear):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
