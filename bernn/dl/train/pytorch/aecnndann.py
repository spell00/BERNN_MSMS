import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function
from bernn.dl.models.pytorch.utils.stochastic import GaussianSample
from bernn.dl.models.pytorch.utils.distributions import log_normal_standard, log_normal_diag, log_gaussian
from bernn.dl.models.pytorch.utils.utils import to_categorical
import pandas as pd

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
    def __init__(self, in_shape=64, out_shape=9, n_layers=2):
        super(Classifier, self).__init__()
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

    def forward(self, x):
        x = self.linear2(x)
        if self.n_layers == 2:
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


class Classifier2(nn.Module):
    def __init__(self, in_shape=64, hidden=64, out_shape=9):
        super(Classifier2, self).__init__()
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

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
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


class Classifier3(nn.Module):
    def __init__(self, in_shape=64, hidden=64, out_shape=9):
        super(Classifier3, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_shape, hidden),
            nn.BatchNorm1d(hidden),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(hidden, out_shape),
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
    def __init__(self, in_shape, layer1, dropout):
        super(Encoder, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(in_shape, layer1),
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
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 0.975)
            #     nn.init.constant_(m.bias, 0.125)


class Encoder2(nn.Module):
    def __init__(self, in_shape, layer1, layer2, dropout):
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

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
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


class ConvEncoder(nn.Module):
    def __init__(self, ni, no, dropout):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ni, 32, kernel_size=(7, 7), stride=(3, 3), padding=0),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=0),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=0),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=0),
            nn.BatchNorm2d(256),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=0),
            nn.BatchNorm2d(512),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(512),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=0),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=2, stride=1, padding=1)
        self.random_init()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = self.conv6(x)
        x = self.maxpool(x)
        x = self.conv7(x)
        x = self.maxpool(x)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
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


class ConvDecoder(nn.Module):
    def __init__(self, in_shape, layer1, layer2, dropout):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape, layer1),
            nn.BatchNorm2d(layer1),
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

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
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
    def __init__(self, in_shape, n_batches, layer1, layer2, dropout):
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

    def forward(self, x, batches=None):
        if batches is not None and self.n_batches > 0:
            x = torch.cat((x, batches), 1)
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        # x2 = torch.sigmoid(x2)
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
    def __init__(self, in_shape, n_batches, layer1, dropout):
        super(Decoder, self).__init__()
        self.linear2 = nn.Sequential(
            nn.Linear(layer1 + n_batches, in_shape),
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


class SHAPAutoEncoder2(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_emb, n_meta, mapper, variational, layer1, layer2, dropout,
                 n_layers, zinb=False, conditional=False, add_noise=False, tied_weights=0, use_gnn=False, device='cuda'):
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

    def forward(self, x, batches=None, sampling=False, beta=1.0):
        if type(x) == pd.core.frame.DataFrame:
            x = torch.Tensor(x.values).to(self.device)
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

class AutoEncoderCNN(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_meta, n_emb, mapper, variational, layer1, dropout, n_layers, zinb=False,
                 conditional=False, add_noise=False, tied_weights=0, use_gnn=False, device='cuda'):
        super(AutoEncoderCNN, self).__init__()
        self.add_noise = add_noise
        self.device = device
        self.use_gnn = use_gnn
        self.use_mapper = mapper
        self.n_batches = n_batches
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'
        # self.gnn1 = GCNConv(in_shape, in_shape)
        self.convEncoder = ConvEncoder(in_shape + n_meta, layer1, dropout)
        self.enc = Encoder(in_shape, layer1, dropout)
        if conditional:
            self.dec = Decoder(in_shape, n_batches, layer1, dropout)
            self.convDecoder = ConvDecoder(in_shape + n_meta, n_batches, layer1, dropout)
        else:
            self.dec = Decoder(in_shape, 0, layer1, dropout)
            self.convDecoder = ConvDecoder(in_shape + n_meta, 0, layer1, dropout)
        
        self.mapper = Classifier(n_batches + 1)

        if variational:
            self.gaussian_sampling = GaussianSample(layer1, layer1, device)
        else:
            self.gaussian_sampling = None
        self.dann_discriminator = Classifier2(layer1, 64, n_batches)
        self.classifier = Classifier(layer1 + n_emb, nb_classes, n_layers=n_layers)
        self._dec_mean = nn.Sequential(nn.Linear(layer1, in_shape + n_meta), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(layer1, in_shape + n_meta), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(layer1, in_shape + n_meta), nn.Sigmoid())
        self.random_init(nn.init.kaiming_uniform_)

    def forward(self, x, to_rec, batches=None, sampling=False, beta=1.0, mapping=True):
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



class Encoder3(nn.Module):
    def __init__(self, in_shape, layer1, layer2, layer3, dropout):
        super(Encoder3, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_shape, layer1),
            nn.BatchNorm1d(layer1),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(layer1, layer2),
            nn.BatchNorm1d(layer2),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(layer2, layer3),
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
            nn.Linear(layer1 + n_batches, layer2),
            nn.BatchNorm1d(layer2),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(layer2 + n_batches, layer3),
            nn.BatchNorm1d(layer3),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        self.linear3 = nn.Sequential(
            nn.Linear(layer3, in_shape),
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


class SHAPAutoEncoder3(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_emb, n_meta, mapper, variational, layer1, layer2, layer3, dropout, zinb=False,
                 conditional=False, add_noise=False, tied_weights=0, use_gnn=False, device='cuda'):
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
        self._dec_mean = nn.Sequential(nn.Linear(layer2, in_shape + n_meta), nn.Sigmoid())
        self._dec_disp = nn.Sequential(nn.Linear(layer2, in_shape + n_meta), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(layer2, in_shape + n_meta), nn.Sigmoid())
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


class AutoEncoder3(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, n_meta, n_emb, mapper, variational, layer1, layer2, layer3, dropout, zinb=False,
                 conditional=False, add_noise=False, tied_weights=0, use_gnn=False, device='cuda'):
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
        self._dec_mean = nn.Sequential(nn.Linear(layer2, in_shape + n_meta), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(layer2, in_shape + n_meta), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(layer2, in_shape + n_meta), nn.Sigmoid())
        self.random_init(nn.init.kaiming_uniform_)

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

