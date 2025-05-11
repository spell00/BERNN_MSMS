import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from ..ekan.src.efficient_kan.kan import KANLinear


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparametrize(self, mu, log_var, train, device, beta=0.):
        if train:
            epsilon = Variable(beta * torch.randn(mu.size()), requires_grad=False)
        else:
            epsilon = Variable(torch.zeros_like(mu), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.to(device)

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        z = std * epsilon + mu
        # z = mu.addcmul(std, epsilon)

        return z


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features, device):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)
        self.device = device

    def forward(self, x, train=False, beta=1.0):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparametrize(mu, log_var, train, self.device, beta), mu, log_var

class GaussianSampleKAN(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features, device):
        super(GaussianSampleKAN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = KANLinear(in_features, out_features)
        self.log_var = KANLinear(in_features, out_features)
        self.device = device

    def forward(self, x, train=False, beta=1.0):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparametrize(mu, log_var, train, self.device, beta), mu, log_var

