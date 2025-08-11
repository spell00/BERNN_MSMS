import math
import torch
import torch.nn.functional as F


def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.

    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy


def log_bernoulli(x, p, eps=0.0):
    """
    Compute log pdf of a Bernoulli distribution with success probability p, at values x.
        .. math:: \log p(x; p) = \log \mathcal{B}(x; p)
    Parameters
    ----------
    x : torch tensor
        Values at which to evaluate pdf.
    p : torch tensor
        Success probability :math:`p(x=1)`, which is also the mean of the Bernoulli distribution.
    eps : float
        Small number used to avoid NaNs by clipping p in range [eps;1-eps].
    Returns
    -------
    torch tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    """
    p = torch.clip(p, eps, 1.0 - eps)
    return -F.binary_cross_entropy(p, x)


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.

    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * torch.log(2 * torch.tensor(math.pi, requires_grad=True)) - log_var / 2 - (x - mu) ** 2 / (
        2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def log_normal_diag(x, mean, log_var, average=False, dim=1):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_normal_standard(x, average=False, dim=1):
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


