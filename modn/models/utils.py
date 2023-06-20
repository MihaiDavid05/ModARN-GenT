import torch


def neg_log_likelihood_1d(y, mu, log_sigma):
    """Compute distribution loss"""
    return torch.sum(log_sigma + (y - mu) ** 2 / (2 * torch.exp(log_sigma) ** 2)) / y.shape[0]


def rmse(yhat, y):
    """Compute simple rmse loss"""
    return torch.sqrt(torch.mean((yhat - y) ** 2))
