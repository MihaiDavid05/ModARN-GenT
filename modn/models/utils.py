import torch


def neg_log_likelihood_1d(y, mu, log_sigma):
    return torch.sum(log_sigma + (y - mu) ** 2 / (2 * torch.exp(log_sigma) ** 2)) / y.shape[0]
