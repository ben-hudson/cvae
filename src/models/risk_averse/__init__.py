import torch

from .shortestpath import *


def get_VaR(alpha, mu, sigma):
    quantile = torch.distributions.Normal(0, 1).icdf(torch.tensor(alpha))
    return mu + quantile * sigma


def get_CVaR(alpha, mu, sigma, tail="right"):
    # right tail CVaR_a = left tail CVaR_{1-a}
    # we flip alpha accordingly and calculate the right tail CVaR
    if tail == "right":
        beta = torch.tensor(alpha)
    elif tail == "left":
        beta = torch.tensor(1 - alpha)
    else:
        raise ValueError(f"tail must be 'left' or 'right'")
    standard_normal = torch.distributions.Normal(0, 1)
    quantile = standard_normal.icdf(beta)
    prob_quantile = standard_normal.log_prob(quantile).exp()

    # right tail CVaR_b = E[X | X >= VaR_b]
    # for a normal distribution (https://en.wikipedia.org/wiki/Expected_shortfall#Normal_distribution)
    return mu + sigma * prob_quantile / (1 - beta)


def get_obj_dist(sol, costs, costs_std):
    costs_cov = torch.diag(costs_std**2)
    obj = torch.dot(costs, sol)
    obj_var = sol.T @ costs_cov @ sol
    return obj, obj_var.sqrt()
