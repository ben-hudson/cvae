import cvxpy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvxpylayers.torch import CvxpyLayer
from distributions import ReparametrizedBernoulli
from torch.distributions import Normal, Gamma
from torchvision.ops import MLP
from typing import List

class Encoder(nn.Module):
    def __init__(self, input_dim: int, cost_dim: int, constr_dim: int, max_hidden_dim: int):
        super().__init__()

        layers = []
        max_out_dim = cost_dim*constr_dim # W input is this dimension
        hidden_dims = _generate_hidden_dims(input_dim, max_out_dim, max_hidden_dim)
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # W is a d_y*d_h binary matrix
        # we model W_ij with a bernoulli distribution
        self.W_logits = nn.Sequential(
            nn.Linear(hidden_dims[-1], cost_dim*constr_dim),
            nn.Unflatten(-1, (constr_dim, cost_dim))
        )

        self.h_shape = nn.Linear(hidden_dims[-1], constr_dim)
        self.h_rate = nn.Linear(hidden_dims[-1], constr_dim)

        self.q_shape = nn.Linear(hidden_dims[-1], cost_dim)
        self.q_rate = nn.Linear(hidden_dims[-1], cost_dim)

    def forward(self, x: torch.Tensor):
        hidden = self.mlp(x)

        W = ReparametrizedBernoulli(logits=self.W_logits(hidden))
        # W = F.gumbel_softmax(self.W_logits(hidden), tau=1, hard=True)
        # h = Normal(F.softplus(self.h_shape(hidden)), torch.exp(self.h_rate(hidden) + eps))
        # q = Normal(F.softplus(self.q_shape(hidden)), torch.exp(self.q_rate(hidden) + eps))
        h = Gamma(F.softplus(self.h_shape(hidden)), F.softplus(self.h_rate(hidden)))
        q = Gamma(F.softplus(self.q_shape(hidden)), F.softplus(self.q_rate(hidden)))

        return W, h, q

class CvxSolver(nn.Module):
    def __init__(self, cost_dim, constr_dim):
        super().__init__()
        # TODO: are all of these nonneg?
        x = cp.Variable(cost_dim, nonneg=True)
        c = cp.Parameter(cost_dim, nonneg=True)
        A = cp.Parameter((constr_dim, cost_dim), nonneg=True)
        b = cp.Parameter(constr_dim, nonneg=True)
        problem = cp.Problem(cp.Minimize(c @ x), [A @ x <= b])
        assert problem.is_dpp(), f'Problem is not DPP'

        self.solver = CvxpyLayer(problem, parameters=[A, b, c], variables=[x])
        # learnable output transformation so we don't have to worry about rotation of learned problem
        self.output_trans = nn.Linear(cost_dim, cost_dim)

    def forward(self, A, b, c):
        solution, = self.solver(A, b, c)
        return self.output_trans(solution)
        # return solution

def _generate_hidden_dims(in_dim: int, out_dim: int, max_hidden_dim: int):
    min_hidden_dim = min(in_dim, out_dim)
    hidden_dims = [max_hidden_dim]
    while hidden_dims[-1] // 2 > min_hidden_dim:
        hidden_dims.append(hidden_dims[-1] // 2)

    if in_dim < out_dim:
        hidden_dims = list(reversed(hidden_dims))

    return hidden_dims


class SolverVAE(nn.Module):
    def __init__(self, y_dim: int, x_dim: int, constr_dim: int, max_hidden_dim: int) -> None:
        super(SolverVAE, self).__init__()

        self.prior_net = Encoder(x_dim, y_dim, constr_dim, max_hidden_dim)
        self.recognition_net = Encoder(y_dim + x_dim, y_dim, constr_dim, max_hidden_dim)
        self.generation_net = CvxSolver(y_dim, constr_dim)

    def sample(self, x: torch.Tensor):
        # first, get the conditioned latent distribution p(z|x)
        W, h, q = self.prior_net(x)
        # take some samples
        # these must be greater or equal to 0 for the problem definition to be valid!
        W_sample = W.rsample(tau=1)
        h_sample = h.rsample()
        q_sample = q.rsample().clamp(1e-3) # must be greater than 0 or problem is unbounded
        # and try to reconstruct the observation p(y|x,z)
        y_sample = self.generation_net(W_sample, h_sample, q_sample)
        return (W, h, q), y_sample

    def forward(self, y: torch.Tensor, x: torch.Tensor):
        # get prior from condition p(z|x), and try to reconstruct observation
        priors, y_hat = self.sample(x)
        # now, we need to get the posterior q(z|x,y)
        obs = torch.cat([y, x], dim=1)
        posteriors = self.recognition_net(obs)

        return priors, posteriors, y_hat
