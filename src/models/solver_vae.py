import cvxpy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvxpylayers.torch import CvxpyLayer
from distributions import ReparametrizedBernoulli
from ilop.linprog_solver import linprog_batch_std
from torch.distributions import Normal, Gamma, kl_divergence

class Encoder(nn.Module):
    def __init__(self, input_dim, cost_dim, constr_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)

        # W is a d_y*d_h binary matrix
        # we model W_ij with a bernoulli distribution
        self.W_logit = nn.Sequential(
            nn.Linear(hidden_dim_2, cost_dim*constr_dim),
            nn.Unflatten(-1, (constr_dim, cost_dim))
        )

        self.h_mean = nn.Linear(hidden_dim_2, constr_dim)
        self.h_logvar = nn.Linear(hidden_dim_2, constr_dim)

        self.q_mean = nn.Linear(hidden_dim_2, cost_dim)
        self.q_logvar = nn.Linear(hidden_dim_2, cost_dim)

    def forward(self, x: torch.Tensor, eps: float = 1e-8):
        hidden = F.silu(self.fc1(x))
        hidden = F.silu(self.fc2(hidden))

        W = ReparametrizedBernoulli(logits=self.W_logit(hidden))
        h = Normal(F.softplus(self.h_mean(hidden)), torch.exp(self.h_logvar(hidden) + eps))
        q = Normal(F.softplus(self.q_mean(hidden)), torch.exp(self.q_logvar(hidden) + eps))

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
        # learnable output transformation so we don't have to worry about permutation and scaling of learned problem
        self.output_trans = nn.Linear(cost_dim, cost_dim)

    def forward(self, A, b, c):
        solution, = self.solver(A, b, c)
        return self.output_trans(solution)


class SolverVAE(nn.Module):
    def __init__(self, y_dim: int, x_dim: int, constr_dim: int, hidden_dim: int) -> None:
        super(SolverVAE, self).__init__()

        self._device_param = nn.Parameter(torch.empty(0)) # https://stackoverflow.com/a/63477353/

        self.prior_net = Encoder(x_dim, y_dim, constr_dim, hidden_dim, hidden_dim // 2)
        self.recognition_net = Encoder(y_dim + x_dim, y_dim, constr_dim, hidden_dim, hidden_dim // 2)
        self.generation_net = CvxSolver(y_dim, constr_dim)

    def sample(self, condition: torch.Tensor):
        # first, get the conditioned latent distribution p(z|x)
        W, h, q = self.prior_net(condition)
        # take some samples
        # these must be greater than 0 for the problem definition to be valid!
        W_sample = F.softplus(W.rsample())
        h_sample = F.softplus(h.rsample())
        q_sample = F.softplus(q.rsample())
        # and try to reconstruct the observation p(y|x,z)
        y_sample = self.generation_net(W_sample, h_sample, q_sample)
        return (W, h, q), y_sample

    def forward(self, obs: torch.Tensor, condition: torch.Tensor):
        # get prior from condition p(z|x), and try to reconstruct observation
        priors, obs_hat = self.sample(condition)
        # now, we need to get the posterior q(z|x,y)
        x = torch.cat([obs, condition], dim=1)
        posteriors = self.recognition_net(  x)

        # want reconstructed observation to be close to the observation
        # is actually doesn't make sense to use MSE because of the permutation (see output_trans in generation_net)
        mse = F.mse_loss(obs_hat, obs).sum()
        # want posterior to be close to the prior
        kld = torch.zeros(1, device=self._device_param.device)
        for p, q in zip(priors, posteriors):
            kld += kl_divergence(p, q).sum()

        return priors, posteriors, obs_hat, mse, kld
