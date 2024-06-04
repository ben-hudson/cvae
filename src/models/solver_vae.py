import cvxpy as cp
import torch
import torch.nn as nn

from cvxpylayers.torch import CvxpyLayer
from distributions import ReparametrizedBernoulli
from torch.distributions import Gamma
from typing import List

from ilop.linprog_solver import linprog_batch_std


def _generate_hidden_dims(in_dim: int, out_dim: int, max_hidden_dim: int):
    min_hidden_dim = min(in_dim, out_dim)
    hidden_dims = [max_hidden_dim]
    while hidden_dims[-1] // 2 > min_hidden_dim:
        hidden_dims.append(hidden_dims[-1] // 2)

    if in_dim < out_dim:
        hidden_dims = list(reversed(hidden_dims))

    return hidden_dims


class Encoder(nn.Module):
    def __init__(self, input_dim: int, decision_dim: int, constr_dim: int, hidden_dims: List[int]):
        super().__init__()

        constr_cols = decision_dim - constr_dim
        assert constr_cols > 0, 'W ends with an nxn identity matrix, so m > n to have anything to learn.'

        # a little MLP
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        hidden_dim = input_dim # what would have been the next input dim from the MLP above

        # W is a binary matrix
        # we model W_ij as a bernoulli distribution and sample using gumbel-softmax trick
        self.W_logits = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Unflatten(-1, (constr_dim, constr_cols))
        )

        self.h = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.h_shape = nn.Sequential(nn.Linear(hidden_dim, constr_dim), nn.Softplus())
        self.h_rate = nn.Sequential(nn.Linear(hidden_dim, constr_dim), nn.Softplus())

        self.q = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.q_shape = nn.Sequential(nn.Linear(hidden_dim, decision_dim), nn.Softplus())
        self.q_rate = nn.Sequential(nn.Linear(hidden_dim, decision_dim), nn.Softplus())

    def forward(self, x: torch.Tensor):
        hidden = self.mlp(x)

        W_logits = self.W_logits(hidden)
        W = ReparametrizedBernoulli(logits=W_logits)

        h_hidden = self.h(hidden)
        h_shape, h_rate = self.h_shape(h_hidden), self.h_rate(h_hidden)
        h = Gamma(h_shape, h_rate)

        q_hidden = self.q(hidden)
        q_shape, q_rate = self.q_shape(q_hidden), self.q_rate(q_hidden)
        q = Gamma(q_shape, q_rate)

        return W, h, q


class CvxSolver(nn.Module):
    def __init__(self, decision_dim, constr_dim):
        super().__init__()
        x = cp.Variable(decision_dim)
        c = cp.Parameter(decision_dim)
        A = cp.Parameter((constr_dim, decision_dim))
        b = cp.Parameter(constr_dim)
        problem = cp.Problem(cp.Minimize(c @ x), [A @ x == b, x >= 0])
        assert problem.is_dpp(), f'Problem is not DPP'

        self.solver = CvxpyLayer(problem, parameters=[A, b, c], variables=[x])

    def forward(self, A, b, c):
        solution, = self.solver(A, b, c)
        return solution


class LinSolver(nn.Module):
    def __init__(self, decision_dim, constr_dim):
        super().__init__()

    def forward(self, W, h, q, tol=1e-3):
        # had to increase tol to avoid numerical stability issues
        y, status, n_iters = linprog_batch_std(q, W, h, tol=tol, cholesky=True)
        assert (status == 0).all()
        return y


class SolverVAE(nn.Module):
    def __init__(self, y_dim: int, x_dim: int, constr_dim: int, max_hidden_dim: int, solver: str) -> None:
        super(SolverVAE, self).__init__()

        self.prior_net = Encoder(x_dim, y_dim, constr_dim, max_hidden_dim)
        self.recognition_net = Encoder(y_dim + x_dim, y_dim, constr_dim, max_hidden_dim)
        if solver == 'cvx':
            self.generation_net = CvxSolver(y_dim, constr_dim)
        elif solver == 'lin':
            self.generation_net = LinSolver(y_dim, constr_dim)
        else:
            raise ValueError(f'unknown solver {solver}.')

    def sample(self, x: torch.Tensor):
        # first, get the conditioned latent distribution p(z|x)
        W, h, q = self.prior_net(x)

        # W must also have at least a single one in each column, otherwise the problem is degenerate (i.e. the
        # constraint is meaningless if it is not related to any variables). Right now, we resolve this by setting the
        # last part of W (the part that usually corresponds to the slack variables) to the identity matrix.
        # TODO: see if we can remove this
        learnable_W = W.rsample(tau=1)
        I = torch.eye(learnable_W.size(1), device=learnable_W.device)
        Is = I.unsqueeze(0).expand(learnable_W.size(0), -1, -1)
        W_sample = torch.cat((learnable_W, Is), dim=2)

        h_sample = h.rsample().clamp(1e-3) # must be greater than 0 or the problem is unfeasible
        q_sample = q.rsample().clamp(1e-3) # must be greater than 0 or the problem is unbounded

        # and try to reconstruct the observation p(y|x,z)
        y_sample = self.generation_net(W_sample, h_sample, q_sample)
        Q_sample = torch.bmm(q_sample.unsqueeze(1), y_sample.unsqueeze(2)).squeeze()
        return (W, h, q), (x, y_sample, W_sample, h_sample, q_sample, Q_sample)

    def forward(self, y: torch.Tensor, x: torch.Tensor):
        # get prior from condition p(z|x), and try to reconstruct observation
        priors, samples = self.sample(x)
        # now, we need to get the posterior q(z|x,y)
        obs = torch.cat([y, x], dim=1)
        posteriors = self.recognition_net(obs)

        return priors, posteriors, samples
