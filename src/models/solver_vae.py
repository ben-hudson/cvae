import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Bernoulli, kl_divergence

from ilop.linprog_solver import linprog_batch_std

class Encoder(nn.Module):
    def __init__(self, input_dim, cost_dim, constr_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)

        # A is a d_y*d_h binary matrix
        # we model A_ij with a bernoulli distribution
        self.W_prob = nn.Linear(hidden_dim_2, cost_dim*constr_dim)

        self.h_mean = nn.Linear(hidden_dim_2, constr_dim)
        self.h_logvar = nn.Linear(hidden_dim_2, constr_dim)

        self.q_mean = nn.Linear(hidden_dim_2, cost_dim)
        self.q_logvar = nn.Linear(hidden_dim_2, cost_dim)

    def forward(self, x: torch.Tensor, eps: float = 1e-8):
        hidden = F.silu(self.fc1(x))
        hidden = F.silu(self.fc2(hidden))

        W = Bernoulli(self.W_prob(hidden))
        h = Normal(loc=self.h_mean(hidden), scale=torch.exp(self.h_logvar(hidden) + eps))
        q = Normal(loc=self.q_mean(hidden), scale=torch.exp(self.q_logvar(hidden) + eps))

        return W, h, q

class Solver(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, b, c):
        x = linprog_batch_std(c, A, b, want_grad=self.training)
        return x


class SolverVAE(nn.Module):
    def __init__(self, y_dim: int, x_dim: int, constr_dim: int, hidden_dim: int) -> None:
        super(SolverVAE, self).__init__()
        self.prior_net = Encoder(x_dim, y_dim, constr_dim, hidden_dim, hidden_dim // 2)
        self.recognition_net = Encoder(y_dim + x_dim, y_dim, constr_dim, hidden_dim, hidden_dim // 2)
        self.generation_net = Solver()

    def sample(self, condition: torch.Tensor):
        # first, get the conditioned latent distribution p(z|x)
        W, h, q = self.prior_net(condition)
        # take some samples
        W_sample, h_sample, q_sample = W.rsample(), h.rsample(), q.rsample()
        # and try to reconstruct the observation p(y|x,z)
        y_sample = self.generation_net(W_sample, h_sample, q_sample)
        return (W, h, q), y_sample

    def forward(self, obs: torch.Tensor, condition: torch.Tensor, compute_loss: bool = True):
        # get prior from condition p(z|x), and try to reconstruct observation
        priors, obs_hat = self.sample(condition)
        # now, we need to get the posterior q(z|x,y)
        x = torch.cat([obs, condition], dim=1)
        posteriors = self.recognition_net(x)

        if compute_loss:
            # want reconstructed observation to be close to the observation
            mse = F.mse_loss(obs_hat, obs).sum()
            # want posterior to be close to the prior
            kld = torch.sum(kl_divergence(p, q) for p, q in zip(priors, posteriors))
            return obs_hat, mse, kld

        else:
            return obs_hat
