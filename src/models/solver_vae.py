import pyepo.func.abcmodule
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import kl_divergence, Normal
from torchvision.ops import MLP
from typing import Dict, Sequence, Tuple

from utils.utils import norm


class SolverVAE(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        decision_dim: int,
        cost_dim: int,
        mlp_hidden_dim: int,
        mlp_hidden_layers: int,
    ) -> None:
        super().__init__()

        self.feat_dim = feat_dim
        self.decision_dim = decision_dim
        self.cost_dim = cost_dim
        self.params_per_cost = 2  # loc, scale

        hidden_layers = [mlp_hidden_dim] * mlp_hidden_layers
        # p_\theta(latent|context) - outputs the parameters of the latent distribution
        self.prior_net = MLP(
            self.feat_dim,
            hidden_layers + [self.cost_dim * self.params_per_cost],
            activation_layer=torch.nn.SiLU,
        )
        # p_\theta(decision|context,latent) - takes latent samples and reconstructs them into decisions
        # this is the perturb-and-MAP solver, which perturbs the input to the solver to get a "smoothed" solution

        # q_\phi(latent|context,decision) - latent samples come from here
        # this is the one we sample from, and then put IT in to the generation net
        self.recognition_net = MLP(
            self.feat_dim + self.decision_dim,
            hidden_layers + [self.cost_dim * self.params_per_cost],
            activation_layer=torch.nn.SiLU,
        )

    def _get_normal(self, mean: torch.Tensor, logvar: torch.Tensor, eps: float = 1e-6) -> Normal:
        var = torch.exp(logvar) + eps
        std = torch.sqrt(var)
        return Normal(mean, std)

    def forward(
        self,
        context: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prior_mean, prior_logvar = self.prior_net(context).chunk(self.params_per_cost, dim=-1)
        posterior_mean, posterior_logvar = self.recognition_net(torch.cat([context, obs], dim=-1)).chunk(
            self.params_per_cost, dim=-1
        )

        prior = self._get_normal(prior_mean, prior_logvar)
        posterior = self._get_normal(posterior_mean, posterior_logvar)

        return prior, posterior

    def predict(
        self,
        context: torch.Tensor,
        point_prediction: bool = True,
    ) -> torch.Tensor:
        prior_mean, prior_logvar = self.prior_net(context).chunk(self.params_per_cost, dim=-1)

        if point_prediction:
            return norm(prior_mean)
        else:
            prior = self._get_normal(prior_mean, prior_logvar)
            return prior
