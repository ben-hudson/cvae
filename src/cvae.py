import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Uniform, Normal
from torchvision.ops import MLP

class CVAE(nn.Module):
    def __init__(self, context_dim: int, obs_dim: int, latent_dim: int, latent_dist: str = "normal") -> None:
        super(CVAE, self).__init__()

        self.context_dim = context_dim
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.latent_dist = latent_dist

        # p_\theta(latent|context) - outputs the parameters of the latent distribution
        self.prior_net = MLP(self.context_dim, [128, 128, 64, 64, 32, 32, self.latent_dim*2], activation_layer=torch.nn.SiLU)
        # p_\theta(decision|context,latent) - takes latent samples and reconstructs them into decisions
        self.generation_net = MLP(self.context_dim + self.latent_dim, [32, 32, 64, 64, 128, 128, self.obs_dim], activation_layer=torch.nn.SiLU)
        # q_\phi(latent|context,decision) - latent samples come from here
        # this is the one we sample from, and then put IT in to the generation net
        self.recognition_net = MLP(self.context_dim + self.obs_dim, [128, 128, 64, 64, 32, 32, self.latent_dim*2], activation_layer=torch.nn.SiLU)

    def forward(self, context: torch.Tensor, obs: torch.Tensor, eps: float=1e-6):
        prior_loc, prior_scale = self.prior_net(context).chunk(2, dim=-1)
        prior_scale = F.softplus(prior_scale) + eps

        posterior_loc, posterior_scale = self.recognition_net(torch.cat([context, obs], dim=-1)).chunk(2, dim=-1)
        posterior_scale = F.softplus(posterior_scale) + eps

        if self.latent_dist == "uniform":
            prior = Uniform(prior_loc, prior_loc + prior_scale)
            posterior = Uniform(posterior_loc, posterior_loc + posterior_scale)

        elif self.latent_dist == "normal":
            prior = Normal(prior_loc, prior_scale)
            posterior = Normal(posterior_loc, posterior_scale)

        else:
            raise ValueError(f"unsupported latent distribution {self.latent_dist}")

        latents = posterior.rsample()
        obs_hat = self.generation_net(torch.cat([context, latents], dim=-1))

        return prior, posterior, obs_hat
