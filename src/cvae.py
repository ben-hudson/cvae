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
        self.prior_net = MLP(self.context_dim, [64, 64, 64, 64, self.latent_dim*2], activation_layer=torch.nn.SiLU)
        # p_\theta(decision|context,latent) - takes latent samples and reconstructs them into decisions
        # in our case, the decision only depends on the latents so we ignore the context here
        # TODO: clarify this
        # self.generation_net = MLP(self.latent_dim, [64, 64, 64, 64, self.obs_dim], activation_layer=torch.nn.SiLU)
        self.generation_net = MLP(self.context_dim + self.latent_dim, [64, 64, 64, 64, self.obs_dim], activation_layer=torch.nn.SiLU)
        # q_\phi(latent|context,decision) - latent samples come from here
        # this is the one we sample from, and then put IT in to the generation net
        self.recognition_net = MLP(self.context_dim + self.obs_dim, [64, 64, 64, 64, self.latent_dim*2], activation_layer=torch.nn.SiLU)

    def forward(self, context: torch.Tensor, obs: torch.Tensor, eps: float=1e-6):
        prior_loc, prior_logvar = self.prior_net(context).chunk(2, dim=-1)
        prior_var = torch.exp(prior_logvar) + eps
        prior_scale = torch.sqrt(prior_var)

        posterior_loc, posterior_logvar = self.recognition_net(torch.cat([context, obs], dim=-1)).chunk(2, dim=-1)
        posterior_var = torch.exp(posterior_logvar) + eps
        posterior_scale = torch.sqrt(posterior_var)

        if self.latent_dist == "normal":
            prior = Normal(prior_loc, prior_scale)
            posterior = Normal(posterior_loc, posterior_scale)

        else:
            raise ValueError(f"unsupported latent distribution {self.latent_dist}")

        latents_hat = posterior.rsample()
        # obs_hat = self.generation_net(latents_hat)
        obs_hat = self.generation_net(torch.cat([context, latents_hat], dim=-1))

        return prior, posterior, latents_hat, obs_hat
