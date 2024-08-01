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
        self.samples_per_latent = 1 # TODO: remove this
        if self.latent_dist == "normal":
            self.params_per_latent = 2 # loc, scale
        elif self.latent_dist == "uniform":
            self.params_per_latent = 1 # we only try to learn (half)width
        else:
            raise ValueError(f"unsupported latent distribution {self.latent_dist}")

        # p_\theta(latent|context) - outputs the parameters of the latent distribution
        self.prior_net = MLP(self.context_dim, [64, 64, 64, 64, self.latent_dim*self.params_per_latent], activation_layer=torch.nn.SiLU)
        # p_\theta(decision|context,latent) - takes latent samples and reconstructs them into decisions
        # in our case, the decision only depends on the latents so we ignore the context here
        # TODO: clarify this
        # self.generation_net = MLP(self.latent_dim, [64, 64, 64, 64, self.obs_dim], activation_layer=torch.nn.SiLU)
        self.generation_net = MLP(self.context_dim + self.latent_dim*self.samples_per_latent, [64, 64, 64, 64, self.obs_dim], activation_layer=torch.nn.SiLU)
        # q_\phi(latent|context,decision) - latent samples come from here
        # this is the one we sample from, and then put IT in to the generation net
        self.recognition_net = MLP(self.context_dim + self.obs_dim, [64, 64, 64, 64, self.latent_dim*self.params_per_latent], activation_layer=torch.nn.SiLU)

    def forward(self, context: torch.Tensor, obs: torch.Tensor, eps: float=1e-6):
        prior_params = self.prior_net(context).chunk(self.params_per_latent, dim=-1)
        posterior_params = self.recognition_net(torch.cat([context, obs], dim=-1)).chunk(self.params_per_latent, dim=-1)

        if self.latent_dist == "normal":
            prior_loc, prior_logvar = prior_params[0], prior_params[1]
            prior_var = torch.exp(prior_logvar) + eps
            prior_scale = torch.sqrt(prior_var)
            prior = Normal(prior_loc, prior_scale)

            posterior_loc, posterior_logvar = posterior_params[0], posterior_params[1]
            posterior_var = torch.exp(posterior_logvar) + eps
            posterior_scale = torch.sqrt(posterior_var)
            posterior = Normal(posterior_loc, posterior_scale)

        elif self.latent_dist == "uniform":
            prior_halfwidth = torch.exp(prior_params[0]) + eps # not sure if exp is the best way to make positive, why not just softplus?
            prior = Uniform(1 - prior_halfwidth, 1 + prior_halfwidth)

            posterior_halfwidth = torch.exp(posterior_params[0]) + eps
            posterior = Uniform(1 - posterior_halfwidth, 1 + posterior_halfwidth)

        else:
            raise ValueError(f"unsupported latent distribution {self.latent_dist}")

        latents_hat = posterior.rsample((self.samples_per_latent, )) # returns shape (samples_per_latent, batch_size, latent_dim)
        latents_hat = latents_hat.permute(1, 2, 0).flatten(1) # reshape to (batch_size, latent_dim*samples_per_latent)
        # obs_hat = self.generation_net(latents_hat)
        obs_hat = self.generation_net(torch.cat([context, latents_hat], dim=-1))
        obs_hat = F.sigmoid(obs_hat)

        return prior, posterior, latents_hat, obs_hat
