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
        if self.latent_dist == "normal":
            self.params_per_latent = 2 # loc, scale
        elif self.latent_dist == "uniform":
            self.params_per_latent = 1 # we only try to learn (half)width
        else:
            raise ValueError(f"unsupported latent distribution {self.latent_dist}")

        # p_\theta(latent|context) - outputs the parameters of the latent distribution
        self.prior_net = MLP(self.context_dim, [64, 64, 64, self.latent_dim*self.params_per_latent], activation_layer=torch.nn.SiLU)
        # p_\theta(decision|context,latent) - takes latent samples and reconstructs them into decisions
        # in our case, the decision only depends on the latents so we ignore the context here
        # TODO: clarify this
        self.generation_net = MLP(self.context_dim + self.latent_dim, [64, 64, 64, self.obs_dim], activation_layer=torch.nn.SiLU)
        # q_\phi(latent|context,decision) - latent samples come from here
        # this is the one we sample from, and then put IT in to the generation net
        self.recognition_net = MLP(self.context_dim + self.obs_dim, [64, 64, 64, self.latent_dim*self.params_per_latent], activation_layer=torch.nn.SiLU)

    def _get_normal(self, loc: torch.Tensor, logvar: torch.Tensor, eps: float=1e-6):
        var = torch.exp(logvar) + eps
        scale = torch.sqrt(var)
        return Normal(loc, scale)

    def _get_uniform(self, loc: torch.Tensor, halfwidth: torch.Tensor, eps: float=1e-6):
        halfwidth += eps
        return Uniform(loc - halfwidth, loc + halfwidth)

    def _generate_obs(self, context: torch.Tensor, latents: torch.Tensor):
        obs_hat = self.generation_net(torch.cat([context, latents], dim=-1))
        return F.sigmoid(obs_hat)

    def forward(self, context: torch.Tensor, obs: torch.Tensor):
        prior_params = self.prior_net(context).chunk(self.params_per_latent, dim=-1)
        posterior_params = self.recognition_net(torch.cat([context, obs], dim=-1)).chunk(self.params_per_latent, dim=-1)

        if self.latent_dist == "normal":
            prior = self._get_normal(prior_params[0], prior_params[1])
            posterior = self._get_normal(posterior_params[0], posterior_params[1])

        elif self.latent_dist == "uniform":
            prior = self._get_uniform(1, prior_params[0])
            posterior = self._get_uniform(1, posterior_params[0])

        else:
            raise ValueError(f"unsupported latent distribution {self.latent_dist}")

        latents_hat = posterior.rsample()
        obs_hat = self._generate_obs(context, latents_hat)

        return prior, posterior, latents_hat, obs_hat

    def sample(self, context: torch.Tensor, mean: bool=False):
        prior_params = self.prior_net(context).chunk(self.params_per_latent, dim=-1)

        if self.latent_dist == "normal":
            prior = self._get_normal(prior_params[0], prior_params[1])
        else:
            raise ValueError(f"unsupported latent distribution {self.latent_dist}")

        if mean:
            latents_hat = prior.loc
        else:
            latents_hat = prior.sample()
        obs_hat = self._generate_obs(context, latents_hat)

        return prior, latents_hat, obs_hat
