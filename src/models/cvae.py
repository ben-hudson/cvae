import torch
import torch.distributions as D

from typing import Tuple

from .vae import VAE


class CVAE(VAE):
    def forward(self, obs: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        posterior = self.recognition_model(obs, context)

        kld = D.kl_divergence(posterior, self.prior).mean()

        latent_sample = posterior.rsample()
        obs_hat = self.generation_model(latent_sample, context)
        loss = self.reconstruction_loss(obs_hat, obs, reduction="mean")

        return kld, loss

    def sample(self, context: torch.Tensor) -> torch.Tensor:
        latent_sample = self.prior.sample()
        obs_hat = self.generation_model(latent_sample, context)
        return obs_hat
