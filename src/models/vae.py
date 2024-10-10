import torch
import torch.distributions as D
import torch.nn as nn

from typing import Callable, Tuple, Union


class VAE(nn.Module):
    def __init__(
        self,
        prior: D.Distribution,
        generation_model: nn.Module,
        recognition_model: nn.Module,
        reconstruction_loss: Union[Callable, nn.Module],
    ) -> None:
        super().__init__()

        # returns the prior distribution p(z)
        self.prior = prior

        # returns the parameters of the generative distribution p(x|z)
        self.generation_model = generation_model

        # estimates the parameters of the (approximate) posterior distribution p(z|x)
        self.recognition_model = recognition_model

        # how to evaluate -log p(x|z)
        self.reconstruction_loss = reconstruction_loss

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        posterior = self.recognition_model(obs)

        kld = D.kl_divergence(posterior, self.prior).mean()

        latent_sample = posterior.rsample()
        obs_hat = self.generation_model(latent_sample)
        loss = self.reconstruction_loss(obs_hat, obs, reduction="mean")

        return kld, loss

    def sample(self) -> torch.Tensor:
        latent_sample = self.prior.sample()
        obs_hat = self.generation_model(latent_sample)
        return obs_hat
