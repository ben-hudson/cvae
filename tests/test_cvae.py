import os
import pathlib
import torch
import torch.nn as nn
import tqdm
import wandb

from collections import defaultdict
from ignite.metrics import Average
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.ops import MLP
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from torchvision.utils import make_grid

from models.cvae import CVAE
from utils.wandb import record_metrics


class Encoder(nn.Module):
    def __init__(self, obs_dim, context_dim, hidden_dims, latent_dim):
        super().__init__()
        self.params_per_latent = 2
        obs_dim_flat = obs_dim[0] * obs_dim[1]
        self.mlp = MLP(
            obs_dim_flat + context_dim,
            hidden_dims + [latent_dim * self.params_per_latent],
            activation_layer=nn.LeakyReLU,
        )

    def forward(self, obs: torch.Tensor, context: torch.Tensor):
        x = torch.cat([obs.flatten(1), context], dim=-1)
        params = self.mlp(x)
        mean, logvar = params.chunk(self.params_per_latent, dim=-1)
        var = logvar.exp() + 1e-6
        return Normal(mean, var.sqrt())


class Decoder(nn.Module):
    def __init__(self, latent_dim, context_dim, hidden_dims, obs_dim):
        super().__init__()
        obs_dim_flat = obs_dim[0] * obs_dim[1]
        self.mlp = MLP(
            latent_dim + context_dim,
            hidden_dims + [obs_dim_flat],
            activation_layer=nn.LeakyReLU,
        )
        self.unflatten = nn.Unflatten(-1, obs_dim)

    def forward(self, latents: torch.Tensor, context: torch.Tensor):
        x = torch.cat([latents, context], dim=-1)
        obs_hat = self.unflatten(self.mlp(x))
        return nn.functional.sigmoid(obs_hat)


def train_step(model: nn.Module, obs: torch.Tensor, context: torch.Tensor, kld_weight: float):
    kld, reconstruction_loss = model(obs, context)
    # maximize elbo = minimize -elbo = minimize kld + nll
    loss = kld_weight * kld + (1 - kld_weight) * reconstruction_loss
    metrics = {"kld": kld.detach(), "reconstruction_loss": reconstruction_loss.detach(), "loss": loss.detach()}
    return loss, metrics


if __name__ == "__main__":
    run = wandb.init(project="test_cvae")

    tmp_dir = pathlib.Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    batch_size = 128
    img_transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    train_loader = DataLoader(
        MNIST(tmp_dir / "MNIST", download=True, train=True, transform=img_transform),
        batch_size=batch_size,
        shuffle=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    obs_dim = (28, 28)
    num_classes = 10
    latent_dim = 2
    # standard normal
    prior = Normal(torch.zeros(latent_dim, device=device), torch.ones(latent_dim, device=device))
    # the encoder goes from observation to latent "codes"
    encoder = Encoder(obs_dim, num_classes, [512, 256, 128, 64], latent_dim)
    # and the decoder goes back
    decoder = Decoder(latent_dim, num_classes, [64, 128, 256, 512], obs_dim)
    model = CVAE(prior, decoder, encoder, nn.functional.binary_cross_entropy)
    model.to(device)

    kld_weight = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = defaultdict(Average)

    for epoch in tqdm.trange(100):
        for obs, labels in train_loader:
            model.train()
            optimizer.zero_grad()

            obs = obs.squeeze().to(device)
            context = nn.functional.one_hot(labels, num_classes).to(device)
            loss, train_metrics = train_step(model, obs, context, kld_weight)

            loss.backward()
            optimizer.step()

            for name, value in train_metrics.items():
                metrics["train/" + name].update(value)

        # for the eval step we will sample the latent space in a grid
        model.eval()
        with torch.no_grad():
            for label in range(num_classes):
                latent_coords = torch.linspace(-2, 2, 8, device=device)
                latent_coords_ij = torch.meshgrid(latent_coords, latent_coords, indexing="ij")
                latent_grid = torch.stack(latent_coords_ij, dim=-1)
                latents = latent_grid.view(-1, 2)

                context = nn.functional.one_hot(torch.as_tensor(label), num_classes)
                context = context.expand((latents.size(0), -1))

                obs = model.generation_model(latents, context)
                obs_grid = make_grid(obs.unsqueeze(1))
                wandb.log({f"{label}_samples": wandb.Image(obs_grid)}, step=epoch)

        record_metrics(metrics, epoch)
