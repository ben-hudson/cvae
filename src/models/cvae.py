import torch
import torch.nn as nn

from torchvision.ops import MLP
from .solver_vae import Encoder # TODO: move this to common file
from collections import namedtuple

CVAEDist = namedtuple("CVAEDist", "W h q")
CVAESample = namedtuple("CVAESample", "y W h q")

class CVAE(nn.Module):
    def __init__(self, y_dim: int, x_dim: int, constr_dim: int, max_hidden_dim: int) -> None:
        super(CVAE, self).__init__()
        enc_hidden_dims = [max_hidden_dim // 2, max_hidden_dim]
        dec_hidden_dims = [max_hidden_dim, max_hidden_dim // 2]
        self.prior_net = Encoder(x_dim, y_dim, constr_dim, enc_hidden_dims)
        self.recognition_net = Encoder(y_dim + x_dim, y_dim, constr_dim, enc_hidden_dims)
        self.generation_net = MLP(y_dim*constr_dim + y_dim + constr_dim, dec_hidden_dims + [y_dim], activation_layer=nn.SiLU)

    def sample(self, x: torch.Tensor):
        # first, get the conditioned latent distribution p(z|x)
        prior = CVAEDist(*self.prior_net(x))

        # W must also have at least a single one in each column, otherwise the problem is degenerate (i.e. the
        # constraint is meaningless if it is not related to any variables). Right now, we resolve this by setting the
        # last part of W (the part that usually corresponds to the slack variables) to the identity matrix.
        # TODO: see if we can remove this with constrained opt approach
        learnable_W = prior.W.rsample(tau=1)
        I = torch.eye(learnable_W.size(1), device=learnable_W.device)
        Is = I.unsqueeze(0).expand(learnable_W.size(0), -1, -1)
        W = torch.cat((learnable_W, Is), dim=2)

        h = prior.h.rsample() # must be greater than 0 or the problem is unfeasible
        q = prior.q.rsample() # must be greater than 0 or the problem is unbounded

        # and try to reconstruct the observation p(y|x,z)
        z = torch.cat([W.flatten(1), h, q], dim=1)
        y = self.generation_net(z)
        sample = CVAESample(y, W, h, q)

        return prior, sample

    def forward(self, y: torch.Tensor, x: torch.Tensor):
        # get prior from condition p(z|x), and try to reconstruct observation
        prior, sample = self.sample(x)
        # now, we need to get the posterior q(z|x,y)
        obs = torch.cat([y, x], dim=1)
        posterior = CVAEDist(*self.recognition_net(obs))

        return prior, posterior, sample
