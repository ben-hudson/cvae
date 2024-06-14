import torch

from torchvision.ops import MLP

class DeterministicLP(torch.nn.Module):
    def __init__(self, context_dim, decision_dim, constr_dim):
        super().__init__()
        self.context_dim = context_dim

        self.u_dim = context_dim
        self.A_dim = (constr_dim, decision_dim)
        self.b_dim = constr_dim
        self.c_dim = decision_dim
        self.x_dim = decision_dim

        latent_dim = self.A_dim[0]*self.A_dim[1] + self.b_dim + self.c_dim
        self.encoder = MLP(self.u_dim, [32, 32, 64, 64, 128, 128, latent_dim], activation_layer=torch.nn.SiLU)
        self.decoder = MLP(latent_dim, [128, 128, 64, 64, 32, 32, self.x_dim], activation_layer=torch.nn.SiLU)

    def forward(self, u, nonneg=False):
        latents = self.encoder(u)
        if nonneg:
            latents = torch.nn.functional.relu(latents)

        A, b, c = torch.split(latents, [self.A_dim[0]*self.A_dim[1], self.b_dim, self.c_dim], dim=1)
        A = A.reshape(-1, *self.A_dim)

        x = self.decoder(latents)
        if nonneg:
            x = torch.nn.functional.relu(x)

        return c, A, b, x
