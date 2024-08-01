import torch

from sklearn.linear_model import LinearRegression

def r2(latents_hat: torch.Tensor, latents: torch.Tensor):
    latents_hat_np = latents_hat.cpu().numpy()
    latents_np = latents.cpu().numpy()

    linear_model = LinearRegression().fit(latents_hat_np, latents_np)
    r2 = linear_model.score(latents_hat_np, latents_np)

    return r2
