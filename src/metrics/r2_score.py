import numpy as np
import torch

from sklearn.linear_model import LinearRegression

# https://pytorch.org/ignite/generated/ignite.metrics.regression.R2Score.html only supports a single output
# It is possible to implement what we want with Ignite metrics, but way more complicated than this
def r2_score(x: torch.Tensor, y: torch.Tensor) -> float:
    fit = LinearRegression(fit_intercept=True).fit(x, y)
    score = fit.score(x, y)

    return score