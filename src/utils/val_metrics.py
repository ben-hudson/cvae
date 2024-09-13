import pyepo
import pyepo.model
import torch
import torch.nn.functional as F
import torch.distributions as D

from collections import namedtuple
from typing import Sequence


ValMetrics = namedtuple("ValMetrics", "cost_err decision_err regret disappointment surprise obj_true obj_realized kld")


def get_val_metrics(
    cost_true: torch.Tensor,
    sol_true: torch.Tensor,
    obj_true: torch.Tensor,
    cost_dist_true: D.Distribution,
    cost_pred: torch.Tensor,
    sol_pred: torch.Tensor,
    obj_pred: torch.Tensor,
    cost_dist_pred: D.Distribution,
    model_sense: int = pyepo.EPO.MINIMIZE,
    is_integer: bool = False,
    class_weights: Sequence = None,
):
    obj_realized = torch.bmm(cost_true.unsqueeze(1), sol_pred.unsqueeze(2)).squeeze(2)

    regret = obj_realized - obj_true
    # disappointment is positive when the realized objective is greater than the predicted one
    disappointment = (obj_realized - obj_pred).clamp(min=0)
    # surprise is positive when the realized objective is less than the predicted one (the predicted one is greater)
    surprise = (obj_pred - obj_realized).clamp(min=0)
    # these definitions flip when maximizing
    if model_sense == pyepo.EPO.MAXIMIZE:
        regret *= -1
        disappointment, surprise = surprise, disappointment

    cost_err = F.mse_loss(cost_pred, cost_true, reduction="mean")
    if is_integer:
        if class_weights is not None:
            sample_weights = torch.empty_like(sol_true)
            sample_weights[sol_true == 0] = class_weights[0]
            sample_weights[sol_true == 1] = class_weights[1]
        else:
            sample_weights = None
        decision_err = F.binary_cross_entropy(sol_pred, sol_true, weight=sample_weights, reduction="mean")
    else:
        decision_err = F.mse_loss(sol_pred, sol_true, reduction="mean")

    kld = D.kl_divergence(cost_dist_true, cost_dist_pred)

    return ValMetrics(
        cost_err,
        decision_err,
        regret.mean(),
        disappointment.mean(),
        surprise.mean(),
        obj_true,
        obj_realized,
        kld.mean(),
    )
