import pyepo
import pyepo.model
import torch
import torch.nn.functional as F
import torch.distributions as D

from collections import namedtuple

ValMetrics = namedtuple("ValMetrics", "cost_err decision_err regret disappointment surprise obj_true obj_realized kld")


def get_sample_val_metrics(
    data_model: pyepo.model.opt.optModel,
    cost_true: torch.Tensor,
    sol_true: torch.Tensor,
    obj_true: float,
    cost_dist_true: D.Distribution,
    cost_pred: torch.Tensor,
    sol_pred: torch.Tensor,
    obj_pred: float,
    cost_dist_pred: D.Distribution,
    is_integer: bool = False,
):
    obj_realized = torch.dot(cost_true, sol_pred)

    regret = obj_realized - obj_true
    # disappointment is positive when the realized objective is greater than the predicted one
    disappointment = (obj_realized - obj_pred).clamp(min=0)
    # surprise is positive when the realized objective is less than the predicted one (the predicted one is greater)
    surprise = (obj_pred - obj_realized).clamp(min=0)
    # these definitions flip when maximizing
    if data_model.modelSense == pyepo.EPO.MAXIMIZE:
        regret *= -1
        disappointment, surprise = surprise, disappointment

    cost_err = F.mse_loss(cost_pred, cost_true, reduction="mean").item()
    if is_integer:
        decision_err = F.binary_cross_entropy(sol_pred, sol_true, reduction="mean").item()
    else:
        decision_err = F.mse_loss(sol_pred, sol_true, reduction="mean").item()

    kld = D.kl_divergence(cost_dist_true, cost_dist_pred).mean()

    return ValMetrics(
        cost_err,
        decision_err,
        regret,
        disappointment,
        surprise,
        abs(obj_true),
        abs(obj_realized),
        kld,
    )
