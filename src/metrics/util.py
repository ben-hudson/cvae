import torch
import torch.nn.functional as F
import pyepo

from collections import namedtuple

ValMetrics = namedtuple("ValMetrics", "cost_err decision_err regret spo_loss abs_obj")


def get_val_metrics_sample(
    data_model,
    cost_true: torch.Tensor,
    sol_true: torch.Tensor,
    obj_true,
    cost_pred: torch.Tensor,
    sol_pred: torch.Tensor,
    obj_pred,
    is_integer: bool = False,
):
    obj_realized = torch.dot(cost_true, sol_pred)

    spo = obj_realized - obj_true
    regret = obj_realized - obj_pred
    if data_model.modelSense == pyepo.EPO.MAXIMIZE:
        spo *= -1
        regret *= -1

    cost_err = F.mse_loss(cost_pred, cost_true, reduction="mean")
    if is_integer:
        decision_err = F.binary_cross_entropy(sol_pred, sol_true, reduction="mean")
    else:
        decision_err = F.mse_loss(sol_pred, sol_true, reduction="mean")

    return ValMetrics(cost_err, decision_err, regret, spo, abs(obj_true))
