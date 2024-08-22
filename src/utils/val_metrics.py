import pyepo
import pyepo.model
import torch
import torch.nn.functional as F

from collections import namedtuple

ValMetrics = namedtuple("ValMetrics", "cost_err decision_err regret spo_loss obj_true obj_realized success")


def get_sample_val_metrics(
    data_model: pyepo.model.opt.optModel,
    cost_true: torch.Tensor,
    sol_true: torch.Tensor,
    obj_true: float,
    cost_pred: torch.Tensor,
    sol_pred: torch.Tensor,
    obj_pred: float,
    is_integer: bool = False,
):
    obj_realized = torch.dot(cost_true, sol_pred).item()

    spo_loss = obj_realized - obj_true
    regret = obj_realized - obj_pred
    if data_model.modelSense == pyepo.EPO.MAXIMIZE:
        spo_loss *= -1
        regret *= -1

    cost_err = F.mse_loss(cost_pred, cost_true, reduction="mean")
    if is_integer:
        decision_err = F.binary_cross_entropy(sol_pred, sol_true, reduction="mean")
    else:
        decision_err = F.mse_loss(sol_pred, sol_true, reduction="mean")

    return ValMetrics(cost_err, decision_err, regret, spo_loss, abs(obj_true), abs(obj_realized), 1.0)
