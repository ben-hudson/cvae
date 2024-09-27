import pyepo
import torch
import torch.nn.functional as F
import torch.distributions as D

from collections import namedtuple
from typing import Sequence

from utils.utils import norm, is_integer
from distributions import expectation, variance

EvalMetrics = namedtuple(
    "EvalMetrics",
    [
        "cost_err",
        "decision_err",
        "regret",
        "disappointment",
        "surprise",
        "obj_true",
        "obj_pred",
        "obj_expected",
        "obj_realized",
        "cost_dist_exp_err",
        "cost_dist_var_err",
    ],
)


def get_eval_metrics(
    cost_true: torch.Tensor,
    sol_true: torch.Tensor,
    cost_dist_true: D.Distribution,
    cost_pred: torch.Tensor,
    sol_pred: torch.Tensor,
    cost_dist_pred: D.Distribution,
    model_sense: int = pyepo.EPO.MINIMIZE,
    class_weights: Sequence = None,
):
    cost_true = norm(cost_true)
    cost_pred = norm(cost_pred)

    assert is_integer(sol_true), "we have made some assumptions that only hold for integer solutions"
    assert is_integer(sol_pred), "we have made some assumptions that only hold for integer solutions"

    cost_dist_exp_true = expectation(cost_dist_true)
    cost_dist_exp_pred = expectation(cost_dist_pred)
    cost_dist_var_true = variance(cost_dist_true)
    cost_dist_var_pred = variance(cost_dist_pred)

    # wait-and-see solution value with true cost realizations
    obj_true = torch.bmm(cost_true.unsqueeze(1), sol_true.unsqueeze(2)).squeeze(2)
    # predicted solution value with learned cost expectations
    obj_pred = torch.bmm(cost_pred.unsqueeze(1), sol_pred.unsqueeze(2)).squeeze(2)
    # predicted solution value with true cost realizations
    obj_realized = torch.bmm(cost_true.unsqueeze(1), sol_pred.unsqueeze(2)).squeeze(2)
    # predicted solution value with true cost expectations
    obj_expected = torch.bmm(cost_dist_exp_true.unsqueeze(1), sol_pred.unsqueeze(2)).squeeze(2)

    regret = obj_realized - obj_true
    # disappointment is positive when the realized objective is greater than the predicted one
    disappointment = (obj_realized - obj_expected).clamp(min=0)
    # surprise is positive when the realized objective is less than the predicted one (the predicted one is greater)
    surprise = (obj_expected - obj_realized).clamp(min=0)
    # these definitions flip when maximizing
    if model_sense == pyepo.EPO.MAXIMIZE:
        regret *= -1
        disappointment, surprise = surprise, disappointment

    cost_err = F.mse_loss(cost_pred, cost_true, reduction="mean")
    if class_weights is not None:
        sample_weights = torch.empty_like(sol_true)
        sample_weights[sol_true == 0] = class_weights[0]
        sample_weights[sol_true == 1] = class_weights[1]
    else:
        sample_weights = None
    decision_err = F.binary_cross_entropy(sol_pred, sol_true, weight=sample_weights, reduction="mean")

    # we compare cost distributions by their expectation and variance
    cost_dist_exp_err = F.mse_loss(cost_dist_exp_pred, cost_dist_exp_true, reduction="mean")
    cost_dist_var_err = F.mse_loss(cost_dist_var_pred, cost_dist_var_true, reduction="mean")

    return EvalMetrics(
        cost_err,
        decision_err,
        regret.mean(),
        disappointment.mean(),
        surprise.mean(),
        obj_true,
        obj_pred,
        obj_expected,
        obj_realized,
        cost_dist_exp_err,
        cost_dist_var_err,
    )
